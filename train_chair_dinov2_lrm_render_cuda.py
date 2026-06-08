#!/usr/bin/env python3
"""
DINOv2 LRM-style render-supervised chair reconstruction.

This is the final architecture path for the Blender chair dataset:

  source RGB view -> pretrained DINOv2 encoder -> learned triplanes
  -> radiance/density decoder -> differentiable volume rendering
  -> target RGB/mask view losses.

It deliberately avoids PyTorch3D/nvdiffrast so it can run on a fresh rented
CUDA server with a normal PyTorch install.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


BOX_MIN = (-1.05, -1.05, -0.08)
BOX_MAX = (1.05, 1.05, 1.92)


def ensure_deps(skip_install: bool) -> None:
    if skip_install:
        return
    pkgs = [
        "torch",
        "torchvision",
        "Pillow",
        "numpy",
        "scikit-image",
        "trimesh",
        "tqdm",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore", *pkgs], check=True)


def read_excluded_uids(path: str) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    values = set()
    for line in p.read_text(encoding="utf-8").replace(",", "\n").splitlines():
        uid = line.strip()
        if uid:
            values.add(uid)
    return values


def read_dataset(dataset_root: Path, exclude_uids: set[str]) -> Dict[str, List[dict]]:
    views_csv = dataset_root / "metadata" / "views.csv"
    if not views_csv.exists():
        raise FileNotFoundError(f"Missing {views_csv}")
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with open(views_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["uid"]
            if uid in exclude_uids:
                continue
            view = int(row.get("view", row.get("view_index", 0)))
            image_path = dataset_root / row.get("image_path", f"renders/{uid}/view_{view:03d}.png")
            mask_path = dataset_root / row.get("mask_path", f"masks/{uid}/view_{view:03d}.png")
            camera_path = dataset_root / row.get("camera_path", f"cameras/{uid}/view_{view:03d}.json")
            mesh_path = dataset_root / "objects" / uid / "normalized.glb"
            points_path = dataset_root / "objects" / uid / "points.npz"
            if image_path.exists() and mask_path.exists() and camera_path.exists() and mesh_path.exists() and points_path.exists():
                grouped[uid].append(
                    {
                        "uid": uid,
                        "view": view,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "camera_path": str(camera_path),
                        "mesh_path": str(mesh_path),
                        "points_path": str(points_path),
                    }
                )
    grouped = {uid: sorted(rows, key=lambda x: x["view"]) for uid, rows in grouped.items() if len(rows) >= 2}
    if not grouped:
        raise RuntimeError(f"No usable multiview objects under {dataset_root}")
    return grouped


def split_uids(dataset_root: Path, grouped: Dict[str, List[dict]], seed: int, train_ratio: float, val_ratio: float):
    available = set(grouped.keys())
    splits_path = dataset_root / "metadata" / "splits.json"
    if splits_path.exists():
        raw = json.loads(splits_path.read_text(encoding="utf-8"))
        train = [uid for uid in raw.get("train", []) if uid in available]
        val = [uid for uid in raw.get("val", []) if uid in available]
        test = [uid for uid in raw.get("test", []) if uid in available]
        used = set(train) | set(val) | set(test)
        leftovers = sorted(available - used)
        train.extend(leftovers)
    else:
        uids = sorted(available)
        rng = random.Random(seed)
        rng.shuffle(uids)
        n_train = max(1, int(len(uids) * train_ratio))
        n_val = max(1, int(len(uids) * val_ratio))
        train = uids[:n_train]
        val = uids[n_train : n_train + n_val]
        test = uids[n_train + n_val :]
    if not val:
        val = train[-max(1, len(train) // 10) :]
        train = train[: -len(val)]
    return train, val, test


def stable_seed(text: str, seed: int) -> int:
    value = seed & 0xFFFFFFFF
    for ch in text:
        value = ((value * 131) + ord(ch)) & 0xFFFFFFFF
    return value


def load_source_image(image_path: str, mask_path: str, image_size: int, crop: bool):
    from PIL import Image, ImageOps

    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if crop:
        mask_arr = np.asarray(mask, dtype=np.uint8)
        if mask_arr.max() > 8:
            ys, xs = np.where(mask_arr > 8)
            pad = int(max(mask_arr.shape[:2]) * 0.08)
            box = (
                max(0, int(xs.min()) - pad),
                max(0, int(ys.min()) - pad),
                min(mask_arr.shape[1], int(xs.max()) + pad + 1),
                min(mask_arr.shape[0], int(ys.max()) + pad + 1),
            )
            img = img.crop(box)
    img = ImageOps.contain(img, (image_size, image_size))
    canvas = Image.new("RGB", (image_size, image_size), (219, 222, 224))
    canvas.paste(img, ((image_size - img.width) // 2, (image_size - img.height) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def load_rgb_mask(image_path: str, mask_path: str):
    from PIL import Image

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    return rgb, mask


def camera_rays(camera_json: str, xs: np.ndarray, ys: np.ndarray):
    cam = json.loads(Path(camera_json).read_text(encoding="utf-8"))
    intr = cam["intrinsics"]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    c2w = np.asarray(cam["camera_matrix_world"], dtype=np.float32).reshape(4, 4)
    rot = c2w[:3, :3]
    origin = c2w[:3, 3]
    dirs_cam = np.stack([(xs - cx) / fx, -(ys - cy) / fy, -np.ones_like(xs)], axis=-1).astype(np.float32)
    dirs_world = dirs_cam @ rot.T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-8
    origins = np.broadcast_to(origin.astype(np.float32), dirs_world.shape)
    return origins.astype(np.float32), dirs_world.astype(np.float32)


def sample_geometry_queries(points_path: str, rng, count: int):
    data = np.load(points_path)
    surface = np.asarray(data["points"], dtype=np.float32)
    if len(surface) == 0:
        raise RuntimeError(f"Empty points file: {points_path}")

    n_surface = count // 3
    n_near = count // 3
    n_uniform = count - n_surface - n_near

    surface_idx = rng.choice(len(surface), n_surface, replace=len(surface) < n_surface)
    surface_pts = surface[surface_idx]
    surface_target = np.ones((n_surface, 1), dtype=np.float32)

    near_idx = rng.choice(len(surface), n_near, replace=len(surface) < n_near)
    near_sigma = rng.choice(
        np.array([0.015, 0.035, 0.07], dtype=np.float32),
        size=n_near,
        p=[0.45, 0.35, 0.20],
    )
    near_pts = surface[near_idx] + rng.normal(size=(n_near, 3)).astype(np.float32) * near_sigma[:, None]
    near_dist = np.linalg.norm(near_pts - surface[near_idx], axis=1, keepdims=True)
    near_target = np.exp(-near_dist / 0.045).astype(np.float32)

    uniform_pts = np.empty((n_uniform, 3), dtype=np.float32)
    uniform_pts[:, 0] = rng.uniform(BOX_MIN[0], BOX_MAX[0], size=n_uniform)
    uniform_pts[:, 1] = rng.uniform(BOX_MIN[1], BOX_MAX[1], size=n_uniform)
    uniform_pts[:, 2] = rng.uniform(BOX_MIN[2], BOX_MAX[2], size=n_uniform)
    uniform_target = np.zeros((n_uniform, 1), dtype=np.float32)

    query = np.concatenate([surface_pts, near_pts, uniform_pts], axis=0)
    target = np.concatenate([surface_target, near_target, uniform_target], axis=0)
    order = rng.permutation(len(query))
    return query[order].astype(np.float32), target[order].astype(np.float32)


class RenderPairDataset:
    def __init__(self, grouped, uids, args, training: bool):
        self.grouped = grouped
        self.uids = list(uids)
        self.args = args
        self.training = training
        self.items = []
        for uid in self.uids:
            for target_idx in range(len(grouped[uid])):
                self.items.append((uid, target_idx))

    def __len__(self):
        return len(self.items)

    def _choose_source(self, rng, rows, target_idx):
        if self.training and rng.random() < self.args.same_view_prob:
            return target_idx
        choices = [i for i in range(len(rows)) if i != target_idx]
        if not choices:
            return target_idx
        return int(rng.choice(choices))

    def _sample_patch(self, rng, mask, patch_size):
        h, w = mask.shape
        if self.training and rng.random() < self.args.foreground_patch_prob and mask.max() > 8 / 255.0:
            ys, xs = np.where(mask > 8 / 255.0)
            center_idx = int(rng.integers(0, len(xs)))
            cx = int(xs[center_idx])
            cy = int(ys[center_idx])
            x0 = np.clip(cx - patch_size // 2, 0, max(0, w - patch_size))
            y0 = np.clip(cy - patch_size // 2, 0, max(0, h - patch_size))
        else:
            x0 = int(rng.integers(0, max(1, w - patch_size + 1)))
            y0 = int(rng.integers(0, max(1, h - patch_size + 1)))
        yy, xx = np.meshgrid(np.arange(y0, y0 + patch_size), np.arange(x0, x0 + patch_size), indexing="ij")
        return xx.reshape(-1).astype(np.float32) + 0.5, yy.reshape(-1).astype(np.float32) + 0.5, yy, xx

    def __getitem__(self, index):
        uid, target_idx = self.items[index]
        seed = stable_seed(f"{uid}:{target_idx}:{index}", self.args.seed + (0 if self.training else 100000))
        rng = np.random.default_rng(seed)
        rows = self.grouped[uid]
        source_idx = self._choose_source(rng, rows, target_idx) if self.training else 0
        if source_idx == target_idx and len(rows) > 1 and not self.training:
            source_idx = 1
        source = rows[source_idx]
        target = rows[target_idx]

        source_image = load_source_image(source["image_path"], source["mask_path"], self.args.image_size, self.args.crop)
        target_rgb, target_mask = load_rgb_mask(target["image_path"], target["mask_path"])

        xs, ys, yy, xx = self._sample_patch(rng, target_mask, self.args.patch_size)
        rays_o, rays_d = camera_rays(target["camera_path"], xs, ys)
        rgb_patch = target_rgb[yy, xx].reshape(-1, 3).astype(np.float32)
        mask_patch = target_mask[yy, xx].reshape(-1, 1).astype(np.float32)
        geo_query, geo_target = sample_geometry_queries(target["points_path"], rng, self.args.geometry_queries)

        return {
            "uid": uid,
            "source_image": source_image.astype(np.float32),
            "rays_o": rays_o,
            "rays_d": rays_d,
            "target_rgb": rgb_patch,
            "target_mask": mask_patch,
            "geo_query": geo_query,
            "geo_target": geo_target,
        }


def collate_batch(batch):
    import torch

    out = {}
    for key in ("source_image", "rays_o", "rays_d", "target_rgb", "target_mask", "geo_query", "geo_target"):
        out[key] = torch.from_numpy(np.stack([item[key] for item in batch], axis=0))
    out["uid"] = [item["uid"] for item in batch]
    return out


def build_model(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DINOv2Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.hub.load("facebookresearch/dinov2", args.dinov2_model)
            dims = {
                "dinov2_vits14": 384,
                "dinov2_vitb14": 768,
                "dinov2_vitl14": 1024,
                "dinov2_vitg14": 1536,
                "dinov2_vits14_reg": 384,
                "dinov2_vitb14_reg": 768,
                "dinov2_vitl14_reg": 1024,
                "dinov2_vitg14_reg": 1536,
            }
            in_dim = dims.get(args.dinov2_model, 1024)
            self.proj = nn.Sequential(
                nn.Linear(in_dim, args.latent_dim),
                nn.LayerNorm(args.latent_dim),
                nn.GELU(),
                nn.Linear(args.latent_dim, args.latent_dim),
            )

        def forward(self, image):
            mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (image - mean) / std
            feat = self.backbone(x)
            if isinstance(feat, dict):
                feat = feat.get("x_norm_clstoken", next(iter(feat.values())))
            return self.proj(feat)

    class TriplaneGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = args.plane_channels
            self.s = args.plane_size
            self.fc = nn.Sequential(nn.Linear(args.latent_dim, 3 * self.c * 8 * 8), nn.GELU())
            blocks = []
            size = 8
            while size < self.s:
                blocks += [
                    nn.ConvTranspose2d(self.c, self.c, 4, 2, 1),
                    nn.GroupNorm(min(16, self.c), self.c),
                    nn.SiLU(),
                    nn.Conv2d(self.c, self.c, 3, padding=1),
                    nn.GroupNorm(min(16, self.c), self.c),
                    nn.SiLU(),
                ]
                size *= 2
            self.up = nn.Sequential(*blocks)

        def forward(self, latent):
            b = latent.shape[0]
            x = self.fc(latent).view(b * 3, self.c, 8, 8)
            x = self.up(x)
            if x.shape[-1] != self.s or x.shape[-2] != self.s:
                x = F.interpolate(x, size=(self.s, self.s), mode="bilinear", align_corners=False)
            return x.view(b, 3, self.c, self.s, self.s)

    class RadianceDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            in_dim = args.plane_channels * 3 + 3
            layers = []
            dim = in_dim
            for _ in range(args.decoder_layers):
                layers += [nn.Linear(dim, args.decoder_hidden), nn.SiLU()]
                dim = args.decoder_hidden
            self.net = nn.Sequential(*layers)
            self.sigma = nn.Linear(dim, 1)
            self.rgb = nn.Linear(dim, 3)
            nn.init.constant_(self.sigma.bias, args.sigma_init_bias)

        def sample_planes(self, planes, pts):
            b, _, c, _, _ = planes.shape
            x = pts[..., 0].clamp(BOX_MIN[0], BOX_MAX[0]) / abs(BOX_MIN[0])
            y = pts[..., 1].clamp(BOX_MIN[1], BOX_MAX[1]) / abs(BOX_MIN[1])
            z = ((pts[..., 2].clamp(BOX_MIN[2], BOX_MAX[2]) - BOX_MIN[2]) / (BOX_MAX[2] - BOX_MIN[2])) * 2.0 - 1.0
            grids = [torch.stack([x, y], dim=-1), torch.stack([x, z], dim=-1), torch.stack([y, z], dim=-1)]
            feats = []
            for i, grid in enumerate(grids):
                sampled = F.grid_sample(
                    planes[:, i],
                    grid.view(b, -1, 1, 2),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                feats.append(sampled.squeeze(-1).transpose(1, 2))
            return torch.cat(feats + [pts], dim=-1)

        def forward(self, planes, pts):
            h = self.net(self.sample_planes(planes, pts))
            sigma = F.softplus(self.sigma(h) + args.sigma_activation_bias)
            rgb = torch.sigmoid(self.rgb(h))
            return rgb, sigma

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DINOv2Encoder()
            self.triplane = TriplaneGenerator()
            self.decoder = RadianceDecoder()

        def planes(self, image):
            return self.triplane(self.encoder(image))

        def forward(self, image):
            return self.planes(image)

    return Model()


def render_rays(model, planes, rays_o, rays_d, args, training: bool):
    import torch

    b, n, _ = rays_o.shape
    t_vals = torch.linspace(args.near, args.far, args.samples_per_ray, device=rays_o.device, dtype=rays_o.dtype)
    if training and args.ray_jitter:
        mids = 0.5 * (t_vals[:-1] + t_vals[1:])
        upper = torch.cat([mids, t_vals[-1:]], dim=0)
        lower = torch.cat([t_vals[:1], mids], dim=0)
        t_vals = lower + (upper - lower) * torch.rand((b, n, args.samples_per_ray), device=rays_o.device, dtype=rays_o.dtype)
    else:
        t_vals = t_vals.view(1, 1, -1).expand(b, n, -1)

    pts = rays_o[:, :, None, :] + rays_d[:, :, None, :] * t_vals[..., None]
    flat_pts = pts.reshape(b, n * args.samples_per_ray, 3)
    rgb, sigma = model.decoder(planes, flat_pts)
    rgb = rgb.reshape(b, n, args.samples_per_ray, 3)
    sigma = sigma.reshape(b, n, args.samples_per_ray, 1)

    box_min = rays_o.new_tensor(BOX_MIN).view(1, 1, 1, 3)
    box_max = rays_o.new_tensor(BOX_MAX).view(1, 1, 1, 3)
    inside = ((pts >= box_min) & (pts <= box_max)).all(dim=-1, keepdim=True)
    sigma = sigma * inside

    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    last = torch.full_like(deltas[..., :1], 1e10)
    deltas = torch.cat([deltas, last], dim=-1)[..., None]
    alpha = 1.0 - torch.exp(-sigma * deltas)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1, :]), 1.0 - alpha + 1e-6], dim=2), dim=2)[..., :-1, :]
    weights = alpha * trans
    color = (weights * rgb).sum(dim=2)
    acc = weights.sum(dim=2).clamp(0.0, 1.0)
    bg = rays_o.new_tensor(args.background).view(1, 1, 3)
    color = color + (1.0 - acc) * bg
    depth = (weights.squeeze(-1) * t_vals).sum(dim=2) / (acc.squeeze(-1) + 1e-6)
    return color, acc, depth, weights.squeeze(-1)


def render_losses(pred_rgb, pred_mask, pred_depth, weights, target_rgb, target_mask, args, stage=None):
    import torch
    import torch.nn.functional as F

    fg_weight = args.background_rgb_weight * (1.0 - target_mask) + args.foreground_rgb_weight * target_mask
    rgb_diff = torch.sqrt((pred_rgb - target_rgb).pow(2) + args.charbonnier_eps)
    rgb_loss = (rgb_diff * fg_weight).sum() / (fg_weight.sum() * pred_rgb.shape[-1] + 1e-6)

    mask_bce_weight = args.background_mask_weight * (1.0 - target_mask) + args.foreground_mask_weight * target_mask
    bce_raw = F.binary_cross_entropy(pred_mask.clamp(1e-4, 1.0 - 1e-4), target_mask, reduction="none")
    bce = (bce_raw * mask_bce_weight).sum() / (mask_bce_weight.sum() + 1e-6)
    inter = (pred_mask * target_mask).sum(dim=1)
    dice = 1.0 - (2.0 * inter + 1.0) / (pred_mask.sum(dim=1) + target_mask.sum(dim=1) + 1.0)
    mask_loss = bce + dice.mean()
    fg_denom = target_mask.sum(dim=1).clamp_min(1.0)
    fg_recall_loss = (((1.0 - pred_mask).clamp_min(0.0) * target_mask).sum(dim=1) / fg_denom).mean()

    opacity_loss = (pred_mask * (1.0 - target_mask)).mean()
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    mean_t = (weights * torch.arange(weights.shape[-1], device=weights.device, dtype=weights.dtype)).sum(dim=-1, keepdim=True) / weights_sum
    distortion = (weights * (torch.arange(weights.shape[-1], device=weights.device, dtype=weights.dtype) - mean_t).abs()).mean()

    stage = stage or {}
    mask_weight = args.mask_weight * stage.get("mask", 1.0)
    opacity_weight = args.opacity_weight * stage.get("opacity", 1.0)
    distortion_weight = args.distortion_weight * stage.get("distortion", 1.0)
    total = (
        rgb_loss
        + mask_weight * mask_loss
        + args.mask_recall_weight * stage.get("mask", 1.0) * fg_recall_loss
        + opacity_weight * opacity_loss
        + distortion_weight * distortion
    )
    return total, {
        "rgb": rgb_loss.detach(),
        "mask": mask_loss.detach(),
        "recall": fg_recall_loss.detach(),
        "opacity": opacity_loss.detach(),
        "distortion": distortion.detach(),
    }


def geometry_density_loss(model, planes, geo_query, geo_target):
    import torch
    import torch.nn.functional as F

    _, sigma = model.decoder(planes, geo_query)
    pred = 1.0 - torch.exp(-sigma * 0.08)
    bce = F.binary_cross_entropy(pred.float().clamp(1e-4, 1.0 - 1e-4), geo_target.float())
    surface = pred[geo_target > 0.95]
    empty = pred[geo_target < 0.05]
    surface_loss = (1.0 - surface).abs().mean() if surface.numel() else bce * 0.0
    empty_loss = empty.abs().mean() if empty.numel() else bce * 0.0
    return bce + 0.25 * surface_loss + 0.25 * empty_loss


def stage_weights(args, epoch: int) -> dict:
    if epoch <= args.geometry_warmup_epochs:
        return {
            "mask": args.warmup_mask_mult,
            "opacity": args.warmup_opacity_mult,
            "distortion": 1.0,
            "geometry": args.warmup_geometry_mult,
            "perceptual": 0.0,
        }
    return {
        "mask": 1.0,
        "opacity": 1.0,
        "distortion": 1.0,
        "geometry": 1.0,
        "perceptual": 1.0,
    }


def build_perceptual_model(args, device, rank: int):
    if args.perceptual_weight <= 0:
        return None
    try:
        import torch
        import torchvision

        weights = torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES
        model = torchvision.models.vgg16(weights=weights).features[:16].eval().to(device)
        for param in model.parameters():
            param.requires_grad_(False)
        if rank == 0:
            print("VGG perceptual loss enabled.", flush=True)
        return model
    except Exception as exc:
        if rank == 0:
            print(f"VGG perceptual loss disabled: {exc}", flush=True)
        return None


def perceptual_patch_loss(perceptual_model, pred_rgb, target_rgb, patch_size: int):
    if perceptual_model is None:
        return pred_rgb.sum() * 0.0
    import torch
    import torch.nn.functional as F

    b = pred_rgb.shape[0]
    pred = pred_rgb.view(b, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous()
    target = target_rgb.view(b, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous()
    if patch_size < 64:
        pred = F.interpolate(pred, size=(64, 64), mode="bilinear", align_corners=False)
        target = F.interpolate(target, size=(64, 64), mode="bilinear", align_corners=False)
    mean = pred.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = pred.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        target_f = perceptual_model((target - mean) / std)
    pred_f = perceptual_model((pred - mean) / std)
    return F.l1_loss(pred_f, target_f)


def set_encoder_trainable(model, trainable: bool):
    module = model.module if hasattr(model, "module") else model
    for p in module.encoder.backbone.parameters():
        p.requires_grad = trainable


def make_optimizer(model, args, encoder_trainable: bool):
    import torch

    module = model.module if hasattr(model, "module") else model
    head_params = list(module.encoder.proj.parameters()) + list(module.triplane.parameters()) + list(module.decoder.parameters())
    params = [{"params": head_params, "lr": args.lr}]
    if encoder_trainable and args.encoder_lr > 0:
        params.append({"params": [p for p in module.encoder.backbone.parameters() if p.requires_grad], "lr": args.encoder_lr})
    return torch.optim.AdamW(params, weight_decay=args.weight_decay)


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, best_val: float):
    import torch

    module = model.module if hasattr(model, "module") else model
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "args": vars(args),
            "best_val": best_val,
        },
        path,
    )


def load_checkpoint_if_requested(path: str, model, optimizer, device, rank: int):
    if not path:
        return 1, float("inf")
    import torch

    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    if rank == 0:
        print(f"resumed checkpoint: {ckpt_path} start_epoch={start_epoch} best_val={best_val:.6f}", flush=True)
    return start_epoch, best_val


def save_validation_preview(model, grouped, val_uids, args, work_dir: Path, epoch: int, device) -> None:
    if args.preview_every <= 0 or epoch % args.preview_every != 0 or not val_uids:
        return
    import torch
    from PIL import Image, ImageDraw

    module = model.module if hasattr(model, "module") else model
    module.eval()

    uid = val_uids[(epoch - 1) % len(val_uids)]
    rows = grouped[uid]
    if len(rows) < 2:
        return
    source = rows[0]
    target = rows[0] if args.preview_same_view else rows[1]

    source_np = load_source_image(source["image_path"], source["mask_path"], args.image_size, args.crop)
    target_rgb, target_mask = load_rgb_mask(target["image_path"], target["mask_path"])
    h, w = target_mask.shape
    preview_size = args.preview_size
    xs, ys = np.meshgrid(
        (np.arange(preview_size, dtype=np.float32) + 0.5) * (w / preview_size),
        (np.arange(preview_size, dtype=np.float32) + 0.5) * (h / preview_size),
        indexing="xy",
    )
    rays_o, rays_d = camera_rays(target["camera_path"], xs.reshape(-1), ys.reshape(-1))

    source_t = torch.from_numpy(source_np[None]).to(device)
    rays_o_t = torch.from_numpy(rays_o[None]).to(device)
    rays_d_t = torch.from_numpy(rays_d[None]).to(device)

    pred_rgb_parts = []
    pred_mask_parts = []
    with torch.no_grad():
        planes = module.planes(source_t)
        for start in range(0, rays_o_t.shape[1], args.preview_ray_chunk):
            ro = rays_o_t[:, start : start + args.preview_ray_chunk]
            rd = rays_d_t[:, start : start + args.preview_ray_chunk]
            rgb, mask, _, _ = render_rays(module, planes, ro, rd, args, training=False)
            pred_rgb_parts.append(rgb.float().cpu())
            pred_mask_parts.append(mask.float().cpu())

    pred_rgb = torch.cat(pred_rgb_parts, dim=1).numpy()[0].reshape(preview_size, preview_size, 3)
    pred_mask = torch.cat(pred_mask_parts, dim=1).numpy()[0].reshape(preview_size, preview_size)

    source_img = Image.fromarray(np.clip(np.transpose(source_np, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)).resize((preview_size, preview_size))
    target_img = Image.fromarray(np.clip(target_rgb * 255, 0, 255).astype(np.uint8)).resize((preview_size, preview_size))
    pred_img = Image.fromarray(np.clip(pred_rgb * 255, 0, 255).astype(np.uint8))
    target_mask_img = Image.fromarray(np.clip(target_mask * 255, 0, 255).astype(np.uint8)).resize((preview_size, preview_size))
    pred_mask_img = Image.fromarray(np.clip(pred_mask * 255, 0, 255).astype(np.uint8))

    labels = ["source", "target", "pred", "target_mask", "pred_mask"]
    images = [source_img, target_img, pred_img, target_mask_img.convert("RGB"), pred_mask_img.convert("RGB")]
    label_h = 24
    sheet = Image.new("RGB", (preview_size * len(images), preview_size + label_h), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, (label, img) in enumerate(zip(labels, images)):
        x = idx * preview_size
        sheet.paste(img, (x, label_h))
        draw.text((x + 6, 4), label, fill=(20, 20, 20))

    out_dir = work_dir / "val_previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"epoch_{epoch:03d}_{uid}.jpg"
    sheet.save(out_path, quality=92)
    print(f"validation preview saved: {out_path}", flush=True)


def train(args):
    import torch  # Подключаем PyTorch: tensors, CUDA, autograd, optimizer, model.to(device).
    from torch.nn.parallel import DistributedDataParallel as DDP  # DDP синхронизирует обучение между несколькими GPU.
    from torch.utils.data import DataLoader  # DataLoader собирает samples из dataset в batch и грузит их в несколько workers.
    from torch.utils.data.distributed import DistributedSampler  # DistributedSampler делит dataset между GPU-процессами в DDP.

    ensure_deps(args.skip_install)  # Проверяем/устанавливаем зависимости, если запуск не был сделан с --skip_install.
    if args.require_cuda and not torch.cuda.is_available():  # Если пользователь потребовал CUDA, но PyTorch не видит GPU.
        raise RuntimeError("CUDA is required but unavailable")  # Сразу останавливаемся, чтобы случайно не учиться на CPU.

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ  # Проверяем, запущен ли скрипт через torchrun/DDP.
    if distributed:  # Ветка для multi-GPU запуска, когда каждый процесс работает со своей видеокартой.
        torch.distributed.init_process_group(backend="nccl")  # Инициализируем связь между GPU-процессами через быстрый NCCL backend.
        rank = int(os.environ["RANK"])  # Глобальный номер процесса: 0 главный, остальные вспомогательные.
        local_rank = int(os.environ["LOCAL_RANK"])  # Номер GPU внутри текущей машины, например cuda:0 или cuda:1.
        world = int(os.environ["WORLD_SIZE"])  # Общее количество процессов/GPU, участвующих в обучении.
        torch.cuda.set_device(local_rank)  # Привязываем текущий процесс к его конкретной GPU.
        device = torch.device("cuda", local_rank)  # Создаем объект device, на который потом переносится модель и batch.
    else:  # Ветка обычного запуска одной командой python3 без torchrun.
        rank = 0  # Единственный процесс считаем главным.
        local_rank = 0  # Локальная GPU по умолчанию первая.
        world = 1  # Всего один процесс, распределенного обучения нет.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Выбираем GPU, если доступна, иначе CPU.

    torch.backends.cuda.matmul.allow_tf32 = True  # Разрешаем быстрый TF32 для matrix multiplication на Ampere/Ada GPU.
    torch.backends.cudnn.allow_tf32 = True  # Разрешаем TF32 в cuDNN для ускорения convolution-like операций.

    dataset_root = Path(args.dataset_root).resolve()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    excluded = read_excluded_uids(args.exclude_uids)
    grouped = read_dataset(dataset_root, excluded)
    if args.max_objects > 0:
        selected_uids = sorted(grouped.keys())[: args.max_objects]
        grouped = {uid: grouped[uid] for uid in selected_uids}
    train_uids, val_uids, test_uids = split_uids(dataset_root, grouped, args.seed, args.train_ratio, args.val_ratio)

    if rank == 0:
        print(f"Dataset root: {dataset_root}", flush=True)
        print(f"Excluded UIDs: {len(excluded)}", flush=True)
        print(f"Objects train={len(train_uids)} val={len(val_uids)} test={len(test_uids)}", flush=True)
        print(f"Views total={sum(len(v) for v in grouped.values())}", flush=True)
        (work_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        (work_dir / "splits_used.json").write_text(
            json.dumps({"train": train_uids, "val": val_uids, "test": test_uids}, indent=2),
            encoding="utf-8",
        )
        print(f"Preparing DINOv2 weights/cache: {args.dinov2_model}", flush=True)
        _ = torch.hub.load("facebookresearch/dinov2", args.dinov2_model)
        del _

    if distributed:
        torch.distributed.barrier()

    resume_epoch = 0
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if resume_path.exists():
            resume_epoch = int(torch.load(resume_path, map_location="cpu").get("epoch", 0))

    encoder_trainable = args.encoder_lr > 0 and (resume_epoch + 1) >= args.unfreeze_encoder_epoch
    model = build_model(args).to(device)
    set_encoder_trainable(model, encoder_trainable)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_ds = RenderPairDataset(grouped, train_uids, args, training=True)
    val_ds = RenderPairDataset(grouped, val_uids, args, training=False)
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=False,
    )

    optimizer = make_optimizer(model, args, encoder_trainable=encoder_trainable)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device.type == "cuda"))
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    perceptual_model = build_perceptual_model(args, device, rank)
    start_epoch, best_val = load_checkpoint_if_requested(args.resume_checkpoint, model, optimizer, device, rank)

    if rank == 0:
        print(f"Device={device} world={world}", flush=True)
        print(f"Training started. steps_per_epoch={len(train_loader)} patch={args.patch_size} samples={args.samples_per_ray}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if epoch == args.unfreeze_encoder_epoch and not encoder_trainable:
            if rank == 0:
                print(f"Unfreezing DINOv2 backbone at epoch {epoch}", flush=True)
            set_encoder_trainable(model, True)
            optimizer = make_optimizer(model, args, encoder_trainable=True)
            encoder_trainable = True

        model.train()
        stage = stage_weights(args, epoch)
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        train_parts = defaultdict(list)
        start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            source = batch["source_image"].to(device, non_blocking=True)
            rays_o = batch["rays_o"].to(device, non_blocking=True)
            rays_d = batch["rays_d"].to(device, non_blocking=True)
            target_rgb = batch["target_rgb"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            geo_query = batch["geo_query"].to(device, non_blocking=True)
            geo_target = batch["geo_target"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(args.amp != "none" and device.type == "cuda")):
                planes = model(source)
                module = model.module if hasattr(model, "module") else model
                pred_rgb, pred_mask, pred_depth, weights = render_rays(module, planes, rays_o, rays_d, args, training=True)
            loss, parts = render_losses(
                pred_rgb.float(),
                pred_mask.float(),
                pred_depth.float(),
                weights.float(),
                target_rgb.float(),
                target_mask.float(),
                args,
                stage,
            )
            if perceptual_model is not None and stage.get("perceptual", 1.0) > 0:
                perc_loss = perceptual_patch_loss(perceptual_model, pred_rgb.float(), target_rgb.float(), args.patch_size)
                loss = loss + args.perceptual_weight * stage["perceptual"] * perc_loss
                parts["perc"] = perc_loss.detach()
            else:
                parts["perc"] = loss.detach() * 0.0
            geo_loss = geometry_density_loss(module, planes.float(), geo_query.float(), geo_target.float())
            loss = loss + args.geometry_weight * stage.get("geometry", 1.0) * geo_loss
            parts["geo"] = geo_loss.detach()
            loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            real_loss = float(loss.detach().cpu()) * args.grad_accum
            train_losses.append(real_loss)
            for key, value in parts.items():
                train_parts[key].append(float(value.cpu()))

            if rank == 0 and (step == 1 or step % args.log_every == 0 or step == len(train_loader)):
                sec = (time.time() - start) / step
                eta = sec * (len(train_loader) - step) / 60.0
                print(
                    f"epoch={epoch:03d}/{args.epochs} step={step:05d}/{len(train_loader)} "
                    f"loss={np.mean(train_losses[-args.log_every:]):.6f} "
                    f"rgb={np.mean(train_parts['rgb'][-args.log_every:]):.5f} "
                    f"mask={np.mean(train_parts['mask'][-args.log_every:]):.5f} "
                    f"recall={np.mean(train_parts['recall'][-args.log_every:]):.5f} "
                    f"geo={np.mean(train_parts['geo'][-args.log_every:]):.5f} "
                    f"perc={np.mean(train_parts['perc'][-args.log_every:]):.5f} "
                    f"sec/step={sec:.3f} eta_min={eta:.1f}",
                    flush=True,
                )

        if distributed:
            torch.distributed.barrier()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                if idx >= args.val_batches:
                    break
                source = batch["source_image"].to(device, non_blocking=True)
                rays_o = batch["rays_o"].to(device, non_blocking=True)
                rays_d = batch["rays_d"].to(device, non_blocking=True)
                target_rgb = batch["target_rgb"].to(device, non_blocking=True)
                target_mask = batch["target_mask"].to(device, non_blocking=True)
                geo_query = batch["geo_query"].to(device, non_blocking=True)
                geo_target = batch["geo_target"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(args.amp != "none" and device.type == "cuda")):
                    planes = model(source)
                    module = model.module if hasattr(model, "module") else model
                    pred_rgb, pred_mask, pred_depth, weights = render_rays(module, planes, rays_o, rays_d, args, training=False)
                loss, _ = render_losses(
                    pred_rgb.float(),
                    pred_mask.float(),
                    pred_depth.float(),
                    weights.float(),
                    target_rgb.float(),
                    target_mask.float(),
                    args,
                    stage,
                )
                if perceptual_model is not None and stage.get("perceptual", 1.0) > 0:
                    perc_loss = perceptual_patch_loss(perceptual_model, pred_rgb.float(), target_rgb.float(), args.patch_size)
                    loss = loss + args.perceptual_weight * stage["perceptual"] * perc_loss
                geo_loss = geometry_density_loss(module, planes.float(), geo_query.float(), geo_target.float())
                loss = loss + args.geometry_weight * stage.get("geometry", 1.0) * geo_loss
                val_losses.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if rank == 0:
            epoch_min = (time.time() - start) / 60.0
            epoch_rgb = float(np.mean(train_parts["rgb"])) if train_parts["rgb"] else 0.0
            epoch_mask = float(np.mean(train_parts["mask"])) if train_parts["mask"] else 0.0
            epoch_recall = float(np.mean(train_parts["recall"])) if train_parts["recall"] else 0.0
            epoch_geo = float(np.mean(train_parts["geo"])) if train_parts["geo"] else 0.0
            epoch_perc = float(np.mean(train_parts["perc"])) if train_parts["perc"] else 0.0
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"rgb={epoch_rgb:.5f} mask={epoch_mask:.5f} recall={epoch_recall:.5f} "
                f"geo={epoch_geo:.5f} perc={epoch_perc:.5f} "
                f"epoch_min={epoch_min:.1f}",
                flush=True,
            )
            save_validation_preview(model, grouped, val_uids, args, work_dir, epoch, device)
            with open(work_dir / "train_log.csv", "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "train_loss",
                        "val_loss",
                        "train_rgb",
                        "train_mask",
                        "train_recall",
                        "train_geo",
                        "train_perc",
                        "epoch_min",
                    ],
                )
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_rgb": epoch_rgb,
                        "train_mask": epoch_mask,
                        "train_recall": epoch_recall,
                        "train_geo": epoch_geo,
                        "train_perc": epoch_perc,
                        "epoch_min": epoch_min,
                    }
                )
            save_checkpoint(work_dir / "latest.pt", model, optimizer, epoch, args, best_val)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(work_dir / "best.pt", model, optimizer, epoch, args, best_val)
                print(f"saved best checkpoint: {work_dir / 'best.pt'}", flush=True)

    if distributed:
        torch.distributed.destroy_process_group()


def predict(args):
    import torch
    import trimesh
    from skimage import measure

    ensure_deps(args.skip_install)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = argparse.Namespace(**ckpt["args"])
    for key in ("image", "mask", "output_dir", "checkpoint", "grid_resolution", "sigma_level", "predict_chunk", "skip_install"):
        setattr(saved_args, key, getattr(args, key, getattr(saved_args, key, None)))
    model = build_model(saved_args).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    source = load_source_image(args.image, args.mask or args.image, saved_args.image_size, saved_args.crop)
    source_t = torch.from_numpy(source[None]).to(device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = np.linspace(BOX_MIN[0], BOX_MAX[0], args.grid_resolution, dtype=np.float32)
    ys = np.linspace(BOX_MIN[1], BOX_MAX[1], args.grid_resolution, dtype=np.float32)
    zs = np.linspace(BOX_MIN[2], BOX_MAX[2], args.grid_resolution, dtype=np.float32)
    field = np.empty((args.grid_resolution, args.grid_resolution, args.grid_resolution), dtype=np.float32)
    with torch.no_grad():
        planes = model.planes(source_t)
        for zi, z in enumerate(zs):
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1), np.full(grid_x.size, z, dtype=np.float32)], axis=-1)
            vals = []
            for start in range(0, len(pts), args.predict_chunk):
                p = torch.from_numpy(pts[start : start + args.predict_chunk][None]).to(device)
                _, sigma = model.decoder(planes, p)
                vals.append(sigma.squeeze(0).squeeze(-1).float().cpu().numpy())
            field[:, :, zi] = np.concatenate(vals, axis=0).reshape(args.grid_resolution, args.grid_resolution)

    np.save(out_dir / "density_grid.npy", field)
    verts, faces, normals, _ = measure.marching_cubes(
        field,
        level=args.sigma_level,
        spacing=(
            (BOX_MAX[0] - BOX_MIN[0]) / (args.grid_resolution - 1),
            (BOX_MAX[1] - BOX_MIN[1]) / (args.grid_resolution - 1),
            (BOX_MAX[2] - BOX_MIN[2]) / (args.grid_resolution - 1),
        ),
    )
    verts[:, 0] += BOX_MIN[0]
    verts[:, 1] += BOX_MIN[1]
    verts[:, 2] += BOX_MIN[2]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    mesh.export(out_dir / "mesh.obj")
    mesh.export(out_dir / "mesh.ply")
    print(f"saved: {out_dir / 'mesh.obj'}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("train", "predict"), default="train")
    p.add_argument("--dataset_root", default="")
    p.add_argument("--work_dir", default="/data/runs/chair_dinov2_lrm_render")
    p.add_argument("--exclude_uids", default="")
    p.add_argument("--max_objects", type=int, default=0)
    p.add_argument("--dinov2_model", default="dinov2_vitl14_reg")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=32)
    p.add_argument("--samples_per_ray", type=int, default=64)
    p.add_argument("--geometry_queries", type=int, default=4096)
    p.add_argument("--near", type=float, default=2.0)
    p.add_argument("--far", type=float, default=7.0)
    p.add_argument("--ray_jitter", action="store_true", default=True)
    p.add_argument("--no_ray_jitter", action="store_false", dest="ray_jitter")
    p.add_argument("--foreground_patch_prob", type=float, default=0.75)
    p.add_argument("--same_view_prob", type=float, default=0.0)
    p.add_argument("--plane_size", type=int, default=128)
    p.add_argument("--plane_channels", type=int, default=48)
    p.add_argument("--decoder_hidden", type=int, default=384)
    p.add_argument("--decoder_layers", type=int, default=5)
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--encoder_lr", type=float, default=2e-6)
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--unfreeze_encoder_epoch", type=int, default=20)
    p.add_argument("--foreground_rgb_weight", type=float, default=4.0)
    p.add_argument("--background_rgb_weight", type=float, default=0.05)
    p.add_argument("--mask_weight", type=float, default=0.35)
    p.add_argument("--foreground_mask_weight", type=float, default=1.0)
    p.add_argument("--background_mask_weight", type=float, default=0.05)
    p.add_argument("--mask_recall_weight", type=float, default=0.0)
    p.add_argument("--opacity_weight", type=float, default=0.04)
    p.add_argument("--distortion_weight", type=float, default=0.002)
    p.add_argument("--geometry_weight", type=float, default=0.08)
    p.add_argument("--perceptual_weight", type=float, default=0.04)
    p.add_argument("--geometry_warmup_epochs", type=int, default=3)
    p.add_argument("--warmup_mask_mult", type=float, default=1.6)
    p.add_argument("--warmup_opacity_mult", type=float, default=1.5)
    p.add_argument("--warmup_geometry_mult", type=float, default=1.8)
    p.add_argument("--sigma_init_bias", type=float, default=0.5)
    p.add_argument("--sigma_activation_bias", type=float, default=0.0)
    p.add_argument("--charbonnier_eps", type=float, default=1e-3)
    p.add_argument("--background", type=float, nargs=3, default=(219 / 255.0, 222 / 255.0, 224 / 255.0))
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--crop", action="store_true", default=True)
    p.add_argument("--no_crop", action="store_false", dest="crop")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_batches", type=int, default=80)
    p.add_argument("--preview_every", type=int, default=1)
    p.add_argument("--preview_size", type=int, default=128)
    p.add_argument("--preview_ray_chunk", type=int, default=4096)
    p.add_argument("--preview_same_view", action="store_true")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_install", action="store_true")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--resume_checkpoint", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--image", default="")
    p.add_argument("--mask", default="")
    p.add_argument("--output_dir", default="/data/runs/chair_lrm_predict")
    p.add_argument("--grid_resolution", type=int, default=160)
    p.add_argument("--sigma_level", type=float, default=8.0)
    p.add_argument("--predict_chunk", type=int, default=262144)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        if not args.dataset_root:
            raise ValueError("--dataset_root is required for training")
        train(args)
    else:
        if not args.checkpoint or not args.image:
            raise ValueError("--checkpoint and --image are required for predict")
        predict(args)


if __name__ == "__main__":
    main()
