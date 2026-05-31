#!/usr/bin/env python3
"""
PyTorch/XLA single-image chair reconstruction with a triplane implicit field.

This is a stronger mesh-oriented architecture than the previous point-cloud
baseline:

  image -> ResNet encoder -> triplane feature field -> UDF decoder -> marching cubes mesh

The model predicts an unsigned distance field (UDF) to the object surface.
At inference it evaluates a dense 3D grid and extracts a mesh with marching
cubes. This is closer to TripoSR's representation family than point regression,
while remaining trainable from the dataset we already have: images plus
surface point clouds.

Kaggle TPU training example:
  !PJRT_DEVICE=TPU python /kaggle/working/repositoryi/train_chair_triplane_udf_xla.py \
    --mode train \
    --dataset_root /kaggle/input/datasets/neixon/objaverse-chairs-2k-24views \
    --work_dir /kaggle/working/chair_triplane_udf \
    --image_size 256 \
    --epochs 40 \
    --batch_size 8 \
    --queries_per_item 8192 \
    --plane_size 64 \
    --plane_channels 32 \
    --lr 2e-4

Inference example:
  !python /kaggle/working/repositoryi/train_chair_triplane_udf_xla.py \
    --mode predict \
    --checkpoint /kaggle/working/chair_triplane_udf/best.pt \
    --image /kaggle/input/my-chair/chair.png \
    --output_dir /kaggle/working/chair_triplane_test \
    --grid_resolution 96 \
    --level 0.025 \
    --crop
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import List

import numpy as np


def ensure_deps(skip_install: bool) -> None:
    if skip_install:
        return
    pkgs = [
        "torch",
        "torchvision",
        "torch_xla[tpu]",
        "Pillow",
        "scipy",
        "scikit-image",
        "trimesh",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore", *pkgs], check=True)


def maybe_extract_dataset(dataset_root: Path, work_dir: Path) -> Path:
    dataset_root = dataset_root.resolve()
    if (dataset_root / "metadata" / "views.csv").exists():
        return dataset_root
    zip_names = ("metadata.zip", "objects.zip", "renders.zip")
    if not all((dataset_root / name).exists() for name in zip_names):
        raise FileNotFoundError(f"No metadata/views.csv and no dataset zips under {dataset_root}")
    extracted = work_dir / "dataset_extracted"
    marker = extracted / ".extract_complete"
    if marker.exists() and (extracted / "metadata" / "views.csv").exists():
        return extracted
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    for name in zip_names:
        print(f"Extracting {name}...", flush=True)
        with zipfile.ZipFile(dataset_root / name, "r") as zf:
            zf.extractall(extracted)
    marker.write_text("ok", encoding="utf-8")
    return extracted


def read_rows(dataset_root: Path) -> List[dict]:
    rows = []
    with open(dataset_root / "metadata" / "views.csv", "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["uid"]
            view_index = int(row["view_index"])
            image_path = dataset_root / "renders" / uid / f"view_{view_index:03d}.png"
            points_path = dataset_root / "objects" / uid / "points.npz"
            if image_path.exists() and points_path.exists():
                rows.append({
                    "uid": uid,
                    "view_index": view_index,
                    "image_path": str(image_path),
                    "points_path": str(points_path),
                })
    if not rows:
        raise RuntimeError(f"No usable rows under {dataset_root}")
    return rows


def split_by_uid(rows: List[dict], seed: int, train_ratio: float, val_ratio: float):
    uids = sorted({r["uid"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(uids)
    n_train = max(1, int(len(uids) * train_ratio))
    n_val = max(1, int(len(uids) * val_ratio))
    train_uids = set(uids[:n_train])
    val_uids = set(uids[n_train:n_train + n_val])
    test_uids = set(uids[n_train + n_val:])
    return (
        [r for r in rows if r["uid"] in train_uids],
        [r for r in rows if r["uid"] in val_uids],
        [r for r in rows if r["uid"] in test_uids],
        {"train_uids": sorted(train_uids), "val_uids": sorted(val_uids), "test_uids": sorted(test_uids)},
    )


def stable_seed(text: str, seed: int) -> int:
    value = seed & 0xFFFFFFFF
    for ch in text:
        value = ((value * 131) + ord(ch)) & 0xFFFFFFFF
    return value


def load_rgb_image(path: str | Path, image_size: int, crop: bool = False):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if crop:
        arr = np.asarray(img, dtype=np.uint8)
        mask = np.any(arr < 245, axis=-1)
        if mask.any():
            ys, xs = np.where(mask)
            pad = int(max(arr.shape[:2]) * 0.05)
            img = img.crop((
                max(0, int(xs.min()) - pad),
                max(0, int(ys.min()) - pad),
                min(arr.shape[1], int(xs.max()) + pad + 1),
                min(arr.shape[0], int(ys.max()) + pad + 1),
            ))
    img = img.resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr, img


class ChairUDFDataset:
    def __init__(
        self,
        rows: List[dict],
        image_size: int,
        queries_per_item: int,
        surface_points: int,
        seed: int,
        training: bool,
        truncation: float,
    ) -> None:
        self.rows = rows
        self.image_size = image_size
        self.queries_per_item = queries_per_item
        self.surface_points = surface_points
        self.seed = seed
        self.training = training
        self.truncation = truncation
        self._cache = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_points(self, path: str):
        if path in self._cache:
            return self._cache[path]
        from scipy.spatial import cKDTree

        data = np.load(path)
        points = np.asarray(data["points"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32) if "normals" in data.files else np.zeros_like(points)
        tree = cKDTree(points)
        self._cache[path] = (points, normals, tree)
        if len(self._cache) > 128:
            self._cache.pop(next(iter(self._cache)))
        return points, normals, tree

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        rng = np.random.default_rng(stable_seed(row["image_path"], self.seed + (997 if self.training else 0)))

        image, _ = load_rgb_image(row["image_path"], self.image_size, crop=False)
        if self.training:
            brightness = rng.uniform(0.90, 1.10)
            contrast = rng.uniform(0.90, 1.10)
            image = np.clip((image - 0.5) * contrast + 0.5, 0.0, 1.0)
            image = np.clip(image * brightness, 0.0, 1.0)

        surface, normals, tree = self._load_points(row["points_path"])
        surf_idx = rng.choice(len(surface), size=self.surface_points, replace=len(surface) < self.surface_points)
        surface_sample = surface[surf_idx]
        normal_sample = normals[surf_idx]

        n_surface = self.queries_per_item // 4
        n_near = self.queries_per_item // 2
        n_uniform = self.queries_per_item - n_surface - n_near

        s_idx = rng.choice(len(surface), size=n_surface, replace=len(surface) < n_surface)
        q_surface = surface[s_idx]
        d_surface = np.zeros((n_surface, 1), dtype=np.float32)

        n_idx = rng.choice(len(surface), size=n_near, replace=len(surface) < n_near)
        sigma = rng.choice(np.array([0.015, 0.035, 0.075], dtype=np.float32), size=n_near, p=[0.45, 0.40, 0.15])
        q_near = surface[n_idx] + rng.normal(size=(n_near, 3)).astype(np.float32) * sigma[:, None]

        q_uniform = np.empty((n_uniform, 3), dtype=np.float32)
        q_uniform[:, 0:2] = rng.uniform(-1.05, 1.05, size=(n_uniform, 2))
        q_uniform[:, 2] = rng.uniform(-0.08, 1.92, size=n_uniform)

        queries = np.concatenate([q_surface, q_near, q_uniform], axis=0).astype(np.float32)
        dist, nearest = tree.query(queries, k=1, workers=1)
        udf = np.clip(dist.astype(np.float32), 0.0, self.truncation)[:, None]
        udf[:n_surface] = d_surface

        # Approximate nearest-surface normals are useful for diagnostics and future normal losses.
        query_normals = normals[np.asarray(nearest, dtype=np.int64)].astype(np.float32)

        order = rng.permutation(len(queries))
        return {
            "image": image.astype(np.float32),
            "query": queries[order],
            "udf": udf[order],
            "query_normal": query_normals[order],
            "surface": surface_sample.astype(np.float32),
            "surface_normal": normal_sample.astype(np.float32),
        }


def collate_batch(items: list[dict]) -> dict:
    import torch

    return {key: torch.from_numpy(np.stack([item[key] for item in items], axis=0)) for key in items[0].keys()}


def build_model(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models

    class ResNetEncoder(nn.Module):
        def __init__(self, latent_dim: int, pretrained: bool) -> None:
            super().__init__()
            weights = None
            if pretrained:
                try:
                    weights = models.ResNet50_Weights.IMAGENET1K_V2
                except Exception:
                    weights = None
            net = models.resnet50(weights=weights)
            self.stem = nn.Sequential(*list(net.children())[:-1])
            self.proj = nn.Sequential(
                nn.Linear(2048, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
            )

        def forward(self, x):
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
            x = self.stem(x).flatten(1)
            return self.proj(x)

    class TriplaneGenerator(nn.Module):
        def __init__(self, latent_dim: int, plane_channels: int, plane_size: int) -> None:
            super().__init__()
            self.plane_channels = plane_channels
            self.plane_size = plane_size
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, 3 * plane_channels * 8 * 8),
                nn.GELU(),
            )
            blocks = []
            channels = plane_channels
            size = 8
            while size < plane_size:
                blocks.extend([
                    nn.ConvTranspose2d(channels, channels, 4, 2, 1),
                    nn.GroupNorm(8 if channels >= 8 else 1, channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.GroupNorm(8 if channels >= 8 else 1, channels),
                    nn.GELU(),
                ])
                size *= 2
            self.up = nn.Sequential(*blocks)

        def forward(self, latent):
            b = latent.shape[0]
            planes = self.fc(latent).view(b * 3, self.plane_channels, 8, 8)
            planes = self.up(planes)
            return planes.view(b, 3, self.plane_channels, self.plane_size, self.plane_size)

    class UDFDecoder(nn.Module):
        def __init__(self, plane_channels: int, hidden: int) -> None:
            super().__init__()
            in_dim = plane_channels * 3 + 3
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )

        def sample_planes(self, planes, query):
            b, _, c, _, _ = planes.shape
            x = query[..., 0].clamp(-1.05, 1.05) / 1.05
            y = query[..., 1].clamp(-1.05, 1.05) / 1.05
            z = (query[..., 2].clamp(-0.08, 1.92) - 0.92) / 1.0
            coords = [
                torch.stack([x, y], dim=-1),
                torch.stack([x, z], dim=-1),
                torch.stack([y, z], dim=-1),
            ]
            feats = []
            for i, grid in enumerate(coords):
                sampled = F.grid_sample(
                    planes[:, i],
                    grid.view(b, -1, 1, 2),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                feats.append(sampled.squeeze(-1).transpose(1, 2))
            return torch.cat(feats, dim=-1)

        def forward(self, planes, query):
            features = self.sample_planes(planes, query)
            x = torch.cat([features, query], dim=-1)
            raw = self.net(x)
            return F.softplus(raw, beta=10.0)

    class ChairTriplaneUDF(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = ResNetEncoder(args.latent_dim, args.pretrained_encoder)
            self.triplane = TriplaneGenerator(args.latent_dim, args.plane_channels, args.plane_size)
            self.decoder = UDFDecoder(args.plane_channels, args.decoder_hidden)

        def forward(self, image, query):
            latent = self.encoder(image)
            planes = self.triplane(latent)
            return self.decoder(planes, query)

    return ChairTriplaneUDF()


def loss_fn(pred, target, truncation: float):
    import torch
    import torch.nn.functional as F

    near_weight = 1.0 + 6.0 * torch.exp(-target / 0.035)
    l1 = (near_weight * torch.abs(pred - target)).mean()
    zero_mask = target < 1e-5
    surface = torch.tensor(0.0, device=pred.device)
    if zero_mask.any():
        surface = F.smooth_l1_loss(pred[zero_mask], target[zero_mask])
    far = 0.05 * torch.relu(pred[target > truncation * 0.95] - truncation).mean() if (target > truncation * 0.95).any() else 0.0
    return l1 + 2.0 * surface + far, {"l1": float(l1.detach().cpu()), "surface": float(surface.detach().cpu())}


def to_cpu_tree(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: to_cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_tree(v) for v in value)
    return value


def train(args) -> None:
    ensure_deps(args.skip_install)

    import torch
    from torch.utils.data import DataLoader

    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        device = torch_xla.device()
        is_xla = str(device).startswith("xla")
    except Exception:
        xm = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_xla = False

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = maybe_extract_dataset(Path(args.dataset_root), work_dir)
    rows = read_rows(dataset_root)
    train_rows, val_rows, test_rows, split = split_by_uid(rows, args.seed, args.train_ratio, args.val_ratio)
    (work_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (work_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    print(f"Device: {device}", flush=True)
    print(f"Dataset root: {dataset_root}", flush=True)
    print(f"Rows train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    train_ds = ChairUDFDataset(train_rows, args.image_size, args.queries_per_item, args.surface_points, args.seed, True, args.truncation)
    val_ds = ChairUDFDataset(val_rows, args.image_size, args.queries_per_item, args.surface_points, args.seed + 999, False, args.truncation)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        collate_fn=collate_batch,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(train_loader)), eta_min=args.lr * 0.03)

    start_epoch = 1
    best_val = float("inf")
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        print(f"Resumed: {args.resume_from}", flush=True)

    log_path = work_dir / "train_log.csv"
    write_header = not log_path.exists()
    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "train_loss", "val_loss", "lr", "epoch_min"])

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            start = time.time()
            train_losses = []
            for step, batch in enumerate(train_loader, start=1):
                image = batch["image"].to(device)
                query = batch["query"].to(device)
                udf = batch["udf"].to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(image, query)
                loss, parts = loss_fn(pred, udf, args.truncation)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if is_xla and xm is not None:
                    xm.optimizer_step(optimizer)
                    torch_xla.sync()
                else:
                    optimizer.step()
                scheduler.step()
                train_losses.append(float(loss.detach().cpu()))
                if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                    elapsed = time.time() - start
                    sec_per_step = elapsed / step
                    eta = sec_per_step * (len(train_loader) - step)
                    print(
                        f"epoch={epoch:03d}/{args.epochs} step={step:04d}/{len(train_loader)} "
                        f"loss={train_losses[-1]:.6f} l1={parts['l1']:.6f} surf={parts['surface']:.6f} "
                        f"sec/step={sec_per_step:.2f} epoch_eta_min={eta / 60:.1f}",
                        flush=True,
                    )

            model.eval()
            val_losses = []
            with torch.no_grad():
                for step, batch in enumerate(val_loader, start=1):
                    if step > args.val_steps:
                        break
                    image = batch["image"].to(device)
                    query = batch["query"].to(device)
                    udf = batch["udf"].to(device)
                    pred = model(image, query)
                    loss, _ = loss_fn(pred, udf, args.truncation)
                    val_losses.append(float(loss.detach().cpu()))
                    if is_xla:
                        torch_xla.sync()

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            epoch_min = (time.time() - start) / 60.0
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch, train_loss, val_loss, lr, epoch_min])
            f.flush()
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.2e} epoch_min={epoch_min:.1f}", flush=True)

            latest = {
                "model": to_cpu_tree(model.state_dict()),
                "optimizer": to_cpu_tree(optimizer.state_dict()),
                "epoch": epoch,
                "best_val": best_val,
                "args": vars(args),
            }
            torch.save(latest, work_dir / "latest.pt")
            if val_loss < best_val:
                best_val = val_loss
                latest["best_val"] = best_val
                torch.save(latest, work_dir / "best.pt")
                print(f"saved best checkpoint: {work_dir / 'best.pt'}", flush=True)


def predict(args) -> None:
    ensure_deps(args.skip_install)

    import torch
    from skimage import measure
    import trimesh

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    if not args.force_cpu:
        try:
            import torch_xla
            device = torch_xla.device()
        except Exception:
            pass

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = argparse.Namespace(**ckpt.get("args", {}))
    for name in ("latent_dim", "plane_channels", "plane_size", "decoder_hidden", "pretrained_encoder"):
        if not hasattr(args, name):
            setattr(args, name, getattr(saved_args, name))
    args.latent_dim = getattr(saved_args, "latent_dim", args.latent_dim)
    args.plane_channels = getattr(saved_args, "plane_channels", args.plane_channels)
    args.plane_size = getattr(saved_args, "plane_size", args.plane_size)
    args.decoder_hidden = getattr(saved_args, "decoder_hidden", args.decoder_hidden)
    args.pretrained_encoder = False

    model = build_model(args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    image_np, resized_img = load_rgb_image(args.image, args.image_size, crop=args.crop)
    image = torch.from_numpy(image_np[None]).to(device)

    xs = np.linspace(-1.05, 1.05, args.grid_resolution, dtype=np.float32)
    ys = np.linspace(-1.05, 1.05, args.grid_resolution, dtype=np.float32)
    zs = np.linspace(-0.08, 1.92, args.grid_resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    values = np.empty((len(grid),), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(grid), args.eval_chunk):
            q = torch.from_numpy(grid[start:start + args.eval_chunk][None]).to(device)
            pred = model(image, q).reshape(-1)
            values[start:start + args.eval_chunk] = pred.detach().cpu().numpy()

    field = values.reshape(args.grid_resolution, args.grid_resolution, args.grid_resolution)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    np.save(output_dir / f"{stem}_udf_grid.npy", field)
    resized_img.save(output_dir / f"{stem}_model_input.png")

    level = float(args.level)
    if field.min() > level:
        level = float(field.min() + 0.2 * (field.max() - field.min()))
        print(f"Requested level too low; using level={level:.6f}", flush=True)
    verts, faces, normals, _ = measure.marching_cubes(
        field,
        level=level,
        spacing=(2.10 / (args.grid_resolution - 1), 2.10 / (args.grid_resolution - 1), 2.00 / (args.grid_resolution - 1)),
    )
    verts[:, 0] += -1.05
    verts[:, 1] += -1.05
    verts[:, 2] += -0.08
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    mesh.remove_unreferenced_vertices()
    mesh_path = output_dir / f"{stem}_mesh.obj"
    ply_path = output_dir / f"{stem}_mesh.ply"
    mesh.export(mesh_path)
    mesh.export(ply_path)
    print(f"Mesh OBJ: {mesh_path}", flush=True)
    print(f"Mesh PLY: {ply_path}", flush=True)
    print(f"Grid: {output_dir / f'{stem}_udf_grid.npy'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", choices=("train", "predict"), default="train")
    parser.add_argument("--dataset_root", default="")
    parser.add_argument("--work_dir", default="/kaggle/working/chair_triplane_udf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--image", default="")
    parser.add_argument("--output_dir", default="/kaggle/working/chair_triplane_predict")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--queries_per_item", type=int, default=8192)
    parser.add_argument("--surface_points", type=int, default=4096)
    parser.add_argument("--truncation", type=float, default=0.20)
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--plane_channels", type=int, default=32)
    parser.add_argument("--plane_size", type=int, default=64)
    parser.add_argument("--decoder_hidden", type=int, default=256)
    parser.add_argument("--pretrained_encoder", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_steps", type=int, default=80)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--grid_resolution", type=int, default=96)
    parser.add_argument("--eval_chunk", type=int, default=65536)
    parser.add_argument("--level", type=float, default=0.025)
    parser.add_argument("--crop", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--skip_install", action="store_true")
    args = parser.parse_args()
    if args.mode == "train" and not args.dataset_root:
        raise ValueError("--dataset_root is required for --mode train")
    if args.mode == "predict" and (not args.checkpoint or not args.image):
        raise ValueError("--checkpoint and --image are required for --mode predict")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
