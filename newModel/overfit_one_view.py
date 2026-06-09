#!/usr/bin/env python3
"""Overfit one rendered view with directly trainable triplanes.

This is a low-level renderer sanity check. It intentionally skips DINOv2 and
the triplane generator. If this script cannot reconstruct one view/mask, the
problem is in rays, bbox, volume rendering, or losses rather than the encoder.
"""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import time
from pathlib import Path

import numpy as np

from newModel.constants import BOX_MAX, BOX_MIN
from newModel.data import camera_rays, load_rgb_mask
from newModel.rendering import render_rays


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--uid", default="")
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--work_dir", default="/kaggle/working/overfit_one_view")
    p.add_argument("--patch_size", type=int, default=96)
    p.add_argument("--samples_per_ray", type=int, default=96)
    p.add_argument("--plane_size", type=int, default=64)
    p.add_argument("--plane_channels", type=int, default=16)
    p.add_argument("--decoder_hidden", type=int, default=128)
    p.add_argument("--decoder_layers", type=int, default=3)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    p.add_argument("--preview_every", type=int, default=50)
    p.add_argument("--preview_size", type=int, default=160)
    p.add_argument("--preview_ray_chunk", type=int, default=8192)
    p.add_argument("--background", type=float, nargs=3, default=(219 / 255.0, 222 / 255.0, 224 / 255.0))
    p.add_argument("--near", type=float, default=2.0)
    p.add_argument("--far", type=float, default=7.0)
    p.add_argument("--ray_jitter", action="store_true", default=True)
    p.add_argument("--sigma_init_bias", type=float, default=2.0)
    p.add_argument("--sigma_activation_bias", type=float, default=1.0)
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--dinov2_model", default="dinov2_vits14")
    return p.parse_args()


def choose_uid(dataset_root: Path, uid: str) -> str:
    if uid:
        return uid
    render_root = dataset_root / "renders"
    uids = sorted(p.name for p in render_root.iterdir() if p.is_dir())
    if not uids:
        raise RuntimeError(f"No render folders under {render_root}")
    return uids[0]


def sample_foreground_patch(rng, mask, patch_size):
    h, w = mask.shape
    ys, xs = np.where(mask > 8 / 255.0)
    if len(xs) == 0:
        raise RuntimeError("Selected mask has no foreground pixels")
    center_idx = int(rng.integers(0, len(xs)))
    cx = int(xs[center_idx])
    cy = int(ys[center_idx])
    x0 = int(np.clip(cx - patch_size // 2, 0, max(0, w - patch_size)))
    y0 = int(np.clip(cy - patch_size // 2, 0, max(0, h - patch_size)))
    yy, xx = np.meshgrid(np.arange(y0, y0 + patch_size), np.arange(x0, x0 + patch_size), indexing="ij")
    return xx.reshape(-1).astype(np.float32) + 0.5, yy.reshape(-1).astype(np.float32) + 0.5, yy, xx


def save_preview(step, module, planes, rgb_np, mask_np, camera_path, args, out_dir, device):
    import torch
    from PIL import Image, ImageDraw

    h, w = mask_np.shape
    size = args.preview_size
    xs = np.linspace(0, w - 1, size, dtype=np.float32)
    ys = np.linspace(0, h - 1, size, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    rays_o, rays_d = camera_rays(str(camera_path), xx.reshape(-1) + 0.5, yy.reshape(-1) + 0.5)
    rays_o_t = torch.from_numpy(rays_o[None]).to(device)
    rays_d_t = torch.from_numpy(rays_d[None]).to(device)

    pred_rgbs = []
    pred_masks = []
    with torch.no_grad():
        for start in range(0, rays_o_t.shape[1], args.preview_ray_chunk):
            ro = rays_o_t[:, start : start + args.preview_ray_chunk]
            rd = rays_d_t[:, start : start + args.preview_ray_chunk]
            pred_rgb, pred_mask, _, _ = render_rays(module, planes, ro, rd, args, training=False)
            pred_rgbs.append(pred_rgb.float().cpu())
            pred_masks.append(pred_mask.float().cpu())

    pred_rgb = torch.cat(pred_rgbs, dim=1).numpy().reshape(size, size, 3)
    pred_mask = torch.cat(pred_masks, dim=1).numpy().reshape(size, size)
    target_img = Image.fromarray(np.clip(rgb_np * 255, 0, 255).astype(np.uint8)).resize((size, size))
    pred_img = Image.fromarray(np.clip(pred_rgb * 255, 0, 255).astype(np.uint8))
    target_mask_img = Image.fromarray(np.clip(mask_np * 255, 0, 255).astype(np.uint8)).convert("RGB").resize((size, size))
    pred_mask_img = Image.fromarray(np.clip(pred_mask * 255, 0, 255).astype(np.uint8)).convert("RGB")

    labels = ["target", "pred", "target_mask", "pred_mask"]
    images = [target_img, pred_img, target_mask_img, pred_mask_img]
    sheet = Image.new("RGB", (size * 4, size + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(zip(labels, images)):
        x = i * size
        draw.text((x + 4, 4), label, fill=(0, 0, 0))
        sheet.paste(img, (x, 24))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"step_{step:05d}.jpg"
    sheet.save(path, quality=92)
    print(f"preview saved: {path}", flush=True)


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyDecoder(nn.Module):
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
            b, _, _, _, _ = planes.shape
            x = pts[..., 0].clamp(BOX_MIN[0], BOX_MAX[0]) / abs(BOX_MIN[0])
            y = pts[..., 1].clamp(BOX_MIN[1], BOX_MAX[1]) / abs(BOX_MIN[1])
            z = ((pts[..., 2].clamp(BOX_MIN[2], BOX_MAX[2]) - BOX_MIN[2]) / (BOX_MAX[2] - BOX_MIN[2])) * 2.0 - 1.0
            grids = [torch.stack([x, y], dim=-1), torch.stack([x, z], dim=-1), torch.stack([y, z], dim=-1)]
            feats = []
            for i, grid in enumerate(grids):
                sampled = F.grid_sample(planes[:, i], grid.view(b, -1, 1, 2), mode="bilinear", padding_mode="border", align_corners=True)
                feats.append(sampled.squeeze(-1).transpose(1, 2))
            return torch.cat(feats + [pts], dim=-1)

        def forward(self, planes, pts):
            h = self.net(self.sample_planes(planes, pts))
            sigma = F.softplus(self.sigma(h) + args.sigma_activation_bias)
            rgb = torch.sigmoid(self.rgb(h))
            return rgb, sigma

    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_root)
    uid = choose_uid(dataset_root, args.uid)
    view = args.view
    rgb_path = dataset_root / "renders" / uid / f"view_{view:03d}.png"
    mask_path = dataset_root / "masks" / uid / f"view_{view:03d}.png"
    camera_path = dataset_root / "cameras" / uid / f"view_{view:03d}.json"
    rgb_np, mask_np = load_rgb_mask(str(rgb_path), str(mask_path))

    print(f"uid={uid} view={view}", flush=True)
    print(f"foreground ratio={float((mask_np > 8 / 255.0).mean()):.4f}", flush=True)

    decoder = TinyDecoder().to(device)
    planes = torch.nn.Parameter(torch.randn(1, 3, args.plane_channels, args.plane_size, args.plane_size, device=device) * 0.02)
    optimizer = torch.optim.AdamW([planes, *decoder.parameters()], lr=args.lr, weight_decay=0.0)
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    out_dir = Path(args.work_dir)
    preview_dir = out_dir / "previews"

    for step in range(1, args.steps + 1):
        xs, ys, yy, xx = sample_foreground_patch(rng, mask_np, args.patch_size)
        rays_o, rays_d = camera_rays(str(camera_path), xs, ys)
        rays_o_t = torch.from_numpy(rays_o[None]).to(device)
        rays_d_t = torch.from_numpy(rays_d[None]).to(device)
        target_rgb = torch.from_numpy(rgb_np[yy, xx].reshape(1, -1, 3).astype(np.float32)).to(device)
        target_mask = torch.from_numpy(mask_np[yy, xx].reshape(1, -1, 1).astype(np.float32)).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(args.amp != "none" and device.type == "cuda")):
            pred_rgb, pred_mask, _, _ = render_rays(DecoderWrapper(decoder), planes, rays_o_t, rays_d_t, args, training=True)

        rgb_weight = 0.1 * (1.0 - target_mask) + 5.0 * target_mask
        rgb_loss = (torch.sqrt((pred_rgb.float() - target_rgb).pow(2) + 1e-3) * rgb_weight).sum() / (rgb_weight.sum() * 3.0 + 1e-6)
        bce = F.binary_cross_entropy(pred_mask.float().clamp(1e-4, 1.0 - 1e-4), target_mask, reduction="none")
        mask_weight = 0.1 * (1.0 - target_mask) + target_mask
        bce = (bce * mask_weight).sum() / (mask_weight.sum() + 1e-6)
        inter = (pred_mask.float() * target_mask).sum(dim=1)
        dice = 1.0 - (2.0 * inter + 1.0) / (pred_mask.float().sum(dim=1) + target_mask.sum(dim=1) + 1.0)
        recall = (((1.0 - pred_mask.float()).clamp_min(0.0) * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp_min(1.0)).mean()
        loss = rgb_loss + 2.0 * (bce + dice.mean()) + 2.0 * recall
        loss.backward()
        torch.nn.utils.clip_grad_norm_([planes, *decoder.parameters()], 1.0)
        optimizer.step()

        if step == 1 or step % 25 == 0:
            print(
                f"step={step:05d}/{args.steps} loss={float(loss.detach().cpu()):.5f} "
                f"rgb={float(rgb_loss.detach().cpu()):.5f} mask={float((bce + dice.mean()).detach().cpu()):.5f} "
                f"recall={float(recall.detach().cpu()):.5f}",
                flush=True,
            )
        if step == 1 or step % args.preview_every == 0 or step == args.steps:
            save_preview(step, DecoderWrapper(decoder), planes, rgb_np, mask_np, camera_path, args, preview_dir, device)


class DecoderWrapper:
    def __init__(self, decoder):
        self.decoder = decoder


if __name__ == "__main__":
    main()
