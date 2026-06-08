"""Validation preview rendering."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import camera_rays, load_rgb_mask, load_source_image
from .rendering import render_rays


def save_validation_preview(model, grouped, val_uids, args, work_dir: Path, epoch: int, device) -> None:
    import torch
    from PIL import Image, ImageDraw

    if args.preview_every <= 0 or epoch % args.preview_every != 0 or not val_uids:
        return

    module = model.module if hasattr(model, "module") else model
    uid = val_uids[epoch % len(val_uids)]
    rows = grouped[uid]
    source = rows[0]
    target = rows[0] if args.preview_same_view else rows[1]

    source_np = load_source_image(source["image_path"], source["mask_path"], args.image_size, args.crop)
    target_rgb, target_mask = load_rgb_mask(target["image_path"], target["mask_path"])
    h, w = target_mask.shape
    preview_size = args.preview_size
    xs = np.linspace(0, w - 1, preview_size, dtype=np.float32)
    ys = np.linspace(0, h - 1, preview_size, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    rays_o, rays_d = camera_rays(target["camera_path"], xx.reshape(-1) + 0.5, yy.reshape(-1) + 0.5)

    source_t = torch.from_numpy(source_np[None]).to(device)
    rays_o_t = torch.from_numpy(rays_o[None]).to(device)
    rays_d_t = torch.from_numpy(rays_d[None]).to(device)

    with torch.no_grad():
        planes = module.planes(source_t)
        pred_rgbs = []
        pred_masks = []
        for start in range(0, rays_o_t.shape[1], args.preview_ray_chunk):
            ro = rays_o_t[:, start : start + args.preview_ray_chunk]
            rd = rays_d_t[:, start : start + args.preview_ray_chunk]
            rgb, mask, _, _ = render_rays(module, planes, ro, rd, args, training=False)
            pred_rgbs.append(rgb.float().cpu())
            pred_masks.append(mask.float().cpu())

    pred_rgb = torch.cat(pred_rgbs, dim=1).numpy().reshape(preview_size, preview_size, 3)
    pred_mask = torch.cat(pred_masks, dim=1).numpy().reshape(preview_size, preview_size)
    source_img = Image.fromarray(np.clip(np.transpose(source_np, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)).resize(
        (preview_size, preview_size)
    )
    target_img = Image.fromarray(np.clip(target_rgb * 255, 0, 255).astype(np.uint8)).resize((preview_size, preview_size))
    pred_img = Image.fromarray(np.clip(pred_rgb * 255, 0, 255).astype(np.uint8))
    target_mask_img = Image.fromarray(np.clip(target_mask * 255, 0, 255).astype(np.uint8)).resize((preview_size, preview_size))
    pred_mask_img = Image.fromarray(np.clip(pred_mask * 255, 0, 255).astype(np.uint8))

    labels = ["source", "target", "pred", "target_mask", "pred_mask"]
    images = [source_img, target_img, pred_img, target_mask_img.convert("RGB"), pred_mask_img.convert("RGB")]
    sheet = Image.new("RGB", (preview_size * len(images), preview_size + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(zip(labels, images)):
        x = i * preview_size
        draw.text((x + 4, 4), label, fill=(0, 0, 0))
        sheet.paste(img, (x, 24))

    out_dir = work_dir / "val_previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"epoch_{epoch:03d}_{uid}.jpg"
    sheet.save(out_path, quality=92)
    print(f"validation preview saved: {out_path}", flush=True)

