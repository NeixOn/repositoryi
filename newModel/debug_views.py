"""Step-level visual diagnostics for Kaggle/debug runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _tensor_image_to_pil(tensor, size: int):
    from PIL import Image

    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        img = Image.fromarray(arr).convert("RGB")
    return img.resize((size, size))


def _flat_patch_to_pil(tensor, patch_size: int, size: int, channels: int):
    from PIL import Image

    arr = tensor.detach().float().cpu().numpy().reshape(patch_size, patch_size, channels)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if channels == 1:
        img = Image.fromarray(arr[..., 0], mode="L").convert("RGB")
    else:
        img = Image.fromarray(arr).convert("RGB")
    return img.resize((size, size))


def save_train_step_preview(
    *,
    work_dir: Path,
    epoch: int,
    step: int,
    batch,
    pred_rgb,
    pred_mask,
    loss_value: float,
    parts: dict,
    patch_size: int,
    preview_size: int,
):
    from PIL import Image, ImageDraw

    out_dir = work_dir / "train_step_previews" / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    source = _tensor_image_to_pil(batch["source_image"][idx], preview_size)
    target_full = Image.open(batch["target_path"][idx]).convert("RGB").resize((preview_size, preview_size))
    target_patch = _flat_patch_to_pil(batch["target_rgb"][idx], patch_size, preview_size, 3)
    pred_patch = _flat_patch_to_pil(pred_rgb[idx], patch_size, preview_size, 3)
    target_mask = _flat_patch_to_pil(batch["target_mask"][idx], patch_size, preview_size, 1)
    pred_mask_img = _flat_patch_to_pil(pred_mask[idx], patch_size, preview_size, 1)

    labels = ["source", "target_full", "target_patch", "pred_patch", "target_mask", "pred_mask"]
    images = [source, target_full, target_patch, pred_patch, target_mask, pred_mask_img]
    header_h = 46
    sheet = Image.new("RGB", (preview_size * len(images), preview_size + header_h), "white")
    draw = ImageDraw.Draw(sheet)

    uid = batch["uid"][idx]
    source_view = batch["source_view"][idx]
    target_view = batch["target_view"][idx]
    info = (
        f"epoch={epoch} step={step} uid={uid} source_view={source_view} target_view={target_view} "
        f"loss={loss_value:.5f} rgb={float(parts['rgb'].detach().cpu()):.5f} "
        f"mask={float(parts['mask'].detach().cpu()):.5f} recall={float(parts['recall'].detach().cpu()):.5f}"
    )
    draw.text((4, 4), info[:180], fill=(0, 0, 0))

    for i, (label, img) in enumerate(zip(labels, images)):
        x = i * preview_size
        draw.text((x + 4, 26), label, fill=(0, 0, 0))
        sheet.paste(img, (x, header_h))

    out_path = out_dir / f"step_{step:05d}_{uid}_s{source_view:03d}_t{target_view:03d}.jpg"
    sheet.save(out_path, quality=92)
    return out_path
