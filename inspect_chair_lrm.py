#!/usr/bin/env python3
"""
Inspect a trained chair LRM-lite checkpoint.

It restores best_orbax, runs inference on validation/test renders, and writes:
  output_dir/
    predictions.csv
    <uid>_viewXXX_input.png
    <uid>_viewXXX_pred.ply
    <uid>_viewXXX_target.ply
    <uid>_viewXXX_preview.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import List

import numpy as np


def ensure_deps(skip_install: bool) -> None:
    if skip_install:
        return
    pkgs = ["flax", "optax", "orbax-checkpoint", "Pillow"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def maybe_extract_dataset(dataset_root: Path, work_dir: Path) -> Path:
    dataset_root = dataset_root.resolve()
    if (dataset_root / "metadata" / "views.csv").exists():
        return dataset_root
    zip_names = ("metadata.zip", "objects.zip", "renders.zip")
    if not all((dataset_root / name).exists() for name in zip_names):
        raise FileNotFoundError(f"No metadata/views.csv or zip dataset under {dataset_root}")
    extracted = work_dir / "dataset_extracted"
    marker = extracted / ".extract_complete"
    if marker.exists() and (extracted / "metadata" / "views.csv").exists():
        return extracted
    if extracted.exists():
        shutil.rmtree(extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    for name in zip_names:
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
        raise RuntimeError(f"No usable rows found under {dataset_root}")
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
    )


def load_image(path: str, image_size: int):
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0, img


def write_ply(path: Path, points: np.ndarray, color):
    color = np.asarray(color, dtype=np.uint8)
    if color.ndim == 1:
        colors = np.tile(color.reshape(1, 3), (len(points), 1))
    else:
        colors = color
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def render_point_preview(points: np.ndarray, size: int, color):
    from PIL import Image

    img = np.full((size, size, 3), 245, dtype=np.uint8)
    pts = points.copy()
    x = (pts[:, 0] + 1.05) / 2.10
    y = 1.0 - (pts[:, 2] / 1.90)
    px = np.clip((x * (size - 1)).astype(np.int32), 0, size - 1)
    py = np.clip((y * (size - 1)).astype(np.int32), 0, size - 1)
    depth = pts[:, 1]
    order = np.argsort(depth)
    color = np.asarray(color, dtype=np.uint8)
    for idx in order:
        cx = int(px[idx])
        cy = int(py[idx])
        img[max(0, cy - 1):min(size, cy + 2), max(0, cx - 1):min(size, cx + 2), :] = color
    return Image.fromarray(img, mode="RGB")


def save_preview(path: Path, input_img, pred: np.ndarray, target: np.ndarray):
    from PIL import Image, ImageDraw

    panel = Image.new("RGB", (768, 288), (255, 255, 255))
    input_img = input_img.resize((256, 256))
    pred_img = render_point_preview(pred, 256, (35, 105, 210))
    target_img = render_point_preview(target, 256, (210, 80, 35))
    panel.paste(input_img, (0, 24))
    panel.paste(pred_img, (256, 24))
    panel.paste(target_img, (512, 24))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 4), "input", fill=(0, 0, 0))
    draw.text((264, 4), "prediction", fill=(0, 0, 0))
    draw.text((520, 4), "target", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def chamfer_numpy(pred: np.ndarray, target: np.ndarray, n: int, seed: int):
    rng = np.random.default_rng(seed)
    p = pred[rng.choice(len(pred), size=min(n, len(pred)), replace=len(pred) < n)]
    t = target[rng.choice(len(target), size=min(n, len(target)), replace=len(target) < n)]
    diff = p[:, None, :] - t[None, :, :]
    dist = np.sum(diff * diff, axis=-1)
    min_pt = dist.min(axis=1)
    min_tp = dist.min(axis=0)
    ch_l2 = float(min_pt.mean() + min_tp.mean())
    ch_l1 = float(np.sqrt(min_pt + 1e-8).mean() + np.sqrt(min_tp + 1e-8).mean())
    out = {"chamfer_l2": ch_l2, "chamfer_l1": ch_l1}
    for thr in (0.02, 0.05):
        thr2 = thr * thr
        precision = float((min_pt < thr2).mean())
        recall = float((min_tp < thr2).mean())
        fscore = 2.0 * precision * recall / (precision + recall + 1e-8)
        out[f"precision_{thr}"] = precision
        out[f"recall_{thr}"] = recall
        out[f"fscore_{thr}"] = fscore
    return out


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--skip_install", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()
    ensure_deps(pre_args.skip_install)

    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax.training import train_state
    import optax
    import orbax.checkpoint as ocp

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="/kaggle/working/chair_lrm_inspect")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--pred_points", type=int, default=8192)
    parser.add_argument("--target_points", type=int, default=8192)
    parser.add_argument("--metric_points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--skip_install", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = maybe_extract_dataset(Path(args.dataset_root), output_dir)
    rows = read_rows(dataset_root)
    train_rows, val_rows, test_rows = split_by_uid(rows, args.seed, args.train_ratio, args.val_ratio)
    split_rows = {"train": train_rows, "val": val_rows, "test": test_rows}[args.split]
    rng = random.Random(args.seed)
    rng.shuffle(split_rows)
    split_rows = split_rows[:args.num_samples]
    if not split_rows:
        raise RuntimeError(f"No rows found for split={args.split}")

    class DownBlock(nn.Module):
        channels: int

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(self.channels, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.LayerNorm(epsilon=1e-6)(x)
            x = nn.gelu(x)
            residual = x
            y = nn.Conv(self.channels, (3, 3), padding="SAME", feature_group_count=self.channels)(x)
            y = nn.LayerNorm(epsilon=1e-6)(y)
            y = nn.Dense(self.channels * 4)(y)
            y = nn.gelu(y)
            y = nn.Dense(self.channels)(y)
            return residual + y * 0.2

    def template_init(count: int):
        side = int(math.ceil(math.sqrt(count)))
        axis = jnp.linspace(-0.85, 0.85, side)
        gx, gy = jnp.meshgrid(axis, axis, indexing="ij")
        xy = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)[:count]
        ids = jnp.arange(count, dtype=jnp.float32)
        z_noise = jnp.mod(jnp.sin(ids * 12.9898) * 43758.5453, 1.0)
        z = 0.04 + 1.68 * z_noise[:, None]
        return jnp.concatenate([xy, z], axis=-1).astype(jnp.float32)

    class ChairLRMLite(nn.Module):
        pred_points: int

        @nn.compact
        def __call__(self, image, training: bool):
            x = image * 2.0 - 1.0
            x = nn.Conv(48, (5, 5), strides=(2, 2), padding="SAME")(x)
            x = nn.LayerNorm(epsilon=1e-6)(x)
            x = nn.gelu(x)
            x = DownBlock(96)(x)
            x = DownBlock(160)(x)
            x = DownBlock(256)(x)
            x = DownBlock(384)(x)
            latent = jnp.mean(x, axis=(1, 2))
            latent = nn.LayerNorm(epsilon=1e-6)(latent)
            latent = nn.Dense(768)(latent)
            latent = nn.gelu(latent)
            latent = nn.Dense(768)(latent)
            latent = nn.gelu(latent)
            template_raw = self.param("template", lambda key: template_init(self.pred_points))
            template = jnp.concatenate([
                jnp.clip(template_raw[:, 0:2], -0.95, 0.95),
                jnp.clip(template_raw[:, 2:3], 0.0, 1.85),
            ], axis=-1)
            q = nn.Dense(128)(template)
            q = nn.gelu(q)
            cond = nn.Dense(256)(latent)[:, None, :]
            y = jnp.broadcast_to(q[None, :, :], (image.shape[0], self.pred_points, 128))
            y = jnp.concatenate([y, jnp.broadcast_to(cond, (image.shape[0], self.pred_points, 256))], axis=-1)
            for width in (384, 384, 256):
                skip = nn.Dense(width)(y)
                y = nn.LayerNorm(epsilon=1e-6)(y)
                y = nn.Dense(width)(y)
                y = nn.gelu(y)
                y = nn.Dense(width)(y)
                y = y + skip * 0.5
            raw = nn.Dense(3, kernel_init=nn.initializers.normal(1e-4))(y)
            xy = jnp.clip(template[None, :, 0:2] + 0.35 * jnp.tanh(raw[:, :, 0:2]), -0.98, 0.98)
            z = jnp.clip(template[None, :, 2:3] + 0.45 * jnp.tanh(raw[:, :, 2:3]), 0.0, 1.85)
            return jnp.concatenate([xy, z], axis=-1)

    class State(train_state.TrainState):
        pass

    model = ChairLRMLite(args.pred_points)
    variables = model.init(jax.random.PRNGKey(args.seed), jnp.ones((1, args.image_size, args.image_size, 3), jnp.float32), training=False)
    schedule = optax.cosine_decay_schedule(5e-5, decay_steps=1, alpha=0.03)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=5e-5),
    )
    state = State.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    state = ocp.PyTreeCheckpointer().restore(Path(args.checkpoint), item=state)

    rows_out = []
    for i, row in enumerate(split_rows):
        image, input_img = load_image(row["image_path"], args.image_size)
        pred = model.apply({"params": state.params}, jnp.asarray(image[None, ...], dtype=jnp.float32), training=False)
        pred = np.asarray(jax.device_get(pred[0]), dtype=np.float32)

        data = np.load(row["points_path"])
        target_all = np.asarray(data["points"], dtype=np.float32)
        local_rng = np.random.default_rng(args.seed + i)
        target = target_all[local_rng.choice(len(target_all), size=args.target_points, replace=len(target_all) < args.target_points)]
        metrics = chamfer_numpy(pred, target, args.metric_points, args.seed + i)

        stem = f"{i:03d}_{row['uid']}_view{int(row['view_index']):03d}"
        input_path = output_dir / f"{stem}_input.png"
        pred_path = output_dir / f"{stem}_pred.ply"
        target_path = output_dir / f"{stem}_target.ply"
        preview_path = output_dir / f"{stem}_preview.png"
        input_img.save(input_path)
        write_ply(pred_path, pred, (35, 105, 210))
        write_ply(target_path, target, (210, 80, 35))
        save_preview(preview_path, input_img, pred, target)

        out_row = {
            "uid": row["uid"],
            "view_index": row["view_index"],
            "input_png": str(input_path),
            "pred_ply": str(pred_path),
            "target_ply": str(target_path),
            "preview_png": str(preview_path),
            **metrics,
        }
        rows_out.append(out_row)
        print(
            f"{stem}: ch_l2={metrics['chamfer_l2']:.6f} ch_l1={metrics['chamfer_l1']:.6f} "
            f"f@0.05={metrics['fscore_0.05']:.4f}",
            flush=True,
        )

    with open(output_dir / "predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    summary = {
        key: float(np.mean([r[key] for r in rows_out]))
        for key in rows_out[0].keys()
        if key.startswith("chamfer") or key.startswith("precision") or key.startswith("recall") or key.startswith("fscore")
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Outputs: {output_dir}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
