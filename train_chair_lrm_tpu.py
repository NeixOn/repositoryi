#!/usr/bin/env python3
"""
TPU-friendly chair reconstruction trainer for the Objaverse chair dataset.

This is not a TripoSR/Hunyuan3D clone. Those systems are trained at foundation
model scale. This script uses the same practical direction for a small
chair-only dataset: a feed-forward image encoder plus a geometry decoder, with
stable point-cloud supervision and proper reconstruction metrics.

Run in Kaggle TPU:
  !python /kaggle/working/repositoryi/train_chair_lrm_tpu.py \
    --dataset_root /kaggle/input/datasets/neixon/objaverse-chairs \
    --work_dir /kaggle/working/chair_lrm_tpu \
    --image_size 256 \
    --pred_points 8192 \
    --target_points 8192 \
    --chamfer_points 4096 \
    --epochs 80 \
    --batch_per_device 1
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
from typing import Iterator, List

import numpy as np


def ensure_deps(skip_install: bool) -> None:
    if skip_install:
        return
    pkgs = ["flax", "optax", "orbax-checkpoint", "Pillow", "tqdm"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def maybe_extract_dataset(dataset_root: Path, work_dir: Path) -> Path:
    dataset_root = dataset_root.resolve()
    if (dataset_root / "metadata" / "views.csv").exists():
        return dataset_root

    zip_names = ("metadata.zip", "objects.zip", "renders.zip")
    if not all((dataset_root / name).exists() for name in zip_names):
        raise FileNotFoundError(
            f"No metadata/views.csv and no Kaggle zip files under {dataset_root}. "
            "Check the path with: !find /kaggle/input -maxdepth 5 -type f | head"
        )

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
    rows: List[dict] = []
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
        {"train_uids": sorted(train_uids), "val_uids": sorted(val_uids), "test_uids": sorted(test_uids)},
    )


def stable_seed(text: str, seed: int) -> int:
    value = seed & 0xFFFFFFFF
    for ch in text:
        value = ((value * 131) + ord(ch)) & 0xFFFFFFFF
    return value


def load_sample(row: dict, image_size: int, target_points: int, seed: int, augment: bool):
    from PIL import Image

    img = Image.open(row["image_path"]).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    image = np.asarray(img, dtype=np.float32) / 255.0

    rng = np.random.default_rng(stable_seed(row["image_path"], seed))
    if augment:
        brightness = rng.uniform(0.90, 1.10)
        contrast = rng.uniform(0.90, 1.10)
        image = np.clip((image - 0.5) * contrast + 0.5, 0.0, 1.0)
        image = np.clip(image * brightness, 0.0, 1.0)

    data = np.load(row["points_path"])
    points = np.asarray(data["points"], dtype=np.float32)
    normals = np.asarray(data["normals"], dtype=np.float32) if "normals" in data.files else np.zeros_like(points)
    idx = rng.choice(len(points), size=target_points, replace=len(points) < target_points)
    return image.astype(np.float32), points[idx].astype(np.float32), normals[idx].astype(np.float32)


def batch_iterator(
    rows: List[dict],
    image_size: int,
    target_points: int,
    batch: int,
    seed: int,
    training: bool,
) -> Iterator[dict]:
    epoch = 0
    while True:
        order = list(range(len(rows)))
        if training:
            rng = random.Random(seed + epoch)
            rng.shuffle(order)
        usable = len(order) - (len(order) % batch)
        for start in range(0, usable, batch):
            ids = order[start:start + batch]
            samples = [load_sample(rows[i], image_size, target_points, seed + epoch, training) for i in ids]
            images, points, normals = zip(*samples)
            yield {
                "image": np.stack(images, axis=0),
                "points": np.stack(points, axis=0),
                "normals": np.stack(normals, axis=0),
            }
        epoch += 1


def shard_batch(batch: dict, devices: int) -> dict:
    return {k: v.reshape((devices, v.shape[0] // devices) + v.shape[1:]) for k, v in batch.items()}


def replicate_tree(tree, devices):
    import jax
    import jax.numpy as jnp

    return jax.tree_util.tree_map(lambda x: jax.device_put(jnp.stack([x] * len(devices), axis=0)), tree)


def unreplicate_tree(tree):
    import jax

    return jax.tree_util.tree_map(lambda x: jax.device_get(x[0]), tree)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--skip_install", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()
    ensure_deps(pre_args.skip_install)

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax.training import train_state
    import optax
    import orbax.checkpoint as ocp

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--work_dir", default="/kaggle/working/chair_lrm_tpu")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--pred_points", type=int, default=8192)
    parser.add_argument("--target_points", type=int, default=8192)
    parser.add_argument("--chamfer_points", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_per_device", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--fscore_thresholds", default="0.02,0.05")
    parser.add_argument("--resume_from", default="", help="Optional Orbax checkpoint directory, e.g. /kaggle/working/chair_lrm_tpu/best_orbax")
    parser.add_argument("--skip_install", action="store_true")
    args = parser.parse_args()

    if args.chamfer_points > args.pred_points or args.chamfer_points > args.target_points:
        raise ValueError("--chamfer_points must be <= --pred_points and <= --target_points")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = maybe_extract_dataset(Path(args.dataset_root), work_dir)
    rows = read_rows(dataset_root)
    train_rows, val_rows, test_rows, split = split_by_uid(rows, args.seed, args.train_ratio, args.val_ratio)

    devices = jax.local_device_count()
    print("JAX devices:", jax.devices(), flush=True)
    print(f"local_device_count={devices}", flush=True)
    if devices < 2:
        print("WARNING: only one JAX device is visible; this is not TPU v5e-8 training.", flush=True)

    global_batch = devices * args.batch_per_device
    steps_per_epoch = max(1, len(train_rows) // global_batch)
    val_steps = max(1, min(80, len(val_rows) // global_batch))
    thresholds = tuple(float(x.strip()) for x in args.fscore_thresholds.split(",") if x.strip())

    print(f"Dataset root: {dataset_root}", flush=True)
    print(f"Rows train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)
    print(f"UIDs train={len(split['train_uids'])} val={len(split['val_uids'])} test={len(split['test_uids'])}", flush=True)
    print(f"global_batch={global_batch} steps_per_epoch={steps_per_epoch} val_steps={val_steps}", flush=True)
    (work_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (work_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

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

    def pairwise_dist(pred, target):
        diff = pred[:, :, None, :] - target[:, None, :, :]
        return jnp.sum(diff * diff, axis=-1)

    def subsample(points, count: int, key):
        idx = jax.random.choice(key, points.shape[1], (count,), replace=False)
        return points[:, idx, :]

    def metrics_from_dist(dist):
        min_pt = jnp.min(dist, axis=2)
        min_tp = jnp.min(dist, axis=1)
        chamfer_l2 = jnp.mean(min_pt) + jnp.mean(min_tp)
        chamfer_l1 = jnp.mean(jnp.sqrt(min_pt + 1e-8)) + jnp.mean(jnp.sqrt(min_tp + 1e-8))
        out = {"chamfer_l2": chamfer_l2, "chamfer_l1": chamfer_l1}
        for thr in thresholds:
            thr2 = thr * thr
            precision = jnp.mean(min_pt < thr2)
            recall = jnp.mean(min_tp < thr2)
            fscore = 2.0 * precision * recall / (precision + recall + 1e-8)
            key = str(thr).replace(".", "p")
            out[f"precision_{key}"] = precision
            out[f"recall_{key}"] = recall
            out[f"fscore_{key}"] = fscore
        return out

    def train_loss_and_metrics(pred, target, key):
        k1, k2 = jax.random.split(key)
        pred_s = subsample(pred, args.chamfer_points, k1)
        target_s = subsample(target, args.chamfer_points, k2)
        dist = pairwise_dist(pred_s, target_s)
        m = metrics_from_dist(dist)
        center_penalty = 0.005 * jnp.mean(jnp.square(jnp.mean(pred, axis=1) - jnp.mean(target, axis=1)))
        range_penalty = 0.001 * (
            jnp.mean(jnp.maximum(jnp.abs(pred[:, :, 0:2]) - 1.0, 0.0) ** 2)
            + jnp.mean(jnp.maximum(pred[:, :, 2:3] - 1.9, 0.0) ** 2)
            + jnp.mean(jnp.maximum(-pred[:, :, 2:3], 0.0) ** 2)
        )
        loss = m["chamfer_l2"] + 0.15 * m["chamfer_l1"] + center_penalty + range_penalty
        m["loss"] = loss
        return loss, m

    class State(train_state.TrainState):
        pass

    model = ChairLRMLite(args.pred_points)
    rng = jax.random.PRNGKey(args.seed)
    variables = model.init(rng, jnp.ones((1, args.image_size, args.image_size, 3), jnp.float32), training=True)

    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, steps_per_epoch * args.warmup_epochs)
    warmup = optax.linear_schedule(0.0, args.lr, warmup_steps)
    cosine = optax.cosine_decay_schedule(args.lr, max(1, total_steps - warmup_steps), alpha=0.03)
    schedule = optax.join_schedules([warmup, cosine], [warmup_steps])
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=args.weight_decay),
    )
    state = State.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_from does not exist: {resume_path}")
        ckptr = ocp.PyTreeCheckpointer()
        state = ckptr.restore(resume_path, item=state)
        print(f"Resumed checkpoint: {resume_path}", flush=True)
    state = replicate_tree(state, jax.local_devices())

    def train_step(state, batch, rng_key):
        def loss_fn(params):
            pred = state.apply_fn({"params": params}, batch["image"], training=True)
            return train_loss_and_metrics(pred, batch["points"], rng_key)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        metrics = jax.lax.pmean(metrics, axis_name="batch")
        state = state.apply_gradients(grads=grads)
        return state, metrics

    def eval_step(state, batch, rng_key):
        pred = state.apply_fn({"params": state.params}, batch["image"], training=False)
        _, metrics = train_loss_and_metrics(pred, batch["points"], rng_key)
        return jax.lax.pmean(metrics, axis_name="batch")

    train_step = jax.pmap(train_step, axis_name="batch")
    eval_step = jax.pmap(eval_step, axis_name="batch")

    train_iter = batch_iterator(train_rows, args.image_size, args.target_points, global_batch, args.seed, True)
    val_iter = batch_iterator(val_rows, args.image_size, args.target_points, global_batch, args.seed + 9999, False)

    metric_names = ["loss", "chamfer_l2", "chamfer_l1"]
    for thr in thresholds:
        key = str(thr).replace(".", "p")
        metric_names.extend([f"precision_{key}", f"recall_{key}", f"fscore_{key}"])

    best_val = float("inf")
    log_path = work_dir / "train_log.csv"
    print("Training started.", flush=True)
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "epoch_min", "lr", *[f"train_{m}" for m in metric_names], *[f"val_{m}" for m in metric_names]])

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_sums = {name: 0.0 for name in metric_names}
            for step in range(1, steps_per_epoch + 1):
                batch = shard_batch(next(train_iter), devices)
                step_key = jax.random.fold_in(rng, (epoch * 100000) + step)
                step_keys = jax.random.split(step_key, devices)
                state, metrics = train_step(state, batch, step_keys)
                host_metrics = jax.device_get(metrics)
                for name in metric_names:
                    train_sums[name] += float(np.asarray(host_metrics[name])[0])
                if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == steps_per_epoch):
                    elapsed = time.time() - epoch_start
                    sec_per_step = elapsed / step
                    epoch_eta = sec_per_step * (steps_per_epoch - step)
                    total_remaining = (args.epochs - epoch) * steps_per_epoch + (steps_per_epoch - step)
                    print(
                        f"epoch={epoch:03d}/{args.epochs} step={step:04d}/{steps_per_epoch} "
                        f"loss={float(np.asarray(host_metrics['loss'])[0]):.6f} "
                        f"ch_l2={float(np.asarray(host_metrics['chamfer_l2'])[0]):.6f} "
                        f"ch_l1={float(np.asarray(host_metrics['chamfer_l1'])[0]):.6f} "
                        f"sec/step={sec_per_step:.2f} "
                        f"epoch_eta_min={epoch_eta / 60:.1f} total_eta_hr={(sec_per_step * total_remaining) / 3600:.2f}",
                        flush=True,
                    )

            val_sums = {name: 0.0 for name in metric_names}
            for step in range(1, val_steps + 1):
                batch = shard_batch(next(val_iter), devices)
                step_key = jax.random.fold_in(rng, 900000000 + epoch * 100000 + step)
                step_keys = jax.random.split(step_key, devices)
                metrics = eval_step(state, batch, step_keys)
                host_metrics = jax.device_get(metrics)
                for name in metric_names:
                    val_sums[name] += float(np.asarray(host_metrics[name])[0])

            train_avg = {name: train_sums[name] / steps_per_epoch for name in metric_names}
            val_avg = {name: val_sums[name] / val_steps for name in metric_names}
            lr_value = float(schedule((epoch - 1) * steps_per_epoch))
            epoch_min = (time.time() - epoch_start) / 60.0
            writer.writerow([
                epoch,
                epoch_min,
                lr_value,
                *[train_avg[name] for name in metric_names],
                *[val_avg[name] for name in metric_names],
            ])
            f.flush()

            fscore_bits = []
            for thr in thresholds:
                key = str(thr).replace(".", "p")
                fscore_bits.append(f"val_f@{thr:g}={val_avg[f'fscore_{key}']:.4f}")
            print(
                f"epoch={epoch:03d} train_loss={train_avg['loss']:.6f} "
                f"val_loss={val_avg['loss']:.6f} val_ch_l2={val_avg['chamfer_l2']:.6f} "
                f"val_ch_l1={val_avg['chamfer_l1']:.6f} {' '.join(fscore_bits)} "
                f"epoch_min={epoch_min:.1f}",
                flush=True,
            )

            if val_avg["loss"] < best_val:
                best_val = val_avg["loss"]
                ckpt_dir = work_dir / "best_orbax"
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                ckptr = ocp.PyTreeCheckpointer()
                ckptr.save(ckpt_dir, unreplicate_tree(state))
                print(f"saved best checkpoint: {ckpt_dir}", flush=True)

    print(f"Done. Best val_loss={best_val:.6f}", flush=True)
    print(f"Log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
