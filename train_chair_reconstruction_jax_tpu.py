#!/usr/bin/env python3
"""
JAX/Flax TPU training for single-image chair -> point cloud reconstruction.

Designed for Kaggle TPU v5e-8 where TensorFlow may not expose TPU devices.

Run:
  !python /kaggle/working/repositoryi/train_chair_reconstruction_jax_tpu.py \
    --dataset_root /kaggle/input/datasets/neixon/objaverse-chairs \
    --work_dir /kaggle/working/chair_recon_jax \
    --pred_points 8192 \
    --target_points 8192 \
    --chamfer_points 4096 \
    --epochs 80
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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List


def ensure_deps(skip_install: bool):
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
            f"No metadata/views.csv or dataset zip files under {dataset_root}. Check with: !find /kaggle/input -maxdepth 4 -type f | head"
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
    rows = []
    with open(dataset_root / "metadata" / "views.csv", "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["uid"]
            view_index = int(row["view_index"])
            image_path = dataset_root / "renders" / uid / f"view_{view_index:03d}.png"
            points_path = dataset_root / "objects" / uid / "points.npz"
            if image_path.exists() and points_path.exists():
                rows.append({"uid": uid, "image_path": str(image_path), "points_path": str(points_path)})
    if not rows:
        raise RuntimeError("No usable training rows found")
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


def load_sample(row: dict, image_size: int, target_points: int, seed: int):
    import numpy as np
    from PIL import Image

    img = Image.open(row["image_path"]).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    image = np.asarray(img, dtype=np.float32) / 255.0
    data = np.load(row["points_path"])
    points = np.asarray(data["points"], dtype=np.float32)
    rng = np.random.default_rng((abs(hash(row["image_path"])) + seed) % (2**32))
    idx = rng.choice(len(points), size=target_points, replace=len(points) < target_points)
    return image, points[idx].astype(np.float32)


def batch_iterator(rows: List[dict], image_size: int, target_points: int, batch: int, seed: int, training: bool) -> Iterator[dict]:
    import numpy as np

    epoch = 0
    while True:
        order = list(range(len(rows)))
        if training:
            rng = random.Random(seed + epoch)
            rng.shuffle(order)
        for start in range(0, len(order) - batch + 1, batch):
            ids = order[start:start + batch]
            images, points = zip(*(load_sample(rows[i], image_size, target_points, seed + epoch) for i in ids))
            yield {"image": np.stack(images, axis=0), "points": np.stack(points, axis=0)}
        epoch += 1


def shard_batch(batch: dict, devices: int):
    import numpy as np

    return {k: v.reshape((devices, v.shape[0] // devices) + v.shape[1:]) for k, v in batch.items()}


def replicate_tree(tree, devices):
    import jax

    return jax.tree_util.tree_map(lambda x: jax.device_put_sharded([x] * len(devices), devices), tree)


def main():
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
    parser.add_argument("--work_dir", default="/kaggle/working/chair_recon_jax")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--pred_points", type=int, default=8192)
    parser.add_argument("--target_points", type=int, default=8192)
    parser.add_argument("--chamfer_points", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_per_device", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--skip_install", action="store_true")
    args = parser.parse_args()

    if args.skip_install:
        pass

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = maybe_extract_dataset(Path(args.dataset_root), work_dir)
    rows = read_rows(dataset_root)
    train_rows, val_rows, test_rows, split = split_by_uid(rows, args.seed, args.train_ratio, args.val_ratio)

    devices = jax.local_device_count()
    print("JAX devices:", jax.devices(), flush=True)
    print(f"local_device_count={devices}", flush=True)
    if devices < 2:
        print("WARNING: only one JAX device is visible. This is not using v5e-8 as expected.", flush=True)

    global_batch = devices * args.batch_per_device
    steps_per_epoch = max(1, len(train_rows) // global_batch)
    val_steps = max(1, min(50, len(val_rows) // global_batch))
    print(f"Rows train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)
    print(f"global_batch={global_batch} steps_per_epoch={steps_per_epoch} val_steps={val_steps}", flush=True)
    (work_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (work_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    class ConvBlock(nn.Module):
        channels: int
        stride: int = 1

        @nn.compact
        def __call__(self, x, training: bool):
            x = nn.Conv(self.channels, (3, 3), strides=(self.stride, self.stride), padding="SAME", use_bias=False)(x)
            x = nn.BatchNorm(use_running_average=not training, momentum=0.9)(x)
            x = nn.silu(x)
            return x

    class ChairNet(nn.Module):
        pred_points: int

        @nn.compact
        def __call__(self, x, training: bool):
            x = ConvBlock(32, 2)(x, training)
            x = ConvBlock(64, 2)(x, training)
            x = ConvBlock(96, 2)(x, training)
            x = ConvBlock(160, 2)(x, training)
            x = ConvBlock(256, 2)(x, training)
            x = jnp.mean(x, axis=(1, 2))
            x = nn.LayerNorm()(x)
            x = nn.Dense(2048)(x)
            x = nn.gelu(x)
            x = nn.Dense(4096)(x)
            x = nn.gelu(x)
            x = nn.Dense(self.pred_points * 3, kernel_init=nn.initializers.normal(1e-4))(x)
            return x.reshape((x.shape[0], self.pred_points, 3))

    def chamfer(pred, target):
        pred = pred.astype(jnp.float32)
        target = target.astype(jnp.float32)
        pidx = jax.random.choice(jax.random.PRNGKey(123), pred.shape[1], (args.chamfer_points,), replace=False)
        tidx = jax.random.choice(jax.random.PRNGKey(456), target.shape[1], (args.chamfer_points,), replace=False)
        pred = pred[:, pidx, :]
        target = target[:, tidx, :]
        diff = pred[:, :, None, :] - target[:, None, :, :]
        dist = jnp.sum(diff * diff, axis=-1)
        return jnp.mean(jnp.min(dist, axis=2)) + jnp.mean(jnp.min(dist, axis=1))

    class State(train_state.TrainState):
        batch_stats: dict

    rng = jax.random.PRNGKey(args.seed)
    model = ChairNet(args.pred_points)
    variables = model.init(rng, jnp.ones((1, args.image_size, args.image_size, 3), jnp.float32), training=True)
    schedule = optax.cosine_decay_schedule(args.lr, decay_steps=max(1, steps_per_epoch * args.epochs), alpha=0.01)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    state = State.create(apply_fn=model.apply, params=variables["params"], tx=tx, batch_stats=variables.get("batch_stats", {}))
    state = replicate_tree(state, jax.local_devices())

    @jax.pmap(axis_name="batch")
    def train_step(state, batch):
        def loss_fn(params):
            vars_in = {"params": params, "batch_stats": state.batch_stats}
            pred, updates = state.apply_fn(vars_in, batch["image"], training=True, mutable=["batch_stats"])
            loss = chamfer(pred, batch["points"])
            return loss, updates
        (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")
        state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])
        return state, loss

    @jax.pmap(axis_name="batch")
    def eval_step(state, batch):
        pred = state.apply_fn({"params": state.params, "batch_stats": state.batch_stats}, batch["image"], training=False)
        loss = chamfer(pred, batch["points"])
        return jax.lax.pmean(loss, axis_name="batch")

    train_iter = batch_iterator(train_rows, args.image_size, args.target_points, global_batch, args.seed, True)
    val_iter = batch_iterator(val_rows, args.image_size, args.target_points, global_batch, args.seed + 999, False)

    best_val = float("inf")
    log_path = work_dir / "train_log.csv"
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_chamfer", "val_chamfer"])

        for epoch in range(1, args.epochs + 1):
            train_losses = []
            for _ in range(steps_per_epoch):
                batch = shard_batch(next(train_iter), devices)
                state, loss = train_step(state, batch)
                train_losses.append(float(np.asarray(loss)[0]))
            val_losses = []
            for _ in range(val_steps):
                batch = shard_batch(next(val_iter), devices)
                val_loss = eval_step(state, batch)
                val_losses.append(float(np.asarray(val_loss)[0]))
            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            writer.writerow([epoch, train_loss, val_loss])
            f.flush()
            print(f"epoch={epoch:03d} train_chamfer={train_loss:.6f} val_chamfer={val_loss:.6f}", flush=True)

            if val_loss < best_val:
                best_val = val_loss
                ckpt_dir = work_dir / "best_orbax"
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                ckptr = ocp.PyTreeCheckpointer()
                ckptr.save(ckpt_dir, jax.device_get(state))
                print(f"saved best checkpoint: {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
