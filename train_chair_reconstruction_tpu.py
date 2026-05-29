#!/usr/bin/env python3
"""
TPU-oriented single-image chair reconstruction training.

Input dataset layout, either extracted:
  dataset_root/
    renders/<uid>/view_000.png
    objects/<uid>/points.npz
    metadata/views.csv
    metadata/objects.csv

or Kaggle Dataset zip layout:
  dataset_root/renders.zip
  dataset_root/objects.zip
  dataset_root/metadata.zip

Run on Kaggle TPU v5e-8:
  !python train_chair_reconstruction_tpu.py \
    --dataset_root /kaggle/input/objaverse-chairs \
    --work_dir /kaggle/working/chair_recon_tpu \
    --pred_points 8192 \
    --target_points 8192 \
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
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def maybe_extract_dataset(dataset_root: Path, work_dir: Path) -> Path:
    dataset_root = dataset_root.resolve()
    if (dataset_root / "metadata" / "views.csv").exists():
        return dataset_root

    zip_names = ("metadata.zip", "objects.zip", "renders.zip")
    if not dataset_root.exists() and Path("/kaggle/input").exists():
        candidates = []
        for root in Path("/kaggle/input").iterdir():
            if not root.is_dir():
                continue
            has_extracted = (root / "metadata" / "views.csv").exists()
            has_zips = all((root / name).exists() for name in zip_names)
            if has_extracted or has_zips:
                candidates.append(root)
        if len(candidates) == 1:
            print(f"Dataset root {dataset_root} not found. Using detected Kaggle input: {candidates[0]}", flush=True)
            dataset_root = candidates[0].resolve()
        elif len(candidates) > 1:
            names = "\n".join(str(c) for c in candidates)
            raise FileNotFoundError(
                f"Dataset root {dataset_root} not found. Multiple possible Kaggle inputs detected:\n{names}\n"
                "Pass the correct one with --dataset_root."
            )

    if not all((dataset_root / name).exists() for name in zip_names):
        raise FileNotFoundError(
            f"Could not find extracted metadata/views.csv or Kaggle zip files under {dataset_root}. "
            "In Kaggle, check the real folder name with: !ls -la /kaggle/input"
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


def read_views(dataset_root: Path) -> List[dict]:
    views_csv = dataset_root / "metadata" / "views.csv"
    rows = []
    with open(views_csv, "r", encoding="utf-8", newline="") as f:
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
        raise RuntimeError(f"No usable rows found in {views_csv}")
    return rows


def split_by_uid(rows: List[dict], seed: int, train_ratio: float, val_ratio: float):
    uids = sorted({row["uid"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(uids)
    n = len(uids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    train_uids = set(uids[:n_train])
    val_uids = set(uids[n_train:n_train + n_val])
    test_uids = set(uids[n_train + n_val:])

    train = [row for row in rows if row["uid"] in train_uids]
    val = [row for row in rows if row["uid"] in val_uids]
    test = [row for row in rows if row["uid"] in test_uids]
    return train, val, test, {"train_uids": sorted(train_uids), "val_uids": sorted(val_uids), "test_uids": sorted(test_uids)}


def load_example_py(image_path_b: bytes, points_path_b: bytes, image_size: int, target_points: int, seed: int):
    image_path = image_path_b.decode("utf-8")
    points_path = points_path_b.decode("utf-8")

    import PIL.Image

    image = PIL.Image.open(image_path).convert("RGB").resize((image_size, image_size), PIL.Image.BILINEAR)
    image = np.asarray(image, dtype=np.float32) / 255.0

    data = np.load(points_path)
    points = np.asarray(data["points"], dtype=np.float32)
    if len(points) == 0:
        points = np.zeros((target_points, 3), dtype=np.float32)
    local_seed = (abs(hash(image_path)) + seed) % (2**32)
    rng = np.random.default_rng(local_seed)
    replace = len(points) < target_points
    idx = rng.choice(len(points), size=target_points, replace=replace)
    points = points[idx].astype(np.float32)
    return image, points


def make_dataset(rows: List[dict], image_size: int, target_points: int, batch_size: int, seed: int, training: bool):
    image_paths = [row["image_path"] for row in rows]
    points_paths = [row["points_path"] for row in rows]
    ds = tf.data.Dataset.from_tensor_slices((image_paths, points_paths))
    if training:
        ds = ds.shuffle(min(len(rows), 8192), seed=seed, reshuffle_each_iteration=True)

    def _load(image_path, points_path):
        image, points = tf.numpy_function(
            func=lambda ip, pp: load_example_py(ip, pp, image_size, target_points, seed),
            inp=[image_path, points_path],
            Tout=[tf.float32, tf.float32],
        )
        image.set_shape((image_size, image_size, 3))
        points.set_shape((target_points, 3))
        if training:
            image = tf.image.random_brightness(image, 0.08)
            image = tf.image.random_contrast(image, 0.85, 1.15)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image, points

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=training)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_model(image_size: int, pred_points: int, encoder: str, weights: str | None):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")

    if encoder == "efficientnetv2b0":
        base = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling="avg",
            include_preprocessing=False,
        )
        features = base.output
    elif encoder == "efficientnetv2s":
        base = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling="avg",
            include_preprocessing=False,
        )
        features = base.output
    elif encoder == "resnet50":
        x = tf.keras.applications.resnet.preprocess_input(inputs * 255.0)
        base = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_tensor=x, pooling="avg")
        features = base.output
    else:
        raise ValueError(f"Unsupported encoder: {encoder}")

    x = tf.keras.layers.LayerNormalization()(features)
    x = tf.keras.layers.Dense(2048, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(4096, activation="gelu")(x)
    x = tf.keras.layers.Dropout(0.10)(x)
    x = tf.keras.layers.Dense(pred_points * 3, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-4))(x)
    outputs = tf.keras.layers.Reshape((pred_points, 3), name="points")(x)
    return tf.keras.Model(inputs, outputs, name=f"chair_recon_{encoder}_{pred_points}")


def chamfer_distance_chunked(pred, target, chunk_size: int):
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)
    pred_count = pred.shape[1]
    target_count = target.shape[1]
    if pred_count is None or target_count is None:
        raise ValueError("Point counts must be statically known for chunked Chamfer loss")

    pred_sq = tf.reduce_sum(tf.square(pred), axis=-1, keepdims=True)
    target_sq_t = tf.transpose(tf.reduce_sum(tf.square(target), axis=-1, keepdims=True), [0, 2, 1])

    min_pred_parts = []
    pred_count = int(pred_count)
    target_count = int(target_count)
    for start in range(0, pred_count, chunk_size):
        chunk = pred[:, start:start + chunk_size, :]
        chunk_sq = tf.reduce_sum(tf.square(chunk), axis=-1, keepdims=True)
        dist = chunk_sq - 2.0 * tf.matmul(chunk, target, transpose_b=True) + target_sq_t
        min_pred_parts.append(tf.reduce_min(dist, axis=2))
    min_pred = tf.concat(min_pred_parts, axis=1)

    min_target_parts = []
    pred_sq_t = tf.transpose(pred_sq, [0, 2, 1])
    for start in range(0, target_count, chunk_size):
        chunk = target[:, start:start + chunk_size, :]
        chunk_sq = tf.reduce_sum(tf.square(chunk), axis=-1, keepdims=True)
        dist = chunk_sq - 2.0 * tf.matmul(chunk, pred, transpose_b=True) + pred_sq_t
        min_target_parts.append(tf.reduce_min(dist, axis=2))
    min_target = tf.concat(min_target_parts, axis=1)

    return tf.reduce_mean(min_pred) + tf.reduce_mean(min_target)


class ChairReconModel(tf.keras.Model):
    def __init__(self, network: tf.keras.Model, chamfer_points: int, chunk_size: int):
        super().__init__()
        self.network = network
        self.chamfer_points = chamfer_points
        self.chunk_size = chunk_size
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.chamfer_tracker = tf.keras.metrics.Mean(name="chamfer")

    @property
    def metrics(self):
        return [self.loss_tracker, self.chamfer_tracker]

    def call(self, inputs, training=False):
        return self.network(inputs, training=training)

    def _subsample(self, points):
        n = tf.shape(points)[1]
        k = tf.minimum(n, self.chamfer_points)
        idx = tf.random.shuffle(tf.range(n))[:k]
        return tf.gather(points, idx, axis=1)

    def train_step(self, data):
        images, target = data
        target_loss = self._subsample(target)
        with tf.GradientTape() as tape:
            pred = self.network(images, training=True)
            pred_loss = self._subsample(pred)
            chamfer = chamfer_distance_chunked(pred_loss, target_loss, self.chunk_size)
            reg = tf.add_n(self.network.losses) if self.network.losses else 0.0
            loss = chamfer + reg
        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.chamfer_tracker.update_state(chamfer)
        return {"loss": self.loss_tracker.result(), "chamfer": self.chamfer_tracker.result()}

    def test_step(self, data):
        images, target = data
        pred = self.network(images, training=False)
        chamfer = chamfer_distance_chunked(self._subsample(pred), self._subsample(target), self.chunk_size)
        self.loss_tracker.update_state(chamfer)
        self.chamfer_tracker.update_state(chamfer)
        return {"loss": self.loss_tracker.result(), "chamfer": self.chamfer_tracker.result()}


def setup_strategy():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("TPU initialized:", resolver.cluster_spec().as_dict(), flush=True)
        return tf.distribute.TPUStrategy(resolver)
    except Exception as exc:
        print(f"TPU not found, using default strategy: {exc}", flush=True)
        return tf.distribute.get_strategy()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--work_dir", default="/kaggle/working/chair_recon_tpu")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--pred_points", type=int, default=8192)
    parser.add_argument("--target_points", type=int, default=8192)
    parser.add_argument("--chamfer_points", type=int, default=4096)
    parser.add_argument("--chamfer_chunk", type=int, default=1024)
    parser.add_argument("--batch_per_replica", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=2e-6)
    parser.add_argument("--encoder", choices=("efficientnetv2b0", "efficientnetv2s", "resnet50"), default="efficientnetv2b0")
    parser.add_argument("--encoder_weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = maybe_extract_dataset(Path(args.dataset_root), work_dir)
    rows = read_views(dataset_root)
    train_rows, val_rows, test_rows, split = split_by_uid(rows, args.seed, args.train_ratio, args.val_ratio)

    print(f"Dataset root: {dataset_root}", flush=True)
    print(f"Rows: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)
    print(f"UIDs: train={len(split['train_uids'])} val={len(split['val_uids'])} test={len(split['test_uids'])}", flush=True)
    (work_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    (work_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    strategy = setup_strategy()
    global_batch = args.batch_per_replica * strategy.num_replicas_in_sync
    print(f"Replicas: {strategy.num_replicas_in_sync}; global_batch={global_batch}", flush=True)

    train_ds = make_dataset(train_rows, args.image_size, args.target_points, global_batch, args.seed, training=True)
    val_ds = make_dataset(val_rows, args.image_size, args.target_points, global_batch, args.seed, training=False)

    steps_per_epoch = max(1, len(train_rows) // global_batch)
    total_steps = max(1, steps_per_epoch * args.epochs)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=args.min_lr / args.lr,
    )

    with strategy.scope():
        weights = None if args.encoder_weights == "none" else args.encoder_weights
        network = build_model(args.image_size, args.pred_points, args.encoder, weights)
        model = ChairReconModel(network, args.chamfer_points, args.chamfer_chunk)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4, global_clipnorm=1.0)
        model.compile(optimizer=optimizer, jit_compile=False)

    ckpt_path = work_dir / "checkpoints" / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_chamfer",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        ),
        tf.keras.callbacks.CSVLogger(str(work_dir / "train_log.csv"), append=args.resume),
        tf.keras.callbacks.TensorBoard(log_dir=str(work_dir / "tb")),
    ]

    latest = tf.train.latest_checkpoint(str(work_dir / "tf_ckpt"))
    if args.resume and latest:
        print(f"Resuming from {latest}", flush=True)
        model.load_weights(latest)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )
    model.network.save(work_dir / "saved_model.keras")
    print(f"Saved model to {work_dir / 'saved_model.keras'}", flush=True)
    print(f"Best weights: {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
