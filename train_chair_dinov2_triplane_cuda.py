#!/usr/bin/env python3
"""
DINOv2 + triplane-UDF chair reconstruction training on CUDA/DDP.

This is the strongest practical from-scratch model in this repository for the
new Blender dataset:

  RGB chair image -> pretrained DINOv2 encoder -> learned triplanes
  -> UDF implicit decoder -> marching cubes mesh.

It intentionally avoids PyTorch3D/nvdiffrast so a fresh A40 server can run it
with a plain PyTorch CUDA environment.

Recommended 2xA40 run:
  torchrun --standalone --nproc_per_node=2 /data/repositoryi/train_chair_dinov2_triplane_cuda.py \
    --mode train \
    --dataset_root /data/datasets/objaverse-chair-blender-dataset \
    --work_dir /data/runs/chair_dinov2_triplane \
    --image_size 518 \
    --dinov2_model dinov2_vitl14_reg \
    --batch_size 1 \
    --grad_accum 8 \
    --queries_per_item 49152 \
    --surface_points 8192 \
    --plane_size 128 \
    --plane_channels 64 \
    --decoder_hidden 512 \
    --latent_dim 1024 \
    --epochs 60 \
    --lr 8e-5 \
    --encoder_lr 2e-6 \
    --unfreeze_encoder_epoch 20 \
    --amp bf16 \
    --num_workers 12 \
    --require_cuda
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
from pathlib import Path
from typing import List

import numpy as np


def ensure_deps(skip_install: bool) -> None:
    if skip_install:
        return
    pkgs = ["torch", "torchvision", "Pillow", "scipy", "scikit-image", "trimesh", "tqdm"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore", *pkgs], check=True)


def read_rows(dataset_root: Path) -> List[dict]:
    rows: List[dict] = []
    views_csv = dataset_root / "metadata" / "views.csv"
    if not views_csv.exists():
        raise FileNotFoundError(f"Missing {views_csv}")

    with open(views_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["uid"]
            view = int(row.get("view", row.get("view_index", 0)))
            image_path = dataset_root / row.get("image_path", f"renders/{uid}/view_{view:03d}.png")
            mask_path = dataset_root / row.get("mask_path", f"masks/{uid}/view_{view:03d}.png")
            points_path = dataset_root / "objects" / uid / "points.npz"
            if image_path.exists() and points_path.exists():
                rows.append(
                    {
                        "uid": uid,
                        "view": view,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path) if mask_path.exists() else "",
                        "points_path": str(points_path),
                    }
                )
    if not rows:
        raise RuntimeError(f"No usable rows under {dataset_root}")
    return rows


def split_by_uid_from_metadata(dataset_root: Path, rows: List[dict], seed: int, train_ratio: float, val_ratio: float):
    splits_path = dataset_root / "metadata" / "splits.json"
    if splits_path.exists():
        with open(splits_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        train_uids = set(raw.get("train", []))
        val_uids = set(raw.get("val", []))
        test_uids = set(raw.get("test", []))
    else:
        uids = sorted({r["uid"] for r in rows})
        rng = random.Random(seed)
        rng.shuffle(uids)
        n_train = max(1, int(len(uids) * train_ratio))
        n_val = max(1, int(len(uids) * val_ratio))
        train_uids = set(uids[:n_train])
        val_uids = set(uids[n_train : n_train + n_val])
        test_uids = set(uids[n_train + n_val :])
    return (
        [r for r in rows if r["uid"] in train_uids],
        [r for r in rows if r["uid"] in val_uids],
        [r for r in rows if r["uid"] in test_uids],
        {"train": sorted(train_uids), "val": sorted(val_uids), "test": sorted(test_uids)},
    )


def stable_seed(text: str, seed: int) -> int:
    value = seed & 0xFFFFFFFF
    for ch in text:
        value = ((value * 131) + ord(ch)) & 0xFFFFFFFF
    return value


def load_image_and_mask(image_path: str, mask_path: str, image_size: int, crop: bool):
    from PIL import Image, ImageOps

    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L") if mask_path else None
    if crop and mask is not None:
        mask_arr = np.asarray(mask, dtype=np.uint8)
        if mask_arr.max() > 0:
            ys, xs = np.where(mask_arr > 8)
            pad = int(max(mask_arr.shape[:2]) * 0.08)
            box = (
                max(0, xs.min() - pad),
                max(0, ys.min() - pad),
                min(mask_arr.shape[1], xs.max() + pad + 1),
                min(mask_arr.shape[0], ys.max() + pad + 1),
            )
            img = img.crop(box)
            mask = mask.crop(box)

    img = ImageOps.contain(img, (image_size, image_size))
    canvas = Image.new("RGB", (image_size, image_size), (219, 222, 224))
    canvas.paste(img, ((image_size - img.width) // 2, (image_size - img.height) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


class ChairDataset:
    def __init__(
        self,
        rows,
        image_size,
        queries_per_item,
        surface_points,
        seed,
        training,
        truncation,
        crop,
        cache_size,
    ):
        self.rows = rows
        self.image_size = image_size
        self.queries_per_item = queries_per_item
        self.surface_points = surface_points
        self.seed = seed
        self.training = training
        self.truncation = truncation
        self.crop = crop
        self.cache_size = cache_size
        self._cache = {}

    def __len__(self):
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
        if len(self._cache) > self.cache_size:
            self._cache.pop(next(iter(self._cache)))
        return points, normals, tree

    def __getitem__(self, idx):
        row = self.rows[idx]
        seed = stable_seed(row["image_path"], self.seed + (1009 if self.training else 0))
        rng = np.random.default_rng(seed)
        image = load_image_and_mask(row["image_path"], row["mask_path"], self.image_size, self.crop)
        if self.training:
            image = np.clip((image - 0.5) * rng.uniform(0.90, 1.12) + 0.5, 0.0, 1.0)
            image = np.clip(image * rng.uniform(0.88, 1.12), 0.0, 1.0)

        surface, normals, tree = self._load_points(row["points_path"])

        n_surface = self.queries_per_item // 4
        n_near = self.queries_per_item // 2
        n_uniform = self.queries_per_item - n_surface - n_near

        s_idx = rng.choice(len(surface), n_surface, replace=len(surface) < n_surface)
        q_surface = surface[s_idx]

        n_idx = rng.choice(len(surface), n_near, replace=len(surface) < n_near)
        sigma = rng.choice(
            np.array([0.008, 0.02, 0.045, 0.09], dtype=np.float32),
            n_near,
            p=[0.25, 0.35, 0.25, 0.15],
        )
        q_near = surface[n_idx] + rng.normal(size=(n_near, 3)).astype(np.float32) * sigma[:, None]

        q_uniform = np.empty((n_uniform, 3), dtype=np.float32)
        q_uniform[:, 0:2] = rng.uniform(-1.05, 1.05, size=(n_uniform, 2))
        q_uniform[:, 2] = rng.uniform(-0.08, 1.92, size=n_uniform)

        queries = np.concatenate([q_surface, q_near, q_uniform], axis=0).astype(np.float32)
        dist, _ = tree.query(queries, k=1, workers=1)
        udf = np.clip(dist.astype(np.float32), 0.0, self.truncation)[:, None]
        udf[:n_surface] = 0.0
        order = rng.permutation(len(queries))

        surf_idx = rng.choice(len(surface), self.surface_points, replace=len(surface) < self.surface_points)
        return {
            "image": image.astype(np.float32),
            "query": queries[order],
            "udf": udf[order],
            "surface": surface[surf_idx].astype(np.float32),
            "surface_normal": normals[surf_idx].astype(np.float32),
        }


def collate_batch(items):
    import torch

    return {key: torch.from_numpy(np.stack([item[key] for item in items], axis=0)) for key in items[0].keys()}


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

        def forward(self, x):
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
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
            return x.view(b, 3, self.c, self.s, self.s)

    class UDFDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            in_dim = args.plane_channels * 3 + 3
            layers = []
            dim = in_dim
            for _ in range(args.decoder_layers):
                layers += [nn.Linear(dim, args.decoder_hidden), nn.SiLU()]
                dim = args.decoder_hidden
            layers += [nn.Linear(dim, 1), nn.Softplus(beta=10)]
            self.net = nn.Sequential(*layers)

        def sample(self, planes, query):
            b, _, c, _, _ = planes.shape
            x = query[..., 0].clamp(-1.05, 1.05) / 1.05
            y = query[..., 1].clamp(-1.05, 1.05) / 1.05
            z = ((query[..., 2].clamp(-0.08, 1.92) + 0.08) / 2.0) * 2.0 - 1.0
            coords = [torch.stack([x, y], dim=-1), torch.stack([x, z], dim=-1), torch.stack([y, z], dim=-1)]
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
            return torch.cat(feats + [query], dim=-1)

        def forward(self, planes, query):
            return self.net(self.sample(planes, query))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DINOv2Encoder()
            self.triplane = TriplaneGenerator()
            self.decoder = UDFDecoder()

        def forward(self, image, query):
            latent = self.encoder(image)
            planes = self.triplane(latent)
            return self.decoder(planes, query)

        def planes(self, image):
            return self.triplane(self.encoder(image))

    return Model()


def set_encoder_trainable(model, trainable: bool):
    module = model.module if hasattr(model, "module") else model
    for p in module.encoder.backbone.parameters():
        p.requires_grad = trainable


def make_optimizer(model, args, encoder_trainable: bool):
    import torch

    module = model.module if hasattr(model, "module") else model
    head_params = list(module.encoder.proj.parameters()) + list(module.triplane.parameters()) + list(module.decoder.parameters())
    params = [{"params": head_params, "lr": args.lr}]
    if encoder_trainable:
        params.append({"params": [p for p in module.encoder.backbone.parameters() if p.requires_grad], "lr": args.encoder_lr})
    return torch.optim.AdamW(params, weight_decay=args.weight_decay)


def udf_loss(pred, target, truncation):
    import torch

    l1 = torch.nn.functional.l1_loss(pred, target)
    l2 = torch.nn.functional.mse_loss(pred, target)
    surf = pred[target < 1e-5].abs().mean() if (target < 1e-5).any() else pred.mean() * 0.0
    far = torch.relu(pred[target > truncation * 0.95] - truncation).mean() if (target > truncation * 0.95).any() else pred.mean() * 0.0
    return l1 + 0.25 * l2 + 0.05 * surf + 0.02 * far, {"l1": l1.detach(), "l2": l2.detach(), "surf": surf.detach()}


def save_checkpoint(path: Path, model, optimizer, epoch, args, best_val):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    module = model.module if hasattr(model, "module") else model
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


def train(args):
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    ensure_deps(args.skip_install)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        world = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset_root = Path(args.dataset_root).resolve()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(dataset_root)
    train_rows, val_rows, test_rows, splits = split_by_uid_from_metadata(dataset_root, rows, args.seed, args.train_ratio, args.val_ratio)
    if rank == 0:
        print(f"Dataset root: {dataset_root}", flush=True)
        print(f"Rows train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)
        print(
            f"UIDs train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}",
            flush=True,
        )
        (work_dir / "splits_used.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
        (work_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"Preparing DINOv2 weights/cache: {args.dinov2_model}", flush=True)
        _ = torch.hub.load("facebookresearch/dinov2", args.dinov2_model)
        del _

    if distributed:
        torch.distributed.barrier()

    model = build_model(args).to(device)
    set_encoder_trainable(model, False)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_ds = ChairDataset(train_rows, args.image_size, args.queries_per_item, args.surface_points, args.seed, True, args.truncation, args.crop, args.cache_size)
    val_ds = ChairDataset(val_rows, args.image_size, args.val_queries, args.surface_points, args.seed + 17, False, args.truncation, args.crop, args.cache_size)
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

    optimizer = make_optimizer(model, args, encoder_trainable=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device.type == "cuda"))
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    best_val = float("inf")

    if rank == 0:
        print(f"Device={device} world={world}", flush=True)
        print(f"Training started. steps_per_epoch={len(train_loader)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if epoch == args.unfreeze_encoder_epoch:
            if rank == 0:
                print(f"Unfreezing DINOv2 backbone at epoch {epoch}", flush=True)
            set_encoder_trainable(model, True)
            optimizer = make_optimizer(model, args, encoder_trainable=True)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            query = batch["query"].to(device, non_blocking=True)
            target = batch["udf"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(args.amp != "none" and device.type == "cuda")):
                pred = model(image, query)
                loss, parts = udf_loss(pred, target, args.truncation)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(float(loss.detach().cpu()) * args.grad_accum)
            if rank == 0 and (step == 1 or step % args.log_every == 0 or step == len(train_loader)):
                sec = (time.time() - start) / step
                eta = sec * (len(train_loader) - step) / 60.0
                print(
                    f"epoch={epoch:03d}/{args.epochs} step={step:05d}/{len(train_loader)} "
                    f"loss={np.mean(train_losses[-args.log_every:]):.6f} sec/step={sec:.3f} eta_min={eta:.1f}",
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
                image = batch["image"].to(device, non_blocking=True)
                query = batch["query"].to(device, non_blocking=True)
                target = batch["udf"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(args.amp != "none" and device.type == "cuda")):
                    pred = model(image, query)
                    loss, _ = udf_loss(pred, target, args.truncation)
                val_losses.append(float(loss.detach().cpu()))

        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        if rank == 0:
            mins = (time.time() - start) / 60.0
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} epoch_min={mins:.1f}", flush=True)
            with open(work_dir / "train_log.csv", "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "epoch_min"])
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "epoch_min": mins})
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
    for key in ("grid_resolution", "level", "image", "output_dir", "checkpoint", "skip_install", "crop"):
        setattr(saved_args, key, getattr(args, key, getattr(saved_args, key, None)))
    model = build_model(saved_args).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    image = load_image_and_mask(args.image, "", saved_args.image_size, args.crop)
    image_t = torch.from_numpy(image[None]).to(device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = np.linspace(-1.05, 1.05, args.grid_resolution, dtype=np.float32)
    ys = np.linspace(-1.05, 1.05, args.grid_resolution, dtype=np.float32)
    zs = np.linspace(-0.08, 1.92, args.grid_resolution, dtype=np.float32)
    field = np.empty((args.grid_resolution, args.grid_resolution, args.grid_resolution), dtype=np.float32)
    with torch.no_grad():
        planes = model.planes(image_t)
        pts = []
        coords = []
        for zi, z in enumerate(zs):
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            q = np.stack([grid_x.reshape(-1), grid_y.reshape(-1), np.full(grid_x.size, z, dtype=np.float32)], axis=-1)
            for start in range(0, len(q), args.predict_chunk):
                query = torch.from_numpy(q[start : start + args.predict_chunk][None]).to(device)
                pred = model.decoder(planes, query).squeeze(0).squeeze(-1).float().cpu().numpy()
                pts.append(pred)
            field[:, :, zi] = np.concatenate(pts, axis=0).reshape(args.grid_resolution, args.grid_resolution)
            pts.clear()
    np.save(out_dir / "udf_grid.npy", field)
    verts, faces, normals, _ = measure.marching_cubes(
        field,
        level=args.level,
        spacing=(2.10 / (args.grid_resolution - 1), 2.10 / (args.grid_resolution - 1), 2.00 / (args.grid_resolution - 1)),
    )
    verts[:, 0] += -1.05
    verts[:, 1] += -1.05
    verts[:, 2] += -0.08
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    mesh.export(out_dir / "mesh.obj")
    mesh.export(out_dir / "mesh.ply")
    print(f"saved: {out_dir / 'mesh.obj'}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("train", "predict"), default="train")
    p.add_argument("--dataset_root", default="")
    p.add_argument("--work_dir", default="/data/runs/chair_dinov2_triplane")
    p.add_argument("--dinov2_model", default="dinov2_vitl14_reg")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--queries_per_item", type=int, default=49152)
    p.add_argument("--val_queries", type=int, default=24576)
    p.add_argument("--surface_points", type=int, default=8192)
    p.add_argument("--plane_size", type=int, default=128)
    p.add_argument("--plane_channels", type=int, default=64)
    p.add_argument("--decoder_hidden", type=int, default=512)
    p.add_argument("--decoder_layers", type=int, default=5)
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--encoder_lr", type=float, default=2e-6)
    p.add_argument("--weight_decay", type=float, default=0.03)
    p.add_argument("--unfreeze_encoder_epoch", type=int, default=20)
    p.add_argument("--truncation", type=float, default=0.16)
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="bf16")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--cache_size", type=int, default=96)
    p.add_argument("--crop", action="store_true", default=True)
    p.add_argument("--no_crop", action="store_false", dest="crop")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--val_batches", type=int, default=80)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_install", action="store_true")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--image", default="")
    p.add_argument("--output_dir", default="/data/runs/chair_dinov2_predict")
    p.add_argument("--grid_resolution", type=int, default=160)
    p.add_argument("--level", type=float, default=0.025)
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
