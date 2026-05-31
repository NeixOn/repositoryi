#!/usr/bin/env python3
"""
CUDA/DDP triplane-UDF training for single-image chair mesh reconstruction.

Recommended server run:

  torchrun --standalone --nproc_per_node=2 train_chair_triplane_udf_cuda.py \
    --dataset_root /data/abo_chairs \
    --work_dir /data/runs/chair_triplane_udf \
    --encoder convnext_base \
    --image_size 256 \
    --batch_size 2 \
    --grad_accum 8 \
    --queries_per_item 32768 \
    --surface_points 8192 \
    --plane_size 128 \
    --plane_channels 48 \
    --decoder_hidden 384 \
    --epochs 80 \
    --amp fp16

This is a mesh-first model:
  image -> pretrained ConvNeXt/ResNet encoder -> triplanes -> UDF MLP -> marching cubes mesh
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
    pkgs = ["torch", "torchvision", "Pillow", "scipy", "scikit-image", "trimesh"]
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
            image_path = dataset_root / row.get("image_path", "")
            if not image_path.exists():
                image_path = dataset_root / "renders" / uid / f"view_{view_index:03d}.png"
            points_path = dataset_root / "objects" / uid / "points.npz"
            if image_path.exists() and points_path.exists():
                rows.append({"uid": uid, "view_index": view_index, "image_path": str(image_path), "points_path": str(points_path)})
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
    from PIL import Image, ImageOps

    img = Image.open(path).convert("RGB")
    if crop:
        arr = np.asarray(img, dtype=np.uint8)
        mask = np.any(arr < 245, axis=-1)
        if mask.any():
            ys, xs = np.where(mask)
            pad = int(max(arr.shape[:2]) * 0.06)
            img = img.crop((max(0, xs.min() - pad), max(0, ys.min() - pad), min(arr.shape[1], xs.max() + pad + 1), min(arr.shape[0], ys.max() + pad + 1)))
    img = ImageOps.contain(img, (image_size, image_size))
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    canvas.paste(img, ((image_size - img.width) // 2, (image_size - img.height) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)), canvas


class ChairUDFDataset:
    def __init__(self, rows, image_size, queries_per_item, surface_points, seed, training, truncation, crop):
        self.rows = rows
        self.image_size = image_size
        self.queries_per_item = queries_per_item
        self.surface_points = surface_points
        self.seed = seed
        self.training = training
        self.truncation = truncation
        self.crop = crop
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
        if len(self._cache) > 64:
            self._cache.pop(next(iter(self._cache)))
        return points, normals, tree

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        rng = np.random.default_rng(stable_seed(row["image_path"], self.seed + (1009 if self.training else 0)))
        image, _ = load_rgb_image(row["image_path"], self.image_size, crop=self.crop)
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
        sigma = rng.choice(np.array([0.01, 0.025, 0.05, 0.10], dtype=np.float32), n_near, p=[0.25, 0.35, 0.25, 0.15])
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

    return {k: torch.from_numpy(np.stack([item[k] for item in items], axis=0)) for k in items[0].keys()}


def build_model(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.models as models

    class ImageEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            if args.encoder == "convnext_base":
                weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if args.pretrained_encoder else None
                net = models.convnext_base(weights=weights)
                self.features = net.features
                in_dim = 1024
            elif args.encoder == "resnet101":
                weights = models.ResNet101_Weights.IMAGENET1K_V2 if args.pretrained_encoder else None
                net = models.resnet101(weights=weights)
                self.features = nn.Sequential(*list(net.children())[:-2])
                in_dim = 2048
            else:
                weights = models.ResNet50_Weights.IMAGENET1K_V2 if args.pretrained_encoder else None
                net = models.resnet50(weights=weights)
                self.features = nn.Sequential(*list(net.children())[:-2])
                in_dim = 2048
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.proj = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, args.latent_dim), nn.LayerNorm(args.latent_dim), nn.GELU(), nn.Linear(args.latent_dim, args.latent_dim))

        def forward(self, x):
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = (x - mean) / std
            return self.proj(self.pool(self.features(x)))

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
            h = args.decoder_hidden
            self.net = nn.Sequential(
                nn.Linear(in_dim, h), nn.LayerNorm(h), nn.SiLU(),
                nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(),
                nn.Linear(h, h), nn.LayerNorm(h), nn.SiLU(),
                nn.Linear(h, h), nn.SiLU(),
                nn.Linear(h, 1),
            )

        def sample(self, planes, q):
            b = q.shape[0]
            x = q[..., 0].clamp(-1.05, 1.05) / 1.05
            y = q[..., 1].clamp(-1.05, 1.05) / 1.05
            z = (q[..., 2].clamp(-0.08, 1.92) - 0.92) / 1.0
            coords = [torch.stack([x, y], -1), torch.stack([x, z], -1), torch.stack([y, z], -1)]
            feats = []
            for i, grid in enumerate(coords):
                sampled = F.grid_sample(planes[:, i], grid.view(b, -1, 1, 2), mode="bilinear", padding_mode="border", align_corners=True)
                feats.append(sampled.squeeze(-1).transpose(1, 2))
            return torch.cat(feats, -1)

        def forward(self, planes, q):
            f = self.sample(planes, q)
            return F.softplus(self.net(torch.cat([f, q], -1)), beta=10.0)

    class ChairTriplaneUDF(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ImageEncoder()
            self.triplane = TriplaneGenerator()
            self.decoder = UDFDecoder()

        def forward(self, image, query):
            latent = self.encoder(image)
            planes = self.triplane(latent)
            return self.decoder(planes, query)

    return ChairTriplaneUDF()


def loss_fn(pred, target, truncation):
    import torch
    import torch.nn.functional as F

    near_weight = 1.0 + 8.0 * torch.exp(-target / 0.035)
    l1 = (near_weight * torch.abs(pred - target)).mean()
    zero = target < 1e-5
    surface = F.smooth_l1_loss(pred[zero], target[zero]) if zero.any() else torch.zeros((), device=pred.device)
    far = torch.zeros((), device=pred.device)
    far_mask = target > truncation * 0.95
    if far_mask.any():
        far = 0.05 * torch.relu(pred[far_mask] - truncation).mean()
    return l1 + 2.0 * surface + far, {"l1": l1.detach(), "surface": surface.detach()}


def setup_distributed():
    import torch
    import torch.distributed as dist

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return True, rank, local_rank, world
    return False, 0, 0, 1


def is_main(rank: int) -> bool:
    return rank == 0


def save_checkpoint(path, model, optimizer, scaler, epoch, best_val, args):
    import torch
    import torch.nn as nn

    state_dict = model.module.state_dict() if isinstance(model, nn.parallel.DistributedDataParallel) else model.state_dict()
    torch.save({"model": state_dict, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict() if scaler else None, "epoch": epoch, "best_val": best_val, "args": vars(args)}, path)


def train(args):
    ensure_deps(args.skip_install)
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler

    ddp, rank, local_rank, world = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("CUDA was required but is not available")

    work_dir = Path(args.work_dir)
    if is_main(rank):
        work_dir.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()
    dataset_root = maybe_extract_dataset(Path(args.dataset_root), work_dir)
    rows = read_rows(dataset_root)
    train_rows, val_rows, test_rows, split = split_by_uid(rows, args.seed, args.train_ratio, args.val_ratio)
    if is_main(rank):
        (work_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
        (work_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"Device={device} world_size={world}", flush=True)
        print(f"Dataset root: {dataset_root}", flush=True)
        print(f"Rows train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    train_ds = ChairUDFDataset(train_rows, args.image_size, args.queries_per_item, args.surface_points, args.seed, True, args.truncation, args.crop)
    val_ds = ChairUDFDataset(val_rows, args.image_size, args.queries_per_item, args.surface_points, args.seed + 999, False, args.truncation, args.crop)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, shuffle=train_sampler is None, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch, drop_last=True, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, num_workers=max(0, min(args.num_workers, 8)), pin_memory=True, collate_fn=collate_batch, drop_last=True, persistent_workers=args.num_workers > 0)

    model = build_model(args).to(device)
    if args.freeze_encoder_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    steps_total = max(1, args.epochs * max(1, len(train_loader) // args.grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_total, eta_min=args.lr * 0.03)
    use_amp = args.amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and args.amp == "fp16"))
    best_val = float("inf")
    start_epoch = 1
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        target = model.module if ddp else model
        target.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))

    log_path = work_dir / "train_log.csv"
    if is_main(rank) and not log_path.exists():
        log_path.write_text("epoch,train_loss,val_loss,lr,epoch_min\n", encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if epoch == args.freeze_encoder_epochs + 1:
            target = model.module if ddp else model
            for p in target.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay, betas=(0.9, 0.95))
        model.train()
        start = time.time()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        for step, batch in enumerate(train_loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            query = batch["query"].to(device, non_blocking=True)
            udf = batch["udf"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(image, query)
                loss, parts = loss_fn(pred, udf, args.truncation)
                loss = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if is_main(rank) and (step == 1 or step % args.log_every == 0 or step == len(train_loader)):
                loss_value = float((loss.detach() * args.grad_accum).cpu())
                train_losses.append(loss_value)
                elapsed = time.time() - start
                sec = elapsed / step
                eta = sec * (len(train_loader) - step)
                print(f"epoch={epoch:03d}/{args.epochs} step={step:05d}/{len(train_loader)} loss={loss_value:.6f} l1={float(parts['l1'].cpu()):.6f} surf={float(parts['surface'].cpu()):.6f} sec/step={sec:.2f} eta_min={eta/60:.1f}", flush=True)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for step, batch in enumerate(val_loader, 1):
                if step > args.val_steps:
                    break
                image = batch["image"].to(device, non_blocking=True)
                query = batch["query"].to(device, non_blocking=True)
                udf = batch["udf"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model(image, query)
                    loss, _ = loss_fn(pred, udf, args.truncation)
                val_losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if ddp:
            tensor = torch.tensor([val_loss], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            val_loss = float(tensor.cpu())
        if is_main(rank):
            epoch_min = (time.time() - start) / 60.0
            lr = optimizer.param_groups[0]["lr"]
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch},{train_loss},{val_loss},{lr},{epoch_min}\n")
            print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.2e} epoch_min={epoch_min:.1f}", flush=True)
            save_checkpoint(work_dir / "latest.pt", model, optimizer, scaler, epoch, best_val, args)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(work_dir / "best.pt", model, optimizer, scaler, epoch, best_val, args)
                print(f"saved best checkpoint: {work_dir / 'best.pt'}", flush=True)
        if ddp:
            dist.barrier()


def predict(args):
    ensure_deps(args.skip_install)
    import torch
    from skimage import measure
    import trimesh

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved = argparse.Namespace(**ckpt.get("args", {}))
    for name in ("encoder", "image_size", "latent_dim", "plane_channels", "plane_size", "decoder_hidden", "pretrained_encoder"):
        setattr(args, name, getattr(saved, name, getattr(args, name)))
    args.pretrained_encoder = False
    model = build_model(args).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    image_np, resized = load_rgb_image(args.image, args.image_size, crop=args.crop)
    image = torch.from_numpy(image_np[None]).to(device)
    xs = np.linspace(-1.05, 1.05, args.grid_resolution, dtype=np.float32)
    ys = np.linspace(-1.05, 1.05, args.grid_resolution, dtype=np.float32)
    zs = np.linspace(-0.08, 1.92, args.grid_resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)
    values = np.empty(len(grid), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(grid), args.eval_chunk):
            q = torch.from_numpy(grid[start:start + args.eval_chunk][None]).to(device)
            values[start:start + args.eval_chunk] = model(image, q).reshape(-1).detach().cpu().numpy()
    field = values.reshape(args.grid_resolution, args.grid_resolution, args.grid_resolution)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    resized.save(out / f"{stem}_model_input.png")
    np.save(out / f"{stem}_udf_grid.npy", field)
    level = args.level
    if float(field.min()) > level:
        level = float(field.min() + 0.2 * (field.max() - field.min()))
    verts, faces, normals, _ = measure.marching_cubes(field, level=level, spacing=(2.10 / (args.grid_resolution - 1), 2.10 / (args.grid_resolution - 1), 2.00 / (args.grid_resolution - 1)))
    verts[:, 0] -= 1.05
    verts[:, 1] -= 1.05
    verts[:, 2] -= 0.08
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    mesh.export(out / f"{stem}_mesh.obj")
    mesh.export(out / f"{stem}_mesh.ply")
    print(f"Mesh OBJ: {out / f'{stem}_mesh.obj'}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("train", "predict"), default="train")
    p.add_argument("--dataset_root", default="")
    p.add_argument("--work_dir", default="/data/runs/chair_triplane_udf")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--resume_from", default="")
    p.add_argument("--image", default="")
    p.add_argument("--output_dir", default="/data/runs/chair_predict")
    p.add_argument("--encoder", choices=("convnext_base", "resnet101", "resnet50"), default="convnext_base")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--queries_per_item", type=int, default=32768)
    p.add_argument("--surface_points", type=int, default=8192)
    p.add_argument("--truncation", type=float, default=0.20)
    p.add_argument("--latent_dim", type=int, default=1024)
    p.add_argument("--plane_channels", type=int, default=48)
    p.add_argument("--plane_size", type=int, default=128)
    p.add_argument("--decoder_hidden", type=int, default=384)
    p.add_argument("--pretrained_encoder", action="store_true", default=True)
    p.add_argument("--no_pretrained_encoder", dest="pretrained_encoder", action="store_false")
    p.add_argument("--freeze_encoder_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", choices=("bf16", "fp16", "none"), default="fp16")
    p.add_argument("--train_ratio", type=float, default=0.88)
    p.add_argument("--val_ratio", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--val_steps", type=int, default=120)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--crop", action="store_true")
    p.add_argument("--grid_resolution", type=int, default=128)
    p.add_argument("--eval_chunk", type=int, default=131072)
    p.add_argument("--level", type=float, default=0.025)
    p.add_argument("--force_cpu", action="store_true")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--skip_install", action="store_true")
    args = p.parse_args()
    if args.mode == "train" and not args.dataset_root:
        raise ValueError("--dataset_root is required for training")
    if args.mode == "predict" and (not args.checkpoint or not args.image):
        raise ValueError("--checkpoint and --image are required for prediction")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.mode == "train":
        train(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
