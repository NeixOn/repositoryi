#!/usr/bin/env python3
"""
Self-contained chair geometry baseline for Kaggle.

This script intentionally does not import any project modules. It trains a
single-image -> triplane -> surface-proximity field model from RGB, masks, and
surface point clouds.

Modes:
  check   - verify paths and split counts
  train   - train / overfit the baseline
  predict - extract a mesh from a checkpoint and one image+mask

The target is not true watertight occupancy. With only surface point clouds,
the robust first step is a soft near-surface field:
  target(x) = exp(-distance_to_surface(x) / truncation)
Marching Cubes at a configurable level extracts a mesh-like surface.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=("check", "train", "predict"), default="train")
    p.add_argument("--dataset_root", default="/kaggle/input/datasets/neixon/objaverse-chair-blender-dataset")
    p.add_argument("--mask_root", default="/kaggle/working/repaired_masks")
    p.add_argument("--clean_uids", default="/kaggle/working/chair_dataset_audit_repaired/clean_uids.txt")
    p.add_argument("--splits_json", default="/kaggle/working/chair_dataset_audit_repaired/splits.json")
    p.add_argument("--work_dir", default="/kaggle/working/chair_geometry_baseline")
    p.add_argument("--views", type=int, default=24)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps_per_epoch", type=int, default=0, help="0 means one pass over dataset items.")
    p.add_argument("--val_steps", type=int, default=50)
    p.add_argument("--max_train_objects", type=int, default=0)
    p.add_argument("--max_val_objects", type=int, default=0)
    p.add_argument("--overfit_objects", type=int, default=0, help="Use the first N train objects for both train and val.")
    p.add_argument("--queries", type=int, default=4096)
    p.add_argument("--positive_fraction", type=float, default=0.55)
    p.add_argument("--near_sigma", type=float, default=0.025)
    p.add_argument("--truncation", type=float, default=0.06)
    p.add_argument("--bounds", type=float, nargs=6, default=(-1.05, -1.05, -1.05, 1.05, 1.05, 1.05))
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--plane_channels", type=int, default=32)
    p.add_argument("--plane_size", type=int, default=64)
    p.add_argument("--decoder_hidden", type=int, default=192)
    p.add_argument("--decoder_layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", choices=("none", "fp16", "bf16"), default="fp16")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--image", default="")
    p.add_argument("--mask", default="")
    p.add_argument("--output_dir", default="/kaggle/working/chair_geometry_prediction")
    p.add_argument("--grid_resolution", type=int, default=96)
    p.add_argument("--mc_level", type=float, default=0.35)
    p.add_argument("--predict_chunk", type=int, default=131072)
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def read_uids(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_first(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    direct = root / filename
    if direct.exists():
        return direct
    for path in root.rglob(filename):
        if path.is_file():
            return path
    return None


def looks_like_dataset_root(path: Path) -> bool:
    return (path / "renders").exists() and (path / "objects").exists()


def resolve_dataset_root(root: Path) -> Path:
    if looks_like_dataset_root(root):
        return root
    if not root.exists():
        return root
    candidates = []
    for path in root.rglob("*"):
        if path.is_dir() and looks_like_dataset_root(path):
            candidates.append(path)
    if candidates:
        candidates = sorted(candidates, key=lambda p: (len(p.parts), str(p)))
        print(f"[paths] auto dataset_root: {candidates[0]}", flush=True)
        return candidates[0]
    return root


def discover_uids_from_layout(dataset_root: Path, mask_root: Path, views: int) -> list[str]:
    sources = []
    for dirname in ("renders", "objects"):
        root = dataset_root / dirname
        if root.exists():
            sources.extend([p.name for p in root.iterdir() if p.is_dir()])
    if mask_root.exists():
        sources.extend([p.name for p in mask_root.iterdir() if p.is_dir()])
    counts = {}
    for uid in sources:
        counts[uid] = counts.get(uid, 0) + 1
    uids = []
    for uid in sorted(counts):
        point_ok = points_path(dataset_root, uid).exists()
        views_ok = all(image_path(dataset_root, uid, v).exists() and mask_path(mask_root, uid, v).exists() for v in range(views))
        if point_ok and views_ok:
            uids.append(uid)
    return uids


def resolve_metadata_paths(args: argparse.Namespace) -> None:
    dataset_root = resolve_dataset_root(Path(args.dataset_root))
    args.dataset_root = str(dataset_root)

    mask_root = Path(args.mask_root)
    if not mask_root.exists():
        candidate = dataset_root / "masks"
        if candidate.exists():
            print(f"[paths] auto mask_root: {candidate}", flush=True)
            args.mask_root = str(candidate)

    clean_path = Path(args.clean_uids)
    if not clean_path.exists():
        found = find_first(dataset_root, "clean_uids.txt")
        if found:
            print(f"[paths] auto clean_uids: {found}", flush=True)
            args.clean_uids = str(found)

    splits_path = Path(args.splits_json)
    if not splits_path.exists():
        found = find_first(dataset_root, "splits.json")
        if found:
            print(f"[paths] auto splits_json: {found}", flush=True)
            args.splits_json = str(found)


def load_splits(args: argparse.Namespace) -> tuple[list[str], list[str], list[str]]:
    resolve_metadata_paths(args)
    splits_path = Path(args.splits_json)
    if splits_path.exists():
        raw = json.loads(splits_path.read_text(encoding="utf-8"))
        train = list(raw.get("train", []))
        val = list(raw.get("val", []))
        test = list(raw.get("test", []))
    else:
        clean_path = Path(args.clean_uids)
        if clean_path.exists():
            uids = read_uids(args.clean_uids)
        else:
            print("[paths] clean_uids/splits not found; discovering UID values from dataset folders.", flush=True)
            uids = discover_uids_from_layout(Path(args.dataset_root), Path(args.mask_root), args.views)
            if not uids:
                raise FileNotFoundError(f"Could not find {clean_path} and could not discover usable UID folders.")
        rng = random.Random(args.seed)
        rng.shuffle(uids)
        n_train = int(len(uids) * 0.8)
        n_val = int(len(uids) * 0.1)
        train = uids[:n_train]
        val = uids[n_train : n_train + n_val]
        test = uids[n_train + n_val :]

    if args.overfit_objects > 0:
        train = train[: args.overfit_objects]
        val = train[:]
        test = []
    if args.max_train_objects > 0:
        train = train[: args.max_train_objects]
    if args.max_val_objects > 0:
        val = val[: args.max_val_objects]
    return train, val, test


def image_path(dataset_root: Path, uid: str, view: int) -> Path:
    return dataset_root / "renders" / uid / f"view_{view:03d}.png"


def mask_path(mask_root: Path, uid: str, view: int) -> Path:
    return mask_root / uid / f"view_{view:03d}.png"


def points_path(dataset_root: Path, uid: str) -> Path:
    return dataset_root / "objects" / uid / "points.npz"


def crop_square_from_mask(img: Image.Image, mask: Image.Image, pad_ratio: float = 0.15) -> tuple[Image.Image, Image.Image]:
    mask_arr = np.asarray(mask, dtype=np.uint8)
    if mask_arr.max() <= 8:
        return img, mask
    ys, xs = np.where(mask_arr > 8)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0)
    pad = int(side * pad_ratio)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    side = side + 2 * pad
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(img.width, left + side)
    bottom = min(img.height, top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    return img.crop((left, top, right, bottom)), mask.crop((left, top, right, bottom))


def load_input_tensor(img_path: Path, m_path: Path, image_size: int) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(m_path).convert("L")
    img, mask = crop_square_from_mask(img, mask)
    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    m = (np.asarray(mask, dtype=np.float32) / 255.0)[..., None]
    bg = np.array([0.86, 0.87, 0.88], dtype=np.float32)
    rgb = rgb * m + bg[None, None, :] * (1.0 - m)
    x = np.concatenate([rgb, m], axis=-1)
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


class PointCache:
    def __init__(self, max_items: int = 128):
        self.max_items = max_items
        self.cache: dict[str, tuple[np.ndarray, object | None]] = {}
        self.order: list[str] = []

    def get(self, path: Path) -> tuple[np.ndarray, object | None]:
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        data = np.load(path)
        pts = np.asarray(data["points"], dtype=np.float32)
        tree = None
        try:
            from scipy.spatial import cKDTree  # type: ignore

            tree = cKDTree(pts)
        except Exception:
            tree = None
        self.cache[key] = (pts, tree)
        self.order.append(key)
        if len(self.order) > self.max_items:
            old = self.order.pop(0)
            self.cache.pop(old, None)
        return pts, tree


def nearest_distance(points: np.ndarray, tree: object | None, query: np.ndarray) -> np.ndarray:
    if tree is not None:
        dist, _ = tree.query(query, k=1)
        return np.asarray(dist, dtype=np.float32)
    out = np.empty((len(query),), dtype=np.float32)
    chunk = 512
    for start in range(0, len(query), chunk):
        q = query[start : start + chunk]
        d2 = ((q[:, None, :] - points[None, :, :]) ** 2).sum(axis=-1)
        out[start : start + chunk] = np.sqrt(d2.min(axis=1))
    return out


def sample_queries(
    surface: np.ndarray,
    tree: object | None,
    rng: np.random.Generator,
    count: int,
    positive_fraction: float,
    near_sigma: float,
    truncation: float,
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_pos = int(count * positive_fraction)
    n_uniform = count - n_pos
    ids = rng.choice(len(surface), size=n_pos, replace=len(surface) < n_pos)
    pos = surface[ids] + rng.normal(0.0, near_sigma, size=(n_pos, 3)).astype(np.float32)

    lo, hi = bounds[:3], bounds[3:]
    uniform = rng.uniform(lo, hi, size=(n_uniform, 3)).astype(np.float32)
    query = np.concatenate([pos, uniform], axis=0).astype(np.float32)
    dist = nearest_distance(surface, tree, query)
    target = np.exp(-dist / max(truncation, 1e-6)).astype(np.float32)
    target = target[:, None]

    order = rng.permutation(len(query))
    return query[order], target[order]


@dataclass
class Row:
    uid: str
    view: int
    image: Path
    mask: Path
    points: Path


class ChairGeometryDataset:
    def __init__(self, args: argparse.Namespace, uids: Iterable[str], training: bool):
        self.args = args
        self.training = training
        dataset_root = Path(args.dataset_root)
        mask_root = Path(args.mask_root)
        self.rows: list[Row] = []
        for uid in uids:
            ppath = points_path(dataset_root, uid)
            if not ppath.exists():
                continue
            for view in range(args.views):
                ipath = image_path(dataset_root, uid, view)
                mpath = mask_path(mask_root, uid, view)
                if ipath.exists() and mpath.exists():
                    self.rows.append(Row(uid, view, ipath, mpath, ppath))
        if not self.rows:
            raise RuntimeError("No usable rows. Check dataset_root, mask_root, clean_uids, and splits_json.")
        self.cache = PointCache(max_items=64)
        self.bounds = np.asarray(args.bounds, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, np.ndarray | str | int]:
        row = self.rows[index % len(self.rows)]
        seed = (hash((row.uid, row.view, index, self.args.seed, self.training)) & 0xFFFFFFFF)
        rng = np.random.default_rng(seed)
        image = load_input_tensor(row.image, row.mask, self.args.image_size)
        surface, tree = self.cache.get(row.points)
        query, target = sample_queries(
            surface,
            tree,
            rng,
            self.args.queries,
            self.args.positive_fraction,
            self.args.near_sigma,
            self.args.truncation,
            self.bounds,
        )
        return {
            "image": image,
            "query": query,
            "target": target,
            "uid": row.uid,
            "view": row.view,
        }


def collate(batch: list[dict[str, np.ndarray | str | int]]) -> dict[str, object]:
    import torch

    return {
        "image": torch.from_numpy(np.stack([b["image"] for b in batch], axis=0)),
        "query": torch.from_numpy(np.stack([b["query"] for b in batch], axis=0)),
        "target": torch.from_numpy(np.stack([b["target"] for b in batch], axis=0)),
        "uid": [b["uid"] for b in batch],
        "view": [b["view"] for b in batch],
    }


def build_model(args: argparse.Namespace):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    bounds = torch.tensor(args.bounds, dtype=torch.float32)
    lo = bounds[:3]
    hi = bounds[3:]

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            channels = [4, 32, 64, 128, 192, 256]
            blocks = []
            for i in range(len(channels) - 1):
                blocks += [
                    nn.Conv2d(channels[i], channels[i + 1], 4, stride=2, padding=1),
                    nn.BatchNorm2d(channels[i + 1]),
                    nn.SiLU(inplace=True),
                ]
            self.net = nn.Sequential(*blocks)
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels[-1], args.latent_dim),
                nn.LayerNorm(args.latent_dim),
                nn.SiLU(inplace=True),
                nn.Linear(args.latent_dim, args.latent_dim),
            )

        def forward(self, image):
            return self.head(self.net(image))

    class TriplaneGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = args.plane_channels
            self.s = args.plane_size
            self.fc = nn.Linear(args.latent_dim, 3 * self.c * 8 * 8)
            layers = []
            size = 8
            while size < self.s:
                layers += [
                    nn.ConvTranspose2d(self.c, self.c, 4, stride=2, padding=1),
                    nn.GroupNorm(min(8, self.c), self.c),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(self.c, self.c, 3, padding=1),
                    nn.GroupNorm(min(8, self.c), self.c),
                    nn.SiLU(inplace=True),
                ]
                size *= 2
            self.up = nn.Sequential(*layers)

        def forward(self, latent):
            b = latent.shape[0]
            x = self.fc(latent).view(b * 3, self.c, 8, 8)
            x = self.up(x)
            if x.shape[-1] != self.s:
                x = F.interpolate(x, size=(self.s, self.s), mode="bilinear", align_corners=False)
            return x.view(b, 3, self.c, self.s, self.s)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [args.plane_channels * 3 + 3] + [args.decoder_hidden] * args.decoder_layers
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.SiLU(inplace=True)]
            self.net = nn.Sequential(*layers)
            self.out = nn.Linear(dims[-1], 1)

        def normalize(self, pts):
            lo_t = lo.to(pts.device)
            hi_t = hi.to(pts.device)
            return ((pts - lo_t) / (hi_t - lo_t).clamp_min(1e-6)) * 2.0 - 1.0

        def sample_planes(self, planes, pts):
            b = pts.shape[0]
            p = self.normalize(pts).clamp(-1.1, 1.1)
            x, y, z = p[..., 0], p[..., 1], p[..., 2]
            grids = [
                torch.stack([x, y], dim=-1),
                torch.stack([x, z], dim=-1),
                torch.stack([y, z], dim=-1),
            ]
            feats = []
            for i, grid in enumerate(grids):
                sampled = F.grid_sample(
                    planes[:, i],
                    grid.view(b, -1, 1, 2),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                feats.append(sampled.squeeze(-1).transpose(1, 2))
            return torch.cat(feats + [p], dim=-1)

        def forward(self, planes, pts):
            feat = self.sample_planes(planes, pts)
            return self.out(self.net(feat))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.triplane = TriplaneGenerator()
            self.decoder = Decoder()

        def planes(self, image):
            return self.triplane(self.encoder(image))

        def forward(self, image, pts):
            planes = self.planes(image)
            return self.decoder(planes, pts)

    return Model()


def make_loader(ds, batch_size: int, workers: int, shuffle: bool):
    import torch

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        collate_fn=collate,
    )


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def amp_dtype(args: argparse.Namespace):
    import torch

    if args.amp == "bf16":
        return torch.bfloat16
    if args.amp == "fp16":
        return torch.float16
    return None


def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F

    train_uids, val_uids, _ = load_splits(args)
    train_ds = ChairGeometryDataset(args, train_uids, training=True)
    val_ds = ChairGeometryDataset(args, val_uids or train_uids[: max(1, min(8, len(train_uids)))], training=False)
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, max(0, min(args.num_workers, 2)), shuffle=False)
    train_iter = cycle_loader(train_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))
    dtype = amp_dtype(args)
    use_amp = device.type == "cuda" and dtype is not None

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    log_path = work_dir / "train_log.csv"
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss", "val_loss", "lr"]).writeheader()

    steps_per_epoch = args.steps_per_epoch or max(1, len(train_loader))
    best = float("inf")
    global_step = 0
    print(f"[train] device={device} train_rows={len(train_ds)} val_rows={len(val_ds)} steps_per_epoch={steps_per_epoch}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        losses = []
        for _ in range(steps_per_epoch):
            batch = next(train_iter)
            image = batch["image"].to(device, non_blocking=True)
            query = batch["query"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                logits = model(image, query)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            global_step += 1
            if global_step % args.log_every == 0:
                print(f"[train] epoch={epoch} step={global_step} loss={statistics_mean(losses):.5f}", flush=True)

        val_loss = validate(model, val_loader, args, device, dtype, use_amp)
        train_loss = statistics_mean(losses)
        elapsed = time.time() - start
        print(
            f"[epoch] {epoch}/{args.epochs} train={train_loss:.5f} val={val_loss:.5f} sec={elapsed:.1f}",
            flush=True,
        )
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss", "val_loss", "lr"])
            writer.writerow({"epoch": epoch, "step": global_step, "train_loss": train_loss, "val_loss": val_loss, "lr": args.lr})

        state = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_loss": val_loss}
        torch.save(state, work_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(state, work_dir / "best.pt")
            print(f"[ckpt] saved best.pt val={best:.5f}", flush=True)


def statistics_mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def validate(model, loader, args, device, dtype, use_amp: bool) -> float:
    import torch
    import torch.nn.functional as F

    model.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.val_steps:
                break
            image = batch["image"].to(device, non_blocking=True)
            query = batch["query"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                logits = model(image, query)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            losses.append(float(loss.detach().cpu()))
    return statistics_mean(losses)


def check(args: argparse.Namespace) -> None:
    train_uids, val_uids, test_uids = load_splits(args)
    print(f"dataset_root: {args.dataset_root}")
    print(f"mask_root:    {args.mask_root}")
    print(f"train_uids:   {len(train_uids)}")
    print(f"val_uids:     {len(val_uids)}")
    print(f"test_uids:    {len(test_uids)}")
    for name, uids in (("train", train_uids), ("val", val_uids), ("test", test_uids)):
        missing = 0
        rows = 0
        for uid in uids:
            if not points_path(Path(args.dataset_root), uid).exists():
                missing += 1
                continue
            for view in range(args.views):
                if image_path(Path(args.dataset_root), uid, view).exists() and mask_path(Path(args.mask_root), uid, view).exists():
                    rows += 1
        print(f"{name}_rows:   {rows}  missing_point_objects: {missing}")


def predict(args: argparse.Namespace) -> None:
    import torch

    try:
        from skimage import measure
        import trimesh
    except Exception as exc:
        raise RuntimeError("predict mode needs scikit-image and trimesh installed in the Kaggle environment.") from exc

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved = argparse.Namespace(**ckpt["args"])
    for key in ("checkpoint", "image", "mask", "output_dir", "grid_resolution", "mc_level", "predict_chunk", "amp"):
        setattr(saved, key, getattr(args, key))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(saved).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_input_tensor(Path(args.image), Path(args.mask), saved.image_size)
    image_t = torch.from_numpy(image[None]).to(device)
    bounds = np.asarray(saved.bounds, dtype=np.float32)
    lo, hi = bounds[:3], bounds[3:]
    xs = np.linspace(lo[0], hi[0], args.grid_resolution, dtype=np.float32)
    ys = np.linspace(lo[1], hi[1], args.grid_resolution, dtype=np.float32)
    zs = np.linspace(lo[2], hi[2], args.grid_resolution, dtype=np.float32)
    field = np.empty((args.grid_resolution, args.grid_resolution, args.grid_resolution), dtype=np.float32)

    with torch.no_grad():
        planes = model.planes(image_t)
        for zi, z in enumerate(zs):
            yy, xx = np.meshgrid(ys, xs, indexing="ij")
            pts = np.stack([xx.reshape(-1), yy.reshape(-1), np.full(xx.size, z, dtype=np.float32)], axis=-1)
            vals = []
            for start in range(0, len(pts), args.predict_chunk):
                p = torch.from_numpy(pts[start : start + args.predict_chunk][None]).to(device)
                logits = model.decoder(planes, p)
                vals.append(torch.sigmoid(logits).squeeze(0).squeeze(-1).float().cpu().numpy())
            field[:, :, zi] = np.concatenate(vals, axis=0).reshape(args.grid_resolution, args.grid_resolution)
            if (zi + 1) % 16 == 0:
                print(f"[predict] grid slice {zi + 1}/{args.grid_resolution}", flush=True)

    np.save(out_dir / "field.npy", field)
    verts, faces, normals, _ = measure.marching_cubes(
        field,
        level=args.mc_level,
        spacing=(
            (hi[0] - lo[0]) / (args.grid_resolution - 1),
            (hi[1] - lo[1]) / (args.grid_resolution - 1),
            (hi[2] - lo[2]) / (args.grid_resolution - 1),
        ),
    )
    verts[:, 0] += lo[0]
    verts[:, 1] += lo[1]
    verts[:, 2] += lo[2]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    mesh.export(out_dir / "mesh.ply")
    mesh.export(out_dir / "mesh.obj")
    print(f"[predict] saved {out_dir / 'mesh.obj'}")


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    if args.mode == "check":
        check(args)
    elif args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
