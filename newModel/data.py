"""Dataset reading, image loading, camera rays, and batch collation."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .constants import BOX_MAX, BOX_MIN


def read_excluded_uids(path: str) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    values = set()
    for line in p.read_text(encoding="utf-8").replace(",", "\n").splitlines():
        uid = line.strip()
        if uid:
            values.add(uid)
    return values


def read_dataset(dataset_root: Path, exclude_uids: set[str]) -> Dict[str, List[dict]]:
    views_csv = dataset_root / "metadata" / "views.csv"
    if not views_csv.exists():
        raise FileNotFoundError(f"Missing {views_csv}")

    grouped: Dict[str, List[dict]] = defaultdict(list)
    with open(views_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["uid"]
            if uid in exclude_uids:
                continue
            view = int(row.get("view", row.get("view_index", 0)))
            image_path = dataset_root / row.get("image_path", f"renders/{uid}/view_{view:03d}.png")
            mask_path = dataset_root / row.get("mask_path", f"masks/{uid}/view_{view:03d}.png")
            camera_path = dataset_root / row.get("camera_path", f"cameras/{uid}/view_{view:03d}.json")
            mesh_path = dataset_root / "objects" / uid / "normalized.glb"
            points_path = dataset_root / "objects" / uid / "points.npz"
            if image_path.exists() and mask_path.exists() and camera_path.exists() and mesh_path.exists() and points_path.exists():
                grouped[uid].append(
                    {
                        "uid": uid,
                        "view": view,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "camera_path": str(camera_path),
                        "mesh_path": str(mesh_path),
                        "points_path": str(points_path),
                    }
                )

    grouped = {uid: sorted(rows, key=lambda x: x["view"]) for uid, rows in grouped.items() if len(rows) >= 2}
    if not grouped:
        raise RuntimeError(f"No usable multiview objects under {dataset_root}")
    return grouped


def split_uids(dataset_root: Path, grouped: Dict[str, List[dict]], seed: int, train_ratio: float, val_ratio: float):
    available = set(grouped.keys())
    splits_path = dataset_root / "metadata" / "splits.json"
    if splits_path.exists():
        raw = json.loads(splits_path.read_text(encoding="utf-8"))
        train = [uid for uid in raw.get("train", []) if uid in available]
        val = [uid for uid in raw.get("val", []) if uid in available]
        test = [uid for uid in raw.get("test", []) if uid in available]
        used = set(train) | set(val) | set(test)
        train.extend(sorted(available - used))
    else:
        uids = sorted(available)
        rng = random.Random(seed)
        rng.shuffle(uids)
        n_train = max(1, int(len(uids) * train_ratio))
        n_val = max(1, int(len(uids) * val_ratio))
        train = uids[:n_train]
        val = uids[n_train : n_train + n_val]
        test = uids[n_train + n_val :]
    if not val:
        val = train[-max(1, len(train) // 10) :]
        train = train[: -len(val)]
    return train, val, test


def stable_seed(text: str, seed: int) -> int:
    value = seed & 0xFFFFFFFF
    for ch in text:
        value = ((value * 131) + ord(ch)) & 0xFFFFFFFF
    return value


def load_source_image(image_path: str, mask_path: str, image_size: int, crop: bool):
    from PIL import Image, ImageOps

    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if crop:
        mask_arr = np.asarray(mask, dtype=np.uint8)
        if mask_arr.max() > 8:
            ys, xs = np.where(mask_arr > 8)
            pad = max(8, int(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 0.12))
            box = (
                max(0, int(xs.min()) - pad),
                max(0, int(ys.min()) - pad),
                min(img.width, int(xs.max()) + pad + 1),
                min(img.height, int(ys.max()) + pad + 1),
            )
            img = img.crop(box)
    img.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (image_size, image_size), (219, 222, 224))
    canvas.paste(img, ((image_size - img.width) // 2, (image_size - img.height) // 2))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def load_rgb_mask(image_path: str, mask_path: str):
    from PIL import Image

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    return rgb, mask


def camera_rays(camera_json: str, xs: np.ndarray, ys: np.ndarray):
    cam = json.loads(Path(camera_json).read_text(encoding="utf-8"))
    intr = cam["intrinsics"]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    c2w = np.asarray(cam["camera_matrix_world"], dtype=np.float32).reshape(4, 4)
    rot = c2w[:3, :3]
    origin = c2w[:3, 3]
    dirs_cam = np.stack([(xs - cx) / fx, -(ys - cy) / fy, -np.ones_like(xs)], axis=-1).astype(np.float32)
    dirs_world = dirs_cam @ rot.T
    dirs_world /= np.linalg.norm(dirs_world, axis=-1, keepdims=True) + 1e-8
    origins = np.broadcast_to(origin.astype(np.float32), dirs_world.shape)
    return origins.astype(np.float32), dirs_world.astype(np.float32)


def sample_geometry_queries(points_path: str, rng, count: int):
    data = np.load(points_path)
    surface = np.asarray(data["points"], dtype=np.float32)
    if len(surface) == 0:
        raise RuntimeError(f"Empty points file: {points_path}")

    n_surface = count // 3
    n_near = count // 3
    n_uniform = count - n_surface - n_near
    surface_idx = rng.choice(len(surface), n_surface, replace=len(surface) < n_surface)
    near_idx = rng.choice(len(surface), n_near, replace=len(surface) < n_near)
    near_sigma = rng.choice(np.array([0.015, 0.035, 0.07], dtype=np.float32), size=n_near, p=[0.45, 0.35, 0.20])

    surface_pts = surface[surface_idx]
    surface_target = np.ones((n_surface, 1), dtype=np.float32)
    near_pts = surface[near_idx] + rng.normal(size=(n_near, 3)).astype(np.float32) * near_sigma[:, None]
    near_dist = np.linalg.norm(near_pts - surface[near_idx], axis=1, keepdims=True)
    near_target = np.exp(-near_dist / 0.045).astype(np.float32)
    uniform_pts = np.empty((n_uniform, 3), dtype=np.float32)
    uniform_pts[:, 0] = rng.uniform(BOX_MIN[0], BOX_MAX[0], size=n_uniform)
    uniform_pts[:, 1] = rng.uniform(BOX_MIN[1], BOX_MAX[1], size=n_uniform)
    uniform_pts[:, 2] = rng.uniform(BOX_MIN[2], BOX_MAX[2], size=n_uniform)
    uniform_target = np.zeros((n_uniform, 1), dtype=np.float32)

    query = np.concatenate([surface_pts, near_pts, uniform_pts], axis=0)
    target = np.concatenate([surface_target, near_target, uniform_target], axis=0)
    order = rng.permutation(len(query))
    return query[order].astype(np.float32), target[order].astype(np.float32)


class RenderPairDataset:
    def __init__(self, grouped, uids, args, training: bool):
        self.grouped = grouped
        self.uids = list(uids)
        self.args = args
        self.training = training
        self.items = []
        for uid in self.uids:
            for target_idx in range(len(grouped[uid])):
                self.items.append((uid, target_idx))

    def __len__(self):
        return len(self.items)

    def _choose_source(self, rng, rows, target_idx):
        if self.training and rng.random() < self.args.same_view_prob:
            return target_idx
        choices = [i for i in range(len(rows)) if i != target_idx]
        return int(rng.choice(choices)) if choices else target_idx

    def _sample_patch(self, rng, mask, patch_size):
        h, w = mask.shape
        if self.training and rng.random() < self.args.foreground_patch_prob and mask.max() > 8 / 255.0:
            ys, xs = np.where(mask > 8 / 255.0)
            center_idx = int(rng.integers(0, len(xs)))
            cx = int(xs[center_idx])
            cy = int(ys[center_idx])
            x0 = np.clip(cx - patch_size // 2, 0, max(0, w - patch_size))
            y0 = np.clip(cy - patch_size // 2, 0, max(0, h - patch_size))
        else:
            x0 = int(rng.integers(0, max(1, w - patch_size + 1)))
            y0 = int(rng.integers(0, max(1, h - patch_size + 1)))
        yy, xx = np.meshgrid(np.arange(y0, y0 + patch_size), np.arange(x0, x0 + patch_size), indexing="ij")
        return xx.reshape(-1).astype(np.float32) + 0.5, yy.reshape(-1).astype(np.float32) + 0.5, yy, xx

    def __getitem__(self, index):
        uid, target_idx = self.items[index]
        seed = stable_seed(f"{uid}:{target_idx}:{index}", self.args.seed + (0 if self.training else 100000))
        rng = np.random.default_rng(seed)
        rows = self.grouped[uid]
        source_idx = self._choose_source(rng, rows, target_idx) if self.training else 0
        if source_idx == target_idx and len(rows) > 1 and not self.training:
            source_idx = 1

        source = rows[source_idx]
        target = rows[target_idx]
        source_image = load_source_image(source["image_path"], source["mask_path"], self.args.image_size, self.args.crop)
        target_rgb, target_mask = load_rgb_mask(target["image_path"], target["mask_path"])
        xs, ys, yy, xx = self._sample_patch(rng, target_mask, self.args.patch_size)
        rays_o, rays_d = camera_rays(target["camera_path"], xs, ys)
        geo_query, geo_target = sample_geometry_queries(target["points_path"], rng, self.args.geometry_queries)

        return {
            "uid": uid,
            "source_path": source["image_path"],
            "target_path": target["image_path"],
            "source_view": source["view"],
            "target_view": target["view"],
            "source_image": source_image.astype(np.float32),
            "rays_o": rays_o,
            "rays_d": rays_d,
            "target_rgb": target_rgb[yy, xx].reshape(-1, 3).astype(np.float32),
            "target_mask": target_mask[yy, xx].reshape(-1, 1).astype(np.float32),
            "geo_query": geo_query,
            "geo_target": geo_target,
        }


def collate_batch(batch):
    import torch

    out = {}
    for key in ("source_image", "rays_o", "rays_d", "target_rgb", "target_mask", "geo_query", "geo_target"):
        out[key] = torch.from_numpy(np.stack([item[key] for item in batch], axis=0))
    out["uid"] = [item["uid"] for item in batch]
    out["source_path"] = [item["source_path"] for item in batch]
    out["target_path"] = [item["target_path"] for item in batch]
    out["source_view"] = [item["source_view"] for item in batch]
    out["target_view"] = [item["target_view"] for item in batch]
    return out
