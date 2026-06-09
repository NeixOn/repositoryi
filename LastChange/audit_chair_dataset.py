#!/usr/bin/env python3
"""
Kaggle-friendly audit and optional mask repair for the chair reconstruction dataset.

Expected dataset layout:
  dataset_root/
    metadata/views.csv
    metadata/objects.csv                 optional
    objects/<uid>/normalized.glb
    objects/<uid>/points.npz
    renders/<uid>/view_000.png ... view_023.png
    masks/<uid>/view_000.png ... view_023.png
    cameras/<uid>/view_000.json ... view_023.json

The script is intentionally dependency-light: Pillow and NumPy are required;
OpenCV and trimesh are used only when installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ISSUE_BLOCK = "block"
ISSUE_WARN = "warn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_root", required=True, help="Path to objaverse_chairs_blender dataset root.")
    parser.add_argument("--bad_uids", default="", help="Optional text file with excluded UID values.")
    parser.add_argument("--out_dir", default="", help="Report directory. Defaults to <dataset_root>/audit_report.")
    parser.add_argument("--views", type=int, default=24)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--points_per_object", type=int, default=32768)
    parser.add_argument("--expected_objects", type=int, default=500)
    parser.add_argument("--horizontal_views", type=int, default=12)
    parser.add_argument("--expected_azimuth_step", type=float, default=30.0)
    parser.add_argument("--min_fg_ratio", type=float, default=0.002)
    parser.add_argument("--max_fg_ratio", type=float, default=0.90)
    parser.add_argument("--max_edge_touch_ratio", type=float, default=0.18)
    parser.add_argument("--mask_binary_tolerance", type=float, default=0.03)
    parser.add_argument("--max_objects", type=int, default=0, help="Audit only the first N discovered objects.")
    parser.add_argument("--preview_objects", type=int, default=80)
    parser.add_argument("--repair_masks_from_rgb", action="store_true")
    parser.add_argument("--repair_force", action="store_true", help="Regenerate masks even if they look valid.")
    parser.add_argument("--mask_output_dir", default="", help="Optional repaired-mask output dir. Empty means overwrite dataset masks.")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--fail_on_blockers", action="store_true")
    return parser.parse_args()


@dataclass
class Issue:
    severity: str
    uid: str
    view: int | None
    code: str
    message: str


@dataclass
class ViewStats:
    uid: str
    view: int
    image_ok: bool = False
    mask_ok: bool = False
    camera_ok: bool = False
    width: int = 0
    height: int = 0
    fg_ratio: float = -1.0
    components: int = -1
    edge_touch_ratio: float = -1.0
    bbox_area_ratio: float = -1.0
    image_hash: str = ""
    azimuth: float | None = None
    elevation: float | None = None
    radius: float | None = None


@dataclass
class ObjectStats:
    uid: str
    usable: bool = False
    excluded: bool = False
    views_ok: int = 0
    mesh_ok: bool = False
    points_ok: bool = False
    point_count: int = 0
    point_min: list[float] = field(default_factory=list)
    point_max: list[float] = field(default_factory=list)
    point_std: list[float] = field(default_factory=list)


def read_bad_uids(path: str) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")
    values = set()
    for part in text.replace(",", "\n").splitlines():
        value = part.strip()
        if value and not value.startswith("#"):
            values.add(value)
    return values


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def discover_uids(dataset_root: Path) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    views_csv = dataset_root / "metadata" / "views.csv"
    for row in read_csv_rows(views_csv):
        uid = row.get("uid", "").strip()
        if uid and uid not in seen:
            seen.add(uid)
            ordered.append(uid)

    for dirname in ("renders", "masks", "cameras", "objects"):
        root = dataset_root / dirname
        if root.exists():
            for child in sorted(root.iterdir()):
                if child.is_dir() and child.name not in seen:
                    seen.add(child.name)
                    ordered.append(child.name)
    return ordered


def relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def add_issue(issues: list[Issue], severity: str, uid: str, view: int | None, code: str, message: str) -> None:
    issues.append(Issue(severity, uid, view, code, message))


def load_image_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def image_short_hash(arr: np.ndarray) -> str:
    small = Image.fromarray(arr).resize((32, 32), Image.Resampling.BILINEAR).convert("L")
    values = np.asarray(small, dtype=np.uint8)
    return hashlib.sha1(values.tobytes()).hexdigest()[:16]


def estimate_background(rgb: np.ndarray, border: int = 12) -> np.ndarray:
    h, w, _ = rgb.shape
    b = max(2, min(border, h // 4, w // 4))
    samples = np.concatenate(
        [
            rgb[:b].reshape(-1, 3),
            rgb[-b:].reshape(-1, 3),
            rgb[:, :b].reshape(-1, 3),
            rgb[:, -b:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(samples.astype(np.float32), axis=0)


def largest_components(mask: np.ndarray) -> tuple[int, int]:
    try:
        import cv2  # type: ignore

        num, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if num <= 1:
            return 0, 0
        areas = stats[1:, cv2.CC_STAT_AREA]
        return int(num - 1), int(areas.max())
    except Exception:
        return -1, -1


def clean_mask(mask: np.ndarray, min_component_area: int = 64) -> np.ndarray:
    try:
        import cv2  # type: ignore

        m = mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        keep = np.zeros_like(m, dtype=np.uint8)
        for label in range(1, num):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                keep[labels == label] = 255

        inv = 255 - keep
        flood = inv.copy()
        h, w = flood.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 0)
        holes = (flood > 0).astype(np.uint8) * 255
        keep = np.maximum(keep, holes)
        return keep
    except Exception:
        return mask.astype(np.uint8) * 255


def repair_mask_from_rgb(rgb_path: Path, mask_path: Path, threshold: float = 14.0) -> None:
    rgb = load_image_rgb(rgb_path).astype(np.float32)
    bg = estimate_background(rgb)
    dist = np.linalg.norm(rgb - bg[None, None, :], axis=-1)
    mask = clean_mask(dist > threshold)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(mask_path)


def mask_stats(mask_path: Path) -> dict[str, Any]:
    arr = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    h, w = arr.shape
    fg = arr > 8
    fg_count = int(fg.sum())
    ratio = fg_count / float(h * w)
    non_binary_ratio = float(((arr > 8) & (arr < 247)).mean())
    components, largest = largest_components(fg)
    edge_count = int(fg[0, :].sum() + fg[-1, :].sum() + fg[:, 0].sum() + fg[:, -1].sum())
    edge_touch_ratio = edge_count / max(1, fg_count)
    bbox_area_ratio = -1.0
    if fg_count:
        ys, xs = np.where(fg)
        bbox_area_ratio = ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / float(h * w)
    return {
        "width": w,
        "height": h,
        "fg_ratio": ratio,
        "non_binary_ratio": non_binary_ratio,
        "components": components,
        "largest_component": largest,
        "edge_touch_ratio": edge_touch_ratio,
        "bbox_area_ratio": bbox_area_ratio,
    }


def read_camera(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def camera_pose_stats(camera: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    raw = camera.get("camera_matrix_world") or camera.get("c2w") or camera.get("extrinsics")
    if raw is None:
        return None, None, None
    mat = np.asarray(raw, dtype=np.float64).reshape(4, 4)
    pos = mat[:3, 3]
    radius_xy = math.hypot(float(pos[0]), float(pos[1]))
    radius = float(np.linalg.norm(pos))
    azimuth = math.degrees(math.atan2(float(pos[1]), float(pos[0]))) % 360.0
    elevation = math.degrees(math.atan2(float(pos[2]), radius_xy))
    return azimuth, elevation, radius


def circular_step_errors(azimuths: list[float], expected_step: float) -> list[float]:
    if len(azimuths) < 2:
        return []
    errors = []
    for idx in range(1, len(azimuths)):
        delta = (azimuths[idx] - azimuths[idx - 1]) % 360.0
        errors.append(abs(delta - expected_step))
    return errors


def check_camera_group(uid: str, view_stats: list[ViewStats], args: argparse.Namespace, issues: list[Issue]) -> None:
    valid = [item for item in sorted(view_stats, key=lambda x: x.view) if item.azimuth is not None]
    if len(valid) != args.views:
        return

    horizontal = valid[: args.horizontal_views]
    upper = valid[args.horizontal_views :]
    for name, group in (("horizontal", horizontal), ("upper", upper)):
        if len(group) < 2:
            continue
        elevations = [float(item.elevation) for item in group if item.elevation is not None]
        azimuths = [float(item.azimuth) for item in group if item.azimuth is not None]
        radii = [float(item.radius) for item in group if item.radius is not None]
        step_errors = circular_step_errors(azimuths, args.expected_azimuth_step)
        if step_errors and max(step_errors) > 2.0:
            add_issue(
                issues,
                ISSUE_WARN,
                uid,
                None,
                f"{name}_camera_azimuth_step",
                f"Max azimuth step error is {max(step_errors):.2f} deg.",
            )
        if elevations and statistics.pstdev(elevations) > 1.5:
            add_issue(
                issues,
                ISSUE_WARN,
                uid,
                None,
                f"{name}_camera_elevation_jitter",
                f"Elevation std is {statistics.pstdev(elevations):.2f} deg.",
            )
        if radii and statistics.pstdev(radii) > 0.03:
            add_issue(
                issues,
                ISSUE_WARN,
                uid,
                None,
                f"{name}_camera_radius_jitter",
                f"Radius std is {statistics.pstdev(radii):.4f}.",
            )

    if horizontal and upper:
        h_el = statistics.mean(float(item.elevation) for item in horizontal if item.elevation is not None)
        u_el = statistics.mean(float(item.elevation) for item in upper if item.elevation is not None)
        if u_el <= h_el + 5.0:
            add_issue(
                issues,
                ISSUE_WARN,
                uid,
                None,
                "upper_views_not_higher",
                f"Upper elevation mean {u_el:.2f} is not clearly above horizontal mean {h_el:.2f}.",
            )


def audit_points(points_path: Path, expected_count: int, uid: str, issues: list[Issue]) -> tuple[bool, int, list[float], list[float], list[float]]:
    if not points_path.exists():
        add_issue(issues, ISSUE_BLOCK, uid, None, "missing_points", f"Missing {points_path.name}.")
        return False, 0, [], [], []
    try:
        data = np.load(points_path)
    except Exception as exc:
        add_issue(issues, ISSUE_BLOCK, uid, None, "unreadable_points", repr(exc))
        return False, 0, [], [], []
    if "points" not in data:
        add_issue(issues, ISSUE_BLOCK, uid, None, "points_key_missing", "points.npz has no 'points' array.")
        return False, 0, [], [], []
    points = np.asarray(data["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        add_issue(issues, ISSUE_BLOCK, uid, None, "bad_points_shape", f"points shape is {points.shape}.")
        return False, int(points.shape[0]) if points.ndim else 0, [], [], []
    if not np.isfinite(points).all():
        add_issue(issues, ISSUE_BLOCK, uid, None, "nonfinite_points", "points array has NaN or inf.")
        return False, int(points.shape[0]), [], [], []
    if points.shape[0] != expected_count:
        add_issue(issues, ISSUE_WARN, uid, None, "unexpected_point_count", f"Expected {expected_count}, got {points.shape[0]}.")
    mins = points.min(axis=0).astype(float).round(6).tolist()
    maxs = points.max(axis=0).astype(float).round(6).tolist()
    stds = points.std(axis=0).astype(float).round(6).tolist()
    if max(abs(v) for v in mins + maxs) > 2.5:
        add_issue(issues, ISSUE_WARN, uid, None, "points_outside_expected_scale", f"bounds min={mins}, max={maxs}.")
    if min(stds) < 0.01:
        add_issue(issues, ISSUE_WARN, uid, None, "collapsed_points_axis", f"std={stds}.")
    return True, int(points.shape[0]), mins, maxs, stds


def audit_mesh(mesh_path: Path, uid: str, issues: list[Issue]) -> bool:
    if not mesh_path.exists():
        add_issue(issues, ISSUE_BLOCK, uid, None, "missing_mesh", f"Missing {mesh_path.name}.")
        return False
    if mesh_path.stat().st_size < 1024:
        add_issue(issues, ISSUE_BLOCK, uid, None, "tiny_mesh_file", f"Mesh file has only {mesh_path.stat().st_size} bytes.")
        return False
    try:
        import trimesh  # type: ignore

        mesh = trimesh.load(mesh_path, force="mesh")
        if getattr(mesh, "vertices", np.empty((0, 3))).shape[0] == 0:
            add_issue(issues, ISSUE_BLOCK, uid, None, "empty_mesh", "trimesh loaded zero vertices.")
            return False
        if getattr(mesh, "faces", np.empty((0, 3))).shape[0] == 0:
            add_issue(issues, ISSUE_WARN, uid, None, "mesh_without_faces", "trimesh loaded zero faces.")
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        if not np.isfinite(bounds).all():
            add_issue(issues, ISSUE_BLOCK, uid, None, "mesh_nonfinite_bounds", "Mesh bounds contain NaN or inf.")
            return False
        if np.max(np.abs(bounds)) > 2.5:
            add_issue(issues, ISSUE_WARN, uid, None, "mesh_outside_expected_scale", f"bounds={bounds.round(4).tolist()}.")
    except ImportError:
        pass
    except Exception as exc:
        add_issue(issues, ISSUE_WARN, uid, None, "mesh_trimesh_load_failed", repr(exc))
    return True


def expected_paths(dataset_root: Path, uid: str, view: int) -> tuple[Path, Path, Path]:
    name = f"view_{view:03d}"
    return (
        dataset_root / "renders" / uid / f"{name}.png",
        dataset_root / "masks" / uid / f"{name}.png",
        dataset_root / "cameras" / uid / f"{name}.json",
    )


def audit_view(
    dataset_root: Path,
    uid: str,
    view: int,
    args: argparse.Namespace,
    issues: list[Issue],
    mask_root: Path,
) -> ViewStats:
    stats = ViewStats(uid=uid, view=view)
    rgb_path, mask_path_default, camera_path = expected_paths(dataset_root, uid, view)
    mask_path = mask_root / uid / f"view_{view:03d}.png" if mask_root != dataset_root / "masks" else mask_path_default

    if not rgb_path.exists():
        add_issue(issues, ISSUE_BLOCK, uid, view, "missing_rgb", f"Missing {relpath(rgb_path, dataset_root)}.")
        return stats
    try:
        with Image.open(rgb_path) as img:
            stats.width, stats.height = img.size
            if img.size != (args.resolution, args.resolution):
                add_issue(issues, ISSUE_WARN, uid, view, "bad_rgb_resolution", f"RGB size is {img.size}.")
        rgb = load_image_rgb(rgb_path)
        stats.image_hash = image_short_hash(rgb)
        stats.image_ok = True
        if float(rgb.std()) < 1.0:
            add_issue(issues, ISSUE_BLOCK, uid, view, "blank_or_constant_rgb", f"RGB std is {float(rgb.std()):.3f}.")
    except Exception as exc:
        add_issue(issues, ISSUE_BLOCK, uid, view, "unreadable_rgb", repr(exc))
        return stats

    if args.repair_masks_from_rgb and (args.repair_force or not mask_path.exists()):
        try:
            repair_mask_from_rgb(rgb_path, mask_path)
        except Exception as exc:
            add_issue(issues, ISSUE_BLOCK, uid, view, "mask_repair_failed", repr(exc))

    if not mask_path.exists():
        add_issue(issues, ISSUE_BLOCK, uid, view, "missing_mask", f"Missing {relpath(mask_path, dataset_root)}.")
    else:
        try:
            m = mask_stats(mask_path)
            stats.fg_ratio = float(m["fg_ratio"])
            stats.components = int(m["components"])
            stats.edge_touch_ratio = float(m["edge_touch_ratio"])
            stats.bbox_area_ratio = float(m["bbox_area_ratio"])
            if (m["width"], m["height"]) != (args.resolution, args.resolution):
                add_issue(issues, ISSUE_WARN, uid, view, "bad_mask_resolution", f"Mask size is {(m['width'], m['height'])}.")
            if not (args.min_fg_ratio <= stats.fg_ratio <= args.max_fg_ratio):
                add_issue(issues, ISSUE_BLOCK, uid, view, "bad_mask_foreground_ratio", f"fg_ratio={stats.fg_ratio:.6f}.")
            if float(m["non_binary_ratio"]) > args.mask_binary_tolerance:
                add_issue(issues, ISSUE_WARN, uid, view, "mask_not_binary", f"non_binary_ratio={m['non_binary_ratio']:.6f}.")
            if stats.components > 8:
                add_issue(issues, ISSUE_WARN, uid, view, "many_mask_components", f"components={stats.components}.")
            if stats.edge_touch_ratio > args.max_edge_touch_ratio:
                add_issue(issues, ISSUE_WARN, uid, view, "mask_touches_image_border", f"edge_touch_ratio={stats.edge_touch_ratio:.6f}.")
            stats.mask_ok = args.min_fg_ratio <= stats.fg_ratio <= args.max_fg_ratio
        except Exception as exc:
            add_issue(issues, ISSUE_BLOCK, uid, view, "unreadable_mask", repr(exc))

    if not camera_path.exists():
        add_issue(issues, ISSUE_BLOCK, uid, view, "missing_camera", f"Missing {relpath(camera_path, dataset_root)}.")
    else:
        try:
            camera = read_camera(camera_path)
            stats.azimuth, stats.elevation, stats.radius = camera_pose_stats(camera)
            intr = camera.get("intrinsics", {})
            for key in ("fx", "fy", "cx", "cy"):
                if key not in intr:
                    add_issue(issues, ISSUE_WARN, uid, view, "camera_intrinsic_missing", f"Missing intrinsics.{key}.")
            if stats.azimuth is None:
                add_issue(issues, ISSUE_WARN, uid, view, "camera_pose_missing", "No camera_matrix_world/c2w/extrinsics found.")
            stats.camera_ok = True
        except Exception as exc:
            add_issue(issues, ISSUE_BLOCK, uid, view, "unreadable_camera", repr(exc))

    return stats


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_preview(dataset_root: Path, out_dir: Path, objects: list[ObjectStats], args: argparse.Namespace, mask_root: Path) -> Path | None:
    selected = [obj.uid for obj in objects if obj.usable and not obj.excluded][: args.preview_objects]
    if not selected:
        return None
    tile = 128
    label_h = 24
    cols = 4
    rows = math.ceil(len(selected) / cols)
    sheet = Image.new("RGB", (cols * tile * 2, rows * (tile + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, uid in enumerate(selected):
        rgb_path = dataset_root / "renders" / uid / "view_000.png"
        mask_path = mask_root / uid / "view_000.png"
        if not rgb_path.exists() or not mask_path.exists():
            continue
        row = idx // cols
        col = idx % cols
        x = col * tile * 2
        y = row * (tile + label_h)
        rgb = Image.open(rgb_path).convert("RGB").resize((tile, tile))
        mask = Image.open(mask_path).convert("L").resize((tile, tile)).convert("RGB")
        sheet.paste(rgb, (x, y + label_h))
        sheet.paste(mask, (x + tile, y + label_h))
        draw.text((x + 4, y + 5), uid[:24], fill=(20, 20, 20))
    out = out_dir / "preview_rgb_mask_view000.jpg"
    sheet.save(out, quality=92)
    return out


def make_splits(out_dir: Path, clean_uids: list[str], args: argparse.Namespace) -> dict[str, list[str]]:
    rng = np.random.default_rng(args.split_seed)
    uids = np.asarray(sorted(clean_uids), dtype=object)
    rng.shuffle(uids)
    n = len(uids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    splits = {
        "train": [str(x) for x in uids[:n_train]],
        "val": [str(x) for x in uids[n_train : n_train + n_val]],
        "test": [str(x) for x in uids[n_train + n_val :]],
    }
    (out_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return splits


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else dataset_root / "audit_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_root = Path(args.mask_output_dir).expanduser().resolve() if args.mask_output_dir else dataset_root / "masks"

    started = time.time()
    issues: list[Issue] = []
    bad_uids = read_bad_uids(args.bad_uids)
    uids = discover_uids(dataset_root)
    if args.max_objects > 0:
        uids = uids[: args.max_objects]

    if args.expected_objects > 0 and len(uids) != args.expected_objects:
        add_issue(
            issues,
            ISSUE_WARN,
            "",
            None,
            "unexpected_object_count",
            f"Expected {args.expected_objects} discovered UID values, got {len(uids)}.",
        )
    missing_bad_uids = sorted(bad_uids - set(uids))
    for uid in missing_bad_uids[:50]:
        add_issue(issues, ISSUE_WARN, uid, None, "bad_uid_not_in_dataset", "UID is listed in bad_uids but was not discovered.")

    for dirname in ("metadata", "objects", "renders", "cameras"):
        if not (dataset_root / dirname).exists():
            add_issue(issues, ISSUE_BLOCK, "", None, "missing_root_dir", f"Missing dataset directory: {dirname}.")
    if not (dataset_root / "masks").exists() and not args.repair_masks_from_rgb:
        add_issue(issues, ISSUE_BLOCK, "", None, "missing_root_dir", "Missing dataset directory: masks.")

    view_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    objects: list[ObjectStats] = []

    print(f"[audit] dataset_root={dataset_root}", flush=True)
    print(f"[audit] discovered_uids={len(uids)} excluded_bad_uids={len(bad_uids)}", flush=True)
    print(f"[audit] out_dir={out_dir}", flush=True)
    print(f"[audit] mask_root={mask_root}", flush=True)

    for idx, uid in enumerate(uids, 1):
        obj = ObjectStats(uid=uid, excluded=uid in bad_uids)
        if obj.excluded:
            obj.usable = False
            add_issue(issues, ISSUE_WARN, uid, None, "excluded_bad_uid", "UID is listed in bad_uids file.")
            objects.append(obj)
            object_rows.append(
                {
                    "uid": uid,
                    "usable": obj.usable,
                    "excluded": obj.excluded,
                    "views_ok": obj.views_ok,
                    "mesh_ok": obj.mesh_ok,
                    "points_ok": obj.points_ok,
                    "point_count": obj.point_count,
                    "point_min": json.dumps(obj.point_min),
                    "point_max": json.dumps(obj.point_max),
                    "point_std": json.dumps(obj.point_std),
                }
            )
            continue

        uid_view_stats = []
        for view in range(args.views):
            st = audit_view(dataset_root, uid, view, args, issues, mask_root)
            uid_view_stats.append(st)
            view_rows.append(
                {
                    "uid": uid,
                    "view": view,
                    "image_ok": st.image_ok,
                    "mask_ok": st.mask_ok,
                    "camera_ok": st.camera_ok,
                    "width": st.width,
                    "height": st.height,
                    "fg_ratio": f"{st.fg_ratio:.8f}",
                    "components": st.components,
                    "edge_touch_ratio": f"{st.edge_touch_ratio:.8f}",
                    "bbox_area_ratio": f"{st.bbox_area_ratio:.8f}",
                    "image_hash": st.image_hash,
                    "azimuth": "" if st.azimuth is None else f"{st.azimuth:.6f}",
                    "elevation": "" if st.elevation is None else f"{st.elevation:.6f}",
                    "radius": "" if st.radius is None else f"{st.radius:.6f}",
                }
            )

        check_camera_group(uid, uid_view_stats, args, issues)
        hash_counts = Counter(st.image_hash for st in uid_view_stats if st.image_hash)
        duplicate_hashes = [h for h, c in hash_counts.items() if c > 1]
        if duplicate_hashes:
            add_issue(issues, ISSUE_WARN, uid, None, "duplicate_view_images", f"{len(duplicate_hashes)} duplicate low-res image hashes.")

        mesh_path = dataset_root / "objects" / uid / "normalized.glb"
        points_path = dataset_root / "objects" / uid / "points.npz"
        obj.mesh_ok = audit_mesh(mesh_path, uid, issues)
        obj.points_ok, obj.point_count, obj.point_min, obj.point_max, obj.point_std = audit_points(
            points_path, args.points_per_object, uid, issues
        )
        obj.views_ok = sum(1 for st in uid_view_stats if st.image_ok and st.mask_ok and st.camera_ok)
        obj.usable = obj.views_ok == args.views and obj.mesh_ok and obj.points_ok
        objects.append(obj)

        object_rows.append(
            {
                "uid": uid,
                "usable": obj.usable,
                "excluded": obj.excluded,
                "views_ok": obj.views_ok,
                "mesh_ok": obj.mesh_ok,
                "points_ok": obj.points_ok,
                "point_count": obj.point_count,
                "point_min": json.dumps(obj.point_min),
                "point_max": json.dumps(obj.point_max),
                "point_std": json.dumps(obj.point_std),
            }
        )

        if idx % 25 == 0 or idx == len(uids):
            print(f"[audit] objects={idx}/{len(uids)} usable_so_far={sum(o.usable for o in objects)}", flush=True)

    clean_uids = [obj.uid for obj in objects if obj.usable and not obj.excluded]
    splits = make_splits(out_dir, clean_uids, args)
    preview = make_preview(dataset_root, out_dir, objects, args, mask_root)

    issue_rows = [
        {
            "severity": item.severity,
            "uid": item.uid,
            "view": "" if item.view is None else item.view,
            "code": item.code,
            "message": item.message,
        }
        for item in issues
    ]
    write_csv(
        out_dir / "issues.csv",
        issue_rows,
        ["severity", "uid", "view", "code", "message"],
    )
    write_csv(
        out_dir / "objects_audit.csv",
        object_rows,
        ["uid", "usable", "excluded", "views_ok", "mesh_ok", "points_ok", "point_count", "point_min", "point_max", "point_std"],
    )
    write_csv(
        out_dir / "views_audit.csv",
        view_rows,
        [
            "uid",
            "view",
            "image_ok",
            "mask_ok",
            "camera_ok",
            "width",
            "height",
            "fg_ratio",
            "components",
            "edge_touch_ratio",
            "bbox_area_ratio",
            "image_hash",
            "azimuth",
            "elevation",
            "radius",
        ],
    )
    (out_dir / "clean_uids.txt").write_text("\n".join(clean_uids) + ("\n" if clean_uids else ""), encoding="utf-8")

    by_code = Counter(item.code for item in issues)
    by_severity = Counter(item.severity for item in issues)
    summary = {
        "dataset_root": str(dataset_root),
        "audited_at_unix": time.time(),
        "elapsed_sec": round(time.time() - started, 3),
        "discovered_objects": len(uids),
        "expected_objects": args.expected_objects,
        "bad_uids": len(bad_uids),
        "usable_objects": len(clean_uids),
        "blocked_objects": sum(1 for obj in objects if not obj.usable and not obj.excluded),
        "excluded_objects": sum(1 for obj in objects if obj.excluded),
        "expected_views_per_object": args.views,
        "usable_views": sum(obj.views_ok for obj in objects if not obj.excluded),
        "issue_counts_by_severity": dict(sorted(by_severity.items())),
        "top_issue_codes": dict(by_code.most_common(30)),
        "splits": {key: len(value) for key, value in splits.items()},
        "preview": str(preview) if preview else "",
        "mask_root": str(mask_root),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[audit] summary:", json.dumps(summary, indent=2), flush=True)
    blockers = by_severity.get(ISSUE_BLOCK, 0)
    if args.fail_on_blockers and blockers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
