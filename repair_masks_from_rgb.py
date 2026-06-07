#!/usr/bin/env python3
"""
Repair masks directly from already rendered RGB images.

This is the safest fast path when RGB renders are correct but mask renders are
bad. It estimates the background color from image borders and writes a binary
foreground mask for each render. No Blender, no camera math.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--views", type=int, default=24)
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max_objects", type=int, default=0)
    p.add_argument("--start_object", type=int, default=0)
    p.add_argument("--threshold", type=float, default=14.0)
    p.add_argument("--border", type=int, default=12)
    p.add_argument("--min_component_area", type=int, default=64)
    p.add_argument("--min_fg_ratio", type=float, default=0.002)
    p.add_argument("--max_fg_ratio", type=float, default=0.95)
    p.add_argument("--preview_count", type=int, default=80)
    p.add_argument("--preview_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def load_uids(dataset_root: Path) -> list[str]:
    views_csv = dataset_root / "metadata" / "views.csv"
    if views_csv.exists():
        uids = []
        seen = set()
        with open(views_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                uid = row.get("uid", "")
                if uid and uid not in seen:
                    seen.add(uid)
                    uids.append(uid)
        return uids
    return sorted(p.name for p in (dataset_root / "renders").iterdir() if p.is_dir())


def mask_is_valid(path: Path, min_fg: float, max_fg: float) -> tuple[bool, float]:
    if not path.exists():
        return False, -1.0
    try:
        from PIL import Image
        import numpy as np

        arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        fg = float((arr > 8).mean())
        return min_fg <= fg <= max_fg, fg
    except Exception:
        return False, -1.0


def build_jobs(dataset_root: Path, views: int, force: bool, max_objects: int, start_object: int, min_fg: float, max_fg: float):
    uids = load_uids(dataset_root)
    if start_object > 0:
        uids = uids[start_object:]
    if max_objects > 0:
        uids = uids[:max_objects]

    jobs = []
    valid = 0
    invalid = 0
    for uid in uids:
        repair_views = []
        for view in range(views):
            rgb_path = dataset_root / "renders" / uid / f"view_{view:03d}.png"
            if not rgb_path.exists():
                continue
            mask_path = dataset_root / "masks" / uid / f"view_{view:03d}.png"
            ok, _ = mask_is_valid(mask_path, min_fg, max_fg)
            if ok and not force:
                valid += 1
            else:
                invalid += 1
                repair_views.append(view)
        if repair_views:
            jobs.append((str(dataset_root), uid, repair_views))
    return jobs, valid, invalid


def estimate_background(rgb, border: int):
    import numpy as np

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


def clean_mask(mask, min_component_area: int):
    import numpy as np

    try:
        import cv2

        m = (mask.astype(np.uint8) * 255)
        kernel = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)

        num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
        keep = np.zeros_like(m, dtype=np.uint8)
        for label in range(1, num):
            if stats[label, cv2.CC_STAT_AREA] >= min_component_area:
                keep[labels == label] = 255

        # Fill holes by flood filling the inverse from image border.
        inv = 255 - keep
        flood = inv.copy()
        h, w = flood.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 0)
        holes = (flood > 0).astype(np.uint8) * 255
        keep = np.maximum(keep, holes)
        return keep
    except Exception:
        return (mask.astype(np.uint8) * 255)


def make_mask_from_rgb(rgb_path: Path, mask_path: Path, threshold: float, border: int, min_component_area: int):
    import numpy as np
    from PIL import Image

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32)
    bg = estimate_background(rgb, border)
    dist = np.linalg.norm(rgb - bg[None, None, :], axis=-1)

    # Also catch low-saturation gray objects by using local contrast from the
    # border-estimated background. Threshold is intentionally modest because
    # final morphology removes isolated noise.
    raw = dist > threshold
    mask = clean_mask(raw, min_component_area)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(mask_path)


def process_job(job, threshold: float, border: int, min_component_area: int, min_fg: float, max_fg: float):
    dataset_root_s, uid, views = job
    dataset_root = Path(dataset_root_s)
    start = time.time()
    bad = []
    for view in views:
        rgb_path = dataset_root / "renders" / uid / f"view_{view:03d}.png"
        mask_path = dataset_root / "masks" / uid / f"view_{view:03d}.png"
        try:
            make_mask_from_rgb(rgb_path, mask_path, threshold, border, min_component_area)
            ok, fg = mask_is_valid(mask_path, min_fg, max_fg)
            if not ok:
                bad.append((view, fg))
        except Exception as exc:
            bad.append((view, repr(exc)))
    return {"uid": uid, "views": len(views), "sec": time.time() - start, "ok": not bad, "bad": bad}


def _worker(args):
    job, threshold, border, min_area, min_fg, max_fg = args
    return process_job(job, threshold, border, min_area, min_fg, max_fg)


def make_preview_sheet(dataset_root: Path, count: int) -> Path | None:
    if count <= 0:
        return None
    from PIL import Image, ImageDraw

    pairs = []
    for uid_dir in sorted((dataset_root / "renders").glob("*")):
        if len(pairs) >= count:
            break
        uid = uid_dir.name
        rgb_path = uid_dir / "view_000.png"
        mask_path = dataset_root / "masks" / uid / "view_000.png"
        if rgb_path.exists() and mask_path.exists():
            pairs.append((uid, rgb_path, mask_path))
    if not pairs:
        return None

    tile = 160
    label_h = 22
    cols = 4
    rows = math.ceil(len(pairs) / cols)
    sheet = Image.new("RGB", (cols * tile * 2, rows * (tile + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, (uid, rgb_path, mask_path) in enumerate(pairs):
        row = idx // cols
        col = idx % cols
        x = col * tile * 2
        y = row * (tile + label_h)
        rgb = Image.open(rgb_path).convert("RGB").resize((tile, tile))
        mask = Image.open(mask_path).convert("L").resize((tile, tile)).convert("RGB")
        sheet.paste(rgb, (x, y + label_h))
        sheet.paste(mask, (x + tile, y + label_h))
        draw.text((x + 4, y + 4), uid[:18], fill=(20, 20, 20))
    out = dataset_root / "mask_rgb_preview.jpg"
    sheet.save(out, quality=92)
    return out


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    if args.preview_only:
        sheet = make_preview_sheet(dataset_root, args.preview_count)
        if sheet:
            print(f"Preview sheet: {sheet}", flush=True)
        return

    jobs, valid, invalid = build_jobs(
        dataset_root,
        args.views,
        args.force,
        args.max_objects,
        args.start_object,
        args.min_fg_ratio,
        args.max_fg_ratio,
    )
    total_views = sum(len(job[2]) for job in jobs)
    print(f"Dataset: {dataset_root}", flush=True)
    print(f"Existing valid views: {valid}", flush=True)
    print(f"Views to repair: {total_views}", flush=True)
    print(f"Objects to repair: {len(jobs)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)
    print(f"Threshold: {args.threshold}", flush=True)

    if args.dry_run or not jobs:
        sheet = make_preview_sheet(dataset_root, args.preview_count)
        if sheet:
            print(f"Preview sheet: {sheet}", flush=True)
        return

    start = time.time()
    failed = []
    payload = [
        (job, args.threshold, args.border, args.min_component_area, args.min_fg_ratio, args.max_fg_ratio)
        for job in jobs
    ]
    done_objects = 0
    done_views = 0
    with mp.Pool(processes=max(1, args.workers)) as pool:
        for result in pool.imap_unordered(_worker, payload):
            done_objects += 1
            done_views += result["views"]
            if not result["ok"]:
                failed.append(result)
            elapsed = time.time() - start
            sec_obj = elapsed / max(done_objects, 1)
            eta = sec_obj * max(len(jobs) - done_objects, 0) / 60.0
            print(
                f"[rgbmask] object={done_objects}/{len(jobs)} uid={result['uid']} "
                f"views={result['views']} obj_sec={result['sec']:.2f} "
                f"status={'ok' if result['ok'] else 'bad'} eta_min={eta:.1f}",
                flush=True,
            )

    print(f"Elapsed minutes: {(time.time() - start) / 60.0:.1f}", flush=True)
    print(f"Processed views: {done_views}", flush=True)
    print(f"Failed objects: {len(failed)}", flush=True)
    for item in failed[:30]:
        print(f"FAILED {item['uid']} bad={item['bad'][:8]}", flush=True)

    jobs_after, valid_after, _ = build_jobs(
        dataset_root,
        args.views,
        False,
        args.max_objects,
        args.start_object,
        args.min_fg_ratio,
        args.max_fg_ratio,
    )
    print(f"Valid views after repair: {valid_after}", flush=True)
    print(f"Still invalid views: {sum(len(job[2]) for job in jobs_after)}", flush=True)
    sheet = make_preview_sheet(dataset_root, args.preview_count)
    if sheet:
        print(f"Preview sheet: {sheet}", flush=True)


if __name__ == "__main__":
    main()
