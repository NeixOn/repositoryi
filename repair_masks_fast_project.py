#!/usr/bin/env python3
"""
Fast mask repair without Blender.

Reads:
  objects/<uid>/normalized.glb
  cameras/<uid>/view_XXX.json
  renders/<uid>/view_XXX.png  (only for preview)

Writes:
  masks/<uid>/view_XXX.png

The mask is a projected triangle silhouette from the normalized mesh and saved
camera matrices. This is much faster than starting Blender for every object.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--views", type=int, default=24)
    p.add_argument("--force", action="store_true")
    p.add_argument("--max_objects", type=int, default=0)
    p.add_argument("--start_object", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--preview_count", type=int, default=80)
    p.add_argument("--preview_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--min_fg_ratio", type=float, default=0.002)
    p.add_argument("--max_fg_ratio", type=float, default=0.95)
    return p.parse_args()


def mask_is_valid(path: Path, min_fg: float, max_fg: float) -> tuple[bool, float]:
    if not path.exists():
        return False, -1.0
    try:
        from PIL import Image
        import numpy as np

        arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
        if arr.size == 0:
            return False, -1.0
        fg = float((arr > 8).mean())
        return min_fg <= fg <= max_fg, fg
    except Exception:
        return False, -1.0


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
    objects_dir = dataset_root / "objects"
    return sorted(p.name for p in objects_dir.iterdir() if (p / "normalized.glb").exists())


def build_jobs(dataset_root: Path, views: int, force: bool, max_objects: int, start_object: int, min_fg: float, max_fg: float):
    uids = load_uids(dataset_root)
    if start_object > 0:
        uids = uids[start_object:]
    if max_objects > 0:
        uids = uids[:max_objects]

    jobs = []
    valid_views = 0
    invalid_views = 0
    for uid in uids:
        mesh_path = dataset_root / "objects" / uid / "normalized.glb"
        if not mesh_path.exists():
            continue
        repair_views = []
        for view in range(views):
            camera_path = dataset_root / "cameras" / uid / f"view_{view:03d}.json"
            if not camera_path.exists():
                continue
            mask_path = dataset_root / "masks" / uid / f"view_{view:03d}.png"
            ok, _ = mask_is_valid(mask_path, min_fg, max_fg)
            if ok and not force:
                valid_views += 1
            else:
                invalid_views += 1
                repair_views.append(view)
        if repair_views:
            jobs.append((str(dataset_root), uid, repair_views, min_fg, max_fg))
    return jobs, valid_views, invalid_views


def load_mesh_arrays(mesh_path: Path):
    import numpy as np
    import trimesh

    loaded = trimesh.load(mesh_path, force="scene")
    meshes = []
    if isinstance(loaded, trimesh.Scene):
        for geom in loaded.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) and len(geom.faces):
                meshes.append(geom)
    elif isinstance(loaded, trimesh.Trimesh):
        meshes.append(loaded)
    if not meshes:
        raise RuntimeError(f"No mesh geometry in {mesh_path}")

    vertices = []
    faces = []
    offset = 0
    for mesh in meshes:
        v = np.asarray(mesh.vertices, dtype=np.float32)
        f = np.asarray(mesh.faces, dtype=np.int64)
        vertices.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.concatenate(vertices, axis=0), np.concatenate(faces, axis=0)


def project_vertices(vertices, camera_path: Path):
    import numpy as np

    data = json.loads(camera_path.read_text(encoding="utf-8"))
    intr = data["intrinsics"]
    width = int(intr.get("width", 512))
    height = int(intr.get("height", 512))
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    w2c = np.asarray(data["world_to_camera"], dtype=np.float32).reshape(4, 4)

    homo = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float32)], axis=1)
    cam = homo @ w2c.T
    zneg = -cam[:, 2]
    valid = zneg > 1e-4
    xy = np.empty((len(vertices), 2), dtype=np.float32)
    xy[:, 0] = fx * (cam[:, 0] / np.maximum(zneg, 1e-4)) + cx
    xy[:, 1] = fy * (-cam[:, 1] / np.maximum(zneg, 1e-4)) + cy
    return xy, valid, width, height


def rasterize_mask(vertices, faces, camera_path: Path, out_path: Path):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter

    xy, valid, width, height = project_vertices(vertices, camera_path)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)

    # Drawing the union of projected triangles is enough for a silhouette mask.
    for tri in faces:
        if not (valid[tri[0]] and valid[tri[1]] and valid[tri[2]]):
            continue
        pts = xy[tri]
        if (
            pts[:, 0].max() < -2
            or pts[:, 0].min() > width + 2
            or pts[:, 1].max() < -2
            or pts[:, 1].min() > height + 2
        ):
            continue
        draw.polygon([(float(x), float(y)) for x, y in pts], fill=255)

    # Tiny dilation closes sub-pixel cracks between triangles.
    image = image.filter(ImageFilter.MaxFilter(3))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def process_job(job):
    dataset_root_s, uid, views, min_fg, max_fg = job
    dataset_root = Path(dataset_root_s)
    mesh_path = dataset_root / "objects" / uid / "normalized.glb"
    start = time.time()
    try:
        vertices, faces = load_mesh_arrays(mesh_path)
        view_stats = []
        for view in views:
            camera_path = dataset_root / "cameras" / uid / f"view_{view:03d}.json"
            out_path = dataset_root / "masks" / uid / f"view_{view:03d}.png"
            rasterize_mask(vertices, faces, camera_path, out_path)
            ok, fg = mask_is_valid(out_path, min_fg, max_fg)
            view_stats.append((view, ok, fg))
        bad = [item for item in view_stats if not item[1]]
        return {
            "uid": uid,
            "views": len(views),
            "sec": time.time() - start,
            "ok": len(bad) == 0,
            "bad": bad,
            "error": "",
        }
    except Exception as exc:
        return {
            "uid": uid,
            "views": len(views),
            "sec": time.time() - start,
            "ok": False,
            "bad": [],
            "error": repr(exc),
        }


def make_preview_sheet(dataset_root: Path, count: int) -> Path | None:
    if count <= 0:
        return None
    from PIL import Image, ImageDraw

    pairs = []
    for uid_dir in sorted((dataset_root / "masks").glob("*")):
        if len(pairs) >= count:
            break
        uid = uid_dir.name
        mask_path = uid_dir / "view_000.png"
        rgb_path = dataset_root / "renders" / uid / "view_000.png"
        if mask_path.exists() and rgb_path.exists():
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
    out_path = dataset_root / "mask_fast_preview.jpg"
    sheet.save(out_path, quality=92)
    return out_path


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    if args.preview_only:
        sheet = make_preview_sheet(dataset_root, args.preview_count)
        if sheet:
            print(f"Preview sheet: {sheet}", flush=True)
        return

    jobs, valid_views, invalid_views = build_jobs(
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
    print(f"Existing valid views: {valid_views}", flush=True)
    print(f"Views to repair: {total_views}", flush=True)
    print(f"Objects to repair: {len(jobs)}", flush=True)
    print(f"Workers: {args.workers}", flush=True)

    if args.dry_run or not jobs:
        sheet = make_preview_sheet(dataset_root, args.preview_count)
        if sheet:
            print(f"Preview sheet: {sheet}", flush=True)
        return

    start = time.time()
    done_objects = 0
    done_views = 0
    failed = []
    workers = max(1, int(args.workers))

    with mp.Pool(processes=workers) as pool:
        for result in pool.imap_unordered(process_job, jobs):
            done_objects += 1
            done_views += int(result["views"])
            elapsed = time.time() - start
            sec_obj = elapsed / max(done_objects, 1)
            eta = sec_obj * max(len(jobs) - done_objects, 0) / 60.0
            status = "ok" if result["ok"] else "bad"
            if not result["ok"]:
                failed.append(result)
            print(
                f"[fastmask] object={done_objects}/{len(jobs)} uid={result['uid']} "
                f"views={result['views']} obj_sec={result['sec']:.2f} status={status} eta_min={eta:.1f}",
                flush=True,
            )

    print(f"Elapsed minutes: {(time.time() - start) / 60.0:.1f}", flush=True)
    print(f"Processed views: {done_views}", flush=True)
    print(f"Failed objects: {len(failed)}", flush=True)
    for item in failed[:30]:
        print(f"FAILED {item['uid']} error={item['error']} bad={item['bad'][:5]}", flush=True)

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
