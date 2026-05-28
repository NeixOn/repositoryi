#!/usr/bin/env python3
"""
Kaggle-ready Objaverse chair dataset builder.

What it creates per accepted object:
  output_dir/
    objects/<uid>/normalized.glb
    objects/<uid>/points.npz
    renders/<uid>/view_000.png ... view_N.png
    metadata/views.csv
    metadata/objects.csv
    metadata/failed_objects.csv

Run in a Kaggle notebook:
  !python prepare_objaverse_chairs_kaggle.py --num_objects 300 --views 12

Run and autosave to a Kaggle Dataset after every processed batch:
  !python prepare_objaverse_chairs_kaggle.py --num_objects 300 --views 12 \
      --publish_to_kaggle --kaggle_dataset_id your_username/objaverse-chairs \
      --kaggle_create_if_missing

The script uses Objaverse LVIS annotations for chair-like categories, downloads
objects in small chunks, normalizes them in Blender, renders transparent RGB
views, saves camera matrices, and samples surface point clouds.
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
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


DEFAULT_OUTPUT_DIR = "/kaggle/working/objaverse_chairs"
DEFAULT_CACHE_DIR = "/kaggle/working/objaverse_cache"
DEFAULT_BLENDER_DIR = "/kaggle/working/blender"
DEFAULT_BLENDER_VERSION = "4.1.1"
DEFAULT_CATEGORIES = (
    "chair",
    "armchair",
    "stool",
    "folding_chair",
    "rocking_chair",
    "swivel_chair",
)


BLENDER_WORKER_CODE = r'''
import argparse
import csv
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path

import bpy
import mathutils
import numpy as np


IMAGE_FORMAT = "PNG"


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--points", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-faces", type=int, required=True)
    parser.add_argument("--max-faces", type=int, required=True)
    parser.add_argument("--camera-radius", type=float, required=True)
    parser.add_argument("--elevation-min", type=float, required=True)
    parser.add_argument("--elevation-max", type=float, required=True)
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)


def configure_render(resolution, use_gpu):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.render.image_settings.file_format = IMAGE_FORMAT
    scene.render.image_settings.color_mode = "RGBA"
    scene.world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world.color = (1.0, 1.0, 1.0)

    if use_gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        enabled = False
        for compute_type in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = compute_type
                prefs.get_devices()
                for device in prefs.devices:
                    if device.type != "CPU":
                        device.use = True
                        enabled = True
                if enabled:
                    scene.cycles.device = "GPU"
                    print(f"[blender] GPU rendering enabled with {compute_type}", flush=True)
                    return
            except Exception:
                pass
        print("[blender] GPU unavailable, falling back to CPU", flush=True)


def import_object(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".glb" or suffix == ".gltf":
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif suffix == ".dae":
        bpy.ops.wm.collada_import(filepath=path)
    else:
        raise RuntimeError(f"Unsupported file extension: {suffix}")


def mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def bake_world_transforms():
    # Objaverse assets often contain nested transforms. Baking them makes
    # normalization, export, and point sampling use one consistent coordinate frame.
    for obj in mesh_objects():
        obj.data = obj.data.copy()
        matrix = obj.matrix_world.copy()
        for vertex in obj.data.vertices:
            vertex.co = matrix @ vertex.co
        obj.parent = None
        obj.matrix_world = mathutils.Matrix.Identity(4)
        obj.data.update()
    bpy.context.view_layer.update()


def world_bbox(objects):
    points = []
    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ mathutils.Vector(corner))
    if not points:
        raise RuntimeError("No mesh bounding boxes")
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    return (
        mathutils.Vector((min(xs), min(ys), min(zs))),
        mathutils.Vector((max(xs), max(ys), max(zs))),
    )


def face_count(objects):
    return sum(len(obj.data.polygons) for obj in objects)


def normalize_scene():
    objects = mesh_objects()
    if not objects:
        raise RuntimeError("No mesh objects after import")

    lo, hi = world_bbox(objects)
    dims = hi - lo
    max_dim = max(dims.x, dims.y, dims.z)
    if not math.isfinite(max_dim) or max_dim <= 1e-7:
        raise RuntimeError(f"Degenerate bounding box: {tuple(dims)}")

    center = (lo + hi) * 0.5
    scale = 1.8 / max_dim
    for obj in objects:
        obj.location = (obj.location - center) * scale
        obj.scale = obj.scale * scale

    bpy.context.view_layer.update()

    lo2, hi2 = world_bbox(objects)
    floor_offset = lo2.z
    for obj in objects:
        obj.location.z -= floor_offset
    bpy.context.view_layer.update()

    lo3, hi3 = world_bbox(objects)
    dims3 = hi3 - lo3
    center3 = (lo3 + hi3) * 0.5
    return {
        "bbox_min": [lo3.x, lo3.y, lo3.z],
        "bbox_max": [hi3.x, hi3.y, hi3.z],
        "bbox_dims": [dims3.x, dims3.y, dims3.z],
        "bbox_center": [center3.x, center3.y, center3.z],
    }


def add_lights():
    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.5, 5.0))
    key = bpy.context.object
    key.name = "Key_Area_Light"
    key.data.energy = 500
    key.data.size = 5.0

    bpy.ops.object.light_add(type="POINT", location=(-3.0, 2.5, 3.0))
    fill = bpy.context.object
    fill.name = "Fill_Light"
    fill.data.energy = 80


def make_camera():
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    camera.data.lens = 55
    camera.data.sensor_width = 32
    bpy.context.scene.camera = camera
    return camera


def look_at(camera, target):
    direction = mathutils.Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def matrix_to_list(matrix):
    return [[float(v) for v in row] for row in matrix]


def camera_intrinsics(camera, resolution):
    lens = camera.data.lens
    sensor_width = camera.data.sensor_width
    focal_px = lens * resolution / sensor_width
    cx = resolution / 2.0
    cy = resolution / 2.0
    return [[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]]


def export_normalized_glb(path):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects():
        obj.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=True,
    )


def triangulated_mesh_data():
    vertices_parts = []
    faces_parts = []
    vertex_offset = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for obj in mesh_objects():
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            mesh.calc_loop_triangles()
            verts = []
            for v in mesh.vertices:
                p = obj.matrix_world @ v.co
                verts.append([p.x, p.y, p.z])
            verts = np.array(verts, dtype=np.float64)
            tris = np.array([[tri.vertices[0], tri.vertices[1], tri.vertices[2]] for tri in mesh.loop_triangles], dtype=np.int64)
            if len(verts) and len(tris):
                vertices_parts.append(verts)
                faces_parts.append(tris + vertex_offset)
                vertex_offset += len(verts)
        finally:
            eval_obj.to_mesh_clear()

    if not vertices_parts:
        raise RuntimeError("No triangulated mesh data")
    return np.concatenate(vertices_parts, axis=0), np.concatenate(faces_parts, axis=0)


def sample_surface_points(count):
    vertices, faces = triangulated_mesh_data()
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = areas > 1e-12
    if not np.any(valid):
        raise RuntimeError("No non-zero-area triangles")

    tri = tri[valid]
    areas = areas[valid]
    normals = cross[valid]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    probs = areas / areas.sum()

    chosen = np.random.choice(len(tri), size=count, replace=True, p=probs)
    tri_chosen = tri[chosen]
    normals_chosen = normals[chosen]

    u = np.random.rand(count, 1)
    v = np.random.rand(count, 1)
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = tri_chosen[:, 0] + u * (tri_chosen[:, 1] - tri_chosen[:, 0]) + v * (tri_chosen[:, 2] - tri_chosen[:, 0])
    return points.astype(np.float32), normals_chosen.astype(np.float32)


def render_views(uid, out_dir, views, resolution, radius, elevation_min, elevation_max, seed):
    render_dir = out_dir / "renders" / uid
    render_dir.mkdir(parents=True, exist_ok=True)
    camera = make_camera()
    add_lights()

    rng = random.Random(seed)
    # One clean orbit plus slight deterministic elevation variation.
    start_azimuth = rng.uniform(0, 360.0)
    view_rows = []
    for view_idx in range(views):
        azimuth = start_azimuth + view_idx * (360.0 / views)
        elevation = rng.uniform(elevation_min, elevation_max)
        az = math.radians(azimuth)
        el = math.radians(elevation)
        x = radius * math.cos(el) * math.cos(az)
        y = radius * math.cos(el) * math.sin(az)
        z = radius * math.sin(el) + 0.75
        camera.location = (x, y, z)
        look_at(camera, (0.0, 0.0, 0.75))

        png_path = render_dir / f"view_{view_idx:03d}.png"
        bpy.context.scene.render.filepath = str(png_path)
        bpy.ops.render.render(write_still=True)

        view_rows.append({
            "uid": uid,
            "view_index": view_idx,
            "image_path": str(png_path),
            "azimuth_deg": float(azimuth % 360.0),
            "elevation_deg": float(elevation),
            "radius": float(radius),
            "camera_location": [float(v) for v in camera.location],
            "camera_matrix_world": matrix_to_list(camera.matrix_world),
            "camera_intrinsics": camera_intrinsics(camera, resolution),
        })
    return view_rows


def process_one(entry, out_dir, args, index):
    uid = entry["uid"]
    path = entry["path"]
    random.seed(args.seed + index)
    np.random.seed(args.seed + index)

    clean_scene()
    import_object(path)
    bake_world_transforms()
    objects = mesh_objects()
    if not objects:
        raise RuntimeError("No meshes imported")

    faces = face_count(objects)
    if faces < args.min_faces:
        raise RuntimeError(f"Too few faces: {faces}")
    if faces > args.max_faces:
        raise RuntimeError(f"Too many faces: {faces}")

    bbox = normalize_scene()
    object_dir = out_dir / "objects" / uid
    object_dir.mkdir(parents=True, exist_ok=True)
    export_normalized_glb(object_dir / "normalized.glb")
    points, normals = sample_surface_points(args.points)
    np.savez_compressed(object_dir / "points.npz", points=points, normals=normals)

    views = render_views(
        uid=uid,
        out_dir=out_dir,
        views=args.views,
        resolution=args.resolution,
        radius=args.camera_radius,
        elevation_min=args.elevation_min,
        elevation_max=args.elevation_max,
        seed=args.seed + index,
    )
    return {
        "uid": uid,
        "source_path": path,
        "normalized_glb": str(object_dir / "normalized.glb"),
        "points_npz": str(object_dir / "points.npz"),
        "face_count": faces,
        **bbox,
    }, views


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    configure_render(args.resolution, args.use_gpu)

    with open(args.manifest, "r", encoding="utf-8") as f:
        entries = json.load(f)

    object_rows = []
    view_rows = []
    failed_rows = []

    for index, entry in enumerate(entries):
        uid = entry["uid"]
        print(f"[blender] Processing {index + 1}/{len(entries)} {uid}", flush=True)
        try:
            object_row, rows = process_one(entry, out_dir, args, index)
            object_rows.append(object_row)
            view_rows.extend(rows)
        except Exception as exc:
            failed_rows.append({
                "uid": uid,
                "source_path": entry.get("path", ""),
                "error": str(exc),
                "traceback": traceback.format_exc(limit=6),
            })
            print(f"[blender] FAILED {uid}: {exc}", flush=True)
        finally:
            clean_scene()

    with open(metadata_dir / "objects_chunk.csv", "w", encoding="utf-8", newline="") as f:
        if object_rows:
            writer = csv.DictWriter(f, fieldnames=list(object_rows[0].keys()))
            writer.writeheader()
            writer.writerows(object_rows)

    with open(metadata_dir / "views_chunk.csv", "w", encoding="utf-8", newline="") as f:
        if view_rows:
            writer = csv.DictWriter(f, fieldnames=list(view_rows[0].keys()))
            writer.writeheader()
            writer.writerows(view_rows)

    with open(metadata_dir / "failed_chunk.csv", "w", encoding="utf-8", newline="") as f:
        if failed_rows:
            writer = csv.DictWriter(f, fieldnames=list(failed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failed_rows)

    print(
        f"[blender] Chunk complete: accepted={len(object_rows)} failed={len(failed_rows)} views={len(view_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''


def run(cmd: Sequence[str], *, check: bool = True, env: Dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(list(cmd), check=check, env=env)


def pip_install(packages: Sequence[str]) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", *packages])


def ensure_dependencies(skip_install: bool) -> None:
    if skip_install:
        return
    pip_install(["objaverse", "tqdm", "kaggle"])


def blender_release_url(version: str) -> str:
    major_minor = ".".join(version.split(".")[:2])
    return f"https://download.blender.org/release/Blender{major_minor}/blender-{version}-linux-x64.tar.xz"


def find_blender(root: Path) -> Path | None:
    system = shutil.which("blender")
    if system:
        return Path(system)
    candidates = sorted(root.glob("**/blender"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def safe_extract_tar(archive: tarfile.TarFile, dst: Path) -> None:
    dst_resolved = dst.resolve()
    for member in archive.getmembers():
        target = (dst / member.name).resolve()
        try:
            target.relative_to(dst_resolved)
        except ValueError:
            raise RuntimeError(f"Unsafe archive member path: {member.name}")
    archive.extractall(dst)


def try_install_blender_with_apt() -> Path | None:
    if os.name != "posix" or not shutil.which("apt-get"):
        return None
    print("Blender executable was not found. Trying apt-get install blender...", flush=True)
    commands = [
        ["apt-get", "update", "-qq"],
        ["apt-get", "install", "-y", "-qq", "blender"],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"apt-get command failed: {' '.join(cmd)}", flush=True)
            return None
    blender = shutil.which("blender")
    if blender:
        return Path(blender)
    return None


def download_with_progress(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Blender from {url}", flush=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 Kaggle Objaverse Dataset Builder",
            "Accept": "application/octet-stream,*/*",
        },
    )
    with urllib.request.urlopen(request) as response, open(dst, "wb") as f:
        total = int(response.headers.get("Content-Length", "0") or "0")
        downloaded = 0
        last_print = time.time()
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if time.time() - last_print > 5:
                if total:
                    print(f"  {downloaded / 1024**2:.1f}/{total / 1024**2:.1f} MB", flush=True)
                else:
                    print(f"  {downloaded / 1024**2:.1f} MB", flush=True)
                last_print = time.time()


def ensure_blender(blender_dir: Path, version: str, skip_download: bool, skip_apt: bool) -> Path:
    blender = find_blender(blender_dir)
    if blender:
        print(f"Using Blender: {blender}", flush=True)
        return blender

    if not skip_apt:
        blender = try_install_blender_with_apt()
        if blender:
            print(f"Using Blender installed by apt: {blender}", flush=True)
            return blender

    if skip_download:
        raise RuntimeError("Blender was not found and --skip_blender_download was set.")

    archive = blender_dir / f"blender-{version}-linux-x64.tar.xz"
    if not archive.exists():
        try:
            download_with_progress(blender_release_url(version), archive)
        except Exception as exc:
            raise RuntimeError(
                "Could not install Blender. apt-get did not provide Blender and the direct Blender download failed. "
                "In Kaggle, try adding a Blender binary as an input dataset or run with internet enabled."
            ) from exc
    print(f"Extracting {archive}", flush=True)
    blender_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tar:
        safe_extract_tar(tar, blender_dir)

    blender = find_blender(blender_dir)
    if not blender:
        raise RuntimeError(f"Could not find Blender executable under {blender_dir}")
    print(f"Using Blender: {blender}", flush=True)
    return blender


def write_blender_worker(output_dir: Path) -> Path:
    scripts_dir = output_dir / "_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    worker_path = scripts_dir / "blender_objaverse_worker.py"
    worker_path.write_text(BLENDER_WORKER_CODE, encoding="utf-8")
    return worker_path


def append_csv(src: Path, dst: Path) -> None:
    if not src.exists() or src.stat().st_size == 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    write_header = not dst.exists() or dst.stat().st_size == 0
    with open(src, "r", encoding="utf-8", newline="") as f_src, open(dst, "a", encoding="utf-8", newline="") as f_dst:
        reader = csv.reader(f_src)
        writer = csv.writer(f_dst)
        header = next(reader, None)
        if header is None:
            return
        if write_header:
            writer.writerow(header)
        for row in reader:
            writer.writerow(row)


def batched(items: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def normalize_categories(raw: str | None) -> List[str]:
    if not raw:
        return list(DEFAULT_CATEGORIES)
    return [part.strip() for part in raw.split(",") if part.strip()]


def select_chair_uids(categories: Sequence[str], seed: int, max_candidates: int | None) -> List[str]:
    import objaverse

    print("Loading Objaverse LVIS annotations...", flush=True)
    lvis = objaverse.load_lvis_annotations()
    lower_to_key = {key.lower(): key for key in lvis.keys()}
    selected_categories = []
    selected_uids = []

    for category in categories:
        key = lower_to_key.get(category.lower())
        if key and key not in selected_categories:
            selected_categories.append(key)
            selected_uids.extend(lvis[key])

    if not selected_uids:
        for key, uids in lvis.items():
            low = key.lower()
            if "chair" in low or "stool" in low:
                selected_categories.append(key)
                selected_uids.extend(uids)

    selected_uids = sorted(set(selected_uids))
    rng = random.Random(seed)
    rng.shuffle(selected_uids)
    if max_candidates:
        selected_uids = selected_uids[:max_candidates]

    print(f"Selected LVIS categories: {selected_categories}", flush=True)
    print(f"Candidate chair-like UIDs: {len(selected_uids)}", flush=True)
    if not selected_uids:
        raise RuntimeError("No chair UIDs found in Objaverse LVIS annotations.")
    return selected_uids


def load_metadata_for_uids(uids: Sequence[str]) -> Dict[str, dict]:
    import objaverse

    try:
        return objaverse.load_annotations(uids=list(uids))
    except TypeError:
        annotations = objaverse.load_annotations()
        return {uid: annotations.get(uid, {}) for uid in uids}


def license_allowed(annotation: dict, allowed: Sequence[str]) -> bool:
    if not allowed:
        return True
    license_value = str(annotation.get("license", "")).lower()
    return any(token.lower() in license_value for token in allowed)


def remove_downloaded_files(paths: Iterable[str]) -> None:
    for path in paths:
        try:
            p = Path(path)
            if p.exists() and p.is_file():
                p.unlink()
        except Exception as exc:
            print(f"Warning: could not remove cached object {path}: {exc}", flush=True)


def count_rows(csv_path: Path) -> int:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def write_dataset_info(args: argparse.Namespace, output_dir: Path, accepted: int) -> None:
    metadata_dir = output_dir / "metadata"
    info = {
        "dataset_root": str(output_dir),
        "objects_csv": str(metadata_dir / "objects.csv"),
        "views_csv": str(metadata_dir / "views.csv"),
        "failed_objects_csv": str(metadata_dir / "failed_objects.csv"),
        "accepted_objects": accepted,
        "views_per_object": args.views,
        "resolution": args.resolution,
        "points_per_object": args.points,
        "categories": normalize_categories(args.categories),
        "kaggle_persistence_note": (
            "This dataset is written under /kaggle/working by default. "
            "Kaggle keeps files from /kaggle/working as notebook output after Save Version/Commit."
        ),
    }
    (metadata_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=True, indent=2), encoding="utf-8")


def write_kaggle_dataset_metadata(args: argparse.Namespace, output_dir: Path) -> None:
    if not args.kaggle_dataset_id:
        return
    dataset_slug = args.kaggle_dataset_id.split("/", 1)[-1]
    title = args.kaggle_dataset_title or dataset_slug.replace("-", " ").replace("_", " ").title()
    metadata = {
        "title": title,
        "id": args.kaggle_dataset_id,
        "licenses": [{"name": args.kaggle_dataset_license}],
    }
    (output_dir / "dataset-metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")


def install_kaggle_json(source: Path) -> None:
    target_dir = Path.home() / ".kaggle"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "kaggle.json"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    try:
        target.chmod(0o600)
    except OSError:
        pass


def configure_kaggle_credentials(args: argparse.Namespace) -> None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return

    explicit = Path(args.kaggle_json_path).expanduser() if args.kaggle_json_path else None
    if explicit and explicit.exists():
        install_kaggle_json(explicit)
        return

    existing = Path.home() / ".kaggle" / "kaggle.json"
    if existing.exists():
        try:
            existing.chmod(0o600)
        except OSError:
            pass
        return

    search_roots = [Path("/kaggle/working"), Path("/kaggle/input")]
    for root in search_roots:
        if not root.exists():
            continue
        matches = list(root.glob("**/kaggle.json"))
        if matches:
            install_kaggle_json(matches[0])
            print(f"Using Kaggle API token from {matches[0]}", flush=True)
            return


def kaggle_credentials_available() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    home = Path.home()
    candidates = [
        home / ".kaggle" / "kaggle.json",
        home / ".config" / "kaggle" / "kaggle.json",
        Path("/root/.kaggle/kaggle.json"),
    ]
    return any(path.exists() for path in candidates)


def kaggle_command_prefix() -> List[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    return [sys.executable, "-m", "kaggle"]


def publish_kaggle_dataset(args: argparse.Namespace, output_dir: Path, message: str, *, final: bool = False) -> None:
    if not args.publish_to_kaggle:
        return
    if not args.kaggle_dataset_id:
        raise RuntimeError("--kaggle_dataset_id is required when --publish_to_kaggle is set.")
    configure_kaggle_credentials(args)
    if not kaggle_credentials_available():
        raise RuntimeError(
            "Kaggle API credentials were not found. Add KAGGLE_USERNAME/KAGGLE_KEY secrets "
            "or place kaggle.json at /root/.kaggle/kaggle.json."
        )

    write_kaggle_dataset_metadata(args, output_dir)
    if not (output_dir / "dataset-metadata.json").exists():
        raise RuntimeError("dataset-metadata.json was not created.")

    base_cmd = [
        *kaggle_command_prefix(),
        "datasets",
        "version",
        "-p",
        str(output_dir),
        "-m",
        message[:190],
        "-r",
        args.kaggle_upload_mode,
    ]

    print(f"Publishing Kaggle Dataset version: {args.kaggle_dataset_id}", flush=True)
    try:
        run(base_cmd)
        return
    except subprocess.CalledProcessError:
        if not args.kaggle_create_if_missing:
            raise
        print("Dataset version upload failed. Trying to create the dataset first...", flush=True)

    create_cmd = [
        *kaggle_command_prefix(),
        "datasets",
        "create",
        "-p",
        str(output_dir),
        "-r",
        args.kaggle_upload_mode,
    ]
    if args.kaggle_public:
        create_cmd.append("-u")
    run(create_cmd)
    if final:
        print("Kaggle Dataset created on final publish.", flush=True)


def build_dataset(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HOME", str(Path(args.cache_dir) / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(args.cache_dir) / "huggingface" / "transformers"))
    os.environ.setdefault("OBJAVERSE_HOME", str(Path(args.cache_dir) / "objaverse"))

    ensure_dependencies(args.skip_install)

    output_dir = Path(args.output_dir)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_kaggle_dataset_metadata(args, output_dir)
    worker_path = write_blender_worker(output_dir)
    blender = ensure_blender(
        Path(args.blender_dir),
        args.blender_version,
        args.skip_blender_download,
        args.skip_apt_blender,
    )

    categories = normalize_categories(args.categories)
    max_candidates = args.max_candidates if args.max_candidates > 0 else None
    candidate_uids = select_chair_uids(categories, args.seed, max_candidates)

    accepted = count_rows(metadata_dir / "objects.csv")
    attempted = set()
    for csv_name in ("objects.csv", "failed_objects.csv"):
        path = metadata_dir / csv_name
        if path.exists():
            with open(path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("uid"):
                        attempted.add(row["uid"])

    print(f"Already accepted: {accepted}; already attempted: {len(attempted)}", flush=True)
    if accepted >= args.num_objects:
        print("Requested number of objects already exists. Nothing to do.", flush=True)
        write_dataset_info(args, output_dir, accepted)
        publish_kaggle_dataset(
            args,
            output_dir,
            f"Objaverse chairs dataset already prepared: {accepted} objects",
            final=True,
        )
        return

    import objaverse

    processed_batches = 0
    last_published_count = -1
    for uid_batch in batched(candidate_uids, args.download_batch):
        remaining = args.num_objects - count_rows(metadata_dir / "objects.csv")
        if remaining <= 0:
            break

        uid_batch = [uid for uid in uid_batch if uid not in attempted]
        if not uid_batch:
            continue

        if args.allowed_licenses:
            annotations = load_metadata_for_uids(uid_batch)
            allowed_tokens = [x.strip() for x in args.allowed_licenses.split(",") if x.strip()]
            uid_batch = [uid for uid in uid_batch if license_allowed(annotations.get(uid, {}), allowed_tokens)]
            if not uid_batch:
                continue

        print(f"Downloading batch of {len(uid_batch)} objects...", flush=True)
        try:
            downloaded = objaverse.load_objects(uids=uid_batch, download_processes=args.download_processes)
        except Exception as exc:
            print(f"Batch download failed: {exc}", flush=True)
            for uid in uid_batch:
                attempted.add(uid)
            continue

        entries = [{"uid": uid, "path": str(path)} for uid, path in downloaded.items() if path and Path(path).exists()]
        if not entries:
            continue

        # Process only a little more than needed, because some objects will be rejected.
        entries = entries[: max(remaining * 2, min(len(entries), args.download_batch))]
        manifest = output_dir / "_scripts" / "chunk_manifest.json"
        manifest.write_text(json.dumps(entries, ensure_ascii=True, indent=2), encoding="utf-8")

        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        cmd = [
            str(blender),
            "--background",
            "--factory-startup",
            "--python",
            str(worker_path),
            "--",
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--views",
            str(args.views),
            "--resolution",
            str(args.resolution),
            "--points",
            str(args.points),
            "--seed",
            str(args.seed),
            "--min-faces",
            str(args.min_faces),
            "--max-faces",
            str(args.max_faces),
            "--camera-radius",
            str(args.camera_radius),
            "--elevation-min",
            str(args.elevation_min),
            "--elevation-max",
            str(args.elevation_max),
        ]
        if args.use_gpu:
            cmd.append("--use-gpu")

        # The worker writes chunk CSVs into metadata/. Move them through a stable
        # temporary location before appending, so reruns remain resumable.
        run(cmd, env=env)

        append_csv(metadata_dir / "objects_chunk.csv", metadata_dir / "objects.csv")
        append_csv(metadata_dir / "views_chunk.csv", metadata_dir / "views.csv")
        append_csv(metadata_dir / "failed_chunk.csv", metadata_dir / "failed_objects.csv")
        for temp_name in ("objects_chunk.csv", "views_chunk.csv", "failed_chunk.csv"):
            temp_path = metadata_dir / temp_name
            if temp_path.exists():
                temp_path.unlink()

        for entry in entries:
            attempted.add(entry["uid"])

        accepted_now = count_rows(metadata_dir / "objects.csv")
        print(f"Progress: accepted {accepted_now}/{args.num_objects}", flush=True)

        if not args.keep_raw:
            remove_downloaded_files(downloaded.values())

        processed_batches += 1
        write_dataset_info(args, output_dir, accepted_now)
        if (
            args.publish_to_kaggle
            and accepted_now > 0
            and accepted_now != last_published_count
            and processed_batches % args.publish_every_batches == 0
        ):
            publish_kaggle_dataset(
                args,
                output_dir,
                f"Objaverse chairs autosave: {accepted_now} accepted objects",
            )
            last_published_count = accepted_now

    final_count = count_rows(metadata_dir / "objects.csv")
    write_dataset_info(args, output_dir, final_count)
    if args.publish_to_kaggle and final_count > 0 and final_count != last_published_count:
        publish_kaggle_dataset(
            args,
            output_dir,
            f"Objaverse chairs final autosave: {final_count} accepted objects",
            final=True,
        )
    print("", flush=True)
    print(f"Done. Accepted objects: {final_count}", flush=True)
    print(f"Dataset root: {output_dir}", flush=True)
    print(f"Views CSV: {metadata_dir / 'views.csv'}", flush=True)
    print(f"Objects CSV: {metadata_dir / 'objects.csv'}", flush=True)
    print(f"Dataset info: {metadata_dir / 'dataset_info.json'}", flush=True)
    if final_count < args.num_objects:
        print(
            "Warning: fewer objects were accepted than requested. Increase --max_candidates or relax filters.",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build a curated Objaverse chair dataset for single-image 3D reconstruction.",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--blender_dir", default=DEFAULT_BLENDER_DIR)
    parser.add_argument("--blender_version", default=DEFAULT_BLENDER_VERSION)
    parser.add_argument("--num_objects", type=int, default=200)
    parser.add_argument("--max_candidates", type=int, default=2000)
    parser.add_argument("--download_batch", type=int, default=24)
    parser.add_argument("--download_processes", type=int, default=8)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument(
        "--allowed_licenses",
        default="",
        help="Comma-separated license substrings to keep. Empty means keep all Objaverse LVIS chairs.",
    )
    parser.add_argument("--min_faces", type=int, default=80)
    parser.add_argument("--max_faces", type=int, default=700000)
    parser.add_argument("--camera_radius", type=float, default=3.0)
    parser.add_argument("--elevation_min", type=float, default=10.0)
    parser.add_argument("--elevation_max", type=float, default=30.0)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--no_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--keep_raw", action="store_true")
    parser.add_argument(
        "--publish_to_kaggle",
        action="store_true",
        help="After each configured batch, publish the current output folder as a Kaggle Dataset version.",
    )
    parser.add_argument(
        "--kaggle_dataset_id",
        default="",
        help="Target Kaggle Dataset id, for example username/objaverse-chairs.",
    )
    parser.add_argument(
        "--kaggle_json_path",
        default="",
        help="Optional path to kaggle.json. If omitted, the script searches /kaggle/working and /kaggle/input.",
    )
    parser.add_argument(
        "--kaggle_dataset_title",
        default="",
        help="Human-readable Kaggle Dataset title. Defaults to a title generated from the slug.",
    )
    parser.add_argument(
        "--kaggle_dataset_license",
        default="CC-BY-4.0",
        help="Kaggle dataset license name used in dataset-metadata.json.",
    )
    parser.add_argument(
        "--kaggle_upload_mode",
        default="zip",
        choices=("skip", "zip", "tar"),
        help="Kaggle CLI upload mode.",
    )
    parser.add_argument(
        "--publish_every_batches",
        type=int,
        default=1,
        help="Publish a Kaggle Dataset version every N processed download/render batches.",
    )
    parser.add_argument(
        "--kaggle_create_if_missing",
        action="store_true",
        help="If version upload fails, try creating the Kaggle Dataset from the current output folder.",
    )
    parser.add_argument(
        "--kaggle_public",
        action="store_true",
        help="Create the Kaggle Dataset as public when --kaggle_create_if_missing is used.",
    )
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_blender_download", action="store_true")
    parser.add_argument(
        "--skip_apt_blender",
        action="store_true",
        help="Do not try installing Blender with apt-get before falling back to direct download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_objects <= 0:
        raise ValueError("--num_objects must be positive")
    if args.views <= 0:
        raise ValueError("--views must be positive")
    if args.resolution <= 0:
        raise ValueError("--resolution must be positive")
    if args.download_batch <= 0:
        raise ValueError("--download_batch must be positive")
    if args.publish_every_batches <= 0:
        raise ValueError("--publish_every_batches must be positive")
    if args.publish_to_kaggle and not args.kaggle_dataset_id:
        raise ValueError("--kaggle_dataset_id is required with --publish_to_kaggle")
    build_dataset(args)


if __name__ == "__main__":
    main()
