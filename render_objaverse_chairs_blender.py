#!/usr/bin/env python3
"""
Render a small Objaverse chair subset with Blender Cycles GPU.

Designed for Kaggle/Vast notebooks after cloning this repository.

Example:
  python3 render_objaverse_chairs_blender.py \
    --output_dir /kaggle/working/objaverse_chair_render_test \
    --cache_dir /kaggle/working/objaverse_cache \
    --blender_path /kaggle/working/blender-4.1.1-linux-x64/blender \
    --num_objects 10 \
    --views 4 \
    --resolution 512 \
    --use_gpu \
    --clean_output

Outputs:
  objects/<uid>/normalized.glb
  renders/<uid>/view_000.png ...
  metadata/views.csv
  metadata/objects.csv
  metadata/failed_objects.csv
  preview_contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_CATEGORIES = (
    "chair",
    "armchair",
    "stool",
    "folding_chair",
    "rocking_chair",
    "swivel_chair",
)

DEFAULT_BLENDER_URL = "https://download.blender.org/release/Blender4.1/blender-4.1.1-linux-x64.tar.xz"


BLENDER_WORKER_CODE = r'''
import argparse
import csv
import json
import math
import sys
import traceback
from pathlib import Path

import bpy
import mathutils


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--use_gpu", action="store_true")
    return parser.parse_args(argv)


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.textures,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def setup_render(resolution, use_gpu):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass

    scene.world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.78, 0.80, 0.84, 1.0)
        bg.inputs["Strength"].default_value = 0.05

    selected_device = "CPU"
    scene.cycles.device = "CPU"

    if use_gpu:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for compute_type in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = compute_type
                prefs.get_devices()
                any_gpu = False
                for device in prefs.devices:
                    device.use = device.type != "CPU"
                    any_gpu = any_gpu or device.use
                    print("device:", device.name, device.type, "use:", device.use, flush=True)
                if any_gpu:
                    scene.cycles.device = "GPU"
                    selected_device = compute_type
                    break
            except Exception as exc:
                print(f"[blender] {compute_type} unavailable: {exc}", flush=True)

    print("[blender] render_device =", selected_device, flush=True)
    print("[blender] scene.cycles.device =", scene.cycles.device, flush=True)


def import_asset(asset_path):
    ext = Path(asset_path).suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=asset_path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=asset_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=asset_path)
    else:
        raise RuntimeError(f"Unsupported asset extension: {ext}")


def mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def normalize_object():
    meshes = mesh_objects()
    if not meshes:
        raise RuntimeError("No mesh objects imported")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    if len(meshes) > 1:
        bpy.ops.object.join()

    obj = bpy.context.object
    obj.name = "chair"

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.context.view_layer.update()

    min_corner = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    max_corner = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))

    for vertex in obj.data.vertices:
        world = obj.matrix_world @ vertex.co
        min_corner.x = min(min_corner.x, world.x)
        min_corner.y = min(min_corner.y, world.y)
        min_corner.z = min(min_corner.z, world.z)
        max_corner.x = max(max_corner.x, world.x)
        max_corner.y = max(max_corner.y, world.y)
        max_corner.z = max(max_corner.z, world.z)

    center = (min_corner + max_corner) * 0.5
    size_x = max_corner.x - min_corner.x
    size_y = max_corner.y - min_corner.y
    size_z = max_corner.z - min_corner.z
    size = max(size_x, size_y, size_z)
    if size <= 0:
        raise RuntimeError("Invalid object size")

    for vertex in obj.data.vertices:
        vertex.co -= center

    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)

    scale = 1.6 / size
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    min_z = min((obj.matrix_world @ v.co).z for v in obj.data.vertices)
    obj.location.z -= min_z
    obj.location.x = 0
    obj.location.y = 0

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    bpy.context.view_layer.update()

    print("[blender] normalized location:", tuple(round(v, 4) for v in obj.location), flush=True)
    print("[blender] normalized dimensions:", tuple(round(v, 4) for v in obj.dimensions), flush=True)
    return obj


def object_bounds(obj):
    bpy.context.view_layer.update()
    points = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    min_corner = mathutils.Vector((
        min(p.x for p in points),
        min(p.y for p in points),
        min(p.z for p in points),
    ))
    max_corner = mathutils.Vector((
        max(p.x for p in points),
        max(p.y for p in points),
        max(p.z for p in points),
    ))
    center = (min_corner + max_corner) * 0.5
    size = max_corner - min_corner
    radius = max(size.x, size.y, size.z) * 0.5
    return min_corner, max_corner, center, size, radius


def add_lights(obj):
    min_corner, max_corner, center, size, radius = object_bounds(obj)

    bpy.ops.object.light_add(
        type="AREA",
        location=(center.x, center.y - radius * 3.5, max_corner.z + radius * 3.0),
    )
    key = bpy.context.object
    key.name = "key_light"
    key.data.energy = 260
    key.data.size = radius * 4.0

    bpy.ops.object.light_add(
        type="AREA",
        location=(center.x - radius * 3.0, center.y + radius * 2.5, max_corner.z + radius * 2.5),
    )
    fill = bpy.context.object
    fill.name = "fill_light"
    fill.data.energy = 70
    fill.data.size = radius * 4.0

    bpy.ops.object.light_add(
        type="AREA",
        location=(center.x + radius * 2.5, center.y + radius * 3.0, max_corner.z + radius * 2.2),
    )
    rim = bpy.context.object
    rim.name = "rim_light"
    rim.data.energy = 70
    rim.data.size = radius * 3.5


def look_at(obj, target):
    loc = mathutils.Vector(obj.location)
    direction = mathutils.Vector(target) - loc
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera():
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 55
    camera.data.sensor_width = 32
    camera.data.dof.use_dof = False
    bpy.context.scene.camera = camera
    return camera


def build_camera_views(requested_views):
    if requested_views == 24:
        views = []
        for i in range(12):
            views.append({"azimuth_deg": i * 30.0, "elevation_deg": 10.0})
        for i in range(8):
            views.append({"azimuth_deg": 22.5 + i * 45.0, "elevation_deg": 25.0})
        for az in (45.0, 135.0, 225.0, 315.0):
            views.append({"azimuth_deg": az, "elevation_deg": 40.0})
        return views

    return [
        {
            "azimuth_deg": i * 360.0 / requested_views,
            "elevation_deg": 13.0,
        }
        for i in range(requested_views)
    ]


def export_normalized(out_dir, uid):
    object_dir = out_dir / "objects" / uid
    object_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects():
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    out_path = object_dir / "normalized.glb"
    bpy.ops.export_scene.gltf(filepath=str(out_path), export_format="GLB", use_selection=True)
    return out_path


def render_views(camera, out_dir, uid, requested_views):
    obj = bpy.data.objects["chair"]
    min_corner, max_corner, center, size, radius = object_bounds(obj)
    camera_views = build_camera_views(requested_views)

    render_dir = out_dir / "renders" / uid
    render_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    camera_radius = max(radius * 5.8, 4.2)
    target = mathutils.Vector((
        center.x,
        center.y,
        min_corner.z + size.z * 0.52,
    ))

    camera.data.lens = 55
    camera.data.sensor_width = 32

    for i, view in enumerate(camera_views):
        az_deg = view["azimuth_deg"]
        elev_deg = view["elevation_deg"]
        az = math.radians(az_deg)
        elev = math.radians(elev_deg)

        camera.location = (
            center.x + camera_radius * math.cos(elev) * math.cos(az),
            center.y + camera_radius * math.cos(elev) * math.sin(az),
            target.z + camera_radius * math.sin(elev),
        )
        look_at(camera, target)

        path = render_dir / f"view_{i:03d}.png"
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)

        rows.append({
            "uid": uid,
            "view": i,
            "image_path": f"renders/{uid}/view_{i:03d}.png",
            "azimuth_deg": az_deg,
            "elevation_deg": elev_deg,
            "radius": camera_radius,
            "target_x": target.x,
            "target_y": target.y,
            "target_z": target.z,
        })
        print("[blender] rendered", path, flush=True)

    return rows


def process_item(item, out_dir, requested_views):
    uid = item["uid"]
    asset_path = item["path"]
    print(f"[blender] Processing {uid}: {asset_path}", flush=True)

    clean_scene()
    import_asset(asset_path)
    obj = normalize_object()
    add_lights(obj)
    camera = add_camera()
    normalized_path = export_normalized(out_dir, uid)
    view_rows = render_views(camera, out_dir, uid, requested_views)

    return {
        "uid": uid,
        "asset_path": asset_path,
        "normalized_path": str(normalized_path),
        "views": len(view_rows),
        "view_rows": view_rows,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_render(args.resolution, args.use_gpu)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    objects = []
    failed = []
    all_views = []

    for item in manifest:
        try:
            result = process_item(item, out_dir, args.views)
            objects.append(result)
            all_views.extend(result["view_rows"])
        except Exception as exc:
            traceback.print_exc()
            failed.append({"uid": item.get("uid"), "asset_path": item.get("path"), "error": repr(exc)})

    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_dir / "views.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["uid", "view", "image_path", "azimuth_deg", "elevation_deg", "radius", "target_x", "target_y", "target_z"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_views)

    with open(metadata_dir / "objects.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["uid", "asset_path", "normalized_path", "views"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in objects:
            writer.writerow({key: row[key] for key in fieldnames})

    with open(metadata_dir / "failed_objects.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = ["uid", "asset_path", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed)

    print(f"[blender] DONE accepted={len(objects)} failed={len(failed)} views={len(all_views)}", flush=True)


if __name__ == "__main__":
    main()
'''


def run(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(list(cmd), check=check)


def pip_install(packages: Sequence[str]) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages])


def batched(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def ensure_blender(args: argparse.Namespace) -> str:
    if args.blender_path and Path(args.blender_path).exists():
        return args.blender_path

    default_kaggle = Path("/kaggle/working/blender-4.1.1-linux-x64/blender")
    if default_kaggle.exists():
        return str(default_kaggle)

    blender = shutil.which("blender")
    if blender and not args.download_blender:
        return blender

    if not args.download_blender:
        raise FileNotFoundError(
            "Blender was not found. Pass --blender_path or add --download_blender."
        )

    install_dir = Path(args.blender_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    archive_path = install_dir / "blender-4.1.1-linux-x64.tar.xz"
    extracted = install_dir / "blender-4.1.1-linux-x64" / "blender"

    if not extracted.exists():
        if not archive_path.exists():
            print(f"Downloading Blender 4.1.1 to {archive_path}", flush=True)
            urllib.request.urlretrieve(DEFAULT_BLENDER_URL, archive_path)
        print(f"Extracting Blender to {install_dir}", flush=True)
        with tarfile.open(archive_path, "r:xz") as tar:
            tar.extractall(install_dir)

    if not extracted.exists():
        raise FileNotFoundError(f"Blender download/extract failed: {extracted}")
    return str(extracted)


def select_chair_uids(categories: Sequence[str], max_candidates: int) -> list[str]:
    import objaverse

    print("Loading Objaverse LVIS annotations...", flush=True)
    lvis = objaverse.load_lvis_annotations()
    lower_to_key = {key.lower(): key for key in lvis.keys()}

    selected_uids: list[str] = []
    selected_categories: list[str] = []
    for category in categories:
        key = lower_to_key.get(category.lower())
        if key:
            selected_categories.append(key)
            selected_uids.extend(lvis[key])

    if not selected_uids:
        for key, uids in lvis.items():
            low = key.lower()
            if "chair" in low or "stool" in low:
                selected_categories.append(key)
                selected_uids.extend(uids)

    selected = sorted(set(selected_uids))
    if max_candidates > 0:
        selected = selected[:max_candidates]

    print(f"Selected LVIS categories: {selected_categories}", flush=True)
    print(f"Chair-like candidate UIDs: {len(selected)}", flush=True)
    return selected


def download_objects(uids: Sequence[str], download_processes: int) -> dict[str, str]:
    import objaverse

    print(f"Downloading {len(uids)} Objaverse objects...", flush=True)
    downloaded = objaverse.load_objects(uids=list(uids), download_processes=download_processes)
    return {uid: str(path) for uid, path in downloaded.items() if path and Path(path).exists()}


def composite_pngs(output_dir: Path, background: tuple[int, int, int]) -> None:
    from PIL import Image

    for path in (output_dir / "renders").glob("*/*.png"):
        img = Image.open(path).convert("RGBA")
        bg = Image.new("RGBA", img.size, (*background, 255))
        composed = Image.alpha_composite(bg, img).convert("RGB")
        composed.save(path)


def make_contact_sheet(output_dir: Path, max_objects: int = 64) -> Path | None:
    from PIL import Image, ImageDraw, ImageFont

    objects_csv = output_dir / "metadata" / "objects.csv"
    if not objects_csv.exists():
        return None

    rows = []
    with open(objects_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    rows = rows[:max_objects]
    if not rows:
        return None

    thumbs = []
    for row in rows:
        uid = row["uid"]
        image_path = output_dir / "renders" / uid / "view_000.png"
        if not image_path.exists():
            continue
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((220, 220), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (240, 270), (238, 238, 238))
        canvas.paste(img, ((240 - img.width) // 2, 8))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 238), uid[:18], fill=(30, 30, 30))
        thumbs.append(canvas)

    if not thumbs:
        return None

    cols = min(5, len(thumbs))
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows_count * 270), (245, 245, 245))
    for idx, thumb in enumerate(thumbs):
        x = (idx % cols) * 240
        y = (idx // cols) * 270
        sheet.paste(thumb, (x, y))

    out_path = output_dir / "preview_contact_sheet.jpg"
    sheet.save(out_path, quality=95)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/kaggle/working/objaverse_chair_render_test")
    parser.add_argument("--cache_dir", default="/kaggle/working/objaverse_cache")
    parser.add_argument("--blender_path", default="")
    parser.add_argument("--blender_dir", default="/kaggle/working")
    parser.add_argument("--download_blender", action="store_true")
    parser.add_argument("--num_objects", type=int, default=10)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--download_processes", type=int, default=4)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--clean_output", action="store_true")
    parser.add_argument("--background", default="219,222,224", help="RGB background after alpha compositing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_install:
        pip_install(["objaverse", "tqdm", "Pillow"])

    os.environ.setdefault("OBJAVERSE_HOME", str(cache_dir / "objaverse"))

    blender = ensure_blender(args)
    print(f"Using Blender: {blender}", flush=True)

    categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    candidates = select_chair_uids(categories, args.max_candidates)
    if not candidates:
        raise RuntimeError("No chair-like candidates were found")

    target_uids = candidates[: args.num_objects]
    downloaded = download_objects(target_uids, args.download_processes)
    manifest = [{"uid": uid, "path": downloaded[uid]} for uid in target_uids if uid in downloaded]

    if not manifest:
        raise RuntimeError("No requested objects were downloaded successfully")

    scripts_dir = output_dir / "_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    worker_path = scripts_dir / "blender_render_worker.py"
    manifest_path = scripts_dir / "manifest.json"
    worker_path.write_text(BLENDER_WORKER_CODE, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(worker_path),
        "--",
        "--manifest",
        str(manifest_path),
        "--out_dir",
        str(output_dir),
        "--views",
        str(args.views),
        "--resolution",
        str(args.resolution),
    ]
    if args.use_gpu:
        cmd.append("--use_gpu")

    run(cmd)

    background = tuple(int(part.strip()) for part in args.background.split(","))
    if len(background) != 3:
        raise ValueError("--background must be R,G,B")
    composite_pngs(output_dir, background)  # type: ignore[arg-type]
    sheet = make_contact_sheet(output_dir)

    print("", flush=True)
    print("Done.", flush=True)
    print(f"Dataset root: {output_dir}", flush=True)
    print(f"Views CSV: {output_dir / 'metadata' / 'views.csv'}", flush=True)
    print(f"Objects CSV: {output_dir / 'metadata' / 'objects.csv'}", flush=True)
    if sheet:
        print(f"Preview contact sheet: {sheet}", flush=True)


if __name__ == "__main__":
    main()
