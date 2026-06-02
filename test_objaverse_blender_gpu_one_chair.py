#!/usr/bin/env python3
"""
Download one chair-like Objaverse asset and render it with Blender Cycles GPU.

This is a small sanity-check script for building a higher-quality Blender-based
dataset pipeline. It intentionally processes one object only.

Example:
  python test_objaverse_blender_gpu_one_chair.py \
    --output_dir /data/objaverse_blender_test \
    --views 4 \
    --resolution 512 \
    --use_gpu

On Kaggle:
  !python /kaggle/working/repositoryi/test_objaverse_blender_gpu_one_chair.py \
    --output_dir /kaggle/working/objaverse_blender_test \
    --views 4 \
    --resolution 512 \
    --use_gpu
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_CATEGORIES = (
    "chair",
    "armchair",
    "stool",
    "folding_chair",
    "rocking_chair",
    "swivel_chair",
)


BLENDER_WORKER = r"""
import argparse
import json
import math
import os
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
    parser.add_argument("--asset_path", required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--output_dir", required=True)
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


def configure_cycles(resolution, use_gpu):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 96
    scene.cycles.use_denoising = True
    scene.cycles.device = "CPU"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "Medium High Contrast"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass

    enabled = "CPU"
    if use_gpu:
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for compute_type in ("OPTIX", "CUDA"):
                try:
                    prefs.compute_device_type = compute_type
                    prefs.get_devices()
                    any_gpu = False
                    for device in prefs.devices:
                        device.use = device.type != "CPU"
                        any_gpu = any_gpu or device.use
                    if any_gpu:
                        scene.cycles.device = "GPU"
                        enabled = compute_type
                        break
                except Exception:
                    continue
        except Exception as exc:
            print(f"[blender] GPU setup failed, using CPU: {exc}", flush=True)

    print(f"[blender] render_device={enabled}", flush=True)
    return enabled


def import_asset(asset_path):
    ext = Path(asset_path).suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=asset_path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=asset_path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=asset_path)
    else:
        raise RuntimeError(f"Unsupported asset extension for this test: {ext}")


def mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def normalize_object():
    meshes = mesh_objects()
    if not meshes:
        raise RuntimeError("No mesh objects were imported")

    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = "normalized_chair"

    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    min_corner = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    max_corner = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))
    for corner in obj.bound_box:
        world = obj.matrix_world @ mathutils.Vector(corner)
        min_corner.x = min(min_corner.x, world.x)
        min_corner.y = min(min_corner.y, world.y)
        min_corner.z = min(min_corner.z, world.z)
        max_corner.x = max(max_corner.x, world.x)
        max_corner.y = max(max_corner.y, world.y)
        max_corner.z = max(max_corner.z, world.z)

    center = (min_corner + max_corner) * 0.5
    size = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)
    if not size or size <= 0:
        raise RuntimeError("Invalid object bounding box")

    obj.location -= center
    scale = 1.6 / size
    obj.scale = (scale, scale, scale)
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

    # Put the object on the ground plane after normalization.
    min_z = min((obj.matrix_world @ mathutils.Vector(corner)).z for corner in obj.bound_box)
    obj.location.z -= min_z
    bpy.context.view_layer.update()

    # Ensure completely black assets are still visible in the test render.
    for slot in obj.material_slots:
        mat = slot.material
        if mat and mat.use_nodes:
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                try:
                    bsdf.inputs["Roughness"].default_value = 0.55
                except Exception:
                    pass

    return obj


def add_world_and_lights():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (0.78, 0.80, 0.84)

    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.5, 4.5))
    key = bpy.context.object
    key.name = "key_area"
    key.data.energy = 550
    key.data.size = 5.0

    bpy.ops.object.light_add(type="AREA", location=(-3.5, 2.5, 3.2))
    fill = bpy.context.object
    fill.name = "fill_area"
    fill.data.energy = 170
    fill.data.size = 5.0

    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, -0.01))
    plane = bpy.context.object
    plane.name = "matte_ground"
    mat = bpy.data.materials.new("ground_light_gray")
    mat.diffuse_color = (0.78, 0.78, 0.76, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.78, 0.78, 0.76, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
    plane.data.materials.append(mat)


def look_at(obj, target):
    loc = mathutils.Vector(obj.location)
    direction = mathutils.Vector(target) - loc
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(radius=3.0, elevation_deg=18.0):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    camera.data.lens = 55
    camera.data.sensor_width = 32
    camera.data.dof.use_dof = False
    return camera


def export_normalized(out_path):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects():
        if obj.name != "matte_ground":
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(filepath=str(out_path), export_format="GLB", use_selection=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    object_dir = output_dir / "objects" / args.uid
    render_dir = output_dir / "renders" / args.uid
    object_dir.mkdir(parents=True, exist_ok=True)
    render_dir.mkdir(parents=True, exist_ok=True)

    clean_scene()
    device = configure_cycles(args.resolution, args.use_gpu)
    import_asset(args.asset_path)
    obj = normalize_object()
    add_world_and_lights()
    camera = add_camera()

    normalized_path = object_dir / "normalized.glb"
    export_normalized(normalized_path)

    camera_rows = []
    for view_idx in range(args.views):
        az = (2.0 * math.pi * view_idx) / max(args.views, 1)
        elevation = math.radians(18.0)
        radius = 3.0
        camera.location = (
            radius * math.cos(elevation) * math.cos(az),
            radius * math.cos(elevation) * math.sin(az),
            radius * math.sin(elevation) + 0.75,
        )
        look_at(camera, (0.0, 0.0, 0.65))
        bpy.context.scene.render.filepath = str(render_dir / f"view_{view_idx:03d}.png")
        bpy.ops.render.render(write_still=True)
        camera_rows.append(
            {
                "uid": args.uid,
                "view": view_idx,
                "image_path": f"renders/{args.uid}/view_{view_idx:03d}.png",
                "azimuth_deg": round(math.degrees(az), 4),
                "elevation_deg": 18.0,
                "radius": radius,
            }
        )

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "test_result.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "uid": args.uid,
                "asset_path": args.asset_path,
                "normalized_glb": str(normalized_path),
                "render_dir": str(render_dir),
                "render_device": device,
                "views": camera_rows,
            },
            f,
            indent=2,
        )

    print(f"[blender] saved_normalized={normalized_path}", flush=True)
    print(f"[blender] saved_renders={render_dir}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
"""


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, check=check)


def pip_install(packages: list[str]) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages])


def ensure_blender(skip_apt: bool = False) -> str:
    blender = shutil.which("blender")
    if blender:
        return blender

    if skip_apt:
        raise FileNotFoundError("Blender executable was not found and --skip_apt_blender was used")

    apt_get = shutil.which("apt-get")
    if not apt_get:
        raise FileNotFoundError("Blender executable was not found and apt-get is unavailable")

    print("Blender executable was not found. Trying apt-get install blender...", flush=True)
    run([apt_get, "update"], check=False)
    run([apt_get, "install", "-y", "blender"])
    blender = shutil.which("blender")
    if not blender:
        raise FileNotFoundError("Blender install finished, but blender is still not on PATH")
    return blender


def select_one_chair_uid(categories: list[str], explicit_uid: str | None) -> str:
    if explicit_uid:
        return explicit_uid

    import objaverse

    print("Loading Objaverse LVIS annotations...", flush=True)
    lvis = objaverse.load_lvis_annotations()
    lower_to_key = {key.lower(): key for key in lvis.keys()}
    selected: list[str] = []
    selected_categories: list[str] = []
    for category in categories:
        key = lower_to_key.get(category.lower())
        if key:
            selected_categories.append(key)
            selected.extend(lvis[key])

    if not selected:
        for key, uids in lvis.items():
            low = key.lower()
            if "chair" in low or "stool" in low:
                selected_categories.append(key)
                selected.extend(uids)

    selected = sorted(set(selected))
    if not selected:
        raise RuntimeError("No chair-like Objaverse LVIS UIDs were found")

    print(f"Selected LVIS categories: {selected_categories}", flush=True)
    print(f"Candidate chair-like UIDs: {len(selected)}", flush=True)
    return selected[0]


def download_uid(uid: str, download_processes: int) -> Path:
    import objaverse

    print(f"Downloading Objaverse object: {uid}", flush=True)
    downloaded = objaverse.load_objects(uids=[uid], download_processes=download_processes)
    path = downloaded.get(uid)
    if not path:
        raise RuntimeError(f"Objaverse did not return a local path for UID {uid}")
    local_path = Path(path)
    if not local_path.exists():
        raise FileNotFoundError(f"Downloaded path does not exist: {local_path}")
    print(f"Downloaded asset path: {local_path}", flush=True)
    return local_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/kaggle/working/objaverse_blender_gpu_test")
    parser.add_argument("--cache_dir", default="/kaggle/working/objaverse_cache")
    parser.add_argument("--uid", default=None, help="Optional explicit Objaverse UID to download")
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--download_processes", type=int, default=4)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--skip_apt_blender", action="store_true")
    parser.add_argument("--clean_output", action="store_true")
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

    blender = ensure_blender(skip_apt=args.skip_apt_blender)
    print(f"Using Blender: {blender}", flush=True)

    categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    uid = select_one_chair_uid(categories, args.uid)
    asset_path = download_uid(uid, args.download_processes)

    scripts_dir = output_dir / "_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    worker_path = scripts_dir / "blender_one_chair_worker.py"
    worker_path.write_text(BLENDER_WORKER, encoding="utf-8")

    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(worker_path),
        "--",
        "--asset_path",
        str(asset_path),
        "--uid",
        uid,
        "--output_dir",
        str(output_dir),
        "--views",
        str(args.views),
        "--resolution",
        str(args.resolution),
    ]
    if args.use_gpu:
        cmd.append("--use_gpu")

    run(cmd)

    print("", flush=True)
    print("Done.", flush=True)
    print(f"UID: {uid}", flush=True)
    print(f"Dataset root: {output_dir}", flush=True)
    print(f"Normalized mesh: {output_dir / 'objects' / uid / 'normalized.glb'}", flush=True)
    print(f"Renders: {output_dir / 'renders' / uid}", flush=True)
    print(f"Metadata: {output_dir / 'metadata' / 'test_result.json'}", flush=True)


if __name__ == "__main__":
    main()
