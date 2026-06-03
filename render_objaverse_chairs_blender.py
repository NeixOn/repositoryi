#!/usr/bin/env python3
"""
Build a Blender-rendered Objaverse chair dataset.

Creates, per object:
  objects/<uid>/normalized.glb
  objects/<uid>/points.npz
  renders/<uid>/view_000.png
  masks/<uid>/view_000.png
  depths/<uid>/view_000.exr
  normals/<uid>/view_000.png
  cameras/<uid>/view_000.json

Example:
  python3 render_objaverse_chairs_blender.py \
    --output_dir /kaggle/working/objaverse_chair_dataset \
    --cache_dir /kaggle/working/objaverse_cache \
    --blender_path /kaggle/working/blender-4.1.1-linux-x64/blender \
    --num_objects 10 \
    --views 4 \
    --resolution 512 \
    --points 32768 \
    --use_gpu \
    --clean_output
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
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
import shutil
import sys
import time
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
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--modalities", default="rgb,mask,depth,normal,camera,mesh,points")
    return parser.parse_args(argv)


def parse_modalities(value):
    return {part.strip().lower() for part in value.split(",") if part.strip()}


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


def setup_render(resolution, samples, use_gpu, modalities):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = "depth" in modalities
    view_layer.use_pass_normal = "normal" in modalities

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
    size = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)
    if size <= 0:
        raise RuntimeError("Invalid object size")

    for vertex in obj.data.vertices:
        vertex.co -= center

    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (1.6 / size, 1.6 / size, 1.6 / size)
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
    min_corner = mathutils.Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_corner = mathutils.Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    center = (min_corner + max_corner) * 0.5
    size = max_corner - min_corner
    radius = max(size.x, size.y, size.z) * 0.5
    return min_corner, max_corner, center, size, radius


def add_lights(obj):
    min_corner, max_corner, center, size, radius = object_bounds(obj)

    bpy.ops.object.light_add(type="AREA", location=(center.x, center.y - radius * 3.5, max_corner.z + radius * 3.0))
    key = bpy.context.object
    key.name = "key_light"
    key.data.energy = 260
    key.data.size = radius * 4.0

    bpy.ops.object.light_add(type="AREA", location=(center.x - radius * 3.0, center.y + radius * 2.5, max_corner.z + radius * 2.5))
    fill = bpy.context.object
    fill.name = "fill_light"
    fill.data.energy = 70
    fill.data.size = radius * 4.0

    bpy.ops.object.light_add(type="AREA", location=(center.x + radius * 2.5, center.y + radius * 3.0, max_corner.z + radius * 2.2))
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

    return [{"azimuth_deg": i * 360.0 / requested_views, "elevation_deg": 13.0} for i in range(requested_views)]


def matrix_to_list(matrix):
    return [[float(matrix[row][col]) for col in range(4)] for row in range(4)]


def camera_intrinsics(camera, scene):
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    sensor_width = camera.data.sensor_width
    sensor_height = sensor_width * height / width
    fx = camera.data.lens / sensor_width * width
    fy = camera.data.lens / sensor_height * height
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(width * 0.5),
        "cy": float(height * 0.5),
        "width": int(width),
        "height": int(height),
        "lens_mm": float(camera.data.lens),
        "sensor_width_mm": float(sensor_width),
    }


def save_camera_json(path, uid, view_idx, camera, scene, az_deg, elev_deg, radius, target):
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix_world = camera.matrix_world.copy()
    world_to_camera = matrix_world.inverted()
    data = {
        "uid": uid,
        "view": view_idx,
        "azimuth_deg": float(az_deg),
        "elevation_deg": float(elev_deg),
        "radius": float(radius),
        "target": [float(target.x), float(target.y), float(target.z)],
        "camera_location": [float(v) for v in camera.location],
        "camera_matrix_world": matrix_to_list(matrix_world),
        "world_to_camera": matrix_to_list(world_to_camera),
        "intrinsics": camera_intrinsics(camera, scene),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


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


def setup_depth_nodes(depth_dir, view_idx):
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    depth_out = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_out.base_path = str(depth_dir)
    depth_out.file_slots[0].path = f"view_{view_idx:03d}_"
    depth_out.format.file_format = "OPEN_EXR"
    # Blender 4.x output-file EXR supports RGB/RGBA here, not BW. The linked
    # depth pass is still written as depth values inside the EXR.
    depth_out.format.color_mode = "RGB"
    tree.links.new(render_layers.outputs["Depth"], depth_out.inputs[0])


def find_depth_output(depth_dir, view_idx):
    candidates = sorted(depth_dir.glob(f"view_{view_idx:03d}_*.exr"))
    if not candidates:
        return None
    final_path = depth_dir / f"view_{view_idx:03d}.exr"
    if final_path.exists():
        final_path.unlink()
    candidates[-1].rename(final_path)
    for extra in candidates[:-1]:
        if extra.exists():
            extra.unlink()
    return final_path


def make_normal_material():
    mat = bpy.data.materials.new("normal_override")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new(type="ShaderNodeOutputMaterial")
    geom = nodes.new(type="ShaderNodeNewGeometry")
    vec_add = nodes.new(type="ShaderNodeVectorMath")
    vec_add.operation = "ADD"
    vec_add.inputs[1].default_value = (1.0, 1.0, 1.0)
    vec_mul = nodes.new(type="ShaderNodeVectorMath")
    vec_mul.operation = "MULTIPLY"
    vec_mul.inputs[1].default_value = (0.5, 0.5, 0.5)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0

    links.new(geom.outputs["Normal"], vec_add.inputs[0])
    links.new(vec_add.outputs["Vector"], vec_mul.inputs[0])
    links.new(vec_mul.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def set_render_png_rgba():
    scene = bpy.context.scene
    scene.use_nodes = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True


def render_rgb_mask_depth_normal(uid, view_idx, out_dir, camera, modalities):
    scene = bpy.context.scene
    view_layer = scene.view_layers[0]
    render_path = out_dir / "renders" / uid / f"view_{view_idx:03d}.png"
    depth_dir = out_dir / "depths" / uid
    normal_path = out_dir / "normals" / uid / f"view_{view_idx:03d}.png"

    render_path.parent.mkdir(parents=True, exist_ok=True)

    set_render_png_rgba()
    view_layer.material_override = None
    scene.render.filepath = str(render_path)
    bpy.ops.render.render(write_still=True)

    depth_path = None
    if "depth" in modalities:
        depth_dir.mkdir(parents=True, exist_ok=True)
        setup_depth_nodes(depth_dir, view_idx)
        view_layer.material_override = None
        scene.render.filepath = str(depth_dir / f"unused_{view_idx:03d}.png")
        bpy.ops.render.render(write_still=False)
        depth_path = find_depth_output(depth_dir, view_idx)
        scene.use_nodes = False

    if "normal" in modalities:
        normal_path.parent.mkdir(parents=True, exist_ok=True)
        set_render_png_rgba()
        normal_mat = make_normal_material()
        view_layer.material_override = normal_mat
        scene.render.filepath = str(normal_path)
        bpy.ops.render.render(write_still=True)
        view_layer.material_override = None
    else:
        normal_path = None

    return render_path, depth_path, normal_path


def render_views(camera, out_dir, uid, requested_views, modalities):
    obj = bpy.data.objects["chair"]
    min_corner, max_corner, center, size, radius = object_bounds(obj)
    camera_views = build_camera_views(requested_views)

    rows = []
    camera_radius = max(radius * 5.8, 4.2)
    target = mathutils.Vector((center.x, center.y, min_corner.z + size.z * 0.52))
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
        bpy.context.view_layer.update()

        camera_path = out_dir / "cameras" / uid / f"view_{i:03d}.json"
        if "camera" in modalities:
            save_camera_json(camera_path, uid, i, camera, bpy.context.scene, az_deg, elev_deg, camera_radius, target)

        rgb_path, depth_path, normal_path = render_rgb_mask_depth_normal(uid, i, out_dir, camera, modalities)
        rows.append({
            "uid": uid,
            "view": i,
            "image_path": f"renders/{uid}/view_{i:03d}.png",
            "mask_path": f"masks/{uid}/view_{i:03d}.png" if "mask" in modalities else "",
            "depth_path": f"depths/{uid}/view_{i:03d}.exr" if depth_path else "",
            "normal_path": f"normals/{uid}/view_{i:03d}.png" if normal_path else "",
            "camera_path": f"cameras/{uid}/view_{i:03d}.json" if "camera" in modalities else "",
            "azimuth_deg": az_deg,
            "elevation_deg": elev_deg,
            "radius": camera_radius,
            "target_x": target.x,
            "target_y": target.y,
            "target_z": target.z,
        })
        print("[blender] rendered", rgb_path, flush=True)

    return rows


def process_item(item, out_dir, requested_views, modalities):
    uid = item["uid"]
    asset_path = item["path"]
    print(f"[blender] Processing {uid}: {asset_path}", flush=True)

    clean_scene()
    import_asset(asset_path)
    obj = normalize_object()
    add_lights(obj)
    camera = add_camera()
    normalized_path = export_normalized(out_dir, uid)
    view_rows = render_views(camera, out_dir, uid, requested_views, modalities)

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
    modalities = parse_modalities(args.modalities)
    setup_render(args.resolution, args.samples, args.use_gpu, modalities)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    objects = []
    failed = []
    all_views = []
    chunk_start = time.time()
    total_items = len(manifest)
    for item_idx, item in enumerate(manifest, start=1):
        item_start = time.time()
        print(f"[blender] Progress object {item_idx}/{total_items}: {item.get('uid')}", flush=True)
        try:
            result = process_item(item, out_dir, args.views, modalities)
            objects.append(result)
            all_views.extend(result["view_rows"])
        except Exception as exc:
            traceback.print_exc()
            failed.append({"uid": item.get("uid"), "asset_path": item.get("path"), "error": repr(exc)})
        finally:
            item_sec = time.time() - item_start
            elapsed = time.time() - chunk_start
            avg_sec = elapsed / item_idx
            remaining = max(total_items - item_idx, 0)
            eta_sec = remaining * avg_sec
            print(
                "[blender] Object timing "
                f"{item_idx}/{total_items}: current={item_sec:.1f}s "
                f"avg={avg_sec:.1f}s eta_min={eta_sec / 60.0:.1f}",
                flush=True,
            )

    metadata_dir = out_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    view_fields = [
        "uid", "view", "image_path", "mask_path", "depth_path", "normal_path", "camera_path",
        "azimuth_deg", "elevation_deg", "radius", "target_x", "target_y", "target_z",
    ]
    views_csv = metadata_dir / "views.csv"
    views_exists = views_csv.exists()
    with open(views_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=view_fields)
        if not views_exists:
            writer.writeheader()
        writer.writerows(all_views)

    objects_csv = metadata_dir / "objects.csv"
    objects_exists = objects_csv.exists()
    with open(objects_csv, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["uid", "asset_path", "normalized_path", "views"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not objects_exists:
            writer.writeheader()
        for row in objects:
            writer.writerow({key: row[key] for key in fieldnames})

    failed_csv = metadata_dir / "failed_objects.csv"
    failed_exists = failed_csv.exists()
    with open(failed_csv, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["uid", "asset_path", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not failed_exists:
            writer.writeheader()
        writer.writerows(failed)

    print(f"[blender] DONE accepted={len(objects)} failed={len(failed)} views={len(all_views)}", flush=True)


if __name__ == "__main__":
    main()
'''


def run(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(list(cmd), check=check)


def run_blender(cmd: Sequence[str], *, verbose: bool = False) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    if verbose:
        subprocess.run(list(cmd), check=True)
        return

    keep_markers = (
        "[blender]",
        "Traceback",
        "Error:",
        "RuntimeError",
        "Exception",
        "FAILED",
        "DONE",
        "Blender quit",
    )
    skip_prefixes = (
        "Fra:",
        "Saved:",
        " Time:",
        "Color management:",
    )
    skip_contains = (
        " | Scene, ViewLayer | ",
        "Sample ",
        "Mem:",
    )

    process = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None

    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        if any(marker in line for marker in keep_markers):
            print(line, flush=True)
            continue
        if line.startswith(skip_prefixes):
            continue
        if any(marker in line for marker in skip_contains):
            continue
        if line.startswith(("device:", "Data are loaded", "glTF import finished")):
            continue

    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, list(cmd))


def pip_install(packages: Sequence[str]) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", *packages])


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
        raise FileNotFoundError("Blender was not found. Pass --blender_path or add --download_blender.")

    install_dir = Path(args.blender_dir)
    install_dir.mkdir(parents=True, exist_ok=True)
    archive_path = install_dir / "blender-4.1.1-linux-x64.tar.xz"
    extracted = install_dir / "blender-4.1.1-linux-x64" / "blender"
    if not extracted.exists():
        if not archive_path.exists():
            print(f"Downloading Blender 4.1.1 to {archive_path}", flush=True)
            wget = shutil.which("wget")
            curl = shutil.which("curl")
            if wget:
                run(
                    [
                        wget,
                        "--user-agent=Mozilla/5.0",
                        "-O",
                        str(archive_path),
                        DEFAULT_BLENDER_URL,
                    ]
                )
            elif curl:
                run(
                    [
                        curl,
                        "-L",
                        "-A",
                        "Mozilla/5.0",
                        "-o",
                        str(archive_path),
                        DEFAULT_BLENDER_URL,
                    ]
                )
            else:
                request = urllib.request.Request(
                    DEFAULT_BLENDER_URL,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(request) as response, open(archive_path, "wb") as f:
                    shutil.copyfileobj(response, f)
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


def read_existing_object_uids(output_dir: Path) -> set[str]:
    objects_csv = output_dir / "metadata" / "objects.csv"
    if not objects_csv.exists():
        return set()
    with open(objects_csv, "r", encoding="utf-8") as f:
        return {row["uid"] for row in csv.DictReader(f) if row.get("uid")}


def object_files_complete(output_dir: Path, uid: str, views: int, modalities: set[str]) -> bool:
    if "mesh" in modalities and not (output_dir / "objects" / uid / "normalized.glb").exists():
        return False
    if "points" in modalities and not (output_dir / "objects" / uid / "points.npz").exists():
        return False

    for view_idx in range(views):
        name = f"view_{view_idx:03d}"
        if "rgb" in modalities and not (output_dir / "renders" / uid / f"{name}.png").exists():
            return False
        if "mask" in modalities and not (output_dir / "masks" / uid / f"{name}.png").exists():
            return False
        if "depth" in modalities and not (output_dir / "depths" / uid / f"{name}.exr").exists():
            return False
        if "normal" in modalities and not (output_dir / "normals" / uid / f"{name}.png").exists():
            return False
        if "camera" in modalities and not (output_dir / "cameras" / uid / f"{name}.json").exists():
            return False
    return True


def filter_resume_uids(output_dir: Path, uids: Sequence[str], views: int, modalities: set[str]) -> list[str]:
    accepted = read_existing_object_uids(output_dir)
    remaining = []
    skipped = 0
    for uid in uids:
        if uid in accepted or object_files_complete(output_dir, uid, views, modalities):
            skipped += 1
            continue
        remaining.append(uid)
    if skipped:
        print(f"Resume: skipped {skipped} already completed objects.", flush=True)
    return remaining


def composite_and_make_masks(output_dir: Path, background: tuple[int, int, int], modalities: set[str]) -> None:
    from PIL import Image

    for path in (output_dir / "renders").glob("*/*.png"):
        uid = path.parent.name

        img = Image.open(path).convert("RGBA")
        alpha = img.getchannel("A")
        if "mask" in modalities:
            mask_dir = output_dir / "masks" / uid
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask_path = mask_dir / path.name
            mask = alpha.point(lambda value: 255 if value > 8 else 0)
            mask.save(mask_path)

        bg = Image.new("RGBA", img.size, (*background, 255))
        composed = Image.alpha_composite(bg, img).convert("RGB")
        composed.save(path)

    for path in (output_dir / "normals").glob("*/*.png"):
        img = Image.open(path).convert("RGBA")
        bg = Image.new("RGBA", img.size, (128, 128, 255, 255))
        composed = Image.alpha_composite(bg, img).convert("RGB")
        composed.save(path)


def create_points_npz(output_dir: Path, points_per_object: int, seed: int) -> None:
    import numpy as np
    import trimesh

    rng = np.random.default_rng(seed)
    objects_csv = output_dir / "metadata" / "objects.csv"
    if not objects_csv.exists():
        return

    with open(objects_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        uid = row["uid"]
        mesh_path = output_dir / "objects" / uid / "normalized.glb"
        out_path = output_dir / "objects" / uid / "points.npz"
        if not mesh_path.exists():
            continue
        scene_or_mesh = trimesh.load(mesh_path, force="scene")
        if isinstance(scene_or_mesh, trimesh.Scene):
            meshes = [geom for geom in scene_or_mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if not meshes:
                continue
            mesh = trimesh.util.concatenate(meshes)
        else:
            mesh = scene_or_mesh
        if len(mesh.faces) == 0:
            continue
        points, face_idx = trimesh.sample.sample_surface(mesh, points_per_object)
        normals = mesh.face_normals[face_idx]
        order = rng.permutation(len(points))
        points = points[order].astype("float32")
        normals = normals[order].astype("float32")
        np.savez_compressed(out_path, points=points, normals=normals)


def make_splits(output_dir: Path, train_ratio: float, val_ratio: float, seed: int) -> None:
    objects_csv = output_dir / "metadata" / "objects.csv"
    if not objects_csv.exists():
        return
    with open(objects_csv, "r", encoding="utf-8") as f:
        uids = [row["uid"] for row in csv.DictReader(f)]
    rng = random.Random(seed)
    rng.shuffle(uids)
    n = len(uids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": uids[:n_train],
        "val": uids[n_train : n_train + n_val],
        "test": uids[n_train + n_val :],
    }
    with open(output_dir / "metadata" / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def write_dataset_info(output_dir: Path, args: argparse.Namespace) -> None:
    objects_csv = output_dir / "metadata" / "objects.csv"
    views_csv = output_dir / "metadata" / "views.csv"
    objects = 0
    views = 0
    if objects_csv.exists():
        with open(objects_csv, "r", encoding="utf-8") as f:
            objects = sum(1 for _ in csv.DictReader(f))
    if views_csv.exists():
        with open(views_csv, "r", encoding="utf-8") as f:
            views = sum(1 for _ in csv.DictReader(f))
    info = {
        "name": "objaverse_chairs_blender",
        "objects": objects,
        "views": views,
        "resolution": args.resolution,
        "requested_views": args.views,
        "points_per_object": args.points,
        "modalities": [part.strip() for part in args.modalities.split(",") if part.strip()],
    }
    with open(output_dir / "metadata" / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def make_contact_sheet(output_dir: Path, max_objects: int = 64) -> Path | None:
    from PIL import Image, ImageDraw

    objects_csv = output_dir / "metadata" / "objects.csv"
    if not objects_csv.exists():
        return None
    with open(objects_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:max_objects]
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
        ImageDraw.Draw(canvas).text((8, 238), uid[:18], fill=(30, 30, 30))
        thumbs.append(canvas)
    if not thumbs:
        return None
    cols = min(5, len(thumbs))
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows_count * 270), (245, 245, 245))
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 240, (idx // cols) * 270))
    out_path = output_dir / "preview_contact_sheet.jpg"
    sheet.save(out_path, quality=95)
    return out_path


def iter_chunks(items: Sequence[dict[str, str]], chunk_size: int) -> Iterable[list[dict[str, str]]]:
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def write_kaggle_dataset_metadata(output_dir: Path, args: argparse.Namespace) -> None:
    if not args.kaggle_dataset_id:
        raise ValueError("--kaggle_dataset_id is required when --publish_to_kaggle is used")
    title = args.kaggle_dataset_title or args.kaggle_dataset_id.split("/", 1)[-1].replace("-", " ").title()
    metadata = {
        "title": title,
        "id": args.kaggle_dataset_id,
        "licenses": [{"name": args.kaggle_dataset_license}],
    }
    with open(output_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def publish_kaggle_dataset(output_dir: Path, args: argparse.Namespace, message: str) -> None:
    write_kaggle_dataset_metadata(output_dir, args)
    kaggle = shutil.which("kaggle")
    if not kaggle:
        raise FileNotFoundError("kaggle CLI was not found. Install it with: python3 -m pip install kaggle")

    version_cmd = [
        kaggle,
        "datasets",
        "version",
        "-p",
        str(output_dir),
        "-m",
        message,
        "-r",
        args.kaggle_upload_mode,
    ]
    create_cmd = [
        kaggle,
        "datasets",
        "create",
        "-p",
        str(output_dir),
        "-r",
        args.kaggle_upload_mode,
    ]
    if args.kaggle_public:
        create_cmd.append("--public")

    print(f"Publishing Kaggle Dataset version: {args.kaggle_dataset_id}", flush=True)
    result = run(version_cmd, check=False)
    if result.returncode == 0:
        print("Kaggle Dataset version upload started.", flush=True)
        return

    if not args.kaggle_create_if_missing:
        raise RuntimeError("Kaggle dataset version upload failed. Add --kaggle_create_if_missing for first publish.")

    print("Dataset version upload failed. Trying to create the dataset first...", flush=True)
    run(create_cmd)
    print("Kaggle Dataset create upload started.", flush=True)


def finalize_dataset_outputs(output_dir: Path, args: argparse.Namespace, background: tuple[int, int, int], modalities: set[str]) -> Path | None:
    composite_and_make_masks(output_dir, background, modalities)  # type: ignore[arg-type]
    if "points" in modalities:
        create_points_npz(output_dir, args.points, args.seed)
    make_splits(output_dir, args.train_ratio, args.val_ratio, args.seed)
    write_dataset_info(output_dir, args)
    return make_contact_sheet(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/kaggle/working/objaverse_chair_dataset")
    parser.add_argument("--cache_dir", default="/kaggle/working/objaverse_cache")
    parser.add_argument("--blender_path", default="")
    parser.add_argument("--blender_dir", default="/kaggle/working")
    parser.add_argument("--download_blender", action="store_true")
    parser.add_argument("--num_objects", type=int, default=10)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument(
        "--modalities",
        default="rgb,mask,depth,normal,camera,mesh,points",
        help=(
            "Comma-separated outputs. Use rgb,mask,camera,mesh,points for a much faster "
            "dataset without depth/normal auxiliary renders."
        ),
    )
    parser.add_argument("--categories", default=",".join(DEFAULT_CATEGORIES))
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--download_processes", type=int, default=4)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--verbose_blender", action="store_true", help="Show full Blender render logs.")
    parser.add_argument("--skip_install", action="store_true")
    parser.add_argument("--clean_output", action="store_true")
    parser.add_argument("--background", default="219,222,224", help="RGB background after alpha compositing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--publish_to_kaggle", action="store_true")
    parser.add_argument("--kaggle_dataset_id", default="")
    parser.add_argument("--kaggle_dataset_title", default="")
    parser.add_argument("--kaggle_dataset_license", default="CC0-1.0")
    parser.add_argument("--kaggle_upload_mode", choices=("zip", "tar"), default="zip")
    parser.add_argument("--kaggle_create_if_missing", action="store_true")
    parser.add_argument("--kaggle_public", action="store_true")
    parser.add_argument(
        "--publish_every_objects",
        type=int,
        default=0,
        help="Autosave to Kaggle after every N attempted objects. 25 or 50 is reasonable for overnight runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    modalities = {part.strip().lower() for part in args.modalities.split(",") if part.strip()}
    if "rgb" not in modalities:
        modalities.add("rgb")
    if "mesh" not in modalities:
        modalities.add("mesh")

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_install:
        packages = ["objaverse", "tqdm", "Pillow", "numpy", "trimesh"]
        if args.publish_to_kaggle:
            packages.append("kaggle")
        pip_install(packages)

    os.environ.setdefault("OBJAVERSE_HOME", str(cache_dir / "objaverse"))

    blender = ensure_blender(args)
    print(f"Using Blender: {blender}", flush=True)

    categories = [part.strip() for part in args.categories.split(",") if part.strip()]
    candidates = select_chair_uids(categories, args.max_candidates)
    if not candidates:
        raise RuntimeError("No chair-like candidates were found")

    target_uids = candidates[: args.num_objects]
    target_uids = filter_resume_uids(output_dir, target_uids, args.views, modalities)
    if not target_uids:
        print("Nothing to render: requested objects are already present.", flush=True)
        finalize_dataset_outputs(output_dir, args, background=tuple(int(part.strip()) for part in args.background.split(",")), modalities=modalities)  # type: ignore[arg-type]
        return

    downloaded = download_objects(target_uids, args.download_processes)
    manifest = [{"uid": uid, "path": downloaded[uid]} for uid in target_uids if uid in downloaded]
    if not manifest:
        raise RuntimeError("No requested objects were downloaded successfully")

    background = tuple(int(part.strip()) for part in args.background.split(","))
    if len(background) != 3:
        raise ValueError("--background must be R,G,B")

    scripts_dir = output_dir / "_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    worker_path = scripts_dir / "blender_render_worker.py"
    worker_path.write_text(BLENDER_WORKER_CODE, encoding="utf-8")

    chunk_size = args.publish_every_objects if args.publish_to_kaggle and args.publish_every_objects > 0 else len(manifest)
    chunk_size = max(1, chunk_size)
    chunks = list(iter_chunks(manifest, chunk_size))

    for chunk_idx, chunk in enumerate(chunks, start=1):
        manifest_path = scripts_dir / f"manifest_chunk_{chunk_idx:04d}.json"
        manifest_path.write_text(json.dumps(chunk, indent=2), encoding="utf-8")

        print(
            f"Rendering chunk {chunk_idx}/{len(chunks)}: {len(chunk)} objects",
            flush=True,
        )
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
            "--samples",
            str(args.samples),
            "--modalities",
            ",".join(sorted(modalities)),
        ]
        if args.use_gpu:
            cmd.append("--use_gpu")
        run_blender(cmd, verbose=args.verbose_blender)

        sheet = finalize_dataset_outputs(output_dir, args, background, modalities)  # type: ignore[arg-type]

        if args.publish_to_kaggle:
            objects_csv = output_dir / "metadata" / "objects.csv"
            accepted = 0
            if objects_csv.exists():
                with open(objects_csv, "r", encoding="utf-8") as f:
                    accepted = sum(1 for _ in csv.DictReader(f))
            publish_kaggle_dataset(
                output_dir,
                args,
                f"Autosave after chunk {chunk_idx}/{len(chunks)}: {accepted} objects",
            )

    sheet = make_contact_sheet(output_dir)

    print("", flush=True)
    print("Done.", flush=True)
    print(f"Dataset root: {output_dir}", flush=True)
    print(f"Views CSV: {output_dir / 'metadata' / 'views.csv'}", flush=True)
    print(f"Objects CSV: {output_dir / 'metadata' / 'objects.csv'}", flush=True)
    print(f"Splits: {output_dir / 'metadata' / 'splits.json'}", flush=True)
    if sheet:
        print(f"Preview contact sheet: {sheet}", flush=True)


if __name__ == "__main__":
    main()
