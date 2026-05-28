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

for _site_path in (
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
    "/usr/lib/python3.10/dist-packages",
    "/usr/local/lib/python3.11/dist-packages",
    "/usr/lib/python3.11/dist-packages",
):
    if os.path.isdir(_site_path) and _site_path not in sys.path:
        sys.path.append(_site_path)

import bpy
import mathutils
import numpy as np


IMAGE_FORMAT = "PNG"


def env_int(name):
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else None


def env_float(name):
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else None


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    elif "--manifest" in argv:
        argv = argv[argv.index("--manifest"):]
    else:
        # Some older Blender builds do not preserve the separator in sys.argv.
        # Keep only our worker flags if Blender passed them through differently.
        known = {
            "--manifest",
            "--output-dir",
            "--views",
            "--resolution",
            "--points",
            "--seed",
            "--min-faces",
            "--max-faces",
            "--camera-radius",
            "--elevation-min",
            "--elevation-max",
            "--use-gpu",
        }
        first = next((i for i, item in enumerate(argv) if item in known), None)
        argv = argv[first:] if first is not None else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.environ.get("OBJAVERSE_WORKER_MANIFEST"))
    parser.add_argument("--output-dir", default=os.environ.get("OBJAVERSE_WORKER_OUTPUT_DIR"))
    parser.add_argument("--views", type=int, default=env_int("OBJAVERSE_WORKER_VIEWS"))
    parser.add_argument("--resolution", type=int, default=env_int("OBJAVERSE_WORKER_RESOLUTION"))
    parser.add_argument("--points", type=int, default=env_int("OBJAVERSE_WORKER_POINTS"))
    parser.add_argument("--seed", type=int, default=env_int("OBJAVERSE_WORKER_SEED"))
    parser.add_argument("--min-faces", type=int, default=env_int("OBJAVERSE_WORKER_MIN_FACES"))
    parser.add_argument("--max-faces", type=int, default=env_int("OBJAVERSE_WORKER_MAX_FACES"))
    parser.add_argument("--camera-radius", type=float, default=env_float("OBJAVERSE_WORKER_CAMERA_RADIUS"))
    parser.add_argument("--elevation-min", type=float, default=env_float("OBJAVERSE_WORKER_ELEVATION_MIN"))
    parser.add_argument("--elevation-max", type=float, default=env_float("OBJAVERSE_WORKER_ELEVATION_MAX"))
    parser.add_argument("--use-gpu", action="store_true", default=os.environ.get("OBJAVERSE_WORKER_USE_GPU") == "1")
    args = parser.parse_args(argv)
    missing = [
        name
        for name in (
            "manifest",
            "output_dir",
            "views",
            "resolution",
            "points",
            "seed",
            "min_faces",
            "max_faces",
            "camera_radius",
            "elevation_min",
            "elevation_max",
        )
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("Missing worker arguments: " + ", ".join(missing))
    return args


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
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.cycles.device = "CPU"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.look = "None"
    except Exception:
        pass
    try:
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass
    scene.render.image_settings.file_format = IMAGE_FORMAT
    scene.render.image_settings.color_mode = "RGBA"
    scene.world = bpy.data.worlds.new("World") if scene.world is None else scene.world
    scene.world.color = (0.78, 0.80, 0.82)

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


def force_cpu_rendering():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 48
    scene.render.film_transparent = False


def apply_clay_material():
    material = bpy.data.materials.new("dataset_visible_warm_gray")
    material.diffuse_color = (0.82, 0.80, 0.74, 1.0)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (0.82, 0.80, 0.74, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    for obj in mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(material)


def import_object(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".glb" or suffix == ".gltf":
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
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
    bpy.ops.object.light_add(type="AREA", location=(0.0, -4.0, 5.0))
    key = bpy.context.object
    key.name = "Key_Area_Light"
    key.data.energy = 700
    key.data.size = 4.5

    bpy.ops.object.light_add(type="AREA", location=(-3.5, 3.0, 3.5))
    fill = bpy.context.object
    fill.name = "Fill_Light"
    fill.data.energy = 180
    fill.data.size = 5.5

    bpy.ops.object.light_add(type="AREA", location=(3.0, 4.0, 5.0))
    rim = bpy.context.object
    rim.name = "Rim_Light"
    rim.data.energy = 130
    rim.data.size = 4.0


def make_camera():
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "Camera"
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2.35
    camera.data.lens = 70
    camera.data.sensor_width = 32
    camera.data.clip_end = 100
    bpy.context.scene.camera = camera
    return camera


def look_at(camera, target):
    direction = mathutils.Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def matrix_to_list(matrix):
    return [[float(v) for v in row] for row in matrix]


def camera_intrinsics(camera, resolution):
    if camera.data.type == "ORTHO":
        scale = float(camera.data.ortho_scale)
        focal_like = resolution / scale
        return [[focal_like, 0.0, resolution / 2.0], [0.0, focal_like, resolution / 2.0], [0.0, 0.0, 1.0]]
    lens = camera.data.lens
    sensor_width = camera.data.sensor_width
    focal_px = lens * resolution / sensor_width
    cx = resolution / 2.0
    cy = resolution / 2.0
    return [[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]]


def rendered_image_stats(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = list(image.pixels)
        if not pixels:
            return {"mean_rgb": 0.0, "max_rgb": 0.0, "mean_alpha": 0.0}
        rgb_values = []
        alpha_values = []
        for i in range(0, len(pixels), 4):
            rgb_values.extend((pixels[i], pixels[i + 1], pixels[i + 2]))
            alpha_values.append(pixels[i + 3])
        return {
            "mean_rgb": float(sum(rgb_values) / max(len(rgb_values), 1)),
            "max_rgb": float(max(rgb_values) if rgb_values else 0.0),
            "mean_alpha": float(sum(alpha_values) / max(len(alpha_values), 1)),
        }
    finally:
        bpy.data.images.remove(image)


def render_still_checked(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    stats = rendered_image_stats(path)
    if stats["max_rgb"] < 0.05:
        print(f"[blender] Render was nearly black, retrying on CPU: {stats}", flush=True)
        force_cpu_rendering()
        bpy.ops.render.render(write_still=True)
        stats = rendered_image_stats(path)
    if stats["max_rgb"] < 0.05:
        raise RuntimeError(f"Rendered image is still nearly black after CPU retry: {stats}")
    return stats


def save_rgba_png(path, rgba):
    image = bpy.data.images.new(name=Path(path).stem, width=rgba.shape[1], height=rgba.shape[0], alpha=True)
    try:
        image.pixels = rgba[::-1, :, :].reshape(-1).astype(np.float32).tolist()
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)


def software_point_render(path, camera, points, normals, resolution):
    scale = float(camera.data.ortho_scale)
    matrix_inv = camera.matrix_world.inverted()
    rot_inv = matrix_inv.to_3x3()

    cam_points = np.empty_like(points, dtype=np.float32)
    cam_normals = np.empty_like(normals, dtype=np.float32)
    for i, point in enumerate(points):
        v = matrix_inv @ mathutils.Vector((float(point[0]), float(point[1]), float(point[2])))
        cam_points[i] = (v.x, v.y, v.z)
    for i, normal in enumerate(normals):
        n = rot_inv @ mathutils.Vector((float(normal[0]), float(normal[1]), float(normal[2])))
        cam_normals[i] = (n.x, n.y, n.z)

    x = ((cam_points[:, 0] / scale) + 0.5) * resolution
    y = (0.5 - (cam_points[:, 1] / scale)) * resolution
    depth = -cam_points[:, 2]
    valid = (depth > 0) & (x >= -3) & (x < resolution + 3) & (y >= -3) & (y < resolution + 3)
    if not np.any(valid):
        raise RuntimeError("Software renderer found no projected points in frame")

    x = x[valid].astype(np.int32)
    y = y[valid].astype(np.int32)
    depth = depth[valid]
    cam_normals = cam_normals[valid]

    order = np.argsort(depth)[::-1]
    x = x[order]
    y = y[order]
    depth = depth[order]
    cam_normals = cam_normals[order]

    rgba = np.ones((resolution, resolution, 4), dtype=np.float32)
    rgba[:, :, 0] = 0.78
    rgba[:, :, 1] = 0.80
    rgba[:, :, 2] = 0.82
    zbuf = np.full((resolution, resolution), np.inf, dtype=np.float32)

    base = np.array([0.82, 0.78, 0.68], dtype=np.float32)
    light = np.array([0.2, -0.35, 0.92], dtype=np.float32)
    light = light / np.linalg.norm(light)
    radius_px = max(1, resolution // 160)

    for px, py, z, normal in zip(x, y, depth, cam_normals):
        if z <= 0:
            continue
        n_norm = np.linalg.norm(normal)
        if n_norm > 1e-6:
            normal = normal / n_norm
        shade = 0.58 + 0.42 * max(0.0, float(np.dot(normal, light)))
        color = np.clip(base * shade, 0.0, 1.0)
        x0 = max(0, px - radius_px)
        x1 = min(resolution - 1, px + radius_px)
        y0 = max(0, py - radius_px)
        y1 = min(resolution - 1, py + radius_px)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if z < zbuf[yy, xx]:
                    zbuf[yy, xx] = z
                    rgba[yy, xx, 0:3] = color

    if float(np.max(rgba[:, :, 0:3])) < 0.05:
        raise RuntimeError("Software renderer produced a nearly black image")
    save_rgba_png(path, rgba)
    rgb = rgba[:, :, 0:3]
    return {
        "mean_rgb": float(np.mean(rgb)),
        "max_rgb": float(np.max(rgb)),
        "mean_alpha": float(np.mean(rgba[:, :, 3])),
    }


def export_normalized_glb(path):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objects():
        obj.select_set(True)
    kwargs = {
        "filepath": str(path),
        "export_format": "GLB",
        "use_selection": True,
        "export_apply": True,
    }
    try:
        bpy.ops.export_scene.gltf(**kwargs, export_yup=True)
    except TypeError:
        try:
            bpy.ops.export_scene.gltf(**kwargs)
        except TypeError:
            kwargs.pop("export_apply", None)
            bpy.ops.export_scene.gltf(**kwargs)


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


def render_views(uid, out_dir, views, resolution, radius, elevation_min, elevation_max, seed, bbox, points, normals):
    render_dir = out_dir / "renders" / uid
    render_dir.mkdir(parents=True, exist_ok=True)
    camera = make_camera()
    add_lights()
    target = mathutils.Vector(bbox["bbox_center"])
    dims = bbox["bbox_dims"]
    max_dim = max(float(dims[0]), float(dims[1]), float(dims[2]))
    camera.data.ortho_scale = max(2.20, max_dim * 1.35)

    rng = random.Random(seed)
    # One clean orbit plus slight deterministic elevation variation.
    start_azimuth = rng.uniform(0, 360.0)
    view_rows = []
    for view_idx in range(views):
        azimuth = start_azimuth + view_idx * (360.0 / views)
        elevation = rng.uniform(elevation_min, elevation_max)
        az = math.radians(azimuth)
        el = math.radians(elevation)
        x = target.x + radius * math.cos(el) * math.cos(az)
        y = target.y + radius * math.cos(el) * math.sin(az)
        z = target.z + radius * math.sin(el)
        camera.location = (x, y, z)
        look_at(camera, target)

        png_path = render_dir / f"view_{view_idx:03d}.png"
        render_stats = software_point_render(png_path, camera, points, normals, resolution)

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
            "camera_type": camera.data.type,
            "ortho_scale": float(camera.data.ortho_scale),
            "render_mean_rgb": render_stats["mean_rgb"],
            "render_max_rgb": render_stats["max_rgb"],
            "render_mean_alpha": render_stats["mean_alpha"],
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
    apply_clay_material()
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
        bbox=bbox,
        points=points,
        normals=normals,
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
    pip_install(["objaverse", "tqdm", "kaggle", "trimesh", "Pillow"])


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
        ["apt-get", "install", "-y", "-qq", "blender", "python3-numpy"],
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
        if not skip_apt and str(blender) == "/usr/bin/blender":
            try_install_blender_with_apt()
            blender = find_blender(blender_dir) or blender
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


def write_rows_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def python_load_mesh(path: str):
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    meshes = []
    if isinstance(loaded, trimesh.Scene):
        dumped = loaded.dump(concatenate=False)
        if isinstance(dumped, trimesh.Trimesh):
            meshes.append(dumped.copy())
        else:
            for geom in dumped:
                if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) and len(geom.faces):
                    meshes.append(geom.copy())
    elif isinstance(loaded, trimesh.Trimesh):
        meshes.append(loaded.copy())
    if not meshes:
        raise RuntimeError("No mesh geometry loaded by trimesh")
    mesh = trimesh.util.concatenate(meshes)
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise RuntimeError("Loaded mesh is empty")
    return mesh


def python_normalize_mesh(mesh):
    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    dims = hi - lo
    max_dim = float(dims.max())
    if not np.isfinite(max_dim) or max_dim <= 1e-9:
        raise RuntimeError(f"Degenerate mesh bounds: {dims.tolist()}")
    center = (lo + hi) * 0.5
    vertices = (vertices - center) * (1.8 / max_dim)
    vertices[:, 2] -= vertices[:, 2].min()
    mesh.vertices = vertices
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    dims = hi - lo
    center = (lo + hi) * 0.5
    return {
        "bbox_min": [float(v) for v in lo],
        "bbox_max": [float(v) for v in hi],
        "bbox_dims": [float(v) for v in dims],
        "bbox_center": [float(v) for v in center],
    }


def python_mesh_visual_colors(mesh, color_mode: str):
    import numpy as np

    fallback = np.array([0.84, 0.78, 0.66], dtype=np.float32)
    if color_mode == "clay":
        return None, fallback
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return None, fallback

    kind = getattr(visual, "kind", None)
    if kind == "vertex":
        colors = np.asarray(visual.vertex_colors, dtype=np.float32)
        if len(colors) == len(mesh.vertices):
            colors = colors[:, :3] / 255.0
            if float(colors.max() - colors.min()) > 0.02:
                return ("vertex", colors), fallback
    if kind == "face":
        colors = np.asarray(visual.face_colors, dtype=np.float32)
        if len(colors) == len(mesh.faces):
            colors = colors[:, :3] / 255.0
            if float(colors.max() - colors.min()) > 0.02:
                return ("face", colors), fallback

    material = getattr(visual, "material", None)
    for attr in ("baseColorFactor", "diffuse", "main_color"):
        value = getattr(material, attr, None) if material is not None else None
        if value is not None:
            color = np.asarray(value, dtype=np.float32).reshape(-1)
            if len(color) >= 3:
                color = color[:3]
                if color.max() > 1.0:
                    color = color / 255.0
                return None, np.clip(color, 0.05, 1.0).astype(np.float32)
    return None, fallback


def python_sample_surface(mesh, count: int, color_mode: str):
    import numpy as np
    import trimesh

    points, face_index = trimesh.sample.sample_surface(mesh, count)
    normals = np.asarray(mesh.face_normals[face_index], dtype=np.float32)
    color_source, fallback_color = python_mesh_visual_colors(mesh, color_mode)
    if color_source is None:
        colors = np.tile(fallback_color.reshape(1, 3), (len(points), 1)).astype(np.float32)
    else:
        mode, color_data = color_source
        if mode == "face":
            colors = color_data[face_index].astype(np.float32)
        else:
            faces = np.asarray(mesh.faces[face_index], dtype=np.int64)
            colors = color_data[faces].mean(axis=1).astype(np.float32)
    if color_mode == "normal":
        colors = np.clip((normals + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    return np.asarray(points, dtype=np.float32), normals, colors


def python_camera_matrix(location, target):
    import numpy as np

    location = np.asarray(location, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - location
    forward /= max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1e-8:
        right = np.array([1.0, 0.0, 0.0])
    right /= max(np.linalg.norm(right), 1e-12)
    up = np.cross(right, forward)
    up /= max(np.linalg.norm(up), 1e-12)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = location
    return matrix


def python_render_points(path: Path, points, normals, colors, camera_matrix, ortho_scale: float, resolution: int):
    import numpy as np
    from PIL import Image

    inv = np.linalg.inv(camera_matrix)
    homog = np.concatenate([points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)], axis=1)
    cam = (inv @ homog.T).T[:, :3]
    rot = inv[:3, :3]
    cam_normals = (rot @ normals.astype(np.float64).T).T

    x = ((cam[:, 0] / ortho_scale) + 0.5) * resolution
    y = (0.5 - (cam[:, 1] / ortho_scale)) * resolution
    depth = -cam[:, 2]
    valid = (x >= -4) & (x < resolution + 4) & (y >= -4) & (y < resolution + 4)
    if not np.any(valid):
        raise RuntimeError("No projected points in frame")

    x = np.clip(x[valid].astype(np.int32), 0, resolution - 1)
    y = np.clip(y[valid].astype(np.int32), 0, resolution - 1)
    depth = depth[valid]
    cam_normals = cam_normals[valid]
    colors = colors[valid]
    order = np.argsort(depth)[::-1]

    rgb = np.zeros((resolution, resolution, 3), dtype=np.float32)
    rgb[:, :, :] = np.array([0.78, 0.80, 0.82], dtype=np.float32)
    zbuf = np.full((resolution, resolution), np.inf, dtype=np.float64)
    light = np.array([0.25, -0.35, 0.9], dtype=np.float64)
    light /= np.linalg.norm(light)
    radius = max(1, resolution // 128)

    for idx in order:
        px = int(x[idx])
        py = int(y[idx])
        z = float(depth[idx])
        normal = cam_normals[idx]
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-9:
            normal = normal / normal_norm
        shade = 0.62 + 0.38 * max(0.0, float(np.dot(normal, light)))
        base = colors[idx].astype(np.float32)
        if float(base.max() - base.min()) < 0.01:
            hue = 0.08 + 0.84 * ((abs(float(normal[0])) * 0.37 + abs(float(normal[1])) * 0.41 + abs(float(normal[2])) * 0.22) % 1.0)
            base = np.clip(base * (0.72 + 0.28 * hue), 0.0, 1.0)
        color = np.clip(base * shade, 0.0, 1.0)
        for yy in range(max(0, py - radius), min(resolution, py + radius + 1)):
            for xx in range(max(0, px - radius), min(resolution, px + radius + 1)):
                if z < zbuf[yy, xx]:
                    zbuf[yy, xx] = z
                    rgb[yy, xx, :] = color

    max_rgb = float(rgb.max())
    if max_rgb < 0.05:
        raise RuntimeError(f"Software renderer produced a black image: max_rgb={max_rgb}")
    img = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return {"mean_rgb": float(rgb.mean()), "max_rgb": max_rgb, "mean_alpha": 1.0}


def python_process_one(entry: dict, output_dir: Path, args: argparse.Namespace, index: int):
    import numpy as np

    uid = entry["uid"]
    source_path = entry["path"]
    random.seed(args.seed + index)
    np.random.seed(args.seed + index)

    mesh = python_load_mesh(source_path)
    faces = int(len(mesh.faces))
    if faces < args.min_faces:
        raise RuntimeError(f"Too few faces: {faces}")
    if faces > args.max_faces:
        raise RuntimeError(f"Too many faces: {faces}")

    bbox = python_normalize_mesh(mesh)
    object_dir = output_dir / "objects" / uid
    object_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = object_dir / "normalized.glb"
    mesh.export(normalized_path)

    points, normals, colors = python_sample_surface(mesh, args.points, args.color_mode)
    points_path = object_dir / "points.npz"
    np.savez_compressed(points_path, points=points, normals=normals, colors=colors)

    render_dir = output_dir / "renders" / uid
    target = np.asarray(bbox["bbox_center"], dtype=np.float64)
    max_dim = max(float(v) for v in bbox["bbox_dims"])
    ortho_scale = max(2.20, max_dim * 1.35)
    rng = random.Random(args.seed + index)
    start_azimuth = rng.uniform(0.0, 360.0)
    rows = []
    for view_idx in range(args.views):
        azimuth = start_azimuth + view_idx * (360.0 / args.views)
        elevation = rng.uniform(args.elevation_min, args.elevation_max)
        az = math.radians(azimuth)
        el = math.radians(elevation)
        radius = float(args.camera_radius)
        location = target + np.array([
            radius * math.cos(el) * math.cos(az),
            radius * math.cos(el) * math.sin(az),
            radius * math.sin(el),
        ])
        camera_matrix = python_camera_matrix(location, target)
        image_path = render_dir / f"view_{view_idx:03d}.png"
        stats = python_render_points(image_path, points, normals, colors, camera_matrix, ortho_scale, args.resolution)
        focal_like = args.resolution / ortho_scale
        rows.append({
            "uid": uid,
            "view_index": view_idx,
            "image_path": str(image_path),
            "azimuth_deg": float(azimuth % 360.0),
            "elevation_deg": float(elevation),
            "radius": radius,
            "camera_location": [float(v) for v in location],
            "camera_matrix_world": [[float(v) for v in row] for row in camera_matrix],
            "camera_intrinsics": [[focal_like, 0.0, args.resolution / 2.0], [0.0, focal_like, args.resolution / 2.0], [0.0, 0.0, 1.0]],
            "camera_type": "ORTHO",
            "ortho_scale": float(ortho_scale),
            "render_mean_rgb": stats["mean_rgb"],
            "render_max_rgb": stats["max_rgb"],
            "render_mean_alpha": stats["mean_alpha"],
        })

    object_row = {
        "uid": uid,
        "source_path": source_path,
        "normalized_glb": str(normalized_path),
        "points_npz": str(points_path),
        "face_count": faces,
        **bbox,
    }
    return object_row, rows


def python_process_chunk(entries: Sequence[dict], output_dir: Path, args: argparse.Namespace) -> None:
    metadata_dir = output_dir / "metadata"
    object_rows = []
    view_rows = []
    failed_rows = []
    for index, entry in enumerate(entries):
        uid = entry["uid"]
        print(f"[python] Processing {index + 1}/{len(entries)} {uid}", flush=True)
        try:
            object_row, rows = python_process_one(entry, output_dir, args, index)
            object_rows.append(object_row)
            view_rows.extend(rows)
        except Exception as exc:
            failed_rows.append({
                "uid": uid,
                "source_path": entry.get("path", ""),
                "error": str(exc),
                "traceback": repr(exc),
            })
            print(f"[python] FAILED {uid}: {exc}", flush=True)

    write_rows_csv(metadata_dir / "objects_chunk.csv", object_rows)
    write_rows_csv(metadata_dir / "views_chunk.csv", view_rows)
    write_rows_csv(metadata_dir / "failed_chunk.csv", failed_rows)
    print(f"[python] Chunk complete: accepted={len(object_rows)} failed={len(failed_rows)} views={len(view_rows)}", flush=True)


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

    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        for username_name, key_name in (
            ("KAGGLE_USERNAME", "KAGGLE_KEY"),
            ("kaggle_username", "kaggle_key"),
            ("KAGGLE_USER", "KAGGLE_API_KEY"),
        ):
            try:
                username = secrets.get_secret(username_name)
                key = secrets.get_secret(key_name)
            except Exception:
                continue
            if username and key:
                os.environ["KAGGLE_USERNAME"] = username
                os.environ["KAGGLE_KEY"] = key
                print(f"Using Kaggle API credentials from Secrets: {username_name}/{key_name}", flush=True)
                return
    except Exception:
        pass

    explicit = Path(args.kaggle_json_path).expanduser() if args.kaggle_json_path else None
    if explicit and explicit.exists():
        install_kaggle_json(explicit)
        print(f"Using Kaggle API token from {explicit}", flush=True)
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
    print(
        "Kaggle API credentials were not found in environment, Kaggle Secrets, /root/.kaggle, /kaggle/working, or /kaggle/input.",
        flush=True,
    )


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
    if args.clean_output and output_dir.exists():
        resolved = output_dir.resolve()
        allowed_roots = [Path("/kaggle/working").resolve(), Path("/tmp").resolve()]
        if not any(str(resolved).startswith(str(root)) for root in allowed_roots):
            raise RuntimeError(f"Refusing to clean output directory outside Kaggle working/tmp: {resolved}")
        print(f"Cleaning existing output directory: {output_dir}", flush=True)
        shutil.rmtree(output_dir)

    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_kaggle_dataset_metadata(args, output_dir)
    worker_path = None
    blender = None
    if args.renderer == "blender":
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

        # Process extra candidates when only a few more accepted objects are needed,
        # because some Objaverse assets are malformed or fail geometry filters.
        process_limit = min(len(entries), max(remaining, min(args.download_batch, remaining * 3 + 6)))
        entries = entries[:process_limit]
        if args.renderer == "python":
            python_process_chunk(entries, output_dir, args)
        else:
            manifest = output_dir / "_scripts" / "chunk_manifest.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(entries, ensure_ascii=True, indent=2), encoding="utf-8")

            env = os.environ.copy()
            env.setdefault("CUDA_VISIBLE_DEVICES", "0")
            env.update(
                {
                    "OBJAVERSE_WORKER_MANIFEST": str(manifest),
                    "OBJAVERSE_WORKER_OUTPUT_DIR": str(output_dir),
                    "OBJAVERSE_WORKER_VIEWS": str(args.views),
                    "OBJAVERSE_WORKER_RESOLUTION": str(args.resolution),
                    "OBJAVERSE_WORKER_POINTS": str(args.points),
                    "OBJAVERSE_WORKER_SEED": str(args.seed),
                    "OBJAVERSE_WORKER_MIN_FACES": str(args.min_faces),
                    "OBJAVERSE_WORKER_MAX_FACES": str(args.max_faces),
                    "OBJAVERSE_WORKER_CAMERA_RADIUS": str(args.camera_radius),
                    "OBJAVERSE_WORKER_ELEVATION_MIN": str(args.elevation_min),
                    "OBJAVERSE_WORKER_ELEVATION_MAX": str(args.elevation_max),
                    "OBJAVERSE_WORKER_USE_GPU": "1" if args.use_gpu else "0",
                }
            )
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
            run(cmd, env=env)
        chunk_outputs = [
            metadata_dir / "objects_chunk.csv",
            metadata_dir / "views_chunk.csv",
            metadata_dir / "failed_chunk.csv",
        ]
        if not any(path.exists() and path.stat().st_size > 0 for path in chunk_outputs):
            raise RuntimeError(
                "Blender worker did not produce any chunk metadata. "
                "Check the Blender log above; the worker likely crashed before processing objects."
            )

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
    parser.add_argument("--points", type=int, default=32768)
    parser.add_argument(
        "--renderer",
        choices=("python", "blender"),
        default="python",
        help="Use python for a Blender-free trimesh renderer, or blender for the legacy Blender worker.",
    )
    parser.add_argument(
        "--color_mode",
        choices=("asset", "normal", "clay"),
        default="asset",
        help="Color PNG renders from mesh asset colors when available, normals, or a single clay color.",
    )
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
    parser.add_argument("--use_gpu", action="store_true", default=False)
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
        "--clean_output",
        action="store_true",
        help="Delete the existing output_dir before processing. Use this after changing render settings.",
    )
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
