#!/usr/bin/env python3
"""
Repair mask PNGs for an already rendered Blender chair dataset.

This does not re-render RGB/depth/normal/points. It uses:
  objects/<uid>/normalized.glb
  cameras/<uid>/view_XXX.json

and writes:
  masks/<uid>/view_XXX.png

The Blender worker renders a fast white silhouette on black background.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Sequence


DEFAULT_BLENDER_URL = "https://download.blender.org/release/Blender4.1/blender-4.1.1-linux-x64.tar.xz"


BLENDER_MASK_WORKER = r'''
import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
import mathutils


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--resolution", type=int, default=0)
    p.add_argument("--engine", choices=("eevee", "workbench"), default="eevee")
    return p.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.textures):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def setup_scene(engine):
    scene = bpy.context.scene
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "BW"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if engine == "workbench":
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "FLAT"
        scene.display.shading.color_type = "MATERIAL"
        scene.display.shading.background_type = "VIEWPORT"
        scene.display.shading.background_color = (0.0, 0.0, 0.0)
    else:
        try:
            scene.render.engine = "BLENDER_EEVEE_NEXT"
        except Exception:
            scene.render.engine = "BLENDER_EEVEE"
        scene.eevee.taa_render_samples = 1 if hasattr(scene, "eevee") else 1
        scene.world = bpy.data.worlds.new("World") if scene.world is None else scene.world
        scene.world.color = (0.0, 0.0, 0.0)


def make_mask_material():
    mat = bpy.data.materials.new("mask_white")
    mat.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def import_mesh(mesh_path):
    bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh objects in {mesh_path}")
    mat = make_mask_material()
    for obj in meshes:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        obj.hide_render = False
    return meshes


def configure_camera_from_json(camera, camera_json, resolution_override):
    data = json.loads(Path(camera_json).read_text(encoding="utf-8"))
    intr = data["intrinsics"]
    width = int(resolution_override or intr.get("width", 512))
    height = int(resolution_override or intr.get("height", 512))

    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100

    camera.data.lens = float(intr.get("lens_mm", 55.0))
    camera.data.sensor_width = float(intr.get("sensor_width_mm", 32.0))
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0
    matrix = mathutils.Matrix(data["camera_matrix_world"])
    camera.matrix_world = matrix
    scene.camera = camera
    return camera


def add_camera():
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    bpy.context.scene.camera = camera
    return camera


def render_object_views(mesh_path, views, dataset_root, uid, resolution_override, engine):
    clear_scene()
    setup_scene(engine)
    import_mesh(mesh_path)
    camera = add_camera()
    out_dir = dataset_root / "masks" / uid
    out_dir.mkdir(parents=True, exist_ok=True)

    for view in views:
        camera_json = dataset_root / "cameras" / uid / f"view_{view:03d}.json"
        out_path = out_dir / f"view_{view:03d}.png"
        configure_camera_from_json(camera, camera_json, resolution_override)
        bpy.context.scene.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    items = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    total = sum(len(item["views"]) for item in items)
    done = 0
    start = time.time()

    for obj_idx, item in enumerate(items, start=1):
        uid = item["uid"]
        mesh_path = dataset_root / "objects" / uid / "normalized.glb"
        obj_start = time.time()
        render_object_views(mesh_path, item["views"], dataset_root, uid, args.resolution, args.engine)
        done += len(item["views"])
        elapsed = time.time() - start
        sec_per_view = elapsed / max(done, 1)
        eta = sec_per_view * max(total - done, 0) / 60.0
        print(
            f"[mask] object={obj_idx}/{len(items)} uid={uid} views={len(item['views'])} "
            f"obj_sec={time.time() - obj_start:.2f} sec/view={sec_per_view:.3f} eta_min={eta:.1f}",
            flush=True,
        )

    print(f"[mask] complete views={done} elapsed_min={(time.time() - start) / 60.0:.1f}", flush=True)


if __name__ == "__main__":
    main()
'''


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--blender_path", default="")
    p.add_argument("--download_blender", action="store_true")
    p.add_argument("--blender_dir", default="")
    p.add_argument("--resolution", type=int, default=0, help="0 keeps camera json resolution")
    p.add_argument("--engine", choices=("eevee", "workbench"), default="eevee")
    p.add_argument("--force", action="store_true", help="Re-render all masks, not only invalid/missing masks")
    p.add_argument("--views", type=int, default=24)
    p.add_argument("--max_objects", type=int, default=0)
    p.add_argument("--start_object", type=int, default=0)
    p.add_argument("--preview_count", type=int, default=48)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def find_blender(args) -> str:
    if args.blender_path:
        path = Path(args.blender_path).expanduser()
        if path.exists():
            return str(path)
        raise FileNotFoundError(path)

    found = shutil.which("blender")
    if found:
        return found

    blender_dir = Path(args.blender_dir or (Path(args.dataset_root).parent / "blender"))
    extracted = blender_dir / "blender-4.1.1-linux-x64" / "blender"
    if extracted.exists():
        return str(extracted)

    if not args.download_blender:
        raise FileNotFoundError("Blender not found. Pass --blender_path or --download_blender.")

    blender_dir.mkdir(parents=True, exist_ok=True)
    archive_path = blender_dir / "blender-4.1.1-linux-x64.tar.xz"
    if not archive_path.exists():
        print(f"Downloading Blender to {archive_path}", flush=True)
        urllib.request.urlretrieve(DEFAULT_BLENDER_URL, archive_path)
    print(f"Extracting Blender to {blender_dir}", flush=True)
    with tarfile.open(archive_path, "r:xz") as tar:
        tar.extractall(blender_dir)
    if not extracted.exists():
        raise FileNotFoundError(extracted)
    return str(extracted)


def mask_is_valid(path: Path) -> tuple[bool, float]:
    if not path.exists():
        return False, -1.0
    from PIL import Image
    import numpy as np

    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    if arr.size == 0:
        return False, -1.0
    fg_ratio = float((arr > 8).mean())
    return 0.002 <= fg_ratio <= 0.95, fg_ratio


def load_uids(dataset_root: Path, views: int) -> list[str]:
    views_csv = dataset_root / "metadata" / "views.csv"
    if views_csv.exists():
        uids = []
        with open(views_csv, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                uid = row.get("uid", "")
                if uid and uid not in uids:
                    uids.append(uid)
        return uids
    return sorted(p.name for p in (dataset_root / "objects").iterdir() if (p / "normalized.glb").exists())


def build_manifest(dataset_root: Path, views: int, force: bool, max_objects: int, start_object: int):
    items = []
    valid_views = 0
    invalid_views = 0
    uids = load_uids(dataset_root, views)
    if start_object > 0:
        uids = uids[start_object:]
    if max_objects > 0:
        uids = uids[:max_objects]

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
            ok, _ = mask_is_valid(mask_path)
            if ok and not force:
                valid_views += 1
            else:
                invalid_views += 1
                repair_views.append(view)
        if repair_views:
            items.append({"uid": uid, "views": repair_views})
    return items, valid_views, invalid_views


def make_preview_sheet(dataset_root: Path, count: int) -> Path | None:
    if count <= 0:
        return None
    from PIL import Image, ImageDraw
    import math

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
    out_path = dataset_root / "mask_repair_preview.jpg"
    sheet.save(out_path, quality=92)
    return out_path


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    script_dir = dataset_root / "_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    worker_path = script_dir / "repair_mask_worker.py"
    manifest_path = script_dir / "repair_mask_manifest.json"

    items, valid_views, invalid_views = build_manifest(
        dataset_root,
        args.views,
        args.force,
        args.max_objects,
        args.start_object,
    )
    manifest_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    total_views = sum(len(item["views"]) for item in items)
    print(f"Dataset: {dataset_root}", flush=True)
    print(f"Existing valid views: {valid_views}", flush=True)
    print(f"Views to repair: {total_views}", flush=True)
    print(f"Objects to repair: {len(items)}", flush=True)

    if args.dry_run or not items:
        return

    worker_path.write_text(BLENDER_MASK_WORKER, encoding="utf-8")
    blender = find_blender(args)
    start = time.time()
    cmd = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(worker_path),
        "--",
        "--manifest",
        str(manifest_path),
        "--dataset_root",
        str(dataset_root),
        "--resolution",
        str(args.resolution),
        "--engine",
        args.engine,
    ]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    repaired_items, valid_after, invalid_after = build_manifest(
        dataset_root,
        args.views,
        False,
        args.max_objects,
        args.start_object,
    )
    print(f"Elapsed minutes: {(time.time() - start) / 60.0:.1f}", flush=True)
    print(f"Valid views after repair: {valid_after}", flush=True)
    print(f"Still invalid views: {sum(len(item['views']) for item in repaired_items)}", flush=True)
    sheet = make_preview_sheet(dataset_root, args.preview_count)
    if sheet:
        print(f"Preview sheet: {sheet}", flush=True)


if __name__ == "__main__":
    main()
