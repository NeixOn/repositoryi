#!/usr/bin/env python3
"""
Fast parallel renderer for chair geometry buffers.

It converts the fixed chair dataset into an InstantMesh-like per-object layout:

  output_root/
    rendering_random_24views/
      <uid>/
        000.png          RGBA copied from dataset RGB + fixed mask
        000_depth.exr    Blender depth pass
        000_normal.png   uint8 world-space normal encoded from [-1, 1] to [0, 255]
        ...
        cameras.npz      camera_matrix_world + intrinsics arrays
    filtered_obj_name.json

The master process starts several Blender background workers. Each worker handles
whole objects, so progress is reported per object with a useful ETA instead of
spamming every rendered view.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


BLENDER_WORKER = r'''
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import bpy
import mathutils
import numpy as np
from PIL import Image


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--mask-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--views", type=int, required=True)
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--total-workers", type=int, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="CPU")
    return p.parse_args()


def clean_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def configure_scene(resolution, device):
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_layers[0].use_pass_z = True
    scene.view_layers[0].use_pass_normal = True
    try:
        scene.eevee.taa_render_samples = 1
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    except Exception:
        pass


def import_mesh(path):
    ext = Path(path).suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        raise RuntimeError(f"Unsupported mesh extension: {ext}")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects imported")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = "chair"
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=False)
    mat = bpy.data.materials.new("matte_white")
    mat.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


def setup_camera(camera_json, resolution):
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    intr = camera_json.get("intrinsics", {})
    fx = float(intr.get("fx", intr.get("fl_x", 560.0)))
    fy = float(intr.get("fy", intr.get("fl_y", fx)))
    cx = float(intr.get("cx", resolution / 2.0))
    cy = float(intr.get("cy", resolution / 2.0))
    mat = np.asarray(camera_json.get("camera_matrix_world") or camera_json.get("c2w"), dtype=np.float64).reshape(4, 4)

    cam_obj.matrix_world = mathutils.Matrix(mat.tolist())
    cam_data.type = "PERSP"
    cam_data.lens_unit = "MILLIMETERS"
    cam_data.lens = float(intr.get("lens_mm", 55.0))
    cam_data.sensor_width = float(intr.get("sensor_width_mm", 32.0))
    cam_data.shift_x = (cx - resolution / 2.0) / resolution
    cam_data.shift_y = -(cy - resolution / 2.0) / resolution
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100.0
    return cam_obj


def copy_rgba(rgb_path, mask_path, out_path, resolution):
    rgb = Image.open(rgb_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if rgb.size != (resolution, resolution):
        rgb = rgb.resize((resolution, resolution), Image.Resampling.BILINEAR)
    if mask.size != (resolution, resolution):
        mask = mask.resize((resolution, resolution), Image.Resampling.NEAREST)
    rgba = Image.merge("RGBA", (*rgb.split(), mask))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out_path)


def setup_depth_nodes(depth_dir, stem):
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    depth_out = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_out.base_path = str(depth_dir)
    depth_out.file_slots[0].path = f"{stem}_depth_"
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_mode = "RGB"
    tree.links.new(render_layers.outputs["Depth"], depth_out.inputs[0])


def find_depth_output(depth_dir, stem):
    candidates = sorted(depth_dir.glob(f"{stem}_depth_*.exr"))
    if not candidates:
        return None
    final_path = depth_dir / f"{stem}_depth.exr"
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


def render_depth_normal(out_dir, stem):
    scene = bpy.context.scene
    view_layer = scene.view_layers[0]

    setup_depth_nodes(out_dir, stem)
    view_layer.material_override = None
    scene.render.filepath = str(out_dir / f"{stem}_unused.png")
    bpy.ops.render.render(write_still=False)
    depth_path = find_depth_output(out_dir, stem)
    scene.use_nodes = False

    set_render_png_rgba()
    normal_mat = make_normal_material()
    view_layer.material_override = normal_mat
    scene.render.filepath = str(out_dir / f"{stem}_normal.png")
    bpy.ops.render.render(write_still=True)
    view_layer.material_override = None
    return depth_path, out_dir / f"{stem}_normal.png"


def process_uid(uid, args, dataset_root, mask_root, out_base):
    uid_out = out_base / uid
    done_flag = uid_out / "_done.json"
    if done_flag.exists() and not args.overwrite:
        return "skip"

    mesh_path = dataset_root / "objects" / uid / "normalized.glb"
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    clean_scene()
    configure_scene(args.resolution, args.device)
    import_mesh(mesh_path)

    cam_mats = []
    intrinsics = []
    uid_out.mkdir(parents=True, exist_ok=True)
    for view in range(args.views):
        stem = f"{view:03d}"
        rgb_path = dataset_root / "renders" / uid / f"view_{view:03d}.png"
        mask_path = mask_root / uid / f"view_{view:03d}.png"
        cam_path = dataset_root / "cameras" / uid / f"view_{view:03d}.json"
        if not rgb_path.exists() or not mask_path.exists() or not cam_path.exists():
            raise FileNotFoundError(f"Missing view inputs for uid={uid} view={view}")

        copy_rgba(rgb_path, mask_path, uid_out / f"{stem}.png", args.resolution)
        cam_json = json.loads(cam_path.read_text(encoding="utf-8"))
        old_cams = [obj for obj in bpy.context.scene.objects if obj.type == "CAMERA"]
        for obj in old_cams:
            bpy.data.objects.remove(obj, do_unlink=True)
        setup_camera(cam_json, args.resolution)
        render_depth_normal(uid_out, stem)

        cam_mats.append(np.asarray(cam_json.get("camera_matrix_world") or cam_json.get("c2w"), dtype=np.float32).reshape(4, 4))
        intr = cam_json.get("intrinsics", {})
        intrinsics.append([float(intr.get("fx", 0)), float(intr.get("fy", 0)), float(intr.get("cx", 0)), float(intr.get("cy", 0))])

    np.savez_compressed(
        uid_out / "cameras.npz",
        camera_matrix_world=np.stack(cam_mats, axis=0).astype(np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
    )
    done_flag.write_text(json.dumps({"uid": uid, "views": args.views, "time": time.time()}), encoding="utf-8")
    return "ok"


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    mask_root = Path(args.mask_root)
    output_root = Path(args.output_root)
    out_base = output_root / "rendering_random_24views"
    out_base.mkdir(parents=True, exist_ok=True)
    uids = [line.strip() for line in Path(args.manifest).read_text(encoding="utf-8").splitlines() if line.strip()]

    start = time.time()
    ok = 0
    skipped = 0
    failed = 0
    for idx, uid in enumerate(uids, 1):
        obj_start = time.time()
        try:
            status = process_uid(uid, args, dataset_root, mask_root, out_base)
            if status == "skip":
                skipped += 1
            else:
                ok += 1
            state = status
        except Exception as exc:
            failed += 1
            state = "fail"
            fail_dir = output_root / "_failures"
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / f"{uid}.txt").write_text(repr(exc), encoding="utf-8")

        elapsed = time.time() - start
        done = ok + skipped + failed
        sec_per_obj = elapsed / max(done, 1)
        eta_min = sec_per_obj * max(len(uids) - done, 0) / 60.0
        print(
            f"[worker {args.worker_id}/{args.total_workers}] object={idx}/{len(uids)} "
            f"uid={uid} status={state} obj_sec={time.time() - obj_start:.1f} "
            f"ok={ok} skip={skipped} fail={failed} eta_min={eta_min:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--mask_root", default="", help="Defaults to <dataset_root>/masks.")
    p.add_argument("--output_root", required=True)
    p.add_argument("--clean_uids", default="", help="Defaults to <dataset_root>/audit/clean_uids.txt if present.")
    p.add_argument("--views", type=int, default=24)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--workers", type=int, default=2, help="Number of parallel Blender processes.")
    p.add_argument("--blender", default="blender", help="Path to Blender executable.")
    p.add_argument("--max_objects", type=int, default=0)
    p.add_argument("--start_object", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="CPU", help="Informational for worker; Eevee is used for speed.")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def read_uids(args: argparse.Namespace) -> list[str]:
    dataset_root = Path(args.dataset_root)
    clean_path = Path(args.clean_uids) if args.clean_uids else dataset_root / "audit" / "clean_uids.txt"
    if clean_path.exists():
        uids = [line.strip() for line in clean_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        uids = sorted(p.name for p in (dataset_root / "objects").iterdir() if p.is_dir())
    if args.start_object > 0:
        uids = uids[args.start_object :]
    if args.max_objects > 0:
        uids = uids[: args.max_objects]
    return uids


def chunks(items: list[str], n: int) -> list[list[str]]:
    n = max(1, n)
    out = [[] for _ in range(n)]
    for idx, item in enumerate(items):
        out[idx % n].append(item)
    return [part for part in out if part]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    mask_root = Path(args.mask_root).resolve() if args.mask_root else dataset_root / "masks"
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    uids = read_uids(args)
    write_json(output_root / "filtered_obj_name.json", {"good_objs": uids})
    worker_script = output_root / "_fast_blender_worker.py"
    worker_script.write_text(BLENDER_WORKER, encoding="utf-8")

    print(f"[master] dataset_root={dataset_root}", flush=True)
    print(f"[master] mask_root={mask_root}", flush=True)
    print(f"[master] output_root={output_root}", flush=True)
    print(f"[master] objects={len(uids)} views={args.views} workers={args.workers}", flush=True)
    print(f"[master] blender={args.blender}", flush=True)

    if args.dry_run:
        for uid in uids[:10]:
            print(f"[dry-run] {uid}")
        return 0

    manifests = []
    for worker_id, part in enumerate(chunks(uids, args.workers)):
        manifest = output_root / f"_manifest_worker_{worker_id:02d}.txt"
        manifest.write_text("\n".join(part) + "\n", encoding="utf-8")
        manifests.append((worker_id, manifest, len(part)))

    procs: list[tuple[int, subprocess.Popen]] = []
    start = time.time()
    for worker_id, manifest, count in manifests:
        cmd = [
            args.blender,
            "--background",
            "--factory-startup",
            "--python",
            str(worker_script),
            "--",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(dataset_root),
            "--mask-root",
            str(mask_root),
            "--output-root",
            str(output_root),
            "--views",
            str(args.views),
            "--resolution",
            str(args.resolution),
            "--worker-id",
            str(worker_id),
            "--total-workers",
            str(len(manifests)),
            "--device",
            args.device,
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"[master] starting worker={worker_id} objects={count}", flush=True)
        procs.append((worker_id, subprocess.Popen(cmd)))

    exit_code = 0
    for worker_id, proc in procs:
        code = proc.wait()
        print(f"[master] worker={worker_id} exited code={code}", flush=True)
        if code != 0:
            exit_code = code

    elapsed = (time.time() - start) / 60.0
    done_files = list((output_root / "rendering_random_24views").glob("*/_done.json"))
    fail_files = list((output_root / "_failures").glob("*.txt")) if (output_root / "_failures").exists() else []
    print(
        f"[master] finished elapsed_min={elapsed:.1f} done_objects={len(done_files)} "
        f"failed_objects={len(fail_files)} output={output_root}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
