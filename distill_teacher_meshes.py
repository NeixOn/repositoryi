#!/usr/bin/env python3
"""
Build a pseudo-3D distillation dataset from a strong teacher model.

The script is teacher-agnostic. You provide a command template that receives an
input image and an output directory. The teacher should write a mesh file
(.obj/.ply/.glb/.gltf) somewhere inside that output directory.

Example with a hypothetical shape-only Hunyuan3D command:

  python distill_teacher_meshes.py \
    --source_dataset /data/abo_chairs \
    --output_dir /data/chair_teacher_distill \
    --teacher_cmd 'python /data/Hunyuan3D/run.py --image {image} --output {out_dir} --no_texture' \
    --gpus 0,1 \
    --workers_per_gpu 1 \
    --max_images 5000 \
    --points 65536

Output layout is compatible with train_chair_triplane_udf_cuda.py:

  output_dir/
    metadata/views.csv
    metadata/objects.csv
    metadata/distillation_log.csv
    renders/<sample_uid>/view_000.png
    objects/<sample_uid>/teacher_mesh.*
    objects/<sample_uid>/normalized.glb
    objects/<sample_uid>/points.npz
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np


MESH_EXTS = (".obj", ".ply", ".glb", ".gltf", ".stl")


def ensure_deps(skip_install: bool) -> None:
    if skip_install:
        return
    pkgs = ["Pillow", "trimesh", "scipy", "tqdm"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore", *pkgs], check=True)


def read_source_rows(dataset_root: Path) -> list[dict]:
    views_csv = dataset_root / "metadata" / "views.csv"
    if not views_csv.exists():
        raise FileNotFoundError(f"Missing {views_csv}")
    rows = []
    with open(views_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["uid"]
            view_index = int(row["view_index"])
            image_path = dataset_root / row.get("image_path", "")
            if not image_path.exists():
                image_path = dataset_root / "renders" / uid / f"view_{view_index:03d}.png"
            if image_path.exists():
                rows.append({"uid": uid, "view_index": view_index, "image_path": str(image_path)})
    if not rows:
        raise RuntimeError(f"No source images found under {dataset_root}")
    return rows


def sample_uid(row: dict) -> str:
    return f"{row['uid']}_v{int(row['view_index']):03d}"


def find_mesh(root: Path) -> Path | None:
    meshes = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in MESH_EXTS and path.stat().st_size > 0:
            meshes.append(path)
    if not meshes:
        return None
    meshes.sort(key=lambda p: p.stat().st_size, reverse=True)
    return meshes[0]


def normalize_mesh(src: Path, dst: Path, points_path: Path, points: int, seed: int) -> tuple[int, float]:
    import trimesh

    obj = trimesh.load(src, force="scene", process=False)
    if isinstance(obj, trimesh.Trimesh):
        mesh = obj
    else:
        meshes = []
        for geom in obj.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                meshes.append(geom)
        if not meshes:
            raise RuntimeError("teacher output has no mesh geometry")
        mesh = trimesh.util.concatenate(meshes)
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) < 20 or len(mesh.faces) < 20:
        raise RuntimeError(f"teacher mesh too small: vertices={len(mesh.vertices)} faces={len(mesh.faces)}")
    bounds = mesh.bounds.astype(np.float64)
    center = (bounds[0] + bounds[1]) * 0.5
    scale = float((bounds[1] - bounds[0]).max())
    if not np.isfinite(scale) or scale <= 1e-8:
        raise RuntimeError("bad teacher mesh scale")
    mesh.vertices = (mesh.vertices - center) / scale
    bounds = mesh.bounds.astype(np.float64)
    zmin, zmax = bounds[0, 2], bounds[1, 2]
    zscale = max(1e-8, zmax - zmin)
    mesh.vertices[:, 2] = (mesh.vertices[:, 2] - zmin) / zscale
    mesh.vertices[:, 0:2] *= 1.8
    mesh.vertices[:, 2] *= 1.8
    mesh.vertices[:, 2] += 0.02
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)
    rng = np.random.default_rng(seed)
    samples, face_idx = trimesh.sample.sample_surface(mesh, points)
    normals = mesh.face_normals[face_idx]
    points_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(points_path, points=samples.astype(np.float32), normals=normals.astype(np.float32))
    return int(len(mesh.faces)), float(scale)


def prepare_input_image(src: Path, dst: Path, image_size: int) -> None:
    from PIL import Image, ImageOps

    img = Image.open(src).convert("RGB")
    img = ImageOps.contain(img, (image_size, image_size))
    canvas = Image.new("RGB", (image_size, image_size), (255, 255, 255))
    canvas.paste(img, ((image_size - img.width) // 2, (image_size - img.height) // 2))
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)


def run_teacher(cmd_template: str, image: Path, out_dir: Path, gpu: str | None, timeout: int) -> tuple[int, str]:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = cmd_template.format(image=str(image), out_dir=str(out_dir))
    proc = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return proc.returncode, proc.stdout[-12000:]


def process_one(row: dict, args, gpu: str | None) -> dict:
    uid = sample_uid(row)
    out = Path(args.output_dir)
    image_dst = out / "renders" / uid / "view_000.png"
    obj_dir = out / "objects" / uid
    teacher_dir = obj_dir / "_teacher_raw"
    normalized = obj_dir / "normalized.glb"
    points_path = obj_dir / "points.npz"

    if points_path.exists() and image_dst.exists() and normalized.exists() and not args.overwrite:
        return {"status": "exists", "uid": uid, "views": 1, "faces": -1, "scale": -1.0, "log": ""}

    prepare_input_image(Path(row["image_path"]), image_dst, args.image_size)

    if not args.reuse_teacher_mesh or not find_mesh(teacher_dir):
        if teacher_dir.exists() and args.overwrite:
            shutil.rmtree(teacher_dir)
        code, log = run_teacher(args.teacher_cmd, image_dst, teacher_dir, gpu, args.teacher_timeout)
        if code != 0:
            return {"status": "teacher_failed", "uid": uid, "views": 0, "faces": 0, "scale": 0.0, "log": log}
    else:
        log = "reuse teacher mesh"

    mesh = find_mesh(teacher_dir)
    if mesh is None:
        return {"status": "no_mesh", "uid": uid, "views": 0, "faces": 0, "scale": 0.0, "log": log}
    teacher_copy = obj_dir / f"teacher_mesh{mesh.suffix.lower()}"
    if mesh.resolve() != teacher_copy.resolve():
        shutil.copy2(mesh, teacher_copy)
    try:
        faces, scale = normalize_mesh(teacher_copy, normalized, points_path, args.points, args.seed + abs(hash(uid)) % 1000000)
    except Exception as exc:
        return {"status": "mesh_failed", "uid": uid, "views": 0, "faces": 0, "scale": 0.0, "log": f"{exc}\n{log}"}
    if args.delete_teacher_raw:
        shutil.rmtree(teacher_dir, ignore_errors=True)
    return {"status": "ok", "uid": uid, "views": 1, "faces": faces, "scale": scale, "log": log}


def writer_thread_fn(out: Path, result_queue: queue.Queue, stop_token: object) -> None:
    meta = out / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    views_path = meta / "views.csv"
    objects_path = meta / "objects.csv"
    log_path = meta / "distillation_log.csv"
    views_exists = views_path.exists()
    objects_exists = objects_path.exists()
    log_exists = log_path.exists()
    with open(views_path, "a", encoding="utf-8", newline="") as vf, open(objects_path, "a", encoding="utf-8", newline="") as of, open(log_path, "a", encoding="utf-8", newline="") as lf:
        vw = csv.DictWriter(vf, fieldnames=["uid", "view_index", "image_path"])
        ow = csv.DictWriter(of, fieldnames=["uid", "source", "faces", "scale"])
        lw = csv.DictWriter(lf, fieldnames=["uid", "status", "faces", "scale", "log"])
        if not views_exists:
            vw.writeheader()
        if not objects_exists:
            ow.writeheader()
        if not log_exists:
            lw.writeheader()
        while True:
            item = result_queue.get()
            if item is stop_token:
                break
            if item["status"] in ("ok", "exists"):
                vw.writerow({"uid": item["uid"], "view_index": 0, "image_path": f"renders/{item['uid']}/view_000.png"})
                ow.writerow({"uid": item["uid"], "source": "teacher", "faces": item["faces"], "scale": item["scale"]})
            lw.writerow({"uid": item["uid"], "status": item["status"], "faces": item["faces"], "scale": item["scale"], "log": item["log"][:2000]})
            vf.flush()
            of.flush()
            lf.flush()


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher_cmd", required=True, help="Command template. Use {image} and {out_dir} placeholders.")
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU ids for teacher processes.")
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--views_per_object", type=int, default=1, help="Use first N source views per source object.")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--points", type=int, default=65536)
    parser.add_argument("--teacher_timeout", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse_teacher_mesh", action="store_true")
    parser.add_argument("--delete_teacher_raw", action="store_true")
    parser.add_argument("--skip_install", action="store_true")
    args = parser.parse_args()
    ensure_deps(args.skip_install)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_source_rows(Path(args.source_dataset))
    by_uid = {}
    for row in rows:
        by_uid.setdefault(row["uid"], []).append(row)
    selected = []
    for uid in sorted(by_uid):
        selected.extend(sorted(by_uid[uid], key=lambda x: x["view_index"])[: args.views_per_object])
    random.Random(args.seed).shuffle(selected)
    if args.max_images > 0:
        selected = selected[: args.max_images]

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    worker_gpus = []
    for gpu in gpus:
        worker_gpus.extend([gpu] * max(1, args.workers_per_gpu))
    if not worker_gpus:
        worker_gpus = [None]

    tasks = queue.Queue()
    results = queue.Queue()
    stop = object()
    for row in selected:
        tasks.put(row)

    def worker(gpu: str | None):
        while True:
            try:
                row = tasks.get_nowait()
            except queue.Empty:
                return
            t0 = time.time()
            result = process_one(row, args, gpu)
            result["seconds"] = time.time() - t0
            results.put(result)

    writer = threading.Thread(target=writer_thread_fn, args=(out, results, stop), daemon=True)
    writer.start()
    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in worker_gpus]
    for th in threads:
        th.start()

    done = 0
    ok = 0
    while any(th.is_alive() for th in threads):
        time.sleep(5)
        done = sum(1 for _ in [])  # keep loop responsive without consuming results
        log_path = out / "metadata" / "distillation_log.csv"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                lines = max(0, sum(1 for _ in f) - 1)
            print(f"progress: logged={lines}/{len(selected)}", flush=True)
    for th in threads:
        th.join()
    results.put(stop)
    writer.join()

    log_path = out / "metadata" / "distillation_log.csv"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", newline="") as f:
            logs = list(csv.DictReader(f))
        ok = sum(1 for r in logs if r["status"] in ("ok", "exists"))
        done = len(logs)
    info = {"source_dataset": args.source_dataset, "selected_images": len(selected), "completed": done, "accepted": ok, "points": args.points}
    (out / "metadata" / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done. accepted={ok}/{len(selected)} output={out}", flush=True)


if __name__ == "__main__":
    main()
