#!/usr/bin/env python3
"""
Distill shape-only meshes from Hunyuan3D-2 into a trainable chair dataset.

Unlike distill_teacher_meshes.py, this script keeps Hunyuan3D loaded once per
GPU worker. That is required for speed: loading the teacher model for every
image would be unusably slow.

Expected setup:
  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /data/Hunyuan3D-2
  cd /data/Hunyuan3D-2
  pip install -r requirements.txt
  pip install -e .

Example:
  python /data/repositoryi/distill_hunyuan3d_shape.py \
    --source_dataset /data/abo_chairs \
    --output_dir /data/chair_hunyuan_distill \
    --hunyuan_repo /data/Hunyuan3D-2 \
    --model_path tencent/Hunyuan3D-2mini \
    --subfolder hunyuan3d-dit-v2-mini-turbo \
    --gpus 0,1 \
    --views_per_object 1 \
    --max_images 1000 \
    --steps 30 \
    --points 65536
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from distill_teacher_meshes import (
    prepare_input_image,
    normalize_mesh,
    read_source_rows,
    sample_uid,
)


def select_rows(dataset_root: Path, views_per_object: int, max_images: int, seed: int) -> list[dict]:
    rows = read_source_rows(dataset_root)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["uid"], []).append(row)
    selected = []
    for uid in sorted(grouped):
        selected.extend(sorted(grouped[uid], key=lambda r: r["view_index"])[:views_per_object])
    random.Random(seed).shuffle(selected)
    if max_images > 0:
        selected = selected[:max_images]
    return selected


def worker_main(gpu: str, tasks: mp.Queue, results: mp.Queue, args_dict: dict) -> None:
    args = argparse.Namespace(**args_dict)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    sys.path.insert(0, str(Path(args.hunyuan_repo).resolve()))

    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {}
    if args.subfolder:
        kwargs["subfolder"] = args.subfolder
    if args.low_vram_mode:
        kwargs["low_vram_mode"] = True
    print(f"[gpu {gpu}] loading Hunyuan3D shape pipeline: {args.model_path} {kwargs}", flush=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path, **kwargs)
    try:
        pipeline = pipeline.to(device)
    except Exception:
        pass
    print(f"[gpu {gpu}] pipeline ready", flush=True)

    out = Path(args.output_dir)
    while True:
        row = tasks.get()
        if row is None:
            break
        uid = sample_uid(row)
        image_dst = out / "renders" / uid / "view_000.png"
        obj_dir = out / "objects" / uid
        teacher_mesh = obj_dir / "teacher_mesh.glb"
        normalized = obj_dir / "normalized.glb"
        points_path = obj_dir / "points.npz"
        if points_path.exists() and normalized.exists() and image_dst.exists() and not args.overwrite:
            results.put({"uid": uid, "status": "exists", "faces": -1, "scale": -1.0, "log": ""})
            continue
        try:
            prepare_input_image(Path(row["image_path"]), image_dst, args.image_size)
            obj_dir.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                call_kwargs = {"image": str(image_dst)}
                if args.steps > 0:
                    call_kwargs["num_inference_steps"] = args.steps
                mesh = pipeline(**call_kwargs)[0]
            mesh.export(teacher_mesh)
            faces, scale = normalize_mesh(teacher_mesh, normalized, points_path, args.points, args.seed + abs(hash(uid)) % 1000000)
            results.put({"uid": uid, "status": "ok", "faces": faces, "scale": scale, "log": ""})
        except Exception as exc:
            results.put({"uid": uid, "status": "failed", "faces": 0, "scale": 0.0, "log": repr(exc)})


def writer_main(out: Path, results: mp.Queue, total: int) -> None:
    meta = out / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    views_path = meta / "views.csv"
    objects_path = meta / "objects.csv"
    log_path = meta / "distillation_log.csv"
    seen = set()
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                seen.add(row["uid"])
    with open(views_path, "w", encoding="utf-8", newline="") as vf, open(objects_path, "w", encoding="utf-8", newline="") as of, open(log_path, "w", encoding="utf-8", newline="") as lf:
        vw = csv.DictWriter(vf, fieldnames=["uid", "view_index", "image_path"])
        ow = csv.DictWriter(of, fieldnames=["uid", "source", "faces", "scale"])
        lw = csv.DictWriter(lf, fieldnames=["uid", "status", "faces", "scale", "log"])
        vw.writeheader()
        ow.writeheader()
        lw.writeheader()
        done = 0
        ok = 0
        while done < total:
            item = results.get()
            done += 1
            if item["status"] in ("ok", "exists"):
                ok += 1
                vw.writerow({"uid": item["uid"], "view_index": 0, "image_path": f"renders/{item['uid']}/view_000.png"})
                ow.writerow({"uid": item["uid"], "source": "hunyuan3d", "faces": item["faces"], "scale": item["scale"]})
            lw.writerow({"uid": item["uid"], "status": item["status"], "faces": item["faces"], "scale": item["scale"], "log": item["log"][:2000]})
            vf.flush()
            of.flush()
            lf.flush()
            print(f"distill progress: {done}/{total} ok={ok} last={item['uid']} status={item['status']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hunyuan_repo", default="/data/Hunyuan3D-2")
    parser.add_argument("--model_path", default="tencent/Hunyuan3D-2mini")
    parser.add_argument("--subfolder", default="hunyuan3d-dit-v2-mini-turbo")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--views_per_object", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--points", type=int, default=65536)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low_vram_mode", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = select_rows(Path(args.source_dataset), args.views_per_object, args.max_images, args.seed)
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        gpus = ["0"]

    ctx = mp.get_context("spawn")
    tasks = ctx.Queue()
    results = ctx.Queue()
    for row in rows:
        tasks.put(row)
    for _ in gpus:
        tasks.put(None)

    writer = ctx.Process(target=writer_main, args=(out, results, len(rows)))
    writer.start()
    workers = [ctx.Process(target=worker_main, args=(gpu, tasks, results, vars(args))) for gpu in gpus]
    for p in workers:
        p.start()
    for p in workers:
        p.join()
    writer.join()

    info = {"source_dataset": args.source_dataset, "selected_images": len(rows), "teacher": "Hunyuan3D", "model_path": args.model_path, "subfolder": args.subfolder}
    (out / "metadata" / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done. output={out}", flush=True)


if __name__ == "__main__":
    main()
