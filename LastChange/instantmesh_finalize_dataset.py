#!/usr/bin/env python3
"""
Finalize rendered chair data for InstantMesh training.

Input layout from fast_render_chair_gbuffers.py:
  rendering_random_24views/<uid>/
    000.png
    000_depth.exr
    000_normal.png
    cameras.npz with camera_matrix_world

InstantMesh ObjaverseData expects:
  rendering_random_24views/<uid>/
    000.png
    000_depth.png   uint8 depth, loaded as /255 * depth_scale
    000_normal.png
    cameras.npz with cam_poses, shape [V, 3, 4], world-to-camera

This script writes missing depth PNG files and augments cameras.npz.
EXR conversion is performed through Blender because many OpenCV builds have EXR
disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


BLENDER_CONVERTER = r'''
import argparse
import json
import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from PIL import Image


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    elif "--manifest" in argv:
        argv = argv[argv.index("--manifest"):]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.environ.get("IM_FINALIZE_MANIFEST"))
    p.add_argument("--root", default=os.environ.get("IM_FINALIZE_ROOT"))
    p.add_argument("--depth-scale", type=float, default=float(os.environ.get("IM_FINALIZE_DEPTH_SCALE", "6.0")))
    p.add_argument("--worker-id", type=int, default=int(os.environ.get("IM_FINALIZE_WORKER_ID", "0")))
    p.add_argument("--total-workers", type=int, default=int(os.environ.get("IM_FINALIZE_TOTAL_WORKERS", "1")))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)
    if not args.manifest or not args.root:
        p.error("Missing --manifest/--root")
    return args


def exr_to_depth_png(exr_path, png_path, depth_scale, overwrite):
    if png_path.exists() and not overwrite:
        return "skip"
    img = bpy.data.images.load(str(exr_path))
    w, h = img.size
    arr = np.asarray(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)[..., 0]
    # Blender image pixels are bottom-up relative to normal image viewers.
    arr = np.flipud(arr)
    valid = np.isfinite(arr) & (arr > 0.0) & (arr < depth_scale)
    out = np.zeros((h, w), dtype=np.uint8)
    if valid.any():
        out[valid] = np.clip(arr[valid] / max(depth_scale, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="L").save(png_path)
    bpy.data.images.remove(img)
    return "ok"


def update_cameras_npz(uid_dir):
    cam_path = uid_dir / "cameras.npz"
    data = np.load(cam_path)
    payload = {key: data[key] for key in data.files}
    if "cam_poses" not in payload:
        if "camera_matrix_world" not in payload:
            raise RuntimeError(f"{cam_path} has no camera_matrix_world/cam_poses")
        c2w = np.asarray(payload["camera_matrix_world"], dtype=np.float32)
        w2c = np.linalg.inv(c2w).astype(np.float32)
        payload["cam_poses"] = w2c[:, :3, :4]
        np.savez_compressed(cam_path, **payload)
    return "ok"


def main():
    args = parse_args()
    root = Path(args.root)
    uids = [line.strip() for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    start = time.time()
    ok = 0
    skip = 0
    fail = 0
    for idx, uid in enumerate(uids, 1):
        uid_dir = root / "rendering_random_24views" / uid
        obj_start = time.time()
        try:
            update_cameras_npz(uid_dir)
            for view in range(24):
                status = exr_to_depth_png(
                    uid_dir / f"{view:03d}_depth.exr",
                    uid_dir / f"{view:03d}_depth.png",
                    args.depth_scale,
                    args.overwrite,
                )
                if status == "skip":
                    skip += 1
                else:
                    ok += 1
            state = "ok"
        except Exception as exc:
            fail += 1
            state = "fail"
            fail_dir = root / "_finalize_failures"
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / f"{uid}.txt").write_text(repr(exc), encoding="utf-8")
        done = idx
        elapsed = time.time() - start
        eta_min = (elapsed / max(done, 1)) * max(len(uids) - done, 0) / 60.0
        print(
            f"[finalize {args.worker_id}/{args.total_workers}] object={idx}/{len(uids)} "
            f"uid={uid} status={state} obj_sec={time.time()-obj_start:.1f} "
            f"depth_ok={ok} depth_skip={skip} fail={fail} eta_min={eta_min:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
'''


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", required=True, help="chairs_instantmesh output root.")
    p.add_argument("--blender", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--depth_scale", type=float, default=6.0)
    p.add_argument("--max_objects", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def split_evenly(items: list[str], parts: int) -> list[list[str]]:
    parts = max(1, min(parts, len(items)))
    shards = [[] for _ in range(parts)]
    for idx, item in enumerate(items):
        shards[idx % parts].append(item)
    return shards


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    meta_path = root / "filtered_obj_name.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    uids = list(data["good_objs"])
    if args.max_objects > 0:
        uids = uids[: args.max_objects]

    worker_path = root / "_instantmesh_finalize_worker.py"
    worker_path.write_text(BLENDER_CONVERTER, encoding="utf-8")

    print(f"[master] root={root} objects={len(uids)} workers={args.workers}", flush=True)
    procs = []
    for worker_id, shard in enumerate(split_evenly(uids, args.workers)):
        manifest = root / f"_finalize_manifest_{worker_id:02d}.txt"
        manifest.write_text("\n".join(shard) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "IM_FINALIZE_MANIFEST": str(manifest),
                "IM_FINALIZE_ROOT": str(root),
                "IM_FINALIZE_DEPTH_SCALE": str(args.depth_scale),
                "IM_FINALIZE_WORKER_ID": str(worker_id),
                "IM_FINALIZE_TOTAL_WORKERS": str(args.workers),
            }
        )
        cmd = [args.blender, "--background", "--factory-startup", "--python", str(worker_path)]
        if args.overwrite:
            cmd += ["--", "--overwrite"]
        print(f"[master] start worker={worker_id} objects={len(shard)}", flush=True)
        procs.append((worker_id, subprocess.Popen(cmd, env=env)))

    code = 0
    for worker_id, proc in procs:
        rc = proc.wait()
        print(f"[master] worker={worker_id} exited code={rc}", flush=True)
        if rc != 0:
            code = rc

    png_count = len(list((root / "rendering_random_24views").glob("*/*_depth.png")))
    print(f"[master] depth_png={png_count} expected={len(uids) * 24}", flush=True)
    return code or (0 if png_count >= len(uids) * 24 else 1)


if __name__ == "__main__":
    raise SystemExit(main())
