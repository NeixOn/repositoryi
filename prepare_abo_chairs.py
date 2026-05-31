#!/usr/bin/env python3
"""
Prepare an ABO chair subset in the same layout as the Objaverse chair dataset.

The script downloads only selected Amazon Berkeley Objects assets instead of
pulling the full 300GB+ dataset. Output layout:

  output_dir/
    metadata/views.csv
    metadata/objects.csv
    objects/<uid>/normalized.glb
    objects/<uid>/points.npz
    renders/<uid>/view_000.png ...

It is intentionally conservative: it filters by chair-like metadata, downloads
matching GLB models and available ABO images, normalizes geometry, samples
surface points, and writes train-ready metadata.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterable

import numpy as np


ABO = "https://amazon-berkeley-objects.s3.amazonaws.com"
DEFAULT_INCLUDE = (
    "chair,stool,bar stool,armchair,office chair,dining chair,accent chair,"
    "folding chair,rocking chair,lounge chair,task chair,gaming chair"
)
DEFAULT_EXCLUDE = (
    "sofa,couch,loveseat,bench,table,desk,cabinet,shelf,bed,mattress,"
    "cover,slipcover,cushion,pillow,blanket,ottoman only,replacement"
)


def ensure_deps(skip_install: bool) -> None:
    if skip_install:
        return
    pkgs = ["pandas", "tqdm", "Pillow", "trimesh", "scipy", "awscli"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore", *pkgs], check=True)


def sync_s3_prefix(prefix: str, dst: Path) -> None:
    if dst.exists() and any(dst.rglob("*")):
        return
    dst.mkdir(parents=True, exist_ok=True)
    aws_bin = shutil.which("aws")
    if not aws_bin:
        candidate = Path(sys.executable).resolve().parent / "aws"
        aws_bin = str(candidate) if candidate.exists() else "aws"
    cmd = [
        aws_bin,
        "s3",
        "cp",
        "--no-sign-request",
        "--recursive",
        f"s3://amazon-berkeley-objects/{prefix.strip('/')}/",
        str(dst),
    ]
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download(url: str, dst: Path, retries: int = 5) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return True
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1024 * 1024)
            tmp.replace(dst)
            return True
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                print(f"[download] 404 not found: {url}", flush=True)
                return False
            if tmp.exists():
                tmp.unlink()
            print(f"[download] failed {attempt}/{retries}: {url} ({exc})", flush=True)
            time.sleep(min(30, 2 ** attempt))
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            print(f"[download] failed {attempt}/{retries}: {url} ({exc})", flush=True)
            time.sleep(min(30, 2 ** attempt))
    return False


def read_jsonl_file(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def flatten_text(value) -> str:
    parts = []
    if isinstance(value, dict):
        for v in value.values():
            parts.append(flatten_text(v))
    elif isinstance(value, list):
        for v in value:
            parts.append(flatten_text(v))
    elif value is not None:
        parts.append(str(value))
    return " ".join(parts)


def chair_score(listing: dict, include: list[str], exclude: list[str]) -> int:
    fields = [
        listing.get("item_name"),
        listing.get("product_type"),
        listing.get("product_description"),
        listing.get("bullet_point"),
        listing.get("node"),
        listing.get("style"),
    ]
    text = flatten_text(fields).lower()
    if any(x and x in text for x in exclude):
        return -999
    score = 0
    for kw in include:
        if kw and kw in text:
            score += 4 if "chair" in kw else 2
    if "seat" in text:
        score += 1
    return score


def get_first(value, keys: tuple[str, ...]):
    if not isinstance(value, dict):
        return None
    for key in keys:
        if key in value and value[key]:
            return value[key]
    return None


def load_listings(cache_dir: Path, include: list[str], exclude: list[str], max_candidates: int) -> list[dict]:
    listings_dir = cache_dir / "metadata" / "listings"
    sync_s3_prefix("listings/metadata", listings_dir)
    candidates = []
    listing_files = sorted(list(listings_dir.glob("*.json")) + list(listings_dir.glob("*.json.gz")))
    for i, path in enumerate(listing_files):
        print(f"Reading listings shard {i + 1}/{len(listing_files)}: {path.name}", flush=True)
        try:
            for listing in read_jsonl_file(path):
                model_id = get_first(listing, ("3dmodel_id", "model_id", "glb_id"))
                if not model_id:
                    continue
                score = chair_score(listing, include, exclude)
                if score <= 0:
                    continue
                asin = get_first(listing, ("item_id", "asin", "listing_id")) or model_id
                image_id = get_first(listing, ("main_image_id", "image_id"))
                candidates.append({
                    "uid": str(model_id),
                    "model_id": str(model_id),
                    "asin": str(asin),
                    "image_id": str(image_id) if image_id else "",
                    "spin_id": str(get_first(listing, ("spin_id",)) or ""),
                    "other_image_id": listing.get("other_image_id", []),
                    "score": score,
                    "title": flatten_text(listing.get("item_name"))[:300],
                })
        except Exception as exc:
            print(f"[warn] could not read {path}: {exc}", flush=True)
    candidates.sort(key=lambda x: (-x["score"], x["uid"]))
    dedup = {}
    for row in candidates:
        dedup.setdefault(row["uid"], row)
    rows = list(dedup.values())[:max_candidates]
    (cache_dir / "chair_candidates.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Chair-like candidates with 3D model id: {len(rows)}", flush=True)
    return rows


def load_metadata_csv(cache_dir: Path, prefix: str):
    import pandas as pd

    dst = cache_dir / "metadata" / prefix.replace("/", "_")
    sync_s3_prefix(f"{prefix}/metadata", dst)
    frames = []
    for path in sorted(list(dst.glob("*.csv")) + list(dst.glob("*.csv.gz"))):
        frames.append(pd.read_csv(path, compression="infer"))
    if not frames:
        raise RuntimeError(f"No metadata CSV files found for {prefix} under {dst}")
    return pd.concat(frames, ignore_index=True)


def best_path_column(df) -> str:
    candidates = ["path", "file_path", "relative_path", "s3_path", "key", "glb_path", "image_path"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        if "path" in col.lower() or "key" in col.lower():
            return col
    raise RuntimeError(f"Could not find path column in columns={list(df.columns)}")


def id_column(df, possible: tuple[str, ...]) -> str:
    for col in possible:
        if col in df.columns:
            return col
    for col in df.columns:
        lower = col.lower()
        if "model" in lower and "id" in lower:
            return col
    raise RuntimeError(f"Could not find id column in columns={list(df.columns)}")


def normalize_mesh(src: Path, dst: Path, points_path: Path, points: int, seed: int) -> tuple[int, float]:
    import trimesh

    scene = trimesh.load(src, force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene
    else:
        meshes = []
        for geom in scene.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.faces) > 0:
                meshes.append(geom)
        if not meshes:
            raise RuntimeError("no mesh geometry")
        mesh = trimesh.util.concatenate(meshes)
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) < 80:
        raise RuntimeError(f"too few faces: {len(mesh.faces)}")
    bounds = mesh.bounds.astype(np.float64)
    center = (bounds[0] + bounds[1]) * 0.5
    scale = float((bounds[1] - bounds[0]).max())
    if not math.isfinite(scale) or scale <= 1e-8:
        raise RuntimeError("bad mesh scale")
    mesh.vertices = (mesh.vertices - center) / scale
    bounds = mesh.bounds.astype(np.float64)
    zmin, zmax = bounds[0, 2], bounds[1, 2]
    mesh.vertices[:, 2] = (mesh.vertices[:, 2] - zmin) / max(1e-8, (zmax - zmin))
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


def copy_or_render_placeholder(image_paths: list[Path], out_dir: Path, max_views: int) -> list[Path]:
    from PIL import Image, ImageOps

    out_dir.mkdir(parents=True, exist_ok=True)
    views = []
    for i, src in enumerate(image_paths[:max_views]):
        try:
            img = Image.open(src).convert("RGB")
            img = ImageOps.contain(img, (512, 512))
            canvas = Image.new("RGB", (512, 512), (255, 255, 255))
            canvas.paste(img, ((512 - img.width) // 2, (512 - img.height) // 2))
            dst = out_dir / f"view_{i:03d}.png"
            canvas.save(dst)
            views.append(dst)
        except Exception:
            continue
    return views


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output_dir", default="/data/abo_chairs")
    parser.add_argument("--cache_dir", default="/data/abo_cache")
    parser.add_argument("--num_objects", type=int, default=3000)
    parser.add_argument("--max_candidates", type=int, default=12000)
    parser.add_argument("--views_per_object", type=int, default=24)
    parser.add_argument("--points", type=int, default=65536)
    parser.add_argument("--include_keywords", default=DEFAULT_INCLUDE)
    parser.add_argument("--exclude_keywords", default=DEFAULT_EXCLUDE)
    parser.add_argument("--max_total_gb", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_install", action="store_true")
    args = parser.parse_args()
    ensure_deps(args.skip_install)

    out = Path(args.output_dir)
    cache = Path(args.cache_dir)
    raw_models = cache / "raw_models"
    raw_images = cache / "raw_images"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    include = [x.strip().lower() for x in args.include_keywords.split(",") if x.strip()]
    exclude = [x.strip().lower() for x in args.exclude_keywords.split(",") if x.strip()]
    candidates = load_listings(cache, include, exclude, args.max_candidates)

    print("Loading ABO 3D model metadata...", flush=True)
    models_df = load_metadata_csv(cache, "3dmodels")
    model_id_col = id_column(models_df, ("3dmodel_id", "model_id", "uid"))
    model_path_col = best_path_column(models_df)
    model_paths = {str(r[model_id_col]): str(r[model_path_col]) for _, r in models_df.iterrows()}

    image_paths_by_model = {}
    try:
        print("Loading ABO image metadata...", flush=True)
        images_df = load_metadata_csv(cache, "images")
        img_path_col = best_path_column(images_df)
        img_id_col = id_column(images_df, ("image_id", "id"))
        image_path_by_id = {str(r[img_id_col]): str(r[img_path_col]) for _, r in images_df.iterrows()}
        for cand in candidates:
            ids = []
            if cand.get("image_id"):
                ids.append(cand["image_id"])
            other = cand.get("other_image_id", [])
            if isinstance(other, list):
                ids.extend(str(x) for x in other)
            for image_id in ids:
                if image_id in image_path_by_id:
                    image_paths_by_model.setdefault(cand["uid"], []).append(image_path_by_id[image_id])
    except Exception as exc:
        print(f"[warn] catalog image metadata unavailable: {exc}", flush=True)

    try:
        print("Loading ABO spin metadata...", flush=True)
        spins_df = load_metadata_csv(cache, "spins")
        spin_path_col = best_path_column(spins_df)
        spin_id_col = id_column(spins_df, ("spin_id",))
        spin_groups = {}
        for _, r in spins_df.iterrows():
            sid = str(r.get(spin_id_col, ""))
            path = str(r.get(spin_path_col, ""))
            if sid and path and sid != "nan" and path != "nan":
                spin_groups.setdefault(sid, []).append(path)
        for cand in candidates:
            sid = cand.get("spin_id", "")
            if sid and sid in spin_groups:
                image_paths_by_model.setdefault(cand["uid"], []).extend(spin_groups[sid])
    except Exception as exc:
        print(f"[warn] spin metadata unavailable: {exc}", flush=True)

    views_csv = out / "metadata" / "views.csv"
    objects_csv = out / "metadata" / "objects.csv"
    out.joinpath("metadata").mkdir(parents=True, exist_ok=True)
    accepted = 0
    total_bytes = 0
    with open(views_csv, "w", encoding="utf-8", newline="") as vf, open(objects_csv, "w", encoding="utf-8", newline="") as of:
        vw = csv.DictWriter(vf, fieldnames=["uid", "view_index", "image_path"])
        ow = csv.DictWriter(of, fieldnames=["uid", "source", "title", "faces", "scale"])
        vw.writeheader()
        ow.writeheader()
        for cand in candidates:
            if accepted >= args.num_objects:
                break
            uid = cand["uid"]
            rel_model = model_paths.get(uid)
            if not rel_model:
                continue
            model_url = f"{ABO}/3dmodels/original/{rel_model.lstrip('/')}"
            raw_model = raw_models / uid / Path(rel_model).name
            if not download(model_url, raw_model):
                continue
            total_bytes += raw_model.stat().st_size
            if total_bytes / (1024 ** 3) > args.max_total_gb:
                print(f"Disk budget reached: {total_bytes / (1024 ** 3):.1f} GB", flush=True)
                break
            try:
                dst_glb = out / "objects" / uid / "normalized.glb"
                pts = out / "objects" / uid / "points.npz"
                faces, scale = normalize_mesh(raw_model, dst_glb, pts, args.points, args.seed + accepted)
            except Exception as exc:
                print(f"[skip] {uid}: {exc}", flush=True)
                continue
            local_images = []
            for rel_img in image_paths_by_model.get(uid, [])[: args.views_per_object * 2]:
                rel_img = rel_img.lstrip("/")
                if rel_img.startswith("spins/") or rel_img.startswith("images/"):
                    url = f"{ABO}/{rel_img}"
                elif "/spin" in rel_img.lower() or rel_img.lower().startswith("spin"):
                    url = f"{ABO}/spins/original/{rel_img}"
                else:
                    url = f"{ABO}/images/small/{rel_img}"
                dst = raw_images / uid / Path(rel_img).name
                if download(url, dst, retries=2):
                    local_images.append(dst)
            views = copy_or_render_placeholder(local_images, out / "renders" / uid, args.views_per_object)
            if not views:
                print(f"[skip] {uid}: no usable images", flush=True)
                shutil.rmtree(out / "objects" / uid, ignore_errors=True)
                continue
            for vi, _ in enumerate(views):
                vw.writerow({"uid": uid, "view_index": vi, "image_path": f"renders/{uid}/view_{vi:03d}.png"})
            ow.writerow({"uid": uid, "source": "ABO", "title": cand["title"], "faces": faces, "scale": scale})
            vf.flush()
            of.flush()
            accepted += 1
            print(f"accepted={accepted}/{args.num_objects} uid={uid} views={len(views)}", flush=True)

    info = {"accepted": accepted, "dataset": "ABO chairs subset", "views_per_object": args.views_per_object, "points": args.points}
    (out / "metadata" / "dataset_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done. Accepted={accepted}. Output={out}", flush=True)


if __name__ == "__main__":
    main()
