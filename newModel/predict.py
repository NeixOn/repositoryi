"""Single-image mesh prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .constants import BOX_MAX, BOX_MIN
from .data import load_source_image
from .deps import ensure_deps
from .model import build_model


def predict(args):
    import torch
    import trimesh
    from skimage import measure

    ensure_deps(args.skip_install)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = argparse.Namespace(**ckpt["args"])
    for key in ("image", "mask", "output_dir", "checkpoint", "grid_resolution", "sigma_level", "predict_chunk", "skip_install"):
        setattr(saved_args, key, getattr(args, key, getattr(saved_args, key, None)))

    model = build_model(saved_args).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    source = load_source_image(args.image, args.mask or args.image, saved_args.image_size, saved_args.crop)
    source_t = torch.from_numpy(source[None]).to(device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs = np.linspace(BOX_MIN[0], BOX_MAX[0], args.grid_resolution, dtype=np.float32)
    ys = np.linspace(BOX_MIN[1], BOX_MAX[1], args.grid_resolution, dtype=np.float32)
    zs = np.linspace(BOX_MIN[2], BOX_MAX[2], args.grid_resolution, dtype=np.float32)
    field = np.empty((args.grid_resolution, args.grid_resolution, args.grid_resolution), dtype=np.float32)
    with torch.no_grad():
        planes = model.planes(source_t)
        for zi, z in enumerate(zs):
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            pts = np.stack([grid_x.reshape(-1), grid_y.reshape(-1), np.full(grid_x.size, z, dtype=np.float32)], axis=-1)
            vals = []
            for start in range(0, len(pts), args.predict_chunk):
                p = torch.from_numpy(pts[start : start + args.predict_chunk][None]).to(device)
                _, sigma = model.decoder(planes, p)
                vals.append(sigma.squeeze(0).squeeze(-1).float().cpu().numpy())
            field[:, :, zi] = np.concatenate(vals, axis=0).reshape(args.grid_resolution, args.grid_resolution)

    np.save(out_dir / "density_grid.npy", field)
    verts, faces, normals, _ = measure.marching_cubes(
        field,
        level=args.sigma_level,
        spacing=(
            (BOX_MAX[0] - BOX_MIN[0]) / (args.grid_resolution - 1),
            (BOX_MAX[1] - BOX_MIN[1]) / (args.grid_resolution - 1),
            (BOX_MAX[2] - BOX_MIN[2]) / (args.grid_resolution - 1),
        ),
    )
    verts[:, 0] += BOX_MIN[0]
    verts[:, 1] += BOX_MIN[1]
    verts[:, 2] += BOX_MIN[2]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    mesh.export(out_dir / "mesh.obj")
    mesh.export(out_dir / "mesh.ply")
    print(f"saved: {out_dir / 'mesh.obj'}")

