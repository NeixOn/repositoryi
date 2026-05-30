#!/usr/bin/env python3
"""
Run chair LRM-lite inference on one arbitrary image.

Example:
  !python /kaggle/working/repositoryi/predict_chair_lrm.py \
    --image /kaggle/input/my-chair/chair.png \
    --checkpoint /kaggle/working/chair_lrm_tpu/best_orbax \
    --output_dir /kaggle/working/chair_lrm_custom \
    --pred_points 8192
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


def ensure_deps(skip_install: bool) -> None:
    if skip_install:
        return
    pkgs = ["flax", "optax", "orbax-checkpoint", "Pillow", "open3d"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def load_image(path: Path, image_size: int, crop: bool):
    from PIL import Image

    original = Image.open(path).convert("RGB")
    cropped = original
    if crop:
        arr = np.asarray(cropped, dtype=np.uint8)
        # If there is a light/white background, crop to non-background pixels.
        mask = np.any(arr < 245, axis=-1)
        if mask.any():
            ys, xs = np.where(mask)
            pad = int(max(arr.shape[0], arr.shape[1]) * 0.05)
            x0 = max(0, int(xs.min()) - pad)
            x1 = min(arr.shape[1], int(xs.max()) + pad + 1)
            y0 = max(0, int(ys.min()) - pad)
            y1 = min(arr.shape[0], int(ys.max()) + pad + 1)
            cropped = cropped.crop((x0, y0, x1, y1))
    resized = cropped.resize((image_size, image_size))
    image = np.asarray(resized, dtype=np.float32) / 255.0
    return image, original, cropped, resized


def write_ply(path: Path, points: np.ndarray, color=(35, 105, 210)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = np.asarray(color, dtype=np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(color[0])} {int(color[1])} {int(color[2])}\n")


def write_mesh_from_points(
    points: np.ndarray,
    mesh_ply_path: Path,
    mesh_obj_path: Path,
    poisson_depth: int,
    density_quantile: float,
    smooth_iterations: int,
) -> tuple[Path, Path]:
    import open3d as o3d

    mesh_ply_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.14, 0.42, 0.82]], dtype=np.float64), (len(points), 1)))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.18, max_nn=48)
    )
    pcd.orient_normals_consistent_tangent_plane(32)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=poisson_depth,
        width=0,
        scale=1.08,
        linear_fit=False,
    )
    densities = np.asarray(densities)
    if len(densities) > 0:
        threshold = np.quantile(densities, density_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = bbox.scale(1.08, bbox.get_center())
    mesh = mesh.crop(bbox)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if smooth_iterations > 0:
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)
    mesh.compute_vertex_normals()

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError("Mesh reconstruction produced an empty mesh")
    o3d.io.write_triangle_mesh(str(mesh_ply_path), mesh, write_ascii=False)
    o3d.io.write_triangle_mesh(str(mesh_obj_path), mesh, write_ascii=True)
    return mesh_ply_path, mesh_obj_path


def render_point_preview(points: np.ndarray, size: int):
    from PIL import Image

    img = np.full((size, size, 3), 245, dtype=np.uint8)
    x = (points[:, 0] + 1.05) / 2.10
    y = 1.0 - (points[:, 2] / 1.90)
    px = np.clip((x * (size - 1)).astype(np.int32), 0, size - 1)
    py = np.clip((y * (size - 1)).astype(np.int32), 0, size - 1)
    order = np.argsort(points[:, 1])
    for idx in order:
        cx = int(px[idx])
        cy = int(py[idx])
        img[max(0, cy - 1):min(size, cy + 2), max(0, cx - 1):min(size, cx + 2), :] = (35, 105, 210)
    return Image.fromarray(img, mode="RGB")


def fit_preview(img, size):
    from PIL import Image

    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    work = img.copy()
    work.thumbnail((size, size), Image.BILINEAR)
    x = (size - work.width) // 2
    y = (size - work.height) // 2
    canvas.paste(work, (x, y))
    return canvas


def save_preview(path: Path, original_img, cropped_img, resized_img, pred: np.ndarray) -> None:
    from PIL import Image, ImageDraw

    panel = Image.new("RGB", (1024, 288), (255, 255, 255))
    panel.paste(fit_preview(original_img, 256), (0, 24))
    panel.paste(fit_preview(cropped_img, 256), (256, 24))
    panel.paste(resized_img.resize((256, 256)), (512, 24))
    panel.paste(render_point_preview(pred, 256), (768, 24))
    draw = ImageDraw.Draw(panel)
    draw.text((8, 4), f"original {original_img.width}x{original_img.height}", fill=(0, 0, 0))
    draw.text((264, 4), f"after crop {cropped_img.width}x{cropped_img.height}", fill=(0, 0, 0))
    draw.text((520, 4), f"model input {resized_img.width}x{resized_img.height}", fill=(0, 0, 0))
    draw.text((776, 4), "prediction", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--skip_install", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()
    ensure_deps(pre_args.skip_install)

    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    from flax.training import train_state
    import optax
    import orbax.checkpoint as ocp

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="/kaggle/working/chair_lrm_custom")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--pred_points", type=int, default=8192)
    parser.add_argument("--crop", action="store_true", help="Crop light/white background around the object before resizing")
    parser.add_argument("--no_mesh", action="store_true", help="Only save point cloud, skip Poisson mesh reconstruction")
    parser.add_argument("--poisson_depth", type=int, default=8)
    parser.add_argument("--mesh_density_quantile", type=float, default=0.04)
    parser.add_argument("--mesh_smooth_iterations", type=int, default=1)
    parser.add_argument("--skip_install", action="store_true")
    args = parser.parse_args()

    class DownBlock(nn.Module):
        channels: int

        @nn.compact
        def __call__(self, x):
            x = nn.Conv(self.channels, (3, 3), strides=(2, 2), padding="SAME")(x)
            x = nn.LayerNorm(epsilon=1e-6)(x)
            x = nn.gelu(x)
            residual = x
            y = nn.Conv(self.channels, (3, 3), padding="SAME", feature_group_count=self.channels)(x)
            y = nn.LayerNorm(epsilon=1e-6)(y)
            y = nn.Dense(self.channels * 4)(y)
            y = nn.gelu(y)
            y = nn.Dense(self.channels)(y)
            return residual + y * 0.2

    def template_init(count: int):
        side = int(math.ceil(math.sqrt(count)))
        axis = jnp.linspace(-0.85, 0.85, side)
        gx, gy = jnp.meshgrid(axis, axis, indexing="ij")
        xy = jnp.stack([gx.reshape(-1), gy.reshape(-1)], axis=-1)[:count]
        ids = jnp.arange(count, dtype=jnp.float32)
        z_noise = jnp.mod(jnp.sin(ids * 12.9898) * 43758.5453, 1.0)
        z = 0.04 + 1.68 * z_noise[:, None]
        return jnp.concatenate([xy, z], axis=-1).astype(jnp.float32)

    class ChairLRMLite(nn.Module):
        pred_points: int

        @nn.compact
        def __call__(self, image, training: bool):
            x = image * 2.0 - 1.0
            x = nn.Conv(48, (5, 5), strides=(2, 2), padding="SAME")(x)
            x = nn.LayerNorm(epsilon=1e-6)(x)
            x = nn.gelu(x)
            x = DownBlock(96)(x)
            x = DownBlock(160)(x)
            x = DownBlock(256)(x)
            x = DownBlock(384)(x)
            latent = jnp.mean(x, axis=(1, 2))
            latent = nn.LayerNorm(epsilon=1e-6)(latent)
            latent = nn.Dense(768)(latent)
            latent = nn.gelu(latent)
            latent = nn.Dense(768)(latent)
            latent = nn.gelu(latent)
            template_raw = self.param("template", lambda key: template_init(self.pred_points))
            template = jnp.concatenate([
                jnp.clip(template_raw[:, 0:2], -0.95, 0.95),
                jnp.clip(template_raw[:, 2:3], 0.0, 1.85),
            ], axis=-1)
            q = nn.Dense(128)(template)
            q = nn.gelu(q)
            cond = nn.Dense(256)(latent)[:, None, :]
            y = jnp.broadcast_to(q[None, :, :], (image.shape[0], self.pred_points, 128))
            y = jnp.concatenate([y, jnp.broadcast_to(cond, (image.shape[0], self.pred_points, 256))], axis=-1)
            for width in (384, 384, 256):
                skip = nn.Dense(width)(y)
                y = nn.LayerNorm(epsilon=1e-6)(y)
                y = nn.Dense(width)(y)
                y = nn.gelu(y)
                y = nn.Dense(width)(y)
                y = y + skip * 0.5
            raw = nn.Dense(3, kernel_init=nn.initializers.normal(1e-4))(y)
            xy = jnp.clip(template[None, :, 0:2] + 0.35 * jnp.tanh(raw[:, :, 0:2]), -0.98, 0.98)
            z = jnp.clip(template[None, :, 2:3] + 0.45 * jnp.tanh(raw[:, :, 2:3]), 0.0, 1.85)
            return jnp.concatenate([xy, z], axis=-1)

    class State(train_state.TrainState):
        pass

    image, original_img, cropped_img, resized_img = load_image(Path(args.image), args.image_size, args.crop)
    model = ChairLRMLite(args.pred_points)
    variables = model.init(jax.random.PRNGKey(42), jnp.ones((1, args.image_size, args.image_size, 3), jnp.float32), training=False)
    schedule = optax.cosine_decay_schedule(5e-5, decay_steps=1, alpha=0.03)
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, b1=0.9, b2=0.95, weight_decay=5e-5),
    )
    state = State.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    state = ocp.PyTreeCheckpointer().restore(Path(args.checkpoint), item=state)

    pred = model.apply({"params": state.params}, jnp.asarray(image[None, ...], dtype=jnp.float32), training=False)
    pred = np.asarray(jax.device_get(pred[0]), dtype=np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    original_path = output_dir / f"{stem}_original.png"
    crop_path = output_dir / f"{stem}_after_crop.png"
    input_path = output_dir / f"{stem}_model_input_{args.image_size}.png"
    pred_path = output_dir / f"{stem}_pred.ply"
    npy_path = output_dir / f"{stem}_pred.npy"
    mesh_ply_path = output_dir / f"{stem}_mesh.ply"
    mesh_obj_path = output_dir / f"{stem}_mesh.obj"
    preview_path = output_dir / f"{stem}_preview.png"

    original_img.save(original_path)
    cropped_img.save(crop_path)
    resized_img.save(input_path)
    write_ply(pred_path, pred)
    np.save(npy_path, pred)
    mesh_ok = False
    if not args.no_mesh:
        try:
            write_mesh_from_points(
                pred,
                mesh_ply_path,
                mesh_obj_path,
                args.poisson_depth,
                args.mesh_density_quantile,
                args.mesh_smooth_iterations,
            )
            mesh_ok = True
        except Exception as exc:
            print(f"Mesh reconstruction failed, point cloud was still saved: {exc}", flush=True)
    save_preview(preview_path, original_img, cropped_img, resized_img, pred)

    print(f"Original: {original_path}", flush=True)
    print(f"After crop: {crop_path}", flush=True)
    print(f"Model input: {input_path}", flush=True)
    print(f"Prediction PLY: {pred_path}", flush=True)
    print(f"Prediction NPY: {npy_path}", flush=True)
    if mesh_ok:
        print(f"Mesh PLY: {mesh_ply_path}", flush=True)
        print(f"Mesh OBJ: {mesh_obj_path}", flush=True)
    print(f"Preview: {preview_path}", flush=True)


if __name__ == "__main__":
    main()
