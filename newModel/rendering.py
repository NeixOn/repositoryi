"""Differentiable ray marcher."""

from __future__ import annotations

from .constants import BOX_MAX, BOX_MIN


def render_rays(model, planes, rays_o, rays_d, args, training: bool):
    import torch

    b, n, _ = rays_o.shape
    t_vals = torch.linspace(args.near, args.far, args.samples_per_ray, device=rays_o.device, dtype=rays_o.dtype)
    if training and args.ray_jitter:
        mids = 0.5 * (t_vals[:-1] + t_vals[1:])
        upper = torch.cat([mids, t_vals[-1:]], dim=0)
        lower = torch.cat([t_vals[:1], mids], dim=0)
        t_vals = lower + (upper - lower) * torch.rand((b, n, args.samples_per_ray), device=rays_o.device, dtype=rays_o.dtype)
    else:
        t_vals = t_vals.view(1, 1, -1).expand(b, n, -1)

    pts = rays_o[:, :, None, :] + rays_d[:, :, None, :] * t_vals[..., None]
    flat_pts = pts.reshape(b, n * args.samples_per_ray, 3)
    rgb, sigma = model.decoder(planes, flat_pts)
    rgb = rgb.reshape(b, n, args.samples_per_ray, 3)
    sigma = sigma.reshape(b, n, args.samples_per_ray, 1)

    box_min = rays_o.new_tensor(BOX_MIN).view(1, 1, 1, 3)
    box_max = rays_o.new_tensor(BOX_MAX).view(1, 1, 1, 3)
    inside = ((pts >= box_min) & (pts <= box_max)).all(dim=-1, keepdim=True)
    sigma = sigma * inside

    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    last = torch.full_like(deltas[..., :1], 1e10)
    deltas = torch.cat([deltas, last], dim=-1)[..., None]
    alpha = 1.0 - torch.exp(-sigma * deltas)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1, :]), 1.0 - alpha + 1e-6], dim=2), dim=2)[..., :-1, :]
    weights = alpha * trans
    color = (weights * rgb).sum(dim=2)
    acc = weights.sum(dim=2).clamp(0.0, 1.0)
    bg = rays_o.new_tensor(args.background).view(1, 1, 3)
    color = color + (1.0 - acc) * bg
    depth = (weights.squeeze(-1) * t_vals).sum(dim=2) / (acc.squeeze(-1) + 1e-6)
    return color, acc, depth, weights.squeeze(-1)

