"""Differentiable ray marcher."""

from __future__ import annotations

from .constants import BOX_MAX, BOX_MIN


def render_rays(model, planes, rays_o, rays_d, args, training: bool):
    import torch

    b, n, _ = rays_o.shape
    box_min = rays_o.new_tensor(BOX_MIN).view(1, 1, 3)
    box_max = rays_o.new_tensor(BOX_MAX).view(1, 1, 3)
    safe_d = torch.where(rays_d.abs() < 1e-6, torch.full_like(rays_d, 1e-6), rays_d)
    inv_d = 1.0 / safe_d
    t0 = (box_min - rays_o) * inv_d
    t1 = (box_max - rays_o) * inv_d
    t_near = torch.minimum(t0, t1).amax(dim=-1)
    t_far = torch.maximum(t0, t1).amin(dim=-1)
    hit = t_far > torch.clamp(t_near, min=0.0)
    t_start = torch.where(hit, torch.clamp(t_near, min=0.0), torch.full_like(t_near, args.near))
    t_end = torch.where(hit, t_far, torch.full_like(t_far, args.far))

    base = torch.linspace(0.0, 1.0, args.samples_per_ray, device=rays_o.device, dtype=rays_o.dtype)
    if training and args.ray_jitter:
        step = 1.0 / max(1, args.samples_per_ray - 1)
        base = (base.view(1, 1, -1) + (torch.rand((b, n, args.samples_per_ray), device=rays_o.device, dtype=rays_o.dtype) - 0.5) * step).clamp(0.0, 1.0)
    else:
        base = base.view(1, 1, -1).expand(b, n, -1)
    t_vals = t_start[..., None] + (t_end - t_start)[..., None].clamp_min(1e-4) * base

    pts = rays_o[:, :, None, :] + rays_d[:, :, None, :] * t_vals[..., None]
    flat_pts = pts.reshape(b, n * args.samples_per_ray, 3)
    rgb, sigma = model.decoder(planes, flat_pts)
    rgb = rgb.reshape(b, n, args.samples_per_ray, 3)
    sigma = sigma.reshape(b, n, args.samples_per_ray, 1)
    sigma = sigma * hit[..., None, None]

    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    last = deltas[..., -1:].clamp_min(1e-4)
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
