"""Photometric, mask, geometry, and perceptual losses."""

from __future__ import annotations


def render_losses(pred_rgb, pred_mask, pred_depth, weights, target_rgb, target_mask, args, stage=None):
    import torch
    import torch.nn.functional as F

    fg_weight = args.background_rgb_weight * (1.0 - target_mask) + args.foreground_rgb_weight * target_mask
    rgb_diff = torch.sqrt((pred_rgb - target_rgb).pow(2) + args.charbonnier_eps)
    rgb_loss = (rgb_diff * fg_weight).sum() / (fg_weight.sum() * pred_rgb.shape[-1] + 1e-6)

    mask_bce_weight = args.background_mask_weight * (1.0 - target_mask) + args.foreground_mask_weight * target_mask
    bce_raw = F.binary_cross_entropy(pred_mask.clamp(1e-4, 1.0 - 1e-4), target_mask, reduction="none")
    bce = (bce_raw * mask_bce_weight).sum() / (mask_bce_weight.sum() + 1e-6)
    inter = (pred_mask * target_mask).sum(dim=1)
    dice = 1.0 - (2.0 * inter + 1.0) / (pred_mask.sum(dim=1) + target_mask.sum(dim=1) + 1.0)
    mask_loss = bce + dice.mean()

    fg_denom = target_mask.sum(dim=1).clamp_min(1.0)
    fg_recall_loss = (((1.0 - pred_mask).clamp_min(0.0) * target_mask).sum(dim=1) / fg_denom).mean()
    opacity_loss = (pred_mask * (1.0 - target_mask)).mean()
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    idx = torch.arange(weights.shape[-1], device=weights.device, dtype=weights.dtype)
    mean_t = (weights * idx).sum(dim=-1, keepdim=True) / weights_sum
    distortion = (weights * (idx - mean_t).abs()).mean()

    stage = stage or {}
    total = (
        rgb_loss
        + args.mask_weight * stage.get("mask", 1.0) * mask_loss
        + args.mask_recall_weight * stage.get("mask", 1.0) * fg_recall_loss
        + args.opacity_weight * stage.get("opacity", 1.0) * opacity_loss
        + args.distortion_weight * stage.get("distortion", 1.0) * distortion
    )
    return total, {
        "rgb": rgb_loss.detach(),
        "mask": mask_loss.detach(),
        "recall": fg_recall_loss.detach(),
        "opacity": opacity_loss.detach(),
        "distortion": distortion.detach(),
    }


def geometry_density_loss(model, planes, geo_query, geo_target):
    import torch
    import torch.nn.functional as F

    _, sigma = model.decoder(planes, geo_query)
    pred = 1.0 - torch.exp(-sigma * 0.08)
    bce = F.binary_cross_entropy(pred.float().clamp(1e-4, 1.0 - 1e-4), geo_target.float())
    surface = pred[geo_target > 0.95]
    empty = pred[geo_target < 0.05]
    surface_loss = (1.0 - surface).abs().mean() if surface.numel() else bce * 0.0
    empty_loss = empty.abs().mean() if empty.numel() else bce * 0.0
    return bce + 0.25 * surface_loss + 0.25 * empty_loss


def stage_weights(args, epoch: int) -> dict:
    if epoch <= args.geometry_warmup_epochs:
        return {
            "mask": args.warmup_mask_mult,
            "opacity": args.warmup_opacity_mult,
            "distortion": 0.0,
            "geometry": args.warmup_geometry_mult,
            "perceptual": 0.0,
        }
    return {"mask": 1.0, "opacity": 1.0, "distortion": 1.0, "geometry": 1.0, "perceptual": 1.0}


def build_perceptual_model(args, device, rank: int):
    if args.perceptual_weight <= 0:
        return None
    import torch
    import torchvision

    try:
        weights = torchvision.models.VGG16_Weights.IMAGENET1K_FEATURES
        model = torchvision.models.vgg16(weights=weights).features[:16].to(device).eval()
    except Exception as exc:
        if rank == 0:
            print(f"VGG perceptual loss disabled: {exc}", flush=True)
        return None
    for p in model.parameters():
        p.requires_grad = False
    if rank == 0:
        print("VGG perceptual loss enabled.", flush=True)
    return model


def perceptual_patch_loss(perceptual_model, pred_rgb, target_rgb, patch_size: int):
    import torch
    import torch.nn.functional as F

    b = pred_rgb.shape[0]
    pred = pred_rgb.view(b, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous()
    target = target_rgb.view(b, patch_size, patch_size, 3).permute(0, 3, 1, 2).contiguous()
    if patch_size < 64:
        pred = F.interpolate(pred, size=(64, 64), mode="bilinear", align_corners=False)
        target = F.interpolate(target, size=(64, 64), mode="bilinear", align_corners=False)
    mean = pred.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = pred.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        target_feat = perceptual_model((target - mean) / std)
    pred_feat = perceptual_model((pred - mean) / std)
    return F.l1_loss(pred_feat, target_feat)
