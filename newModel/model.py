"""DINOv2 encoder, triplane generator, and radiance/density decoder."""

from __future__ import annotations

from .constants import BOX_MAX, BOX_MIN


def build_model(args):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class DINOv2Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.hub.load("facebookresearch/dinov2", args.dinov2_model)
            dims = {
                "dinov2_vits14": 384,
                "dinov2_vitb14": 768,
                "dinov2_vitl14": 1024,
                "dinov2_vitg14": 1536,
                "dinov2_vits14_reg": 384,
                "dinov2_vitb14_reg": 768,
                "dinov2_vitl14_reg": 1024,
                "dinov2_vitg14_reg": 1536,
            }
            in_dim = dims.get(args.dinov2_model, 1024)
            self.proj = nn.Sequential(
                nn.Linear(in_dim, args.latent_dim),
                nn.LayerNorm(args.latent_dim),
                nn.GELU(),
                nn.Linear(args.latent_dim, args.latent_dim),
            )

        def forward(self, image):
            mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            feat = self.backbone((image - mean) / std)
            if isinstance(feat, dict):
                feat = feat.get("x_norm_clstoken", next(iter(feat.values())))
            return self.proj(feat)

    class TriplaneGenerator(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = args.plane_channels
            self.s = args.plane_size
            self.fc = nn.Sequential(nn.Linear(args.latent_dim, 3 * self.c * 8 * 8), nn.GELU())
            blocks = []
            size = 8
            while size < self.s:
                blocks += [
                    nn.ConvTranspose2d(self.c, self.c, 4, 2, 1),
                    nn.GroupNorm(min(16, self.c), self.c),
                    nn.SiLU(),
                    nn.Conv2d(self.c, self.c, 3, padding=1),
                    nn.GroupNorm(min(16, self.c), self.c),
                    nn.SiLU(),
                ]
                size *= 2
            self.up = nn.Sequential(*blocks)

        def forward(self, latent):
            b = latent.shape[0]
            x = self.fc(latent).view(b * 3, self.c, 8, 8)
            x = self.up(x)
            if x.shape[-1] != self.s or x.shape[-2] != self.s:
                x = F.interpolate(x, size=(self.s, self.s), mode="bilinear", align_corners=False)
            return x.view(b, 3, self.c, self.s, self.s)

    class RadianceDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            in_dim = args.plane_channels * 3 + 3
            layers = []
            dim = in_dim
            for _ in range(args.decoder_layers):
                layers += [nn.Linear(dim, args.decoder_hidden), nn.SiLU()]
                dim = args.decoder_hidden
            self.net = nn.Sequential(*layers)
            self.sigma = nn.Linear(dim, 1)
            self.rgb = nn.Linear(dim, 3)
            nn.init.constant_(self.sigma.bias, args.sigma_init_bias)

        def sample_planes(self, planes, pts):
            b, _, _, _, _ = planes.shape
            x = pts[..., 0].clamp(BOX_MIN[0], BOX_MAX[0]) / abs(BOX_MIN[0])
            y = pts[..., 1].clamp(BOX_MIN[1], BOX_MAX[1]) / abs(BOX_MIN[1])
            z = ((pts[..., 2].clamp(BOX_MIN[2], BOX_MAX[2]) - BOX_MIN[2]) / (BOX_MAX[2] - BOX_MIN[2])) * 2.0 - 1.0
            grids = [torch.stack([x, y], dim=-1), torch.stack([x, z], dim=-1), torch.stack([y, z], dim=-1)]
            feats = []
            for i, grid in enumerate(grids):
                sampled = F.grid_sample(
                    planes[:, i],
                    grid.view(b, -1, 1, 2),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                feats.append(sampled.squeeze(-1).transpose(1, 2))
            return torch.cat(feats + [pts], dim=-1)

        def forward(self, planes, pts):
            h = self.net(self.sample_planes(planes, pts))
            sigma = F.softplus(self.sigma(h) + args.sigma_activation_bias)
            rgb = torch.sigmoid(self.rgb(h))
            return rgb, sigma

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DINOv2Encoder()
            self.triplane = TriplaneGenerator()
            self.decoder = RadianceDecoder()

        def planes(self, image):
            return self.triplane(self.encoder(image))

        def forward(self, image):
            return self.planes(image)

    return Model()


def set_encoder_trainable(model, trainable: bool):
    module = model.module if hasattr(model, "module") else model
    for p in module.encoder.backbone.parameters():
        p.requires_grad = trainable


def make_optimizer(model, args, encoder_trainable: bool):
    import torch

    module = model.module if hasattr(model, "module") else model
    encoder_params = list(module.encoder.backbone.parameters()) if encoder_trainable else []
    other_params = [p for n, p in module.named_parameters() if not n.startswith("encoder.backbone.") and p.requires_grad]
    groups = [{"params": other_params, "lr": args.lr}]
    if encoder_params:
        groups.append({"params": encoder_params, "lr": args.encoder_lr})
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))

