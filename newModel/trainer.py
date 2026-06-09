"""Training loop orchestration."""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .checkpoints import load_checkpoint_if_requested, save_checkpoint
from .data import RenderPairDataset, collate_batch, read_dataset, read_excluded_uids, split_uids
from .debug_views import save_train_step_preview
from .deps import ensure_deps
from .losses import build_perceptual_model, geometry_density_loss, perceptual_patch_loss, render_losses, stage_weights
from .model import build_model, make_optimizer, set_encoder_trainable
from .preview import save_validation_preview
from .rendering import render_rays


def setup_device(args):
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    ensure_deps(args.skip_install)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        world = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return DDP, distributed, rank, local_rank, world, device


def prepare_dataset(args, rank: int):
    dataset_root = Path(args.dataset_root).resolve()
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    excluded = read_excluded_uids(args.exclude_uids)
    grouped = read_dataset(dataset_root, excluded)
    if args.max_objects > 0:
        selected_uids = sorted(grouped.keys())[: args.max_objects]
        grouped = {uid: grouped[uid] for uid in selected_uids}
    train_uids, val_uids, test_uids = split_uids(dataset_root, grouped, args.seed, args.train_ratio, args.val_ratio)

    if rank == 0:
        print(f"Dataset root: {dataset_root}", flush=True)
        print(f"Excluded UIDs: {len(excluded)}", flush=True)
        print(f"Objects train={len(train_uids)} val={len(val_uids)} test={len(test_uids)}", flush=True)
        print(f"Views total={sum(len(v) for v in grouped.values())}", flush=True)
        (work_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        (work_dir / "splits_used.json").write_text(
            json.dumps({"train": train_uids, "val": val_uids, "test": test_uids}, indent=2),
            encoding="utf-8",
        )
    return dataset_root, work_dir, grouped, train_uids, val_uids, test_uids


def build_loaders(args, grouped, train_uids, val_uids, distributed: bool, world: int, rank: int):
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    train_ds = RenderPairDataset(grouped, train_uids, args, training=True)
    val_ds = RenderPairDataset(grouped, val_uids, args, training=False)
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=True,
        collate_fn=collate_batch,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


def train_one_epoch(
    model,
    train_loader,
    train_sampler,
    optimizer,
    scaler,
    perceptual_model,
    args,
    epoch: int,
    device,
    amp_dtype,
    rank: int,
    work_dir: Path,
):
    import torch

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    model.train()
    stage = stage_weights(args, epoch)
    optimizer.zero_grad(set_to_none=True)
    train_losses = []
    train_parts = defaultdict(list)
    start = time.time()

    for step, batch in enumerate(train_loader, start=1):
        source = batch["source_image"].to(device, non_blocking=True)
        rays_o = batch["rays_o"].to(device, non_blocking=True)
        rays_d = batch["rays_d"].to(device, non_blocking=True)
        target_rgb = batch["target_rgb"].to(device, non_blocking=True)
        target_mask = batch["target_mask"].to(device, non_blocking=True)
        geo_query = batch["geo_query"].to(device, non_blocking=True)
        geo_target = batch["geo_target"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(args.amp != "none" and device.type == "cuda")):
            planes = model(source)
            module = model.module if hasattr(model, "module") else model
            pred_rgb, pred_mask, pred_depth, weights = render_rays(module, planes, rays_o, rays_d, args, training=True)

        loss, parts = render_losses(
            pred_rgb.float(),
            pred_mask.float(),
            pred_depth.float(),
            weights.float(),
            target_rgb.float(),
            target_mask.float(),
            args,
            stage,
        )
        if perceptual_model is not None and stage.get("perceptual", 1.0) > 0:
            perc_loss = perceptual_patch_loss(perceptual_model, pred_rgb.float(), target_rgb.float(), args.patch_size)
            loss = loss + args.perceptual_weight * stage["perceptual"] * perc_loss
            parts["perc"] = perc_loss.detach()
        else:
            parts["perc"] = loss.detach() * 0.0

        geo_loss = geometry_density_loss(module, planes.float(), geo_query.float(), geo_target.float())
        loss = loss + args.geometry_weight * stage.get("geometry", 1.0) * geo_loss
        parts["geo"] = geo_loss.detach()

        real_loss_for_preview = float(loss.detach().cpu())
        if rank == 0 and args.train_preview_every > 0 and (step == 1 or step % args.train_preview_every == 0):
            preview_path = save_train_step_preview(
                work_dir=work_dir,
                epoch=epoch,
                step=step,
                batch=batch,
                pred_rgb=pred_rgb.float(),
                pred_mask=pred_mask.float(),
                loss_value=real_loss_for_preview,
                parts=parts,
                patch_size=args.patch_size,
                preview_size=args.train_preview_size,
                inline_display=args.inline_train_preview,
            )
            print(f"train step preview saved: {preview_path}", flush=True)

        loss = loss / args.grad_accum

        scaler.scale(loss).backward()
        if step % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        real_loss = float(loss.detach().cpu()) * args.grad_accum
        train_losses.append(real_loss)
        for key, value in parts.items():
            train_parts[key].append(float(value.cpu()))

        if rank == 0 and (step == 1 or step % args.log_every == 0 or step == len(train_loader)):
            sec = (time.time() - start) / step
            eta = sec * (len(train_loader) - step) / 60.0
            print(
                f"epoch={epoch:03d}/{args.epochs} step={step:05d}/{len(train_loader)} "
                f"loss={np.mean(train_losses[-args.log_every:]):.6f} "
                f"rgb={np.mean(train_parts['rgb'][-args.log_every:]):.5f} "
                f"mask={np.mean(train_parts['mask'][-args.log_every:]):.5f} "
                f"recall={np.mean(train_parts['recall'][-args.log_every:]):.5f} "
                f"geo={np.mean(train_parts['geo'][-args.log_every:]):.5f} "
                f"perc={np.mean(train_parts['perc'][-args.log_every:]):.5f} "
                f"sec/step={sec:.3f} eta_min={eta:.1f}",
                flush=True,
            )

    return train_losses, train_parts, stage, start


def validate(model, val_loader, perceptual_model, args, stage, device, amp_dtype):
    import torch

    model.eval()
    val_losses = []
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= args.val_batches:
                break
            source = batch["source_image"].to(device, non_blocking=True)
            rays_o = batch["rays_o"].to(device, non_blocking=True)
            rays_d = batch["rays_d"].to(device, non_blocking=True)
            target_rgb = batch["target_rgb"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            geo_query = batch["geo_query"].to(device, non_blocking=True)
            geo_target = batch["geo_target"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(args.amp != "none" and device.type == "cuda")):
                planes = model(source)
                module = model.module if hasattr(model, "module") else model
                pred_rgb, pred_mask, pred_depth, weights = render_rays(module, planes, rays_o, rays_d, args, training=False)
            loss, _ = render_losses(
                pred_rgb.float(),
                pred_mask.float(),
                pred_depth.float(),
                weights.float(),
                target_rgb.float(),
                target_mask.float(),
                args,
                stage,
            )
            if perceptual_model is not None and stage.get("perceptual", 1.0) > 0:
                loss = loss + args.perceptual_weight * stage["perceptual"] * perceptual_patch_loss(
                    perceptual_model, pred_rgb.float(), target_rgb.float(), args.patch_size
                )
            geo_loss = geometry_density_loss(module, planes.float(), geo_query.float(), geo_target.float())
            loss = loss + args.geometry_weight * stage.get("geometry", 1.0) * geo_loss
            val_losses.append(float(loss.detach().cpu()))
    return val_losses


def write_epoch_outputs(work_dir, model, optimizer, args, epoch, best_val, train_loss, val_loss, train_parts, epoch_min):
    epoch_rgb = float(np.mean(train_parts["rgb"])) if train_parts["rgb"] else 0.0
    epoch_mask = float(np.mean(train_parts["mask"])) if train_parts["mask"] else 0.0
    epoch_recall = float(np.mean(train_parts["recall"])) if train_parts["recall"] else 0.0
    epoch_geo = float(np.mean(train_parts["geo"])) if train_parts["geo"] else 0.0
    epoch_perc = float(np.mean(train_parts["perc"])) if train_parts["perc"] else 0.0
    print(
        f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
        f"rgb={epoch_rgb:.5f} mask={epoch_mask:.5f} recall={epoch_recall:.5f} "
        f"geo={epoch_geo:.5f} perc={epoch_perc:.5f} epoch_min={epoch_min:.1f}",
        flush=True,
    )
    with open(work_dir / "train_log.csv", "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "val_loss", "train_rgb", "train_mask", "train_recall", "train_geo", "train_perc", "epoch_min"],
        )
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rgb": epoch_rgb,
                "train_mask": epoch_mask,
                "train_recall": epoch_recall,
                "train_geo": epoch_geo,
                "train_perc": epoch_perc,
                "epoch_min": epoch_min,
            }
        )
    save_checkpoint(work_dir / "latest.pt", model, optimizer, epoch, args, best_val)


def train(args):
    import torch

    DDP, distributed, rank, local_rank, world, device = setup_device(args)
    dataset_root, work_dir, grouped, train_uids, val_uids, _ = prepare_dataset(args, rank)

    if rank == 0:
        print(f"Preparing DINOv2 weights/cache: {args.dinov2_model}", flush=True)
        _ = torch.hub.load("facebookresearch/dinov2", args.dinov2_model)
        del _
    if distributed:
        torch.distributed.barrier()

    resume_epoch = 0
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if resume_path.exists():
            resume_epoch = int(torch.load(resume_path, map_location="cpu").get("epoch", 0))

    encoder_trainable = args.encoder_lr > 0 and (resume_epoch + 1) >= args.unfreeze_encoder_epoch
    model = build_model(args).to(device)
    set_encoder_trainable(model, encoder_trainable)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    train_loader, val_loader, train_sampler = build_loaders(args, grouped, train_uids, val_uids, distributed, world, rank)
    optimizer = make_optimizer(model, args, encoder_trainable=encoder_trainable)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device.type == "cuda"))
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    perceptual_model = build_perceptual_model(args, device, rank)
    start_epoch, best_val = load_checkpoint_if_requested(args.resume_checkpoint, model, optimizer, device, rank)

    if rank == 0:
        print(f"Device={device} world={world}", flush=True)
        print(f"Training started. steps_per_epoch={len(train_loader)} patch={args.patch_size} samples={args.samples_per_ray}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.unfreeze_encoder_epoch and not encoder_trainable:
            if rank == 0:
                print(f"Unfreezing DINOv2 backbone at epoch {epoch}", flush=True)
            set_encoder_trainable(model, True)
            optimizer = make_optimizer(model, args, encoder_trainable=True)
            encoder_trainable = True

        train_losses, train_parts, stage, start = train_one_epoch(
            model, train_loader, train_sampler, optimizer, scaler, perceptual_model, args, epoch, device, amp_dtype, rank, work_dir
        )
        if distributed:
            torch.distributed.barrier()

        val_losses = validate(model, val_loader, perceptual_model, args, stage, device, amp_dtype)
        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if rank == 0:
            epoch_min = (time.time() - start) / 60.0
            write_epoch_outputs(work_dir, model, optimizer, args, epoch, best_val, train_loss, val_loss, train_parts, epoch_min)
            preview_uids = train_uids if args.preview_train else val_uids
            save_validation_preview(model, grouped, preview_uids, args, work_dir, epoch, device)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(work_dir / "best.pt", model, optimizer, epoch, args, best_val)
                print(f"saved best checkpoint: {work_dir / 'best.pt'}", flush=True)

    if distributed:
        torch.distributed.destroy_process_group()
