"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path


def save_checkpoint(path: Path, model, optimizer, epoch: int, args, best_val: float):
    import torch

    module = model.module if hasattr(model, "module") else model
    payload = {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "best_val": best_val,
    }
    torch.save(payload, path)


def load_checkpoint_if_requested(path: str, model, optimizer, device, rank: int):
    import torch

    if not path:
        return 1, float("inf")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    ckpt = torch.load(p, map_location=device)
    module = model.module if hasattr(model, "module") else model
    module.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_val = float(ckpt.get("best_val", float("inf")))
    if rank == 0:
        print(f"resumed checkpoint: {p} start_epoch={start_epoch} best_val={best_val:.6f}", flush=True)
    return start_epoch, best_val

