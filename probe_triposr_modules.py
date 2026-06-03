#!/usr/bin/env python3
"""
Inspect official TripoSR modules before writing the fine-tuning loop.

Run after setup_triposr_finetune_env.sh:
  source /data/venv/bin/activate
  PYTHONPATH=/data/TripoSR python /data/repositoryi/probe_triposr_modules.py \
    --model_name stabilityai/TripoSR \
    --out /data/runs/triposr_probe/modules.json

This script does not train. It records exact module names/shapes so the
fine-tuning script can freeze/unfreeze the right components for the installed
TripoSR version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="stabilityai/TripoSR")
    parser.add_argument("--out", default="/data/runs/triposr_probe/modules.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from tsr.system import TSR

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TSR.from_pretrained(args.model_name, config_name="config.yaml", weight_name="model.ckpt")
    model = model.to(device)
    model.eval()

    modules = []
    params = []
    for name, module in model.named_modules():
        direct_params = sum(p.numel() for p in module.parameters(recurse=False))
        if direct_params:
            modules.append({"name": name, "class": module.__class__.__name__, "direct_params": int(direct_params)})
    for name, param in model.named_parameters():
        params.append({"name": name, "shape": list(param.shape), "requires_grad": bool(param.requires_grad)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"modules": modules, "parameters": params}, indent=2), encoding="utf-8")
    print(f"saved module probe: {out}")
    print("top modules:")
    for item in modules[:50]:
        print(f"{item['name']}: {item['class']} params={item['direct_params']}")


if __name__ == "__main__":
    main()
