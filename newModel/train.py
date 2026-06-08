#!/usr/bin/env python3
"""Entry point for the split newModel training/prediction package."""

from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

from newModel.args import parse_args
from newModel.predict import predict
from newModel.trainer import train


def main():
    args = parse_args()
    if args.mode == "train":
        if not args.dataset_root:
            raise ValueError("--dataset_root is required for training")
        train(args)
    else:
        if not args.checkpoint or not args.image:
            raise ValueError("--checkpoint and --image are required for predict")
        predict(args)


if __name__ == "__main__":
    main()

