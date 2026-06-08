"""Dependency bootstrap helpers."""

from __future__ import annotations

import subprocess
import sys


def ensure_deps(skip_install: bool) -> None:
    if skip_install:
        return
    pkgs = [
        "torch",
        "torchvision",
        "Pillow",
        "numpy",
        "scikit-image",
        "trimesh",
        "tqdm",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore", *pkgs], check=True)

