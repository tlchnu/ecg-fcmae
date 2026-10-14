#!/usr/bin/env python
"""Fuse reparameterizable large-kernel branches and export an inference checkpoint."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_project.models import ECGConvNeXt1D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_config" not in checkpoint or "model" not in checkpoint:
        raise ValueError("Expected a downstream checkpoint with model_config and model")
    model_config = dict(checkpoint["model_config"])
    model_config["deploy"] = False
    model = ECGConvNeXt1D(**model_config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.switch_reparameterizable_layers_to_deploy()

    output = {
        "model": model.state_dict(),
        "model_config": model.model_config,
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    print(f"Deploy checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
