"""Central model registry for reproducible architecture replacement."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .convnext import ECGConvNeXt1D


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "convnext_tiny": {
        "depths": (3, 3, 9, 3),
        "dims": (64, 128, 256, 512),
        "stage_kernel_sizes": (7, 7, 7, 7),
        "block_type": "convnext",
        "layer_scale_init": 0.0,
    },
    "convnext_layerscale": {
        "depths": (3, 3, 9, 3),
        "dims": (64, 128, 256, 512),
        "stage_kernel_sizes": (7, 7, 7, 7),
        "block_type": "convnext",
        "layer_scale_init": 1e-6,
    },
    "reparam_lk13": {
        "depths": (3, 3, 9, 3),
        "dims": (64, 128, 256, 512),
        "stage_kernel_sizes": (7, 13, 13, 13),
        "block_type": "reparam_lk",
        "layer_scale_init": 1e-6,
    },
    "reparam_lk31_mid": {
        "depths": (3, 3, 9, 3),
        "dims": (64, 128, 256, 512),
        "stage_kernel_sizes": (7, 13, 31, 13),
        "block_type": "reparam_lk",
        "layer_scale_init": 1e-6,
    },
    "inception1d": {
        "depths": (3, 3, 9, 3),
        "dims": (64, 128, 256, 512),
        "stage_kernel_sizes": (7, 7, 7, 7),
        "block_type": "inception1d",
        "layer_scale_init": 1e-6,
        "inception_kernels": (3, 7, 15),
        "inception_branch_ratio": 0.25,
    },
    "period_spectral_convnext": {
        "depths": (3, 3, 9, 3),
        "dims": (64, 128, 256, 512),
        "stage_kernel_sizes": (7, 7, 7, 7),
        "block_type": "period_spectral",
        "layer_scale_init": 1e-6,
    },
    "leadaware_convnext": {
        "depths": (3, 3, 9, 3),
        "dims": (64, 128, 256, 512),
        "stage_kernel_sizes": (7, 7, 7, 7),
        "block_type": "convnext",
        "layer_scale_init": 1e-6,
        "lead_stem": True,
        "lead_mixer_heads": 4,
        "lead_dropout": 0.1,
    },
}


def get_model_config(name: str, **overrides: Any) -> dict[str, Any]:
    """Return a detached preset dictionary with explicit overrides applied."""
    if name not in MODEL_PRESETS:
        raise KeyError(f"Unknown model preset {name!r}. Available: {sorted(MODEL_PRESETS)}")
    config = deepcopy(MODEL_PRESETS[name])
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def build_model(name: str = "convnext_tiny", **overrides: Any) -> ECGConvNeXt1D:
    """Build an ECG model from a named preset."""
    return ECGConvNeXt1D(**get_model_config(name, **overrides))
