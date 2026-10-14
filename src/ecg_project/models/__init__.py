"""Model registry and reusable ECG model components."""

from .blocks import ConvNeXtBlock1D
from .convnext import ECGConvNeXt1D
from .factory import MODEL_PRESETS, build_model, get_model_config
from .layers import DropPath, GRN, LayerNorm1D

Block = ConvNeXtBlock1D
ECGConvNeXtV3 = ECGConvNeXt1D
LayerNorm1d = LayerNorm1D

__all__ = [
    "Block",
    "DropPath",
    "ECGConvNeXt1D",
    "ECGConvNeXtV3",
    "GRN",
    "LayerNorm1D",
    "LayerNorm1d",
    "MODEL_PRESETS",
    "build_model",
    "get_model_config",
]
