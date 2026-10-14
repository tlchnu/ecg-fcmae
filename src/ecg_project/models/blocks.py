"""ConvNeXt blocks with replaceable temporal mixers."""

import torch
import torch.nn as nn

from .heads import ECA1D
from .layers import DropPath, GRN
from .mixers import build_temporal_mixer


class ConvNeXtBlock1D(nn.Module):
    """A 1D ConvNeXt V2 block with optional LayerScale and ECA."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        drop_path: float = 0.0,
        mixer_type: str = "convnext",
        layer_scale_init: float = 0.0,
        use_eca: bool = False,
        inception_kernels: tuple[int, int, int] = (3, 7, 15),
        inception_branch_ratio: float = 0.25,
        deploy: bool = False,
    ):
        super().__init__()
        # Keep the historical ``dwconv`` name for baseline checkpoint compatibility.
        self.dwconv = build_temporal_mixer(
            mixer_type=mixer_type,
            channels=dim,
            kernel_size=kernel_size,
            inception_kernels=inception_kernels,
            inception_branch_ratio=inception_branch_ratio,
            deploy=deploy,
        )
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.activation = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale = (
            nn.Parameter(layer_scale_init * torch.ones(dim))
            if layer_scale_init > 0
            else None
        )
        self.eca = ECA1D(dim) if use_eca else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.dwconv(x)
        residual = residual.permute(0, 2, 1)
        residual = self.norm(residual)
        residual = self.pwconv1(residual)
        residual = self.activation(residual)
        residual = self.grn(residual)
        residual = self.pwconv2(residual)
        if self.layer_scale is not None:
            residual = residual * self.layer_scale
        residual = residual.permute(0, 2, 1)
        residual = self.eca(residual)
        return x + self.drop_path(residual)
