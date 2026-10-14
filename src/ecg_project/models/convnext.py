"""Configurable 1D ConvNeXt backbone and ECG classification head."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .blocks import ConvNeXtBlock1D
from .heads import AttentionPool1D, AveragePool1D, FiLM
from .layers import LayerNorm1D, LeadAwareStem1D
from .mixers import DilatedReparamDWConv1D


class ECGConvNeXt1D(nn.Module):
    """Replaceable ConvNeXt-style model for multi-lead ECG classification."""

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 12,
        tabular_features: int = 7,
        depths: Sequence[int] = (3, 3, 9, 3),
        dims: Sequence[int] = (64, 128, 256, 512),
        stage_kernel_sizes: Sequence[int] = (7, 7, 7, 7),
        block_type: str = "convnext",
        drop_path_rate: float = 0.2,
        layer_scale_init: float = 0.0,
        use_eca: bool = False,
        multi_scale_feat: bool = True,
        tab_hidden: int = 64,
        head_dropout: float = 0.2,
        pool: str = "attn",
        fusion: str = "film",
        stem_kernel_size: int = 4,
        stem_stride: int = 4,
        lead_stem: bool = False,
        lead_mixer_heads: int = 4,
        lead_dropout: float = 0.0,
        inception_kernels: Sequence[int] = (3, 7, 15),
        inception_branch_ratio: float = 0.25,
        deploy: bool = False,
    ):
        super().__init__()
        if not (len(depths) == len(dims) == len(stage_kernel_sizes)):
            raise ValueError("depths, dims, and stage_kernel_sizes must have equal length")
        if pool not in {"attn", "avg"}:
            raise ValueError(f"Unknown pool: {pool}")
        if fusion not in {"film", "concat", "none"}:
            raise ValueError(f"Unknown fusion: {fusion}")

        self.model_config = {
            "in_channels": in_channels,
            "num_classes": num_classes,
            "tabular_features": tabular_features,
            "depths": tuple(depths),
            "dims": tuple(dims),
            "stage_kernel_sizes": tuple(stage_kernel_sizes),
            "block_type": block_type,
            "drop_path_rate": drop_path_rate,
            "layer_scale_init": layer_scale_init,
            "use_eca": use_eca,
            "multi_scale_feat": multi_scale_feat,
            "tab_hidden": tab_hidden,
            "head_dropout": head_dropout,
            "pool": pool,
            "fusion": fusion,
            "stem_kernel_size": stem_kernel_size,
            "stem_stride": stem_stride,
            "lead_stem": lead_stem,
            "lead_mixer_heads": lead_mixer_heads,
            "lead_dropout": lead_dropout,
            "inception_kernels": tuple(inception_kernels),
            "inception_branch_ratio": inception_branch_ratio,
            "deploy": deploy,
        }
        self.n_stages = len(dims)
        self.multi_scale_feat = multi_scale_feat
        self.fusion = fusion
        self.use_tabular = tabular_features > 0 and fusion != "none"

        stem_padding = max(0, (stem_kernel_size - stem_stride) // 2)
        if lead_stem:
            stem_layer = LeadAwareStem1D(
                in_channels=in_channels,
                out_channels=dims[0],
                kernel_size=stem_kernel_size,
                stride=stem_stride,
                padding=stem_padding,
                num_heads=lead_mixer_heads,
                lead_dropout=lead_dropout,
            )
        else:
            stem_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    dims[0],
                    kernel_size=stem_kernel_size,
                    stride=stem_stride,
                    padding=stem_padding,
                ),
                LayerNorm1D(dims[0]),
            )
        self.downsample_layers = nn.ModuleList(
            [
                stem_layer
            ]
        )
        for stage_index in range(self.n_stages - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm1D(dims[stage_index]),
                    nn.Conv1d(
                        dims[stage_index],
                        dims[stage_index + 1],
                        kernel_size=2,
                        stride=2,
                    ),
                )
            )

        drop_path_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        block_index = 0
        for stage_index, stage_depth in enumerate(depths):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock1D(
                        dim=dims[stage_index],
                        kernel_size=stage_kernel_sizes[stage_index],
                        drop_path=drop_path_rates[block_index + offset],
                        mixer_type=block_type,
                        layer_scale_init=layer_scale_init,
                        use_eca=use_eca,
                        inception_kernels=tuple(inception_kernels),
                        inception_branch_ratio=inception_branch_ratio,
                        deploy=deploy,
                    )
                    for offset in range(stage_depth)
                ]
            )
            self.stages.append(stage)
            block_index += stage_depth

        feature_dim = dims[-1] + (dims[-2] if multi_scale_feat else 0)
        pool_class = AttentionPool1D if pool == "attn" else AveragePool1D
        self.pool_last = pool_class(dims[-1]) if pool == "attn" else pool_class()
        self.pool_penult = (
            pool_class(dims[-2]) if pool == "attn" else pool_class()
        ) if multi_scale_feat else None

        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_dropout = nn.Dropout(head_dropout)
        if self.use_tabular:
            self.tabular_mlp = nn.Sequential(
                nn.Linear(tabular_features, tab_hidden),
                nn.GELU(),
                nn.Linear(tab_hidden, tab_hidden),
                nn.GELU(),
            )
            self.film = FiLM(tab_hidden, feature_dim) if fusion == "film" else None
        else:
            self.tabular_mlp = None
            self.film = None
        classifier_dim = (
            feature_dim + tab_hidden
            if self.use_tabular and fusion == "concat"
            else feature_dim
        )
        self.classifier = nn.Linear(classifier_dim, num_classes)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_encoder(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, 12, T)`` waveforms into ``(B, C, T')`` features."""
        features = waveform
        for downsample, stage in zip(self.downsample_layers, self.stages):
            features = stage(downsample(features))
        return features

    def forward_features(
        self, waveform: torch.Tensor, tabular: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = waveform
        penultimate = None
        for stage_index, (downsample, stage) in enumerate(
            zip(self.downsample_layers, self.stages)
        ):
            features = stage(downsample(features))
            if self.multi_scale_feat and stage_index == self.n_stages - 2:
                penultimate = features

        waveform_features = self.pool_last(features)
        if self.multi_scale_feat:
            waveform_features = torch.cat(
                [waveform_features, self.pool_penult(penultimate)], dim=-1
            )
        waveform_features = self.feature_dropout(self.feature_norm(waveform_features))

        if not self.use_tabular:
            return waveform_features
        if tabular is None:
            tabular = waveform_features.new_zeros(
                waveform_features.shape[0],
                self.tabular_mlp[0].in_features,
            )
        tabular_features = self.tabular_mlp(tabular)
        if self.fusion == "film":
            return self.film(waveform_features, tabular_features)
        if self.fusion == "concat":
            return torch.cat([waveform_features, tabular_features], dim=-1)
        return waveform_features

    def forward(
        self, waveform: torch.Tensor, tabular: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.classifier(self.forward_features(waveform, tabular))

    @torch.no_grad()
    def switch_reparameterizable_layers_to_deploy(self) -> None:
        """Fuse all reparameterizable large-kernel mixers for inference."""
        for module in self.modules():
            if isinstance(module, DilatedReparamDWConv1D):
                module.switch_to_deploy()
        self.model_config["deploy"] = True
