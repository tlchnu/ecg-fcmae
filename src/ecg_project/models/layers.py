"""Shared neural-network layers for 1D ECG models."""

import torch
import torch.nn as nn


class LayerNorm1D(nn.Module):
    """Channel-first LayerNorm for tensors shaped ``(B, C, T)``."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight[:, None] * normalized + self.bias[:, None]


class LeadAwareStem1D(nn.Module):
    """Shared per-lead stem with lightweight cross-lead self-attention.

    Parameters
    ----------
    in_channels:
        Number of ECG leads in the input waveform.
    out_channels:
        Number of feature channels produced for the ConvNeXt stages.

    Input shape is ``(B, L, T)`` and output shape is ``(B, C, T')``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        num_heads: int = 4,
        lead_dropout: float = 0.0,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive for LeadAwareStem1D")
        if not 0.0 <= lead_dropout < 1.0:
            raise ValueError("lead_dropout must be in [0, 1)")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lead_dropout = lead_dropout
        self.shared_projection = nn.Conv1d(
            1,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.lead_embedding = nn.Parameter(
            torch.zeros(1, in_channels, out_channels, 1)
        )
        self.cross_lead_attention = nn.MultiheadAttention(
            out_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = LayerNorm1D(out_channels)

    def _apply_lead_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.lead_dropout == 0.0:
            return x
        keep_probability = 1.0 - self.lead_dropout
        mask = (
            torch.rand(x.shape[0], x.shape[1], 1, 1, device=x.device)
            < keep_probability
        ).to(dtype=x.dtype)
        empty_rows = mask.sum(dim=1, keepdim=True) == 0
        if empty_rows.any():
            mask[:, :1] = torch.where(empty_rows, torch.ones_like(mask[:, :1]), mask[:, :1])
        return x * mask / keep_probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("LeadAwareStem1D expects input shaped (B, L, T)")
        batch_size, n_leads, n_samples = x.shape
        if n_leads != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} leads, received {n_leads}"
            )

        projected = self.shared_projection(x.reshape(batch_size * n_leads, 1, n_samples))
        _, feature_dim, n_tokens = projected.shape
        projected = projected.reshape(batch_size, n_leads, feature_dim, n_tokens)
        projected = self._apply_lead_dropout(projected + self.lead_embedding)

        lead_tokens = projected.permute(0, 3, 1, 2).reshape(
            batch_size * n_tokens,
            n_leads,
            feature_dim,
        )
        mixed_tokens, _ = self.cross_lead_attention(
            lead_tokens,
            lead_tokens,
            lead_tokens,
            need_weights=False,
        )
        mixed = mixed_tokens.reshape(batch_size, n_tokens, n_leads, feature_dim)
        features = mixed.mean(dim=2).permute(0, 2, 1)
        return self.norm(features)


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXt V2."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        response = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * normalized) + self.beta + x


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor_()
