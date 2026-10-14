"""Optional attention, pooling, and tabular-fusion heads."""

import torch
import torch.nn as nn


class ECA1D(nn.Module):
    """Efficient Channel Attention used only as a downstream ablation."""

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.avg_pool(x).transpose(-1, -2)
        weights = self.conv(weights).transpose(-1, -2)
        return x * torch.sigmoid(weights)


class AttentionPool1D(nn.Module):
    """Learned temporal attention pooling."""

    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = max(1, dim // 4)
        self.attention = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attention(x).softmax(dim=-1)
        return (x * weights).sum(dim=-1)


class AveragePool1D(nn.Module):
    """Mean pooling over the temporal dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation for tabular-waveform fusion."""

    def __init__(self, tabular_dim: int, feature_dim: int):
        super().__init__()
        self.projection = nn.Linear(tabular_dim, feature_dim * 2)

    def forward(self, waveform: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.projection(tabular).chunk(2, dim=-1)
        return waveform * (1.0 + gamma) + beta
