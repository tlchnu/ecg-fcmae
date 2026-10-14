"""Replaceable temporal token mixers for ECG ConvNeXt blocks."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _default_reparam_branches(kernel_size: int) -> list[tuple[int, int]]:
    if kernel_size <= 7:
        return [(3, 1), (3, 2)]
    if kernel_size <= 13:
        return [(5, 1), (5, 2), (3, 3), (3, 5)]
    return [(5, 1), (5, 2), (5, 3), (3, 5), (3, 7), (3, 9)]


class DilatedReparamDWConv1D(nn.Module):
    """Training-time multi-branch depthwise convolution.

    Inspired by UniRepLKNet, small dilated branches complement a large depthwise
    kernel during training. ``switch_to_deploy`` fuses all branches into one
    depthwise convolution with an exactly equivalent output.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        branch_specs: Iterable[tuple[int, int]] | None = None,
        deploy: bool = False,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Large-kernel size must be odd")
        self.channels = channels
        self.kernel_size = kernel_size
        self.deploy = deploy
        specs = list(branch_specs or _default_reparam_branches(kernel_size))
        for branch_kernel, dilation in specs:
            effective = dilation * (branch_kernel - 1) + 1
            if branch_kernel % 2 == 0 or effective > kernel_size:
                raise ValueError(
                    f"Invalid branch (kernel={branch_kernel}, dilation={dilation}) "
                    f"for target kernel {kernel_size}"
                )
        self.branch_specs = specs

        if deploy:
            self.reparam_conv = self._make_conv(kernel_size)
        else:
            self.large_conv = self._make_conv(kernel_size)
            self.branches = nn.ModuleList(
                [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=branch_kernel,
                        padding=dilation * (branch_kernel - 1) // 2,
                        dilation=dilation,
                        groups=channels,
                        bias=True,
                    )
                    for branch_kernel, dilation in specs
                ]
            )

    def _make_conv(self, kernel_size: int) -> nn.Conv1d:
        return nn.Conv1d(
            self.channels,
            self.channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam_conv(x)
        output = self.large_conv(x)
        for branch in self.branches:
            output = output + branch(x)
        return output

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fused depthwise kernel and bias."""
        if self.deploy:
            return self.reparam_conv.weight, self.reparam_conv.bias

        fused_kernel = self.large_conv.weight.detach().clone()
        fused_bias = self.large_conv.bias.detach().clone()
        for branch, (branch_kernel, dilation) in zip(self.branches, self.branch_specs):
            effective_size = dilation * (branch_kernel - 1) + 1
            expanded = branch.weight.new_zeros(
                branch.weight.shape[0], 1, effective_size
            )
            expanded[:, :, ::dilation] = branch.weight
            left_pad = (self.kernel_size - effective_size) // 2
            fused_kernel[:, :, left_pad:left_pad + effective_size] += expanded
            fused_bias += branch.bias
        return fused_kernel, fused_bias

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        """Fuse training branches into a single inference convolution."""
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam_conv = self._make_conv(self.kernel_size).to(kernel.device, kernel.dtype)
        reparam_conv.weight.copy_(kernel)
        reparam_conv.bias.copy_(bias)
        self.reparam_conv = reparam_conv
        del self.large_conv
        del self.branches
        self.deploy = True


class InceptionDWConv1D(nn.Module):
    """Partial-channel multi-scale depthwise mixer inspired by InceptionNeXt."""

    def __init__(
        self,
        channels: int,
        kernels: tuple[int, int, int] = (3, 7, 15),
        branch_ratio: float = 0.25,
    ):
        super().__init__()
        if not 0 < branch_ratio <= 1 / len(kernels):
            raise ValueError("branch_ratio must leave room for the identity branch")
        branch_channels = max(1, int(channels * branch_ratio))
        while branch_channels * len(kernels) > channels:
            branch_channels -= 1
        self.branch_channels = branch_channels
        self.identity_channels = channels - branch_channels * len(kernels)
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    branch_channels,
                    branch_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=branch_channels,
                )
                for kernel in kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split_sizes = [self.branch_channels] * len(self.branches)
        split_sizes.append(self.identity_channels)
        chunks = torch.split(x, split_sizes, dim=1)
        outputs = [branch(chunk) for branch, chunk in zip(self.branches, chunks)]
        if self.identity_channels:
            outputs.append(chunks[-1])
        return torch.cat(outputs, dim=1)


class PeriodSpectralMixer1D(nn.Module):
    """Depthwise temporal convolution with a lightweight spectral residual branch.

    The spectral branch applies an rFFT over the temporal axis, gates low/mid/high
    frequency-bin groups, and returns to the time domain with an inverse FFT. The
    output shape matches the input shape ``(B, C, T)``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        band_ratios: tuple[float, float] = (0.15, 0.40),
        spectral_scale_init: float = 1e-3,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for PeriodSpectralMixer1D")
        low_ratio, mid_ratio = band_ratios
        if not 0.0 < low_ratio < mid_ratio < 1.0:
            raise ValueError("band_ratios must satisfy 0 < low < mid < 1")
        self.channels = channels
        self.kernel_size = kernel_size
        self.band_ratios = band_ratios
        self.time_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.band_gates = nn.Parameter(torch.zeros(1, channels, 3))
        self.spectral_scale = nn.Parameter(torch.full((), spectral_scale_init))

    def _band_edges(self, n_frequency_bins: int) -> tuple[int, int]:
        low_edge = max(1, int(n_frequency_bins * self.band_ratios[0]))
        mid_edge = max(low_edge + 1, int(n_frequency_bins * self.band_ratios[1]))
        return min(low_edge, n_frequency_bins), min(mid_edge, n_frequency_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_features = self.time_conv(x)
        spectrum = torch.fft.rfft(x.float(), dim=-1)
        low_edge, mid_edge = self._band_edges(spectrum.shape[-1])
        gates = 1.0 + torch.tanh(self.band_gates.float())

        gated_spectrum = torch.empty_like(spectrum)
        gated_spectrum[..., :low_edge] = spectrum[..., :low_edge] * gates[..., 0:1]
        gated_spectrum[..., low_edge:mid_edge] = (
            spectrum[..., low_edge:mid_edge] * gates[..., 1:2]
        )
        gated_spectrum[..., mid_edge:] = spectrum[..., mid_edge:] * gates[..., 2:3]
        spectral_features = torch.fft.irfft(
            gated_spectrum,
            n=x.shape[-1],
            dim=-1,
        ).to(dtype=x.dtype)
        return time_features + self.spectral_scale.to(dtype=x.dtype) * spectral_features


def build_temporal_mixer(
    mixer_type: str,
    channels: int,
    kernel_size: int,
    inception_kernels: tuple[int, int, int] = (3, 7, 15),
    inception_branch_ratio: float = 0.25,
    deploy: bool = False,
) -> nn.Module:
    """Build a temporal mixer by registry name."""
    if mixer_type == "convnext":
        return nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
    if mixer_type == "reparam_lk":
        return DilatedReparamDWConv1D(
            channels=channels,
            kernel_size=kernel_size,
            deploy=deploy,
        )
    if mixer_type == "inception1d":
        return InceptionDWConv1D(
            channels=channels,
            kernels=inception_kernels,
            branch_ratio=inception_branch_ratio,
        )
    if mixer_type == "period_spectral":
        return PeriodSpectralMixer1D(
            channels=channels,
            kernel_size=kernel_size,
        )
    raise ValueError(f"Unknown mixer_type: {mixer_type}")
