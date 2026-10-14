"""Tests for architecture-matched true reduced-lead encoder initialization."""

from pathlib import Path

import torch

from ecg_project.models import ECGConvNeXt1D
from ecg_project.utils.checkpoint import load_encoder_checkpoint


def test_load_encoder_checkpoint_selects_requested_stem_channels(tmp_path: Path) -> None:
    source = ECGConvNeXt1D(in_channels=12)
    with torch.no_grad():
        stem = source.downsample_layers[0][0].weight
        for channel in range(12):
            stem[:, channel].fill_(float(channel))
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"encoder": source.state_dict()}, checkpoint)
    target = ECGConvNeXt1D(in_channels=4)

    loaded, skipped = load_encoder_checkpoint(
        target,
        checkpoint,
        verbose=False,
        input_channel_indices=[0, 1, 6, 10],
    )

    assert loaded > 0
    assert skipped == 0
    target_stem = target.downsample_layers[0][0].weight
    assert torch.allclose(target_stem[:, 0], torch.zeros_like(target_stem[:, 0]))
    assert torch.allclose(target_stem[:, 1], torch.ones_like(target_stem[:, 1]))
    assert torch.allclose(target_stem[:, 2], torch.full_like(target_stem[:, 2], 6.0))
    assert torch.allclose(target_stem[:, 3], torch.full_like(target_stem[:, 3], 10.0))


def test_load_encoder_checkpoint_can_skip_stem_and_load_rest(tmp_path: Path) -> None:
    source = ECGConvNeXt1D(in_channels=12)
    checkpoint = tmp_path / "encoder.pt"
    torch.save({"encoder": source.state_dict()}, checkpoint)
    target = ECGConvNeXt1D(in_channels=1)
    stem_before = target.downsample_layers[0][0].weight.detach().clone()

    loaded, skipped = load_encoder_checkpoint(
        target,
        checkpoint,
        verbose=False,
        input_channel_indices=[0],
        stem_mode="skip",
    )

    assert loaded > 0
    assert skipped >= 1
    assert torch.equal(target.downsample_layers[0][0].weight, stem_before)
    assert torch.equal(
        target.downsample_layers[1][0].weight,
        source.downsample_layers[1][0].weight,
    )


def test_single_channel_checkpoint_loads_directly_for_original_lead_index(tmp_path: Path) -> None:
    source = ECGConvNeXt1D(in_channels=1)
    with torch.no_grad():
        source.downsample_layers[0][0].weight.fill_(3.0)
    checkpoint = tmp_path / "single_lead_encoder.pt"
    torch.save({"encoder": source.state_dict()}, checkpoint)
    target = ECGConvNeXt1D(in_channels=1)

    loaded, skipped = load_encoder_checkpoint(
        target,
        checkpoint,
        verbose=False,
        input_channel_indices=[6],
        stem_mode="select",
    )

    assert loaded > 0
    assert skipped == 0
    assert torch.allclose(
        target.downsample_layers[0][0].weight,
        torch.full_like(target.downsample_layers[0][0].weight, 3.0),
    )
