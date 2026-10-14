"""Checkpoint helpers shared by SSL and downstream training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


ENCODER_PREFIXES = ("downsample_layers.", "stages.")


def encoder_state_dict(
    model: nn.Module,
    prefixes: Iterable[str] = ENCODER_PREFIXES,
) -> dict[str, torch.Tensor]:
    """Extract a CPU encoder state dictionary."""
    prefix_tuple = tuple(prefixes)
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith(prefix_tuple)
    }


def load_encoder_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    verbose: bool = True,
    input_channel_indices: Iterable[int] | None = None,
    stem_mode: str = "select",
) -> tuple[int, int]:
    """Load encoder tensors, optionally selecting or skipping the first stem."""
    if stem_mode not in {"select", "skip"}:
        raise ValueError("stem_mode must be 'select' or 'skip'")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("encoder", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    own_state = model.state_dict()
    loaded = 0
    skipped = 0
    selected_channels = (
        list(input_channel_indices) if input_channel_indices is not None else None
    )
    for key, value in state.items():
        if key == "downsample_layers.0.0.weight" and stem_mode == "skip":
            skipped += 1
            continue
        if (
            selected_channels is not None
            and key == "downsample_layers.0.0.weight"
            and key in own_state
            and value.ndim == 3
            and own_state[key].ndim == 3
            and value.shape[0] == own_state[key].shape[0]
            and value.shape[2] == own_state[key].shape[2]
            and own_state[key].shape[1] == len(selected_channels)
            and value.shape != own_state[key].shape
        ):
            value = value[:, selected_channels, :]
        if key in own_state and own_state[key].shape == value.shape:
            own_state[key] = value
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(own_state)
    if verbose:
        message = f"[checkpoint] loaded encoder tensors={loaded}"
        if skipped:
            message += f", skipped={skipped}"
        print(message)
    return loaded, skipped
