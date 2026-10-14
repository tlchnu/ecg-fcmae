"""Tests for deterministic 12-channel zero-fill robustness masks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "evaluate_lead_robustness.py"
SPEC = importlib.util.spec_from_file_location("evaluate_lead_robustness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_named_anatomical4_mask_is_i_ii_v1_v5() -> None:
    masks = MODULE.build_lead_masks(random_seeds=[7])
    keep = np.flatnonzero(masks["anatomical4"])

    assert keep.tolist() == [0, 1, 6, 10]


def test_prediction_archive_resumes_only_with_identical_condition_order(tmp_path) -> None:
    path = tmp_path / "predictions.npz"
    labels = np.zeros((3, 12), dtype=np.float32)
    probabilities = np.zeros((2, 3, 12), dtype=np.float32)
    np.savez_compressed(
        path,
        labels=labels,
        condition_names=np.asarray(["full12", "limb6"]),
        probabilities=probabilities,
    )

    loaded_labels, loaded = MODULE.load_prediction_archive(
        path, ["full12", "limb6"]
    )

    assert np.array_equal(loaded_labels, labels)
    assert set(loaded) == {"full12", "limb6"}
    assert loaded["limb6"].shape == (3, 12)


def test_random_missing_masks_are_deterministic_and_exact() -> None:
    first = MODULE.build_lead_masks(random_seeds=[123])
    second = MODULE.build_lead_masks(random_seeds=[123])

    for missing_count in (1, 2, 4, 6):
        name = f"random_missing_{missing_count}_seed123"
        assert np.array_equal(first[name], second[name])
        assert int(np.sum(~first[name])) == missing_count


def test_apply_lead_mask_zero_fills_only_unavailable_channels() -> None:
    waveform = torch.ones(2, 12, 5)
    keep_mask = np.zeros(12, dtype=bool)
    keep_mask[[0, 1, 6, 10]] = True

    masked = MODULE.apply_lead_mask(waveform, keep_mask)

    assert torch.all(masked[:, keep_mask] == 1)
    assert torch.all(masked[:, ~keep_mask] == 0)
