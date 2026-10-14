"""Tests for fixed sampling and attribution stability summaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "analyze_attribution_stability.py"
SPEC = importlib.util.spec_from_file_location("analyze_attribution_stability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_balanced_indices_are_fixed_and_model_independent() -> None:
    labels = np.array(
        [
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 0],
            [1, 1],
        ],
        dtype=np.int64,
    )

    first = MODULE.select_balanced_indices(labels, samples_per_label=4, seed=17)
    second = MODULE.select_balanced_indices(labels, samples_per_label=4, seed=17)

    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    for label_index, indices in enumerate(first):
        selected_labels = labels[indices, label_index]
        assert int(np.sum(selected_labels == 1)) == 2
        assert int(np.sum(selected_labels == 0)) == 2


def test_attribution_summary_preserves_lead_and_time_dimensions() -> None:
    attribution = np.ones((3, 12, 20), dtype=np.float32)
    attribution[:, 2] *= 3

    lead_importance, temporal = MODULE.summarize_attribution(attribution, temporal_bins=5)

    assert lead_importance.shape == (12,)
    assert temporal.shape == (5,)
    assert lead_importance[2] > lead_importance[0]


def test_safe_spearman_handles_constant_vectors() -> None:
    assert np.isnan(MODULE.safe_spearman(np.ones(3), np.arange(3)))
    assert MODULE.safe_spearman(np.arange(4), np.arange(4)) == 1.0

