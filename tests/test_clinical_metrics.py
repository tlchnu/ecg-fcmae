"""Tests for validation-frozen SHD screening and calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_project.utils.metrics import (
    brier_score,
    calibration_table,
    screening_metrics,
    select_threshold_for_sensitivity,
)


def test_threshold_selection_meets_sensitivity_and_breaks_ties_high() -> None:
    targets = np.array([1, 1, 1, 0, 0], dtype=np.int64)
    probabilities = np.array([0.9, 0.8, 0.7, 0.7, 0.1], dtype=float)

    result = select_threshold_for_sensitivity(targets, probabilities, min_sensitivity=2 / 3)

    assert result["threshold"] == pytest.approx(0.8)
    assert result["sensitivity"] == pytest.approx(2 / 3)
    assert result["specificity"] == pytest.approx(1.0)


def test_screening_metrics_use_probability_greater_equal_threshold() -> None:
    targets = np.array([1, 0, 1, 0], dtype=np.int64)
    probabilities = np.array([0.5, 0.5, 0.49, 0.1], dtype=float)

    result = screening_metrics(targets, probabilities, threshold=0.5)

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["tn"] == 1
    assert result["fn"] == 1
    assert result["sensitivity"] == pytest.approx(0.5)
    assert result["specificity"] == pytest.approx(0.5)


def test_brier_and_fixed_ten_bin_ece() -> None:
    targets = np.array([0, 1], dtype=np.int64)
    probabilities = np.array([0.05, 0.95], dtype=float)

    assert brier_score(targets, probabilities) == pytest.approx(0.0025)
    table = calibration_table(targets, probabilities, n_bins=10)
    assert len(table) == 10
    assert int(table["n"].sum()) == 2
    assert float(table.attrs["ece"]) == pytest.approx(0.05)


def test_threshold_selection_rejects_degenerate_validation_targets() -> None:
    with pytest.raises(ValueError, match="both positive and negative"):
        select_threshold_for_sensitivity(
            np.ones(4, dtype=np.int64),
            np.linspace(0.1, 0.9, 4),
            min_sensitivity=0.9,
        )
