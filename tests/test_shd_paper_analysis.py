"""Unit tests for formal three-seed SHD paper analysis helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "analyze_shd_paper_results.py"
SPEC = importlib.util.spec_from_file_location("analyze_shd_paper_results", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_benjamini_hochberg_is_monotone_in_sorted_order() -> None:
    adjusted = MODULE.benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.20]))

    order = np.argsort([0.01, 0.04, 0.03, 0.20])
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))
    assert adjusted[0] == pytest.approx(0.04)


def test_validate_alignment_rejects_changed_row_order() -> None:
    reference = {
        "row_id": np.array([0, 1, 2]),
        "labels": np.array([[0], [1], [0]]),
    }
    changed = {
        "row_id": np.array([1, 0, 2]),
        "labels": np.array([[1], [0], [0]]),
    }

    with pytest.raises(ValueError, match="row_id"):
        MODULE.validate_prediction_alignment(reference, changed)


def test_cluster_bootstrap_keeps_patient_rows_together() -> None:
    patient_clusters = np.array([10, 10, 20, 30, 30], dtype=np.int64)
    rng = np.random.default_rng(5)

    indices = MODULE.cluster_bootstrap_indices(patient_clusters, rng)

    for patient in np.unique(patient_clusters):
        original_count = int(np.sum(patient_clusters == patient))
        sampled_count = int(np.sum(patient_clusters[indices] == patient))
        assert sampled_count % original_count == 0


def test_mean_seed_difference_uses_seed_metrics_not_probability_ensemble() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    ssl = [
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.4, 0.3, 0.6, 0.7]),
    ]
    scratch = [
        np.array([0.2, 0.8, 0.3, 0.7]),
        np.array([0.1, 0.6, 0.4, 0.9]),
    ]

    value = MODULE.mean_seed_metric_difference(labels, ssl, scratch, metric="auroc")
    expected = np.mean(
        [
            MODULE.safe_metric(labels, ssl_seed, "auroc")
            - MODULE.safe_metric(labels, scratch_seed, "auroc")
            for ssl_seed, scratch_seed in zip(ssl, scratch)
        ]
    )
    assert value == pytest.approx(expected)


def test_parallel_paired_bootstrap_matches_single_process() -> None:
    labels = np.array(
        [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [1, 1]], dtype=np.int64
    )
    ssl = [
        np.array([[0.1, 0.2], [0.2, 0.8], [0.8, 0.3], [0.9, 0.9], [0.3, 0.1], [0.7, 0.7]]),
        np.array([[0.2, 0.1], [0.3, 0.7], [0.7, 0.2], [0.8, 0.8], [0.4, 0.2], [0.6, 0.6]]),
    ]
    scratch = [1.0 - seed_prob for seed_prob in ssl]
    clusters = np.arange(len(labels))

    _, single = MODULE.paired_bootstrap(
        labels, clusters, ssl, scratch, primary_index=1, iterations=20, seed=19, n_jobs=1
    )
    _, parallel = MODULE.paired_bootstrap(
        labels, clusters, ssl, scratch, primary_index=1, iterations=20, seed=19, n_jobs=2
    )

    assert np.allclose(
        single[["point_difference", "ci_low", "ci_high"]],
        parallel[["point_difference", "ci_low", "ci_high"]],
        equal_nan=True,
    )
