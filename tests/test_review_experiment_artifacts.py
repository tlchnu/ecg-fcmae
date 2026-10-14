"""Tests for auditable post-review prediction artifacts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SPEC = importlib.util.spec_from_file_location(
    "export_shd_predictions", PROJECT_ROOT / "scripts" / "export_shd_predictions.py"
)
EXPORT = importlib.util.module_from_spec(EXPORT_SPEC)
assert EXPORT_SPEC.loader is not None
EXPORT_SPEC.loader.exec_module(EXPORT)

ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "review_experiments_v1", PROJECT_ROOT / "analysis" / "review_experiments_v1.py"
)
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)


def test_patient_identifier_changes_alignment_hash() -> None:
    row_id = np.arange(3)
    labels = np.asarray([[0, 1], [1, 0], [1, 1]], dtype=np.float32)
    first = EXPORT.stable_split_hash(row_id, labels, np.asarray(["p0", "p1", "p2"]))
    second = EXPORT.stable_split_hash(row_id, labels, np.asarray(["p1", "p0", "p2"]))
    assert first != second


def test_exact_metric_record_excludes_aggregate_by_name() -> None:
    assert ANALYSIS_SPEC.loader is not None
    ANALYSIS_SPEC.loader.exec_module(ANALYSIS)
    labels = np.asarray(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    probabilities = np.asarray(
        [
            [0.1, 0.2, 0.1],
            [0.2, 0.8, 0.7],
            [0.8, 0.3, 0.8],
            [0.9, 0.9, 0.9],
        ],
        dtype=np.float32,
    )
    record = ANALYSIS.compute_exact_metrics(
        labels,
        probabilities,
        ["subtype_a", "subtype_b", "shd_moderate_or_greater_flag"],
    )
    assert record["macro_auroc_11"] == 1.0
    assert record["macro_auroc_12"] == 1.0
    assert record["primary_shd_auroc"] == 1.0


def test_summary_uses_sample_standard_deviation() -> None:
    assert ANALYSIS_SPEC.loader is not None
    if not hasattr(ANALYSIS, "summarize_seed_metrics"):
        ANALYSIS_SPEC.loader.exec_module(ANALYSIS)
    rows = [
        {"primary_shd_auroc": 0.8, "primary_shd_auprc": 0.7, "macro_auroc_11": 0.6, "macro_auroc_12": 0.7},
        {"primary_shd_auroc": 0.9, "primary_shd_auprc": 0.8, "macro_auroc_11": 0.7, "macro_auroc_12": 0.8},
        {"primary_shd_auroc": 1.0, "primary_shd_auprc": 0.9, "macro_auroc_11": 0.8, "macro_auroc_12": 0.9},
    ]
    summary = ANALYSIS.summarize_seed_metrics(rows)
    assert np.isclose(summary["primary_shd_auroc_std"], 0.1)
