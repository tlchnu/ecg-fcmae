"""Tests for the EchoNext 11-subtype Macro AUROC evaluator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "calculate_subtype_macro_auroc.py"
SPEC = importlib.util.spec_from_file_location("calculate_subtype_macro_auroc", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

AGGREGATE_LABEL = "shd_moderate_or_greater_flag"
LABEL_NAMES = [f"subtype_{index}" for index in range(11)] + [AGGREGATE_LABEL]


def test_select_subtype_indices_excludes_aggregate_by_name() -> None:
    shuffled = [AGGREGATE_LABEL, *LABEL_NAMES[:-1]]

    indices = MODULE.select_subtype_indices(shuffled, AGGREGATE_LABEL)

    assert len(indices) == 11
    assert all(shuffled[index] != AGGREGATE_LABEL for index in indices)


def test_compute_subtype_aurocs_matches_sklearn() -> None:
    y_true = np.tile(np.array([[0], [1], [0], [1]], dtype=float), (1, 12))
    y_score = np.tile(np.array([[0.1], [0.9], [0.2], [0.8]], dtype=float), (1, 12))

    result = MODULE.compute_subtype_aurocs(
        y_true, y_score, LABEL_NAMES, AGGREGATE_LABEL
    )

    expected = roc_auc_score(y_true[:, 0], y_score[:, 0])
    assert result["n_subtypes"] == 11
    assert result["subtype_macro_auroc"] == pytest.approx(expected)
    assert result["fallback_labels"] == []
    assert set(result["per_label_auroc"]) == set(LABEL_NAMES[:-1])


def test_compute_subtype_aurocs_uses_half_for_single_class_label() -> None:
    y_true = np.tile(np.array([[0], [1], [0], [1]], dtype=float), (1, 12))
    y_score = np.tile(np.array([[0.1], [0.9], [0.2], [0.8]], dtype=float), (1, 12))
    y_true[:, 3] = 0

    result = MODULE.compute_subtype_aurocs(
        y_true, y_score, LABEL_NAMES, AGGREGATE_LABEL
    )

    assert result["per_label_auroc"]["subtype_3"] == 0.5
    assert result["fallback_labels"] == ["subtype_3"]
    assert result["subtype_macro_auroc"] == pytest.approx(10.5 / 11)


def test_derive_subtype_macro_recovers_known_mean() -> None:
    subtype_values = np.linspace(0.70, 0.90, 11)
    aggregate_auroc = 0.95
    macro_12 = (subtype_values.sum() + aggregate_auroc) / 12

    result = MODULE.derive_subtype_macro(macro_12, aggregate_auroc)

    assert result == pytest.approx(subtype_values.mean())


def test_load_prediction_bundle_supports_targets_probabilities_aliases(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / "predictions.npz"
    targets = np.zeros((3, 12), dtype=float)
    probabilities = np.full((3, 12), 0.25, dtype=float)
    np.savez(
        bundle_path,
        targets=targets,
        probabilities=probabilities,
        label_names=np.asarray(LABEL_NAMES),
    )

    actual_targets, actual_probabilities, actual_labels = MODULE.load_prediction_bundle(
        bundle_path
    )

    np.testing.assert_array_equal(actual_targets, targets)
    np.testing.assert_array_equal(actual_probabilities, probabilities)
    assert actual_labels == LABEL_NAMES


def test_load_prediction_bundle_supports_remote_labels_key(tmp_path: Path) -> None:
    bundle_path = tmp_path / "remote_predictions.npz"
    targets = np.zeros((3, 12), dtype=float)
    probabilities = np.full((3, 12), 0.25, dtype=float)
    np.savez(
        bundle_path,
        labels=targets,
        probabilities=probabilities,
        label_names=np.asarray(LABEL_NAMES),
    )

    actual_targets, actual_probabilities, actual_labels = MODULE.load_prediction_bundle(
        bundle_path
    )

    np.testing.assert_array_equal(actual_targets, targets)
    np.testing.assert_array_equal(actual_probabilities, probabilities)
    assert actual_labels == LABEL_NAMES


def test_load_prediction_bundle_rejects_ambiguous_scores(tmp_path: Path) -> None:
    bundle_path = tmp_path / "ambiguous.npz"
    np.savez(
        bundle_path,
        y_true=np.zeros((3, 12)),
        y_score=np.zeros((3, 12)),
        probabilities=np.zeros((3, 12)),
        label_names=np.asarray(LABEL_NAMES),
    )

    with pytest.raises(ValueError, match="ambiguous"):
        MODULE.load_prediction_bundle(bundle_path)


def test_build_derived_record_preserves_audit_metadata() -> None:
    record = MODULE.build_derived_record(
        experiment="exp_a",
        seed=42,
        macro_12=0.83,
        aggregate_auroc=0.85,
        source_path="experiments/exp_a/metrics.csv",
    )

    assert record["experiment"] == "exp_a"
    assert record["seed"] == 42
    assert record["source_path"] == "experiments/exp_a/metrics.csv"
    assert record["evidence_source"] == "derived_from_rounded_metrics"
    assert record["rounding_uncertainty_max"] == pytest.approx(0.00006)
    assert record["subtype_macro_auroc"] == pytest.approx((12 * 0.83 - 0.85) / 11)


def test_run_manifest_preserves_group_in_per_label_output(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.npz"
    y_true = np.tile(np.array([[0], [1], [0], [1]], dtype=float), (1, 12))
    y_score = np.tile(np.array([[0.1], [0.9], [0.2], [0.8]], dtype=float), (1, 12))
    np.savez(
        prediction_path,
        labels=y_true,
        probabilities=y_score,
        label_names=np.asarray(LABEL_NAMES),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "experiments": [
                    {
                        "experiment": "exp_a",
                        "method": "method_a",
                        "experiment_group": "main_methods",
                        "seed": 42,
                        "prediction_path": prediction_path.name,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "output"

    MODULE.run_manifest(manifest_path, tmp_path, output_dir)

    per_label = pd.read_csv(output_dir / "per_label_auroc.csv")
    assert set(per_label["experiment_group"]) == {"main_methods"}
