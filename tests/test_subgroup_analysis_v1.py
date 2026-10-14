"""Tests for strict EchoNext subgroup analysis v1."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "analysis" / "subgroup_utils_v1.py"
SPEC = importlib.util.spec_from_file_location("subgroup_utils_v1", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

ANALYSIS_PATH = PROJECT_ROOT / "analysis" / "subgroup_analysis_v1.py"
ANALYSIS_SPEC = importlib.util.spec_from_file_location("subgroup_analysis_v1", ANALYSIS_PATH)
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
assert ANALYSIS_SPEC.loader is not None
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


class SubgroupDefinitionTests(unittest.TestCase):
    def test_age_boundaries_are_exact(self) -> None:
        ages = np.asarray([18, 59, 60, 69, 70, 79, 80, 91], dtype=float)
        actual = MODULE.assign_age_group(ages).tolist()
        self.assertEqual(
            actual,
            ["18-59", "18-59", "60-69", "60-69", "70-79", "70-79", "80+", "80+"],
        )

    def test_unknown_categories_are_retained(self) -> None:
        race = MODULE.normalize_race_ethnicity(
            pd.Series(["white", "HISPANIC", "", None, "declined", "asian"])
        )
        self.assertEqual(
            race.tolist(), ["White", "Hispanic", "Unknown", "Unknown", "Other", "Asian"]
        )
        context = MODULE.normalize_clinical_context(
            pd.Series(["emergency", "procedural", None, "unexpected"])
        )
        self.assertEqual(
            context.tolist(), ["Emergency", "Procedural", "Unknown", "Unknown"]
        )


class MetricTests(unittest.TestCase):
    def test_three_seed_mean_probability_is_rowwise(self) -> None:
        arrays = [
            np.asarray([0.1, 0.8]),
            np.asarray([0.2, 0.7]),
            np.asarray([0.3, 0.6]),
        ]
        np.testing.assert_allclose(MODULE.ensemble_probabilities(arrays), [0.2, 0.7])

    def test_single_class_metric_is_nan(self) -> None:
        value = MODULE.safe_binary_metric(np.ones(4), np.linspace(0.1, 0.9, 4), "auroc")
        self.assertTrue(np.isnan(value))

    def test_probability_validation_rejects_logits(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside.*0, 1"):
            MODULE.validate_probabilities(np.asarray([-0.1, 0.2, 2.0]), "synthetic")

    def test_bh_adjustment_is_monotone_in_sorted_order(self) -> None:
        p_values = np.asarray([0.01, 0.04, 0.03, 0.2])
        q_values = MODULE.benjamini_hochberg(p_values)
        np.testing.assert_allclose(q_values, [0.04, 0.0533333333333, 0.0533333333333, 0.2])

    def test_bootstrap_pvalue_has_plus_one_correction(self) -> None:
        differences = np.asarray([-0.2, -0.1, 0.1, 0.2])
        self.assertAlmostEqual(MODULE.bootstrap_two_sided_pvalue(differences), 1.0)


class AlignmentTests(unittest.TestCase):
    def make_inputs(self):
        metadata = pd.DataFrame(
            {
                "patient_key": ["p0", "p1", "p2"],
                "ecg_key": ["e0", "e1", "e2"],
                "age_at_ecg": [59, 60, 80],
                "sex": ["female", "male", "female"],
                "race_ethnicity": ["white", "black", "unknown"],
                "location_setting": ["outpatient", "inpatient", "emergency"],
            }
        )
        labels = np.asarray([[0, 0], [1, 1], [0, 1]], dtype=np.float32)
        order = np.asarray([2, 0, 1])
        bundle = {
            "row_id": order,
            "patient_id": metadata["patient_key"].to_numpy()[order],
            "record_id": metadata["ecg_key"].to_numpy()[order],
            "labels": labels[order],
            "probabilities": np.asarray([[0.3, 0.8], [0.1, 0.2], [0.7, 0.9]], dtype=np.float32),
            "label_names": np.asarray(["subtype", "shd_moderate_or_greater_flag"]),
        }
        return metadata, labels, bundle

    def test_reordered_predictions_align_by_row_id(self) -> None:
        metadata, labels, bundle = self.make_inputs()
        aligned = MODULE.align_prediction_bundle(
            metadata,
            labels,
            bundle,
            ["subtype", "shd_moderate_or_greater_flag"],
            "shd_moderate_or_greater_flag",
        )
        self.assertEqual(aligned["patient_key"].tolist(), ["p0", "p1", "p2"])
        np.testing.assert_allclose(aligned["primary_probability"], [0.2, 0.9, 0.8])

    def test_missing_prediction_row_is_rejected(self) -> None:
        metadata, labels, bundle = self.make_inputs()
        bundle = {key: value[:2] if np.asarray(value).ndim else value for key, value in bundle.items()}
        with self.assertRaisesRegex(ValueError, "row count"):
            MODULE.align_prediction_bundle(
                metadata,
                labels,
                bundle,
                ["subtype", "shd_moderate_or_greater_flag"],
                "shd_moderate_or_greater_flag",
            )

    def test_duplicate_patient_is_rejected(self) -> None:
        metadata, labels, bundle = self.make_inputs()
        metadata.loc[2, "patient_key"] = "p1"
        with self.assertRaisesRegex(ValueError, "duplicate patient"):
            MODULE.align_prediction_bundle(
                metadata,
                labels,
                bundle,
                ["subtype", "shd_moderate_or_greater_flag"],
                "shd_moderate_or_greater_flag",
            )

    def test_prediction_patient_or_record_mismatch_is_rejected(self) -> None:
        metadata, labels, bundle = self.make_inputs()
        bundle["patient_id"] = bundle["patient_id"].copy()
        bundle["patient_id"][0] = "wrong_patient"
        with self.assertRaisesRegex(ValueError, "patient_id"):
            MODULE.align_prediction_bundle(
                metadata,
                labels,
                bundle,
                ["subtype", "shd_moderate_or_greater_flag"],
                "shd_moderate_or_greater_flag",
            )

        metadata, labels, bundle = self.make_inputs()
        bundle["record_id"] = bundle["record_id"].copy()
        bundle["record_id"][1] = "wrong_record"
        with self.assertRaisesRegex(ValueError, "record_id"):
            MODULE.align_prediction_bundle(
                metadata,
                labels,
                bundle,
                ["subtype", "shd_moderate_or_greater_flag"],
                "shd_moderate_or_greater_flag",
            )

    def test_primary_label_must_be_twelfth_for_real_schema(self) -> None:
        names = [f"label_{index}" for index in range(12)]
        names[10] = "shd_moderate_or_greater_flag"
        with self.assertRaisesRegex(ValueError, "index 11"):
            MODULE.validate_primary_label_position(names, "shd_moderate_or_greater_flag")


class PairedBootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_reuses_indices_for_models(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1])
        scratch = np.asarray([0.1, 0.3, 0.6, 0.8, 0.4, 0.7])
        fcmae = np.asarray([0.05, 0.2, 0.75, 0.9, 0.3, 0.8])
        result = MODULE.paired_bootstrap_metrics(
            labels,
            scratch,
            fcmae,
            iterations=20,
            seed=7,
            max_attempt_multiplier=100,
        )
        self.assertEqual(result["scratch_auroc"].shape, (20,))
        self.assertEqual(result["delta_auroc"].shape, (20,))
        np.testing.assert_allclose(
            result["delta_auroc"], result["fcmae_auroc"] - result["scratch_auroc"]
        )
        np.testing.assert_allclose(
            result["delta_auprc"], result["fcmae_auprc"] - result["scratch_auprc"]
        )


class ArtifactTests(unittest.TestCase):
    def test_build_subgroups_uses_requested_order(self) -> None:
        metadata = pd.DataFrame(
            {
                "age_at_ecg": [59, 60, 70, 80],
                "sex": ["female", "male", "female", "male"],
                "race_ethnicity": ["hispanic", "white", "black", "unknown"],
                "location_setting": ["emergency", "inpatient", "outpatient", "procedural"],
            }
        )
        groups = ANALYSIS.build_subgroups(metadata)
        names = [(category, subgroup) for category, subgroup, _ in groups]
        self.assertEqual(names[:4], [("Age", "18-59"), ("Age", "60-69"), ("Age", "70-79"), ("Age", "80+")])
        self.assertIn(("Race/Ethnicity", "Unknown"), names)
        self.assertNotIn(("Clinical context", "Unknown"), names)

    def test_metric_ci_format_handles_nan(self) -> None:
        self.assertEqual(ANALYSIS.format_metric_ci(np.nan, np.nan, np.nan), "NA")
        self.assertEqual(ANALYSIS.format_metric_ci(0.81234, 0.7012, 0.8999), "0.812 (0.701-0.900)")

    def test_paired_output_contains_required_columns(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1])
        scratch = np.asarray([0.1, 0.3, 0.6, 0.8, 0.4, 0.7])
        fcmae = np.asarray([0.05, 0.2, 0.75, 0.9, 0.3, 0.8])
        result = ANALYSIS.analyze_one_subgroup(
            "Age", "18-59", labels, scratch, fcmae, iterations=20, seed=11
        )
        required = {
            "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high", "auroc_p",
            "delta_auprc", "delta_auprc_ci_low", "delta_auprc_ci_high", "auprc_p",
        }
        self.assertTrue(required.issubset(result["paired_row"]))
        self.assertEqual(result["paired_row"]["bootstrap_n"], 20)

    def test_paired_latex_table_includes_prevalence(self) -> None:
        paired = pd.DataFrame(
            [
                {
                    "subgroup_category": "Age",
                    "subgroup": "18-59",
                    "n": 100,
                    "prevalence": 0.32,
                    "delta_auroc": 0.01,
                    "delta_auroc_ci_low": 0.001,
                    "delta_auroc_ci_high": 0.02,
                    "auroc_p": 0.02,
                    "auroc_q": 0.04,
                    "delta_auprc": 0.02,
                    "delta_auprc_ci_low": 0.002,
                    "delta_auprc_ci_high": 0.03,
                    "auprc_p": 0.01,
                    "auprc_q": 0.03,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            ANALYSIS.write_paired_table(paired, output_dir)
            table = (output_dir / "subgroup_table_paired.tex").read_text(encoding="utf-8")
        self.assertIn("Prevalence", table)
        self.assertIn("0.320", table)


if __name__ == "__main__":
    unittest.main()
