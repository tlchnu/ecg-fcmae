"""Statistical and alignment utilities for EchoNext subgroup analysis v1."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PRIMARY_LABEL = "shd_moderate_or_greater_flag"


def assign_age_group(ages: np.ndarray | pd.Series) -> np.ndarray:
    """Map ages to prespecified bins, retaining invalid/missing ages as Unknown."""
    values = np.asarray(ages, dtype=float)
    groups = np.full(values.shape, "Unknown", dtype="U16")
    groups[(values >= 18) & (values < 60)] = "18-59"
    groups[(values >= 60) & (values < 70)] = "60-69"
    groups[(values >= 70) & (values < 80)] = "70-79"
    groups[values >= 80] = "80+"
    return groups


def _normalize_category(
    values: pd.Series | Sequence[object], mapping: dict[str, str], other: str = "Unknown"
) -> np.ndarray:
    normalized = pd.Series(values, dtype="object").fillna("").astype(str).str.strip().str.lower()
    return normalized.map(mapping).fillna(other).to_numpy(dtype="U32")


def normalize_sex(values: pd.Series | Sequence[object]) -> np.ndarray:
    """Normalize audited EchoNext sex values without guessing numeric encodings."""
    return _normalize_category(values, {"female": "Female", "f": "Female", "male": "Male", "m": "Male"})


def normalize_race_ethnicity(values: pd.Series | Sequence[object]) -> np.ndarray:
    """Normalize race/ethnicity while retaining unknown and unexpected values."""
    mapping = {
        "hispanic": "Hispanic",
        "white": "White",
        "black": "Black",
        "asian": "Asian",
        "other": "Other",
        "unknown": "Unknown",
        "missing": "Unknown",
        "": "Unknown",
        "declined": "Other",
    }
    return _normalize_category(values, mapping)


def normalize_clinical_context(values: pd.Series | Sequence[object]) -> np.ndarray:
    """Normalize the audited EchoNext location-setting categories."""
    mapping = {
        "emergency": "Emergency",
        "inpatient": "Inpatient",
        "outpatient": "Outpatient",
        "procedural": "Procedural",
        "unknown": "Unknown",
        "missing": "Unknown",
        "": "Unknown",
    }
    return _normalize_category(values, mapping)


def validate_probabilities(probabilities: np.ndarray, source: str) -> None:
    """Reject NaNs, infinities, and logits rather than silently applying sigmoid."""
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: probabilities contain non-finite values")
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        raise ValueError(f"{source}: values outside [0, 1]; logits are not accepted")


def validate_primary_label_position(label_names: Sequence[str], primary_label: str = PRIMARY_LABEL) -> int:
    """Return the primary index and require it to be the final output (11 for 12 labels)."""
    names = [str(name) for name in label_names]
    if primary_label not in names:
        raise ValueError(f"Primary label missing: {primary_label}")
    index = names.index(primary_label)
    required = len(names) - 1
    if index != required:
        required_text = "index 11" if len(names) == 12 else f"index {required}"
        raise ValueError(f"Primary label must be the final output at {required_text}; found {index}")
    return index


def ensemble_probabilities(probabilities_by_seed: Sequence[np.ndarray]) -> np.ndarray:
    """Average same-patient probabilities across fitted seeds without concatenation."""
    if not probabilities_by_seed:
        raise ValueError("At least one seed probability array is required")
    arrays = [np.asarray(values, dtype=float) for values in probabilities_by_seed]
    if any(values.shape != arrays[0].shape for values in arrays[1:]):
        raise ValueError("Seed probability arrays have different shapes")
    return np.mean(np.stack(arrays, axis=0), axis=0)


def safe_binary_metric(labels: np.ndarray, probabilities: np.ndarray, metric: str) -> float:
    """Compute AUROC or average precision, returning NaN for single-class labels."""
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    if np.unique(y_true).size < 2:
        return float("nan")
    if metric == "auroc":
        return float(roc_auc_score(y_true, y_prob))
    if metric == "auprc":
        return float(average_precision_score(y_true, y_prob))
    raise ValueError(f"Unknown metric: {metric}")


def benjamini_hochberg(p_values: Iterable[float]) -> np.ndarray:
    """Return BH-FDR adjusted q-values while preserving input order and NaNs."""
    values = np.asarray(list(p_values), dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        return result
    order = valid[np.argsort(values[valid])]
    ranked = values[order]
    adjusted = ranked * valid.size / np.arange(1, valid.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[order] = np.minimum(adjusted, 1.0)
    return result


def bootstrap_two_sided_pvalue(differences: np.ndarray) -> float:
    """Compute a plus-one corrected two-sided bootstrap tail probability around zero."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    lower = (np.count_nonzero(values <= 0.0) + 1) / (values.size + 1)
    upper = (np.count_nonzero(values >= 0.0) + 1) / (values.size + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def align_prediction_bundle(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    bundle: dict[str, np.ndarray],
    expected_label_names: Sequence[str],
    primary_label: str = PRIMARY_LABEL,
) -> pd.DataFrame:
    """Align one NPZ bundle to metadata via row IDs after strict integrity checks."""
    required_columns = {"patient_key", "ecg_key"}
    missing_columns = required_columns.difference(metadata.columns)
    if missing_columns:
        raise ValueError(f"Metadata missing columns: {sorted(missing_columns)}")
    if metadata["patient_key"].duplicated().any():
        raise ValueError("Metadata contains duplicate patient_id values")
    required_bundle_keys = {
        "row_id",
        "patient_id",
        "record_id",
        "labels",
        "probabilities",
        "label_names",
    }
    missing_bundle_keys = required_bundle_keys.difference(bundle)
    if missing_bundle_keys:
        raise ValueError(f"Prediction bundle missing keys: {sorted(missing_bundle_keys)}")
    n_rows = len(metadata)
    row_id = np.asarray(bundle["row_id"], dtype=np.int64)
    patient_ids = np.asarray(bundle["patient_id"]).astype(str)
    record_ids = np.asarray(bundle["record_id"]).astype(str)
    bundle_labels = np.asarray(bundle["labels"])
    probabilities = np.asarray(bundle["probabilities"])
    if not (
        len(row_id)
        == len(patient_ids)
        == len(record_ids)
        == len(bundle_labels)
        == len(probabilities)
        == n_rows
    ):
        raise ValueError("Prediction row count does not match metadata/labels")
    if np.unique(row_id).size != n_rows or set(row_id.tolist()) != set(range(n_rows)):
        raise ValueError("row_id values must be a unique permutation of 0..n-1")
    label_names = [str(value) for value in np.asarray(bundle["label_names"]).tolist()]
    expected = [str(value) for value in expected_label_names]
    if label_names != expected:
        raise ValueError("Prediction label names/order do not match expected labels")
    primary_index = validate_primary_label_position(label_names, primary_label)
    order = np.argsort(row_id)
    expected_patient_ids = metadata["patient_key"].fillna("missing").astype(str).to_numpy()
    expected_record_ids = metadata["ecg_key"].fillna("missing").astype(str).to_numpy()
    if not np.array_equal(patient_ids[order], expected_patient_ids):
        raise ValueError("Prediction patient_id values do not align with metadata")
    if not np.array_equal(record_ids[order], expected_record_ids):
        raise ValueError("Prediction record_id values do not align with metadata")
    aligned_labels = bundle_labels[order]
    expected_labels = np.asarray(labels)
    if aligned_labels.shape != expected_labels.shape or not np.array_equal(aligned_labels, expected_labels):
        raise ValueError("Prediction labels do not align with label CSV")
    aligned_probabilities = probabilities[order]
    validate_probabilities(aligned_probabilities, "prediction bundle")
    output = metadata.reset_index(drop=True).copy()
    output["primary_label"] = expected_labels[:, primary_index].astype(int)
    output["primary_probability"] = aligned_probabilities[:, primary_index].astype(float)
    return output


def paired_bootstrap_metrics(
    labels: np.ndarray,
    scratch_probabilities: np.ndarray,
    fcmae_probabilities: np.ndarray,
    iterations: int,
    seed: int,
    max_attempt_multiplier: int = 50,
) -> dict[str, np.ndarray]:
    """Draw shared patient indices and return model metrics and paired deltas."""
    y_true = np.asarray(labels, dtype=int)
    scratch = np.asarray(scratch_probabilities, dtype=float)
    fcmae = np.asarray(fcmae_probabilities, dtype=float)
    if not (len(y_true) == len(scratch) == len(fcmae)):
        raise ValueError("Labels and model predictions must have equal lengths")
    if np.unique(y_true).size < 2:
        return {name: np.asarray([], dtype=float) for name in (
            "scratch_auroc", "fcmae_auroc", "delta_auroc",
            "scratch_auprc", "fcmae_auprc", "delta_auprc",
        )}
    rng = np.random.default_rng(seed)
    collected = {name: [] for name in (
        "scratch_auroc", "fcmae_auroc", "delta_auroc",
        "scratch_auprc", "fcmae_auprc", "delta_auprc",
    )}
    attempts = 0
    maximum_attempts = max(iterations * max_attempt_multiplier, iterations)
    while len(collected["delta_auroc"]) < iterations and attempts < maximum_attempts:
        attempts += 1
        indices = rng.integers(0, len(y_true), size=len(y_true))
        sampled_labels = y_true[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        scratch_auroc = safe_binary_metric(sampled_labels, scratch[indices], "auroc")
        fcmae_auroc = safe_binary_metric(sampled_labels, fcmae[indices], "auroc")
        scratch_auprc = safe_binary_metric(sampled_labels, scratch[indices], "auprc")
        fcmae_auprc = safe_binary_metric(sampled_labels, fcmae[indices], "auprc")
        collected["scratch_auroc"].append(scratch_auroc)
        collected["fcmae_auroc"].append(fcmae_auroc)
        collected["delta_auroc"].append(fcmae_auroc - scratch_auroc)
        collected["scratch_auprc"].append(scratch_auprc)
        collected["fcmae_auprc"].append(fcmae_auprc)
        collected["delta_auprc"].append(fcmae_auprc - scratch_auprc)
    if len(collected["delta_auroc"]) != iterations:
        raise RuntimeError(
            f"Only {len(collected['delta_auroc'])}/{iterations} valid bootstrap draws "
            f"after {attempts} attempts"
        )
    return {name: np.asarray(values, dtype=float) for name, values in collected.items()}
