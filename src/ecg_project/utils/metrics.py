"""Multi-label classification metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _binary_arrays(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return flat binary targets and finite probabilities."""
    y_true = np.asarray(targets, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(probabilities, dtype=float).reshape(-1)
    if y_true.shape != y_prob.shape:
        raise ValueError("targets and probabilities must have identical shapes")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("targets must be binary")
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("probabilities must be finite")
    return y_true, np.clip(y_prob, 0.0, 1.0)


def screening_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Compute screening metrics using the fixed rule ``probability >= threshold``."""
    y_true, y_prob = _binary_arrays(targets, probabilities)
    y_pred = y_prob >= float(threshold)
    positive = y_true == 1
    negative = ~positive
    tp = int(np.sum(y_pred & positive))
    fp = int(np.sum(y_pred & negative))
    tn = int(np.sum(~y_pred & negative))
    fn = int(np.sum(~y_pred & positive))

    def divide(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "ppv": divide(tp, tp + fp),
        "npv": divide(tn, tn + fn),
    }


def select_threshold_for_sensitivity(
    validation_targets: np.ndarray,
    validation_probabilities: np.ndarray,
    min_sensitivity: float = 0.90,
) -> dict[str, float | int]:
    """Select a threshold using validation data only.

    Among thresholds meeting the sensitivity target, specificity is maximized;
    ties are resolved in favor of the higher threshold.
    """
    y_true, y_prob = _binary_arrays(validation_targets, validation_probabilities)
    if len(np.unique(y_true)) != 2:
        raise ValueError("validation targets must contain both positive and negative classes")
    if not (0.0 <= min_sensitivity <= 1.0):
        raise ValueError("min_sensitivity must be in [0, 1]")
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, np.nextafter(1.0, np.inf)], dtype=float),
                y_prob,
            ]
        )
    )
    feasible = []
    for threshold in candidates:
        metrics = screening_metrics(y_true, y_prob, float(threshold))
        if float(metrics["sensitivity"]) + 1e-12 >= min_sensitivity:
            feasible.append(metrics)
    if not feasible:
        raise RuntimeError("no validation threshold meets the sensitivity target")
    return max(
        feasible,
        key=lambda row: (float(row["specificity"]), float(row["threshold"])),
    )


def brier_score(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Return the binary Brier score after finite-boundary clipping."""
    y_true, y_prob = _binary_arrays(targets, probabilities)
    return float(np.mean((y_prob - y_true) ** 2))


def calibration_table(
    targets: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Return fixed equal-width calibration bins and attach weighted ECE."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    y_true, y_prob = _binary_arrays(targets, probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.minimum(np.floor(y_prob * n_bins).astype(int), n_bins - 1)
    rows = []
    ece = 0.0
    for index in range(n_bins):
        mask = bin_index == index
        count = int(np.sum(mask))
        mean_probability = float(np.mean(y_prob[mask])) if count else float("nan")
        event_rate = float(np.mean(y_true[mask])) if count else float("nan")
        if count:
            ece += (count / len(y_true)) * abs(mean_probability - event_rate)
        rows.append(
            {
                "bin_index": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "right_closed": bool(index == n_bins - 1),
                "n": count,
                "mean_probability": mean_probability,
                "event_rate": event_rate,
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["ece"] = float(ece)
    return table


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    primary_index: int | None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute per-label and macro metrics without using test data for tuning."""
    aurocs = []
    auprcs = []
    for label_index in range(targets.shape[1]):
        if len(np.unique(targets[:, label_index])) > 1:
            aurocs.append(
                roc_auc_score(targets[:, label_index], probabilities[:, label_index])
            )
            auprcs.append(
                average_precision_score(
                    targets[:, label_index], probabilities[:, label_index]
                )
            )
        else:
            aurocs.append(float("nan"))
            auprcs.append(float("nan"))
    aurocs_array = np.asarray(aurocs)
    auprcs_array = np.asarray(auprcs)
    macro_auroc = float(np.nanmean(aurocs_array))
    macro_auprc = float(np.nanmean(auprcs_array))
    predictions = (probabilities >= threshold).astype(np.int32)
    macro_f1 = float(f1_score(targets, predictions, average="macro", zero_division=0))
    micro_f1 = float(f1_score(targets, predictions, average="micro", zero_division=0))
    if primary_index is None:
        primary_auroc = macro_auroc
        primary_auprc = macro_auprc
    else:
        primary_auroc = float(aurocs_array[primary_index])
        primary_auprc = float(auprcs_array[primary_index])
    return {
        "primary_auroc": primary_auroc,
        "primary_auprc": primary_auprc,
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
    }
