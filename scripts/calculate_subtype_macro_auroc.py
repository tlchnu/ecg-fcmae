#!/usr/bin/env python
"""Calculate EchoNext Macro AUROC over 11 SHD subtypes.

The aggregate ``shd_moderate_or_greater_flag`` endpoint is excluded by name.
Prediction-backed rows are calculated exactly. Historical rows can be recovered
from the stored 12-label macro and aggregate AUROC, but are explicitly marked as
derived from rounded metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


AGGREGATE_LABEL = "shd_moderate_or_greater_flag"
N_SUBTYPES = 11
DERIVED_ROUNDING_UNCERTAINTY_MAX = 0.00006
TARGET_KEY_ALIASES = ("y_true", "targets", "labels")
SCORE_KEY_ALIASES = ("y_score", "probabilities", "predictions", "probs")
LABEL_KEY_ALIASES = ("label_names", "class_names")


def select_subtype_indices(
    label_names: Sequence[str], aggregate_label: str = AGGREGATE_LABEL
) -> list[int]:
    """Return indices of exactly 11 subtype labels, excluding aggregate SHD."""
    normalized = [str(label) for label in label_names]
    if len(normalized) != len(set(normalized)):
        raise ValueError("label_names contains duplicate values")
    aggregate_count = normalized.count(aggregate_label)
    if aggregate_count != 1:
        raise ValueError(
            f"expected aggregate label {aggregate_label!r} exactly once, found {aggregate_count}"
        )
    indices = [index for index, label in enumerate(normalized) if label != aggregate_label]
    if len(indices) != N_SUBTYPES:
        raise ValueError(f"expected {N_SUBTYPES} subtype labels, found {len(indices)}")
    return indices


def _validate_prediction_arrays(
    y_true: np.ndarray, y_score: np.ndarray, label_names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    if targets.ndim != 2 or scores.ndim != 2:
        raise ValueError("y_true and y_score must both be two-dimensional")
    if targets.shape != scores.shape:
        raise ValueError(
            f"y_true and y_score shapes differ: {targets.shape} versus {scores.shape}"
        )
    if targets.shape[1] != len(label_names):
        raise ValueError(
            f"array has {targets.shape[1]} labels but label_names has {len(label_names)}"
        )
    if not np.isfinite(targets).all() or not np.isfinite(scores).all():
        raise ValueError("prediction arrays contain non-finite values")
    unique_targets = set(np.unique(targets).tolist())
    if not unique_targets.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"y_true must be binary, found values {sorted(unique_targets)}")
    return targets.astype(int, copy=False), scores


def _binary_auroc_with_fallback(
    targets: np.ndarray, scores: np.ndarray
) -> tuple[float, bool]:
    if np.unique(targets).size < 2:
        return 0.5, True
    return float(roc_auc_score(targets, scores)), False


def compute_subtype_aurocs(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_names: Sequence[str],
    aggregate_label: str = AGGREGATE_LABEL,
) -> dict[str, Any]:
    """Compute per-subtype and 11-label Macro AUROC using a 0.5 fallback."""
    normalized_labels = [str(label) for label in label_names]
    subtype_indices = select_subtype_indices(normalized_labels, aggregate_label)
    targets, scores = _validate_prediction_arrays(y_true, y_score, normalized_labels)
    per_label_auroc: dict[str, float] = {}
    fallback_labels: list[str] = []
    for index in subtype_indices:
        label = normalized_labels[index]
        auroc, used_fallback = _binary_auroc_with_fallback(
            targets[:, index], scores[:, index]
        )
        per_label_auroc[label] = auroc
        if used_fallback:
            fallback_labels.append(label)
    return {
        "subtype_macro_auroc": float(np.mean(list(per_label_auroc.values()))),
        "per_label_auroc": per_label_auroc,
        "fallback_labels": fallback_labels,
        "n_subtypes": len(subtype_indices),
        "aggregate_label": aggregate_label,
    }


def derive_subtype_macro(macro_12: float, aggregate_auroc: float) -> float:
    """Recover the 11-subtype macro from 12-label macro and aggregate AUROC."""
    macro_value = float(macro_12)
    aggregate_value = float(aggregate_auroc)
    if not np.isfinite([macro_value, aggregate_value]).all():
        raise ValueError("macro_12 and aggregate_auroc must be finite")
    result = (12.0 * macro_value - aggregate_value) / 11.0
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"derived subtype macro is outside [0, 1]: {result}")
    return float(result)


def _unique_npz_key(files: Sequence[str], aliases: Sequence[str], role: str) -> str:
    matches = [key for key in aliases if key in files]
    if len(matches) > 1:
        raise ValueError(f"ambiguous {role} arrays in prediction bundle: {matches}")
    if not matches:
        raise ValueError(f"prediction bundle has no supported {role} key: {aliases}")
    return matches[0]


def load_prediction_bundle(
    path: Path, label_names: Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load targets, scores, and label names from a supported NumPy bundle."""
    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        target_key = _unique_npz_key(bundle.files, TARGET_KEY_ALIASES, "target")
        score_key = _unique_npz_key(bundle.files, SCORE_KEY_ALIASES, "score")
        label_matches = [key for key in LABEL_KEY_ALIASES if key in bundle.files]
        if len(label_matches) > 1:
            raise ValueError(f"ambiguous label-name arrays in prediction bundle: {label_matches}")
        if label_matches:
            loaded_labels = [str(value) for value in bundle[label_matches[0]].tolist()]
            if label_names is not None and loaded_labels != [str(value) for value in label_names]:
                raise ValueError("bundle label names differ from explicitly supplied label names")
        elif label_names is not None:
            loaded_labels = [str(value) for value in label_names]
        else:
            raise ValueError("prediction bundle has no label names and none were supplied")
        targets = np.asarray(bundle[target_key])
        scores = np.asarray(bundle[score_key], dtype=float)
    _validate_prediction_arrays(targets, scores, loaded_labels)
    return targets, scores, loaded_labels


def build_derived_record(
    *,
    experiment: str,
    seed: int,
    macro_12: float,
    aggregate_auroc: float,
    source_path: str,
    method: str | None = None,
    experiment_group: str | None = None,
) -> dict[str, Any]:
    """Create one auditable row derived from stored rounded metrics."""
    return {
        "experiment": experiment,
        "method": method or experiment,
        "experiment_group": experiment_group or "other",
        "seed": int(seed),
        "subtype_macro_auroc": derive_subtype_macro(macro_12, aggregate_auroc),
        "primary_shd_auroc": float(aggregate_auroc),
        "macro_12_auroc": float(macro_12),
        "n_subtypes": N_SUBTYPES,
        "fallback_count": np.nan,
        "fallback_labels": "",
        "evidence_source": "derived_from_rounded_metrics",
        "rounding_uncertainty_max": DERIVED_ROUNDING_UNCERTAINTY_MAX,
        "source_path": source_path,
    }


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _exact_record(
    entry: dict[str, Any], project_root: Path, aggregate_label: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prediction_path = _resolve_path(project_root, entry["prediction_path"])
    targets, scores, labels = load_prediction_bundle(
        prediction_path, entry.get("label_names")
    )
    result = compute_subtype_aurocs(targets, scores, labels, aggregate_label)
    aggregate_index = labels.index(aggregate_label)
    aggregate_auroc, aggregate_fallback = _binary_auroc_with_fallback(
        targets[:, aggregate_index], scores[:, aggregate_index]
    )
    macro_12 = (11.0 * result["subtype_macro_auroc"] + aggregate_auroc) / 12.0
    evidence_source = entry.get("evidence_source", "exact_from_predictions")
    if evidence_source not in {"exact_from_predictions", "exact_from_checkpoint"}:
        raise ValueError(f"invalid exact evidence_source: {evidence_source}")
    record = {
        "experiment": entry["experiment"],
        "method": entry.get("method", entry["experiment"]),
        "experiment_group": entry.get("experiment_group", "other"),
        "seed": int(entry["seed"]),
        "subtype_macro_auroc": result["subtype_macro_auroc"],
        "primary_shd_auroc": aggregate_auroc,
        "macro_12_auroc": macro_12,
        "n_subtypes": result["n_subtypes"],
        "fallback_count": len(result["fallback_labels"]) + int(aggregate_fallback),
        "fallback_labels": ";".join(result["fallback_labels"]),
        "evidence_source": evidence_source,
        "rounding_uncertainty_max": 0.0,
        "source_path": str(prediction_path),
    }
    per_label_rows = [
        {
            "experiment": entry["experiment"],
            "method": record["method"],
            "experiment_group": record["experiment_group"],
            "seed": int(entry["seed"]),
            "label": label,
            "auroc": auroc,
            "used_single_class_fallback": label in result["fallback_labels"],
            "source_path": str(prediction_path),
        }
        for label, auroc in result["per_label_auroc"].items()
    ]
    return record, per_label_rows


def _derived_record_from_entry(
    entry: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    if "macro_12" in entry and "aggregate_auroc" in entry:
        macro_12 = float(entry["macro_12"])
        aggregate_auroc = float(entry["aggregate_auroc"])
        source_path = str(entry.get("source_path", "manifest"))
    else:
        metrics_path = _resolve_path(project_root, entry["metrics_path"])
        frame = pd.read_csv(metrics_path)
        row_index = int(entry.get("row_index", 0))
        row = frame.iloc[row_index]
        macro_12 = float(row[entry.get("macro_12_column", "Test_Macro_AUROC")])
        aggregate_auroc = float(
            row[entry.get("aggregate_column", "Test_Primary_AUROC")]
        )
        source_path = str(metrics_path)
    return build_derived_record(
        experiment=entry["experiment"],
        method=entry.get("method"),
        experiment_group=entry.get("experiment_group"),
        seed=int(entry["seed"]),
        macro_12=macro_12,
        aggregate_auroc=aggregate_auroc,
        source_path=source_path,
    )


def _summarize_by_method(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (group, method), method_frame in frame.groupby(
        ["experiment_group", "method"], dropna=False, sort=True
    ):
        values = method_frame["subtype_macro_auroc"].astype(float)
        rows.append(
            {
                "experiment_group": group,
                "method": method,
                "n_seeds": int(method_frame["seed"].nunique()),
                "subtype_macro_auroc_mean": float(values.mean()),
                "subtype_macro_auroc_std": (
                    float(values.std(ddof=1)) if len(values) > 1 else np.nan
                ),
                "subtype_macro_auroc_min": float(values.min()),
                "subtype_macro_auroc_max": float(values.max()),
                "evidence_sources": ";".join(
                    sorted(method_frame["evidence_source"].astype(str).unique())
                ),
            }
        )
    return pd.DataFrame(rows)


def run_manifest(manifest_path: Path, project_root: Path, output_dir: Path) -> None:
    """Evaluate all manifest entries and write auditable result artifacts."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    aggregate_label = manifest.get("aggregate_label", AGGREGATE_LABEL)
    records: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    for entry in manifest["experiments"]:
        if "prediction_path" in entry:
            record, label_rows = _exact_record(entry, project_root, aggregate_label)
            records.append(record)
            per_label_rows.extend(label_rows)
        else:
            records.append(_derived_record_from_entry(entry, project_root))
    by_seed = pd.DataFrame(records).sort_values(
        ["experiment_group", "method", "seed"], kind="stable"
    )
    summary = _summarize_by_method(by_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_label_rows).to_csv(output_dir / "per_label_auroc.csv", index=False)
    by_seed.to_csv(output_dir / "subtype_macro_auroc_by_seed.csv", index=False)
    summary.to_csv(output_dir / "subtype_macro_auroc_summary.csv", index=False)
    summary.sort_values(
        "subtype_macro_auroc_mean", ascending=False, kind="stable"
    ).to_csv(output_dir / "all_experiments_subtype_macro_auroc.csv", index=False)
    calculation_manifest = {
        **manifest,
        "metric_definition": "arithmetic mean of 11 subtype one-vs-rest AUROCs",
        "aggregate_exclusion": aggregate_label,
        "single_class_fallback": 0.5,
        "derived_formula": "(12 * macro_12_auroc - primary_shd_auroc) / 11",
        "derived_rounding_uncertainty_max": DERIVED_ROUNDING_UNCERTAINTY_MAX,
        "n_rows": int(len(by_seed)),
        "n_exact_rows": int(
            by_seed["evidence_source"].str.startswith("exact_").sum()
        ),
    }
    (output_dir / "calculation_manifest.json").write_text(
        json.dumps(calculation_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    readme = (
        "# EchoNext 11-subtype Macro AUROC\n\n"
        "The aggregate SHD endpoint is excluded by exact label name. "
        "Rows marked `exact_from_predictions` are recomputed from test predictions; "
        "rows marked `derived_from_rounded_metrics` use "
        "`(12 * Macro12 - SHD_AUROC) / 11` and carry a conservative ±0.00006 "
        "rounding uncertainty. A single-class subtype contributes AUROC 0.5, "
        "matching ECG-FM benchmarking behavior.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    hash_lines = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "sha256sums.txt":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hash_lines.append(f"{digest}  {path.name}")
    (output_dir / "sha256sums.txt").write_text(
        "\n".join(hash_lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/20260721_subtype_macro_auroc"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else project_root / args.output_dir
    )
    run_manifest(args.manifest, project_root, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
