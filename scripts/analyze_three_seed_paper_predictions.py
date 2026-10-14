"""Create three-seed and ensemble paper statistics from aligned predictions.

This analysis never selects a checkpoint on the test set. It consumes fixed
validation-selected checkpoints, verifies exact row/label alignment, reports
per-seed point estimates, and uses 2,000 patient-level bootstrap resamples for
ensemble 95% confidence intervals and paired Full-vs-Scratch differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PRIMARY_LABEL = "shd_moderate_or_greater_flag"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {"labels", "probabilities", "label_names", "split_hash", "seed"}
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"{path} missing {sorted(missing)}")
        result = {key: bundle[key] for key in bundle.files}
    labels = np.asarray(result["labels"])
    probabilities = np.asarray(result["probabilities"], dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 2:
        raise ValueError(f"invalid shapes in {path}: {labels.shape}, {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"non-finite probabilities in {path}")
    return result


def metric_pair(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    if np.unique(labels).size < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(labels, probabilities)),
        float(average_precision_score(labels, probabilities)),
    )


def per_label_points(
    labels: np.ndarray,
    probabilities: np.ndarray,
    names: list[str],
    seed: str,
) -> list[dict[str, object]]:
    rows = []
    for index, name in enumerate(names):
        auroc, auprc = metric_pair(labels[:, index], probabilities[:, index])
        rows.append(
            {
                "seed": seed,
                "label_index": index,
                "label": name,
                "n": len(labels),
                "positive_n": int(labels[:, index].sum()),
                "prevalence": float(labels[:, index].mean()),
                "auroc": auroc,
                "auprc": auprc,
            }
        )
    return rows


def aggregate_points(labels: np.ndarray, probabilities: np.ndarray, names: list[str]) -> dict[str, float]:
    pairs = [metric_pair(labels[:, index], probabilities[:, index]) for index in range(labels.shape[1])]
    primary_index = names.index(PRIMARY_LABEL)
    subtype_indices = [index for index in range(len(names)) if index != primary_index]
    return {
        "primary_auroc": pairs[primary_index][0],
        "primary_auprc": pairs[primary_index][1],
        "macro_auroc_11": float(np.nanmean([pairs[index][0] for index in subtype_indices])),
        "macro_auroc_12": float(np.nanmean([pair[0] for pair in pairs])),
        "macro_auprc_12": float(np.nanmean([pair[1] for pair in pairs])),
    }


def percentile_interval(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean(array)), float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def bootstrap_ensemble(
    labels: np.ndarray,
    full: np.ndarray,
    scratch: np.ndarray,
    names: list[str],
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    per_label_draws: dict[tuple[int, str], list[float]] = {
        (index, metric): [] for index in range(labels.shape[1]) for metric in ("auroc", "auprc")
    }
    aggregate_draws: dict[str, list[float]] = {
        key: []
        for key in (
            "full_primary_auroc",
            "full_primary_auprc",
            "full_macro_auroc_11",
            "full_macro_auroc_12",
            "full_macro_auprc_12",
            "delta_primary_auroc",
            "delta_primary_auprc",
            "delta_macro_auroc_11",
            "delta_macro_auroc_12",
            "delta_macro_auprc_12",
        )
    }
    completed = 0
    attempts = 0
    max_attempts = iterations * 100
    while completed < iterations and attempts < max_attempts:
        attempts += 1
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        full_points = aggregate_points(sampled_labels, full[indices], names)
        scratch_points = aggregate_points(sampled_labels, scratch[indices], names)
        if not all(np.isfinite(list(full_points.values()) + list(scratch_points.values()))):
            continue
        for key, value in full_points.items():
            aggregate_draws[f"full_{key}"].append(value)
            aggregate_draws[f"delta_{key}"].append(value - scratch_points[key])
        valid = True
        current: dict[tuple[int, str], float] = {}
        for index in range(labels.shape[1]):
            auroc, auprc = metric_pair(sampled_labels[:, index], full[indices, index])
            if not np.isfinite(auroc) or not np.isfinite(auprc):
                valid = False
                break
            current[(index, "auroc")] = auroc
            current[(index, "auprc")] = auprc
        if not valid:
            for values in aggregate_draws.values():
                if len(values) > completed:
                    values.pop()
            continue
        for key, value in current.items():
            per_label_draws[key].append(value)
        completed += 1
    if completed != iterations:
        raise RuntimeError(f"only {completed}/{iterations} valid bootstrap samples after {attempts} attempts")

    label_rows = []
    for index, name in enumerate(names):
        point_auroc, point_auprc = metric_pair(labels[:, index], full[:, index])
        for metric, point in (("auroc", point_auroc), ("auprc", point_auprc)):
            _, low, high = percentile_interval(per_label_draws[(index, metric)])
            label_rows.append(
                {
                    "label_index": index,
                    "label": name,
                    "metric": metric,
                    "point": point,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_n": iterations,
                }
            )

    full_points = aggregate_points(labels, full, names)
    scratch_points = aggregate_points(labels, scratch, names)
    full_rows = []
    paired_rows = []
    for key, point in full_points.items():
        _, low, high = percentile_interval(aggregate_draws[f"full_{key}"])
        full_rows.append(
            {"metric": key, "point": point, "ci_low": low, "ci_high": high, "bootstrap_n": iterations}
        )
        mean_delta, delta_low, delta_high = percentile_interval(aggregate_draws[f"delta_{key}"])
        paired_rows.append(
            {
                "comparison": "full_ensemble - strict_no_pretraining_ensemble",
                "metric": key,
                "point_difference": point - scratch_points[key],
                "bootstrap_mean_difference": mean_delta,
                "ci_low": delta_low,
                "ci_high": delta_high,
                "bootstrap_n": iterations,
            }
        )
    raw = {key: np.asarray(value, dtype=np.float32) for key, value in aggregate_draws.items()}
    for (index, metric), values in per_label_draws.items():
        raw[f"full_label_{index}_{metric}"] = np.asarray(values, dtype=np.float32)
    return pd.DataFrame(label_rows), pd.DataFrame(full_rows), pd.DataFrame(paired_rows), raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-template", required=True)
    parser.add_argument("--scratch-template", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 3407])
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_reference = None
    names_reference = None
    split_hash_reference = None
    full_probabilities = []
    scratch_probabilities = []
    seed_rows = []
    alignment_rows = []
    for seed in args.seeds:
        full_path = Path(args.full_template.format(seed=seed))
        scratch_path = Path(args.scratch_template.format(seed=seed))
        full = load_prediction(full_path)
        scratch = load_prediction(scratch_path)
        labels = np.asarray(full["labels"])
        names = [str(value) for value in full["label_names"].tolist()]
        split_hash = str(np.asarray(full["split_hash"]).item())
        if int(np.asarray(full["seed"]).item()) != seed or int(np.asarray(scratch["seed"]).item()) != seed:
            raise ValueError(f"seed metadata mismatch for seed {seed}")
        if labels_reference is None:
            labels_reference, names_reference, split_hash_reference = labels, names, split_hash
        if not np.array_equal(labels_reference, labels) or names_reference != names or split_hash_reference != split_hash:
            raise ValueError(f"Full prediction alignment mismatch for seed {seed}")
        if not np.array_equal(labels_reference, scratch["labels"]):
            raise ValueError(f"Scratch labels do not align for seed {seed}")
        if names_reference != [str(value) for value in scratch["label_names"].tolist()]:
            raise ValueError(f"Scratch label names do not align for seed {seed}")
        full_probability = np.asarray(full["probabilities"], dtype=np.float64)
        scratch_probability = np.asarray(scratch["probabilities"], dtype=np.float64)
        full_probabilities.append(full_probability)
        scratch_probabilities.append(scratch_probability)
        seed_rows.extend(per_label_points(labels, full_probability, names, str(seed)))
        alignment_rows.extend(
            [
                {"model": "full", "seed": seed, "path": str(full_path), "sha256": file_sha256(full_path), "split_hash": split_hash},
                {"model": "strict_no_pretraining", "seed": seed, "path": str(scratch_path), "sha256": file_sha256(scratch_path), "split_hash": str(np.asarray(scratch["split_hash"]).item())},
            ]
        )

    full_ensemble = np.mean(np.stack(full_probabilities), axis=0)
    scratch_ensemble = np.mean(np.stack(scratch_probabilities), axis=0)
    seed_rows.extend(per_label_points(labels_reference, full_ensemble, names_reference, "ensemble"))
    pd.DataFrame(seed_rows).to_csv(args.output_dir / "full_per_label_seed_and_ensemble.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(args.output_dir / "prediction_alignment_sha256.csv", index=False)
    label_ci, full_ci, paired_ci, raw = bootstrap_ensemble(
        labels_reference,
        full_ensemble,
        scratch_ensemble,
        names_reference,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    label_ci.to_csv(args.output_dir / "full_ensemble_per_label_95ci.csv", index=False)
    full_ci.to_csv(args.output_dir / "full_ensemble_aggregate_95ci.csv", index=False)
    paired_ci.to_csv(args.output_dir / "full_vs_strict_ensemble_paired_95ci.csv", index=False)
    np.savez_compressed(args.output_dir / "bootstrap_raw.npz", **raw)
    status = {
        "status": "complete",
        "seeds": args.seeds,
        "n_test": int(len(labels_reference)),
        "n_labels": int(labels_reference.shape[1]),
        "bootstrap_iterations": args.bootstrap_iters,
        "bootstrap_unit": "patient row; one row per patient verified by prediction manifests",
        "confidence_interval": "percentile 95%",
        "paired_indices": True,
    }
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
