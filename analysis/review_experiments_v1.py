#!/usr/bin/env python
"""Summarize auditable post-review experiment prediction bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ANALYSIS_ROOT = Path(__file__).resolve().parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from subgroup_utils_v1 import paired_bootstrap_metrics


PRIMARY_LABEL = "shd_moderate_or_greater_flag"
METRIC_COLUMNS = (
    "primary_shd_auroc",
    "primary_shd_auprc",
    "macro_auroc_11",
    "macro_auroc_12",
)


def compute_exact_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    label_names: Sequence[str],
) -> dict[str, float]:
    """Return exact aggregate and subtype metrics from one prediction bundle."""
    y_true = np.asarray(targets)
    y_prob = np.asarray(probabilities, dtype=float)
    names = [str(value) for value in label_names]
    if y_true.shape != y_prob.shape or y_true.ndim != 2:
        raise ValueError("targets and probabilities must be matching 2-D arrays")
    if y_true.shape[1] != len(names):
        raise ValueError("label_names length does not match prediction width")
    if names.count(PRIMARY_LABEL) != 1:
        raise ValueError(f"expected {PRIMARY_LABEL!r} exactly once")
    if not np.isfinite(y_prob).all() or np.min(y_prob) < 0.0 or np.max(y_prob) > 1.0:
        raise ValueError("probabilities must be finite and within [0, 1]")
    per_label = []
    for index in range(y_true.shape[1]):
        if np.unique(y_true[:, index]).size < 2:
            per_label.append(float("nan"))
        else:
            per_label.append(float(roc_auc_score(y_true[:, index], y_prob[:, index])))
    primary_index = names.index(PRIMARY_LABEL)
    subtype_indices = [index for index in range(len(names)) if index != primary_index]
    return {
        "primary_shd_auroc": per_label[primary_index],
        "primary_shd_auprc": float(
            average_precision_score(y_true[:, primary_index], y_prob[:, primary_index])
        ),
        "macro_auroc_11": float(np.nanmean(np.asarray(per_label)[subtype_indices])),
        "macro_auroc_12": float(np.nanmean(per_label)),
    }


def summarize_seed_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float | int]:
    """Return means and sample standard deviations across downstream seeds."""
    if not rows:
        raise ValueError("at least one seed metric row is required")
    summary: dict[str, float | int] = {"n_seeds": len(rows)}
    for column in METRIC_COLUMNS:
        values = np.asarray([float(row[column]) for row in rows], dtype=float)
        summary[f"{column}_mean"] = float(np.mean(values))
        summary[f"{column}_std"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        )
    return summary


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    """Load and validate a prediction NPZ without pickle support."""
    with np.load(path, allow_pickle=False) as bundle:
        required = {"labels", "probabilities", "label_names", "split_hash", "seed"}
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        return {key: bundle[key] for key in bundle.files}


def analyze_experiment(
    experiment_dir: Path,
    seeds: Sequence[int],
    reference_template: str | None,
    bootstrap_iters: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Analyze one experiment directory and write reproducible artifacts."""
    prediction_dir = experiment_dir / "predictions"
    rows: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    raw_bootstrap: dict[str, np.ndarray] = {}
    for seed in seeds:
        bundles = {}
        for split in ("val", "test"):
            path = prediction_dir / f"seed_{seed}_{split}.npz"
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            bundle = load_bundle(path)
            if int(np.asarray(bundle["seed"]).item()) != int(seed):
                raise ValueError(f"seed metadata mismatch in {path}")
            bundles[split] = bundle
            alignment_rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "split_hash": str(np.asarray(bundle["split_hash"]).item()),
                    "prediction_sha256": file_sha256(path),
                    "n_rows": int(bundle["labels"].shape[0]),
                }
            )
        test_bundle = bundles["test"]
        metrics = compute_exact_metrics(
            test_bundle["labels"],
            test_bundle["probabilities"],
            test_bundle["label_names"].tolist(),
        )
        rows.append({"seed": seed, **metrics})
        if reference_template:
            reference_path = Path(reference_template.format(seed=seed))
            reference = load_bundle(reference_path)
            hashes_match = str(np.asarray(reference["split_hash"]).item()) == str(
                np.asarray(test_bundle["split_hash"]).item()
            )
            arrays_align = (
                np.array_equal(reference["labels"], test_bundle["labels"])
                and np.array_equal(reference.get("row_id"), test_bundle.get("row_id"))
            )
            if not hashes_match and not arrays_align:
                raise ValueError(f"reference rows or labels do not align for seed {seed}")
            names = [str(value) for value in test_bundle["label_names"].tolist()]
            primary_index = names.index(PRIMARY_LABEL)
            boot = paired_bootstrap_metrics(
                test_bundle["labels"][:, primary_index],
                reference["probabilities"][:, primary_index],
                test_bundle["probabilities"][:, primary_index],
                iterations=bootstrap_iters,
                seed=bootstrap_seed + int(seed),
                max_attempt_multiplier=100,
            )
            paired_rows.append(
                {
                    "seed": seed,
                    "reference_path": str(reference_path),
                    "delta_auroc": float(np.mean(boot["delta_auroc"])),
                    "delta_auroc_ci_low": float(np.percentile(boot["delta_auroc"], 2.5)),
                    "delta_auroc_ci_high": float(np.percentile(boot["delta_auroc"], 97.5)),
                    "delta_auprc": float(np.mean(boot["delta_auprc"])),
                    "delta_auprc_ci_low": float(np.percentile(boot["delta_auprc"], 2.5)),
                    "delta_auprc_ci_high": float(np.percentile(boot["delta_auprc"], 97.5)),
                    "bootstrap_n": bootstrap_iters,
                }
            )
            for key, values in boot.items():
                raw_bootstrap[f"seed_{seed}_{key}"] = values
    summary = summarize_seed_metrics(rows)
    pd.DataFrame(rows).to_csv(experiment_dir / "exact_seed_metrics.csv", index=False)
    pd.DataFrame([summary]).to_csv(experiment_dir / "exact_summary.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(experiment_dir / "alignment_hashes.csv", index=False)
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(experiment_dir / "paired_bootstrap.csv", index=False)
        np.savez_compressed(experiment_dir / "paired_bootstrap_raw.npz", **raw_bootstrap)
    status = {
        "status": "complete",
        "experiment_dir": str(experiment_dir),
        "seeds": [int(seed) for seed in seeds],
        "n_seed_rows": len(rows),
        "bootstrap_iterations": bootstrap_iters if paired_rows else 0,
        "reference_template": reference_template,
    }
    (experiment_dir / "review_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--reference-template", default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    status = analyze_experiment(
        args.experiment_dir,
        args.expected_seeds,
        args.reference_template,
        args.bootstrap_iters,
        args.bootstrap_seed,
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
