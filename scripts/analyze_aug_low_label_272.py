#!/usr/bin/env python
"""Audit augmented low-label runs and compute paired three-seed statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from subgroup_utils_v1 import paired_bootstrap_metrics  # noqa: E402


PRIMARY_LABEL = "shd_moderate_or_greater_flag"
FRACTIONS = ((0.01, "01"), (0.05, "05"), (0.10, "10"), (0.30, "30"))
METHODS = ("ssl", "scratch")


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(path: Path) -> dict[str, np.ndarray]:
    """Load a prediction bundle without pickle."""
    with np.load(path, allow_pickle=False) as bundle:
        required = {
            "row_id", "patient_id", "labels", "probabilities", "label_names",
            "split_hash", "seed",
        }
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"{path} missing arrays: {sorted(missing)}")
        return {key: bundle[key] for key in bundle.files}


def primary_index(bundle: dict[str, np.ndarray]) -> int:
    """Return the unique overall-SHD label index."""
    names = [str(value) for value in bundle["label_names"].tolist()]
    if names.count(PRIMARY_LABEL) != 1:
        raise ValueError(f"expected {PRIMARY_LABEL} once, found {names}")
    return names.index(PRIMARY_LABEL)


def primary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Return exact binary AUROC and AUPRC."""
    return (
        float(roc_auc_score(labels, probabilities)),
        float(average_precision_score(labels, probabilities)),
    )


def validate_pair(
    scratch: dict[str, np.ndarray],
    ssl: dict[str, np.ndarray],
    expected_seed: int,
) -> int:
    """Require identical ordered patients, labels, and label schema."""
    for name, bundle in (("scratch", scratch), ("ssl", ssl)):
        if int(np.asarray(bundle["seed"]).item()) != expected_seed:
            raise ValueError(f"{name} prediction seed mismatch")
        probabilities = np.asarray(bundle["probabilities"], dtype=float)
        if not np.isfinite(probabilities).all() or np.min(probabilities) < 0 or np.max(probabilities) > 1:
            raise ValueError(f"{name} probabilities invalid")
    for key in ("row_id", "patient_id", "labels", "label_names"):
        if not np.array_equal(scratch[key], ssl[key]):
            raise ValueError(f"SSL/scratch mismatch for {key}, seed {expected_seed}")
    if str(np.asarray(scratch["split_hash"]).item()) != str(np.asarray(ssl["split_hash"]).item()):
        raise ValueError(f"split hash mismatch for seed {expected_seed}")
    return primary_index(ssl)


def validate_subsets(low_root: Path) -> pd.DataFrame:
    """Validate paired subset identity, permutation identity, and set nesting."""
    rows: list[dict[str, object]] = []
    indices_by_fraction: dict[float, np.ndarray] = {}
    permutation_hashes: set[str] = set()
    for fraction, tag in FRACTIONS:
        paired_indices = None
        for method in METHODS:
            directory = low_root / f"{method}_{tag}pct"
            manifest_path = directory / "train_subset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            candidates = sorted(directory.glob("train_subset_fraction_*.npy"))
            if len(candidates) != 1:
                raise ValueError(f"expected one subset index file in {directory}")
            index_path = candidates[0]
            indices = np.asarray(np.load(index_path), dtype=np.int64)
            if len(np.unique(indices)) != len(indices):
                raise ValueError(f"duplicate subset indices in {index_path}")
            actual_hash = sha256_file(index_path)
            if actual_hash != manifest["indices_sha256"]:
                raise ValueError(f"subset SHA mismatch in {directory}")
            if paired_indices is None:
                paired_indices = indices
            elif not np.array_equal(paired_indices, indices):
                raise ValueError(f"SSL/scratch subset mismatch at fraction {fraction}")
            permutation_hashes.add(str(manifest["permutation_sha256"]))
            rows.append(
                {
                    "fraction": fraction,
                    "method": method,
                    "n_total": int(manifest["n_total"]),
                    "n_selected": int(manifest["n_selected"]),
                    "permutation_seed": int(manifest["permutation_seed"]),
                    "permutation_sha256": manifest["permutation_sha256"],
                    "indices_sha256": actual_hash,
                    "index_path": str(index_path.resolve()),
                }
            )
        indices_by_fraction[fraction] = np.asarray(paired_indices)
    if len(permutation_hashes) != 1:
        raise ValueError(f"multiple permutation hashes: {sorted(permutation_hashes)}")
    ordered = sorted(indices_by_fraction)
    for smaller, larger in zip(ordered, ordered[1:]):
        if not set(indices_by_fraction[smaller]).issubset(set(indices_by_fraction[larger])):
            raise ValueError(f"subsets are not nested: {smaller} !<= {larger}")
    return pd.DataFrame(rows)


def require_review(experiment_dir: Path, seeds: list[int]) -> None:
    """Require a completed three-seed review for one experiment."""
    status = json.loads((experiment_dir / "review_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete" or status.get("seeds") != seeds:
        raise ValueError(f"incomplete review: {experiment_dir}: {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--strict-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 3407])
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    low_root = args.low_root.resolve()
    full_dir = args.full_dir.resolve()
    strict_dir = args.strict_dir.resolve()
    seeds = [int(seed) for seed in args.seeds]
    analysis_dir = low_root / "analysis"
    provenance_dir = low_root / "provenance"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    subset_frame = validate_subsets(low_root)
    subset_frame.to_csv(provenance_dir / "subset_provenance.csv", index=False)

    metric_frames: list[pd.DataFrame] = []
    alignment_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    raw_bootstrap: dict[str, np.ndarray] = {}

    experiment_dirs: dict[tuple[float, str], Path] = {}
    for fraction, tag in FRACTIONS:
        for method in METHODS:
            directory = low_root / f"{method}_{tag}pct"
            require_review(directory, seeds)
            experiment_dirs[(fraction, method)] = directory
            metrics = pd.read_csv(directory / "exact_seed_metrics.csv")
            if metrics["seed"].astype(int).tolist() != seeds:
                raise ValueError(f"unexpected seed rows in {directory}")
            metrics.insert(0, "fraction", fraction)
            metrics.insert(1, "method", method)
            metric_frames.append(metrics)
            for seed in seeds:
                for split in ("val", "test"):
                    path = directory / "predictions" / f"seed_{seed}_{split}.npz"
                    bundle = load_bundle(path)
                    expected_rows = 4626 if split == "val" else 5442
                    patient_ids = bundle["patient_id"].astype(str)
                    if len(patient_ids) != expected_rows or len(set(patient_ids)) != expected_rows:
                        raise ValueError(f"row/patient count failure: {path}")
                    alignment_rows.append(
                        {
                            "fraction": fraction,
                            "method": method,
                            "seed": seed,
                            "split": split,
                            "n_rows": expected_rows,
                            "split_hash": str(np.asarray(bundle["split_hash"]).item()),
                            "prediction_sha256": sha256_file(path),
                            "path": str(path.resolve()),
                        }
                    )

    # Add 100% aliases from the already frozen Full/Strict experiments.
    for method, directory in (("ssl", full_dir), ("scratch", strict_dir)):
        require_review(directory, seeds)
        metrics = pd.read_csv(directory / "exact_seed_metrics.csv")
        metrics.insert(0, "fraction", 1.0)
        metrics.insert(1, "method", method)
        metric_frames.append(metrics)
        experiment_dirs[(1.0, method)] = directory

    seed_metrics = pd.concat(metric_frames, ignore_index=True)
    seed_metrics.sort_values(["fraction", "method", "seed"]).to_csv(
        analysis_dir / "seed_metrics.csv", index=False
    )
    summary = seed_metrics.groupby(["fraction", "method"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        primary_auroc_mean=("primary_shd_auroc", "mean"),
        primary_auroc_std=("primary_shd_auroc", "std"),
        primary_auprc_mean=("primary_shd_auprc", "mean"),
        primary_auprc_std=("primary_shd_auprc", "std"),
        macro_auroc_11_mean=("macro_auroc_11", "mean"),
        macro_auroc_11_std=("macro_auroc_11", "std"),
        macro_auroc_12_mean=("macro_auroc_12", "mean"),
        macro_auroc_12_std=("macro_auroc_12", "std"),
    )
    summary.to_csv(analysis_dir / "summary.csv", index=False)

    for fraction, tag in (*FRACTIONS, (1.0, "100")):
        probabilities: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
        labels = None
        pindex = None
        for seed in seeds:
            bundles = {
                method: load_bundle(
                    experiment_dirs[(fraction, method)] / "predictions" / f"seed_{seed}_test.npz"
                )
                for method in METHODS
            }
            pindex = validate_pair(bundles["scratch"], bundles["ssl"], seed)
            labels = bundles["ssl"]["labels"][:, pindex].astype(int)
            scratch_prob = bundles["scratch"]["probabilities"][:, pindex].astype(float)
            ssl_prob = bundles["ssl"]["probabilities"][:, pindex].astype(float)
            probabilities["scratch"].append(scratch_prob)
            probabilities["ssl"].append(ssl_prob)
            scratch_auroc, scratch_auprc = primary_metrics(labels, scratch_prob)
            ssl_auroc, ssl_auprc = primary_metrics(labels, ssl_prob)
            boot = paired_bootstrap_metrics(
                labels,
                scratch_prob,
                ssl_prob,
                iterations=args.bootstrap_iters,
                seed=args.bootstrap_seed + int(round(fraction * 1000)) + seed,
                max_attempt_multiplier=100,
            )
            prefix = f"fraction_{tag}_seed_{seed}"
            raw_bootstrap.update({f"{prefix}_{key}": value for key, value in boot.items()})
            paired_rows.append(
                {
                    "fraction": fraction,
                    "level": "seed",
                    "seed": str(seed),
                    "scratch_auroc": scratch_auroc,
                    "ssl_auroc": ssl_auroc,
                    "delta_auroc": ssl_auroc - scratch_auroc,
                    "delta_auroc_ci_low": float(np.percentile(boot["delta_auroc"], 2.5)),
                    "delta_auroc_ci_high": float(np.percentile(boot["delta_auroc"], 97.5)),
                    "scratch_auprc": scratch_auprc,
                    "ssl_auprc": ssl_auprc,
                    "delta_auprc": ssl_auprc - scratch_auprc,
                    "delta_auprc_ci_low": float(np.percentile(boot["delta_auprc"], 2.5)),
                    "delta_auprc_ci_high": float(np.percentile(boot["delta_auprc"], 97.5)),
                    "bootstrap_n": args.bootstrap_iters,
                }
            )
        scratch_ensemble = np.mean(np.stack(probabilities["scratch"]), axis=0)
        ssl_ensemble = np.mean(np.stack(probabilities["ssl"]), axis=0)
        scratch_auroc, scratch_auprc = primary_metrics(labels, scratch_ensemble)
        ssl_auroc, ssl_auprc = primary_metrics(labels, ssl_ensemble)
        boot = paired_bootstrap_metrics(
            labels,
            scratch_ensemble,
            ssl_ensemble,
            iterations=args.bootstrap_iters,
            seed=args.bootstrap_seed + int(round(fraction * 1000)) + 9999,
            max_attempt_multiplier=100,
        )
        prefix = f"fraction_{tag}_ensemble"
        raw_bootstrap.update({f"{prefix}_{key}": value for key, value in boot.items()})
        paired_rows.append(
            {
                "fraction": fraction,
                "level": "ensemble",
                "seed": "ensemble",
                "scratch_auroc": scratch_auroc,
                "ssl_auroc": ssl_auroc,
                "delta_auroc": ssl_auroc - scratch_auroc,
                "delta_auroc_ci_low": float(np.percentile(boot["delta_auroc"], 2.5)),
                "delta_auroc_ci_high": float(np.percentile(boot["delta_auroc"], 97.5)),
                "scratch_auprc": scratch_auprc,
                "ssl_auprc": ssl_auprc,
                "delta_auprc": ssl_auprc - scratch_auprc,
                "delta_auprc_ci_low": float(np.percentile(boot["delta_auprc"], 2.5)),
                "delta_auprc_ci_high": float(np.percentile(boot["delta_auprc"], 97.5)),
                "bootstrap_n": args.bootstrap_iters,
            }
        )

    pd.DataFrame(alignment_rows).to_csv(
        provenance_dir / "prediction_alignment_sha256.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(analysis_dir / "paired_bootstrap.csv", index=False)
    np.savez_compressed(analysis_dir / "paired_bootstrap_raw.npz", **raw_bootstrap)

    # Hash all provenance-critical inputs and outputs, including checkpoints and predictions.
    artifact_rows = []
    for path in sorted(low_root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_sha256.csv", "LOW_LABEL_COMPLETE.json"}:
            artifact_rows.append(
                {
                    "path": path.relative_to(low_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    pd.DataFrame(artifact_rows).to_csv(provenance_dir / "artifact_sha256.csv", index=False)
    status = {
        "status": "complete",
        "seeds": seeds,
        "fractions": [fraction for fraction, _ in FRACTIONS] + [1.0],
        "methods": list(METHODS),
        "trained_conditions": 8,
        "trained_runs": 8 * len(seeds),
        "full_fraction_aliases": {"ssl": str(full_dir), "scratch": str(strict_dir)},
        "bootstrap_iterations": args.bootstrap_iters,
        "bootstrap_unit": "patient row; one row per patient in test predictions",
        "paired_indices": True,
        "nested_subset_validation": "passed",
        "prediction_alignment_validation": "passed",
    }
    (analysis_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (low_root / "LOW_LABEL_COMPLETE.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
