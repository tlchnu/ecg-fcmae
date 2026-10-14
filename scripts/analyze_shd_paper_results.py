#!/usr/bin/env python
"""Formal three-seed paired, screening, calibration, and subgroup analyses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_project.utils.metrics import (
    brier_score,
    calibration_table,
    screening_metrics,
    select_threshold_for_sensitivity,
)


DEFAULT_SEEDS = (42, 2024, 3407)
METRICS = ("primary_auroc", "primary_auprc", "macro_auroc", "macro_auprc")


def safe_metric(y_true: np.ndarray, y_score: np.ndarray, metric: str) -> float:
    """Compute a binary ranking metric, returning NaN for a single-class sample."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if metric == "auroc":
        return float(roc_auc_score(y_true, y_score))
    if metric == "auprc":
        return float(average_precision_score(y_true, y_score))
    raise ValueError(f"Unknown metric: {metric}")


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("p_values must be a finite one-dimensional array")
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def load_prediction_bundle(path: Path) -> dict[str, np.ndarray]:
    """Load a prediction NPZ without pickle-backed object arrays."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_prediction_alignment(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> None:
    """Require exact row, target, and split order before paired analysis."""
    for field in ("row_id", "labels"):
        if field not in reference or field not in candidate:
            raise ValueError(f"Missing required alignment field: {field}")
        if not np.array_equal(reference[field], candidate[field]):
            raise ValueError(f"Prediction {field} arrays are not exactly aligned")
    for optional_field in ("split_hash", "split"):
        if optional_field in reference and optional_field in candidate:
            if not np.array_equal(reference[optional_field], candidate[optional_field]):
                raise ValueError(f"Prediction {optional_field} arrays are not exactly aligned")


def cluster_bootstrap_indices(
    patient_clusters: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample patients with replacement and keep all rows of each sampled patient."""
    clusters = np.asarray(patient_clusters)
    unique_clusters = np.unique(clusters)
    sampled = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
    return np.concatenate([np.flatnonzero(clusters == cluster) for cluster in sampled])


def mean_seed_metric_difference(
    labels: np.ndarray,
    ssl_probabilities: list[np.ndarray],
    scratch_probabilities: list[np.ndarray],
    metric: str,
) -> float:
    """Compute SSL-minus-scratch per seed and then average the seed differences."""
    differences = [
        safe_metric(labels, ssl_seed, metric) - safe_metric(labels, scratch_seed, metric)
        for ssl_seed, scratch_seed in zip(ssl_probabilities, scratch_probabilities)
    ]
    return float(np.nanmean(differences))


def multilabel_summary(labels: np.ndarray, probabilities: np.ndarray, primary_index: int) -> dict[str, float]:
    """Return Primary and macro ranking metrics from one prediction matrix."""
    aurocs = [safe_metric(labels[:, index], probabilities[:, index], "auroc") for index in range(labels.shape[1])]
    auprcs = [safe_metric(labels[:, index], probabilities[:, index], "auprc") for index in range(labels.shape[1])]
    return {
        "primary_auroc": float(aurocs[primary_index]),
        "primary_auprc": float(auprcs[primary_index]),
        "macro_auroc": float(np.nanmean(aurocs)),
        "macro_auprc": float(np.nanmean(auprcs)),
    }


def percentile_interval(values: list[float]) -> tuple[float, float, int]:
    """Return a 95% percentile interval and effective finite count."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return float("nan"), float("nan"), 0
    return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)), int(len(finite))


def paired_bootstrap(
    labels: np.ndarray,
    clusters: np.ndarray,
    ssl_by_seed: list[np.ndarray],
    scratch_by_seed: list[np.ndarray],
    primary_index: int,
    iterations: int,
    seed: int,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap shared patient indices and preserve mean-of-seed differences."""
    point_rows = []
    for seed_index, (ssl_prob, scratch_prob) in enumerate(zip(ssl_by_seed, scratch_by_seed)):
        ssl_metrics = multilabel_summary(labels, ssl_prob, primary_index)
        scratch_metrics = multilabel_summary(labels, scratch_prob, primary_index)
        for metric in METRICS:
            point_rows.append(
                {
                    "seed_index": seed_index,
                    "metric": metric,
                    "ssl": ssl_metrics[metric],
                    "scratch": scratch_metrics[metric],
                    "difference": ssl_metrics[metric] - scratch_metrics[metric],
                }
            )
    rng = np.random.default_rng(seed)
    resamples = [cluster_bootstrap_indices(clusters, rng) for _ in range(iterations)]

    def evaluate_resample(indices: np.ndarray) -> dict[str, float]:
        sample_labels = labels[indices]
        result = {}
        for metric in METRICS:
            seed_differences = []
            for ssl_prob, scratch_prob in zip(ssl_by_seed, scratch_by_seed):
                ssl_metrics = multilabel_summary(sample_labels, ssl_prob[indices], primary_index)
                scratch_metrics = multilabel_summary(sample_labels, scratch_prob[indices], primary_index)
                seed_differences.append(ssl_metrics[metric] - scratch_metrics[metric])
            result[metric] = float(np.nanmean(seed_differences))
        return result

    evaluated = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_resample)(indices) for indices in resamples
    )
    samples = {
        metric: [row[metric] for row in evaluated]
        for metric in METRICS
    }
    interval_rows = []
    points = pd.DataFrame(point_rows)
    for metric in METRICS:
        low, high, effective = percentile_interval(samples[metric])
        interval_rows.append(
            {
                "metric": metric,
                "point_difference": float(points.loc[points.metric == metric, "difference"].mean()),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_n": effective,
            }
        )
    return points, pd.DataFrame(interval_rows)


def labelwise_bootstrap(
    labels: np.ndarray,
    clusters: np.ndarray,
    ssl_by_seed: list[np.ndarray],
    scratch_by_seed: list[np.ndarray],
    label_names: np.ndarray,
    iterations: int,
    seed: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Compute per-label method CIs and paired-difference CIs."""
    rng = np.random.default_rng(seed)
    resamples = [cluster_bootstrap_indices(clusters, rng) for _ in range(iterations)]
    def analyze_label_metric(label_index: int, label_name: str, metric: str) -> dict[str, float | int | str]:
        y_true = labels[:, label_index]
        ssl_points = [safe_metric(y_true, probs[:, label_index], metric) for probs in ssl_by_seed]
        scratch_points = [safe_metric(y_true, probs[:, label_index], metric) for probs in scratch_by_seed]
        ssl_samples: list[float] = []
        scratch_samples: list[float] = []
        difference_samples: list[float] = []
        for indices in resamples:
            sampled_labels = y_true[indices]
            ssl_values = [safe_metric(sampled_labels, probs[indices, label_index], metric) for probs in ssl_by_seed]
            scratch_values = [safe_metric(sampled_labels, probs[indices, label_index], metric) for probs in scratch_by_seed]
            ssl_mean = float(np.nanmean(ssl_values))
            scratch_mean = float(np.nanmean(scratch_values))
            ssl_samples.append(ssl_mean)
            scratch_samples.append(scratch_mean)
            difference_samples.append(ssl_mean - scratch_mean)
        ssl_low, ssl_high, ssl_n = percentile_interval(ssl_samples)
        scratch_low, scratch_high, scratch_n = percentile_interval(scratch_samples)
        diff_low, diff_high, diff_n = percentile_interval(difference_samples)
        return {
            "label_index": label_index,
            "label": str(label_name),
            "prevalence": float(np.mean(y_true)),
            "metric": metric,
            "ssl": float(np.nanmean(ssl_points)),
            "ssl_ci_low": ssl_low,
            "ssl_ci_high": ssl_high,
            "ssl_bootstrap_n": ssl_n,
            "scratch": float(np.nanmean(scratch_points)),
            "scratch_ci_low": scratch_low,
            "scratch_ci_high": scratch_high,
            "scratch_bootstrap_n": scratch_n,
            "difference": float(np.nanmean(np.asarray(ssl_points) - np.asarray(scratch_points))),
            "difference_ci_low": diff_low,
            "difference_ci_high": diff_high,
            "difference_bootstrap_n": diff_n,
        }

    rows = Parallel(n_jobs=n_jobs)(
        delayed(analyze_label_metric)(label_index, str(label_name), metric)
        for label_index, label_name in enumerate(label_names)
        for metric in ("auroc", "auprc")
    )
    return pd.DataFrame(rows)


def cluster_swap_mask(clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one paired-method swap decision per patient cluster."""
    unique_clusters, inverse = np.unique(clusters, return_inverse=True)
    decisions = rng.integers(0, 2, size=len(unique_clusters), dtype=np.int8).astype(bool)
    return decisions[inverse]


def labelwise_permutation_tests(
    labels: np.ndarray,
    clusters: np.ndarray,
    ssl_by_seed: list[np.ndarray],
    scratch_by_seed: list[np.ndarray],
    label_names: np.ndarray,
    iterations: int,
    seed: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run paired patient-level method swaps and apply BH-FDR per metric family."""
    def test_label_metric(label_index: int, label_name: str, metric_offset: int, metric: str) -> dict[str, float | int | str]:
        y_true = labels[:, label_index]
        observed = mean_seed_metric_difference(
            y_true,
            [prob[:, label_index] for prob in ssl_by_seed],
            [prob[:, label_index] for prob in scratch_by_seed],
            metric,
        )
        rng = np.random.default_rng(seed + 1009 * label_index + metric_offset)
        extreme = 0
        effective = 0
        for _ in range(iterations):
            swap = cluster_swap_mask(clusters, rng)
            differences = []
            for ssl_prob, scratch_prob in zip(ssl_by_seed, scratch_by_seed):
                ssl_score = ssl_prob[:, label_index]
                scratch_score = scratch_prob[:, label_index]
                perm_ssl = np.where(swap, scratch_score, ssl_score)
                perm_scratch = np.where(swap, ssl_score, scratch_score)
                value = safe_metric(y_true, perm_ssl, metric) - safe_metric(y_true, perm_scratch, metric)
                differences.append(value)
            statistic = float(np.nanmean(differences))
            if np.isfinite(statistic):
                effective += 1
                extreme += int(abs(statistic) >= abs(observed) - 1e-15)
        p_value = (extreme + 1) / (effective + 1) if effective else float("nan")
        return {
            "label_index": label_index,
            "label": str(label_name),
            "metric": metric,
            "observed_difference": observed,
            "p_value": p_value,
            "permutation_n": effective,
        }

    rows = Parallel(n_jobs=n_jobs)(
        delayed(test_label_metric)(label_index, str(label_name), metric_offset, metric)
        for label_index, label_name in enumerate(label_names)
        for metric_offset, metric in enumerate(("auroc", "auprc"))
    )
    table = pd.DataFrame(rows)
    table["p_value_bh_fdr"] = np.nan
    for metric in ("auroc", "auprc"):
        mask = (table.metric == metric) & table.p_value.notna()
        table.loc[mask, "p_value_bh_fdr"] = benjamini_hochberg(table.loc[mask, "p_value"].to_numpy())
    table["reject_fdr_0p05"] = table["p_value_bh_fdr"] <= 0.05
    return table


def bootstrap_scalar_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    clusters: np.ndarray,
    threshold: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap ranking, calibration, and frozen-threshold screening metrics."""
    rng = np.random.default_rng(seed)
    names = ("auroc", "auprc", "brier", "sensitivity", "specificity", "ppv", "npv")
    values = {name: [] for name in names}
    for _ in range(iterations):
        indices = cluster_bootstrap_indices(clusters, rng)
        y_true = labels[indices]
        y_prob = probabilities[indices]
        screening = screening_metrics(y_true, y_prob, threshold)
        values["auroc"].append(safe_metric(y_true, y_prob, "auroc"))
        values["auprc"].append(safe_metric(y_true, y_prob, "auprc"))
        values["brier"].append(brier_score(y_true, y_prob))
        for name in ("sensitivity", "specificity", "ppv", "npv"):
            values[name].append(float(screening[name]))
    rows = []
    for name in names:
        low, high, effective = percentile_interval(values[name])
        rows.append({"metric": name, "ci_low": low, "ci_high": high, "bootstrap_n": effective})
    return pd.DataFrame(rows)


def subgroup_masks(bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return predefined, test-independent age, sex, and setting subgroups."""
    age = bundle["age_at_ecg"].astype(float)
    sex = bundle["sex"].astype(str)
    setting = bundle.get("location_setting", np.asarray(["missing"] * len(age))).astype(str)
    masks = {
        "age_lt40": age < 40,
        "age_40_64": (age >= 40) & (age < 65),
        "age_65_79": (age >= 65) & (age < 80),
        "age_ge80": age >= 80,
        "sex_female": np.char.lower(sex) == "female",
        "sex_male": np.char.lower(sex) == "male",
    }
    for value in sorted(np.unique(setting)):
        masks[f"setting_{value}"] = setting == value
    return masks


def analyze_subgroups(
    bundle: dict[str, np.ndarray],
    probabilities: np.ndarray,
    primary_index: int,
    threshold: float,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    """Compute subgroup discrimination, calibration, screening metrics, and CIs."""
    rows = []
    labels = bundle["labels"][:, primary_index].astype(int)
    clusters = bundle["patient_cluster_index"]
    for subgroup, mask in subgroup_masks(bundle).items():
        n = int(np.sum(mask))
        if n == 0:
            continue
        y_true = labels[mask]
        y_prob = probabilities[mask, primary_index]
        subgroup_clusters = clusters[mask]
        screening = screening_metrics(y_true, y_prob, threshold)
        point = {
            "subgroup": subgroup,
            "n": n,
            "positive": int(np.sum(y_true)),
            "negative": int(n - np.sum(y_true)),
            "prevalence": float(np.mean(y_true)),
            "auroc": safe_metric(y_true, y_prob, "auroc"),
            "auprc": safe_metric(y_true, y_prob, "auprc"),
            "brier": brier_score(y_true, y_prob),
            "ece": float(calibration_table(y_true, y_prob).attrs["ece"]),
            **screening,
        }
        intervals = bootstrap_scalar_metrics(
            y_true,
            y_prob,
            subgroup_clusters,
            threshold,
            iterations,
            seed + len(rows),
        )
        for _, interval in intervals.iterrows():
            metric = str(interval["metric"])
            point[f"{metric}_ci_low"] = interval["ci_low"]
            point[f"{metric}_ci_high"] = interval["ci_high"]
            point[f"{metric}_bootstrap_n"] = int(interval["bootstrap_n"])
        rows.append(point)
    return pd.DataFrame(rows)


def sha256_outputs(directory: Path) -> None:
    """Write hashes for every analysis artifact except the manifest itself."""
    manifest_path = directory / "analysis" / "sha256sums.txt"
    lines = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path != manifest_path:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(directory).as_posix()}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_paired_forest(table: pd.DataFrame, output_stem: Path) -> None:
    """Plot paired mean-of-seed differences with 95% bootstrap CIs."""
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    y = np.arange(len(table))
    point = table["point_difference"].to_numpy()
    errors = np.vstack([point - table["ci_low"], table["ci_high"] - point])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(point, y, xerr=errors, fmt="o", color="#0072B2", capsize=4)
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y, table["metric"].str.replace("_", " "))
    ax.set_xlabel("FCMAE(ECA) minus scratch")
    ax.set_title("Paired three-seed performance differences")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_calibration(bins: pd.DataFrame, output_stem: Path) -> None:
    """Plot per-seed SHD calibration curves."""
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Ideal")
    for seed, group in bins.groupby("seed"):
        valid = group["n"] > 0
        ax.plot(group.loc[valid, "mean_probability"], group.loc[valid, "event_rate"], marker="o", label=f"Seed {seed}")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze three-seed FCMAE(ECA) versus scratch prediction bundles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--primary-label", default="shd_moderate_or_greater_flag")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--permutation-iters", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--sensitivity-target", type=float, default=0.90)
    parser.add_argument("--n-jobs", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_dir = args.output_dir / "analysis"
    figures_dir = args.output_dir / "figures"
    failures_dir = args.output_dir / "failures"
    for directory in (analysis_dir, figures_dir, failures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config = {
        "seeds": args.seeds,
        "bootstrap_iterations": args.bootstrap_iters,
        "permutation_iterations": args.permutation_iters,
        "bootstrap_seed": args.bootstrap_seed,
        "sensitivity_target": args.sensitivity_target,
        "threshold_selection": "validation sensitivity >= target; maximize specificity; tie -> higher threshold",
        "classification_rule": "probability >= threshold",
        "ece_bins": "[0,.1),...,[.9,1]",
        "age_subgroups": ["<40", "40-64", "65-79", ">=80"],
        "primary_estimator": "mean of per-seed metrics/differences",
    }
    (analysis_dir / "analysis_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    data: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    reference_test = None
    for seed in args.seeds:
        seed_data = {}
        for split_name, split_tag in (("validation", "validation"), ("test", "test")):
            for method in ("ssl", "scratch"):
                path = args.input_dir / f"seed_{seed}" / f"{method}_{split_tag}.npz"
                if not path.exists():
                    raise FileNotFoundError(path)
                seed_data[f"{method}_{split_name}"] = load_prediction_bundle(path)
            validate_prediction_alignment(seed_data[f"ssl_{split_name}"], seed_data[f"scratch_{split_name}"])
        if reference_test is None:
            reference_test = seed_data["ssl_test"]
        else:
            validate_prediction_alignment(reference_test, seed_data["ssl_test"])
        data[seed] = seed_data
    assert reference_test is not None
    label_names = reference_test["label_names"].astype(str)
    if args.primary_label not in set(label_names):
        raise ValueError(f"Primary label not found: {args.primary_label}")
    primary_index = int(np.flatnonzero(label_names == args.primary_label)[0])
    labels = reference_test["labels"]
    clusters = reference_test["patient_cluster_index"]
    one_row_per_patient = bool(len(np.unique(clusters)) == len(clusters))
    (analysis_dir / "patient_cluster_audit.json").write_text(
        json.dumps(
            {
                "test_rows": int(len(clusters)),
                "unique_test_patients": int(len(np.unique(clusters))),
                "one_row_per_patient": one_row_per_patient,
                "bootstrap_unit": "record (equivalent to patient)" if one_row_per_patient else "patient cluster",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    ssl_test = [data[seed]["ssl_test"]["probabilities"] for seed in args.seeds]
    scratch_test = [data[seed]["scratch_test"]["probabilities"] for seed in args.seeds]

    seed_points, paired_ci = paired_bootstrap(
        labels,
        clusters,
        ssl_test,
        scratch_test,
        primary_index,
        args.bootstrap_iters,
        args.bootstrap_seed,
        args.n_jobs,
    )
    seed_points["seed"] = np.repeat(args.seeds, len(METRICS))
    seed_points.to_csv(analysis_dir / "seed_metrics.csv", index=False)
    paired_ci.to_csv(analysis_dir / "paired_difference_ci.csv", index=False)
    paired_ci.to_csv(analysis_dir / "paired_bootstrap_metrics.csv", index=False)

    labelwise = labelwise_bootstrap(
        labels,
        clusters,
        ssl_test,
        scratch_test,
        label_names,
        args.bootstrap_iters,
        args.bootstrap_seed + 1,
        args.n_jobs,
    )
    labelwise.to_csv(analysis_dir / "labelwise_metrics_ci.csv", index=False)
    permutation = labelwise_permutation_tests(
        labels,
        clusters,
        ssl_test,
        scratch_test,
        label_names,
        args.permutation_iters,
        args.bootstrap_seed + 2,
        args.n_jobs,
    )
    permutation.to_csv(analysis_dir / "labelwise_paired_tests_bh_fdr.csv", index=False)

    threshold_rows = []
    screening_rows = []
    screening_ci_rows = []
    calibration_rows = []
    calibration_bins_rows = []
    subgroup_frames = []
    for seed_index, seed in enumerate(args.seeds):
        validation_bundle = data[seed]["ssl_validation"]
        test_bundle = data[seed]["ssl_test"]
        validation_labels = validation_bundle["labels"][:, primary_index]
        validation_probabilities = validation_bundle["probabilities"][:, primary_index]
        threshold_result = select_threshold_for_sensitivity(
            validation_labels,
            validation_probabilities,
            args.sensitivity_target,
        )
        threshold = float(threshold_result["threshold"])
        threshold_rows.append({"seed": seed, **threshold_result})
        test_labels = test_bundle["labels"][:, primary_index]
        test_probabilities = test_bundle["probabilities"][:, primary_index]
        screening = screening_metrics(test_labels, test_probabilities, threshold)
        screening_rows.append({"seed": seed, **screening})
        intervals = bootstrap_scalar_metrics(
            test_labels,
            test_probabilities,
            test_bundle["patient_cluster_index"],
            threshold,
            args.bootstrap_iters,
            args.bootstrap_seed + 100 + seed_index,
        )
        intervals.insert(0, "seed", seed)
        screening_ci_rows.append(intervals)
        bins = calibration_table(test_labels, test_probabilities, n_bins=10)
        bins.insert(0, "seed", seed)
        calibration_bins_rows.append(bins)
        calibration_rows.append(
            {
                "seed": seed,
                "brier": brier_score(test_labels, test_probabilities),
                "ece": float(bins.attrs["ece"]),
            }
        )
        subgroup = analyze_subgroups(
            test_bundle,
            test_bundle["probabilities"],
            primary_index,
            threshold,
            args.bootstrap_iters,
            args.bootstrap_seed + 1000 + seed_index,
        )
        subgroup.insert(0, "seed", seed)
        subgroup_frames.append(subgroup)

    threshold_payload = {
        "selection_split": "validation",
        "test_used_for_selection": False,
        "target_sensitivity": args.sensitivity_target,
        "classification_rule": "probability >= threshold",
        "per_seed": threshold_rows,
    }
    (analysis_dir / "threshold.json").write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")
    pd.DataFrame(screening_rows).to_csv(analysis_dir / "screening_metrics.csv", index=False)
    pd.concat(screening_ci_rows, ignore_index=True).to_csv(analysis_dir / "screening_bootstrap_ci.csv", index=False)
    pd.DataFrame(calibration_rows).to_json(analysis_dir / "calibration_metrics.json", orient="records", indent=2)
    calibration_bins = pd.concat(calibration_bins_rows, ignore_index=True)
    calibration_bins.to_csv(analysis_dir / "calibration_bins.csv", index=False)
    subgroup_table = pd.concat(subgroup_frames, ignore_index=True)
    subgroup_table.to_csv(analysis_dir / "subgroup_metrics.csv", index=False)
    (analysis_dir / "subgroup_definition.yaml").write_text(
        yaml.safe_dump(
            {
                "age": ["age < 40", "40 <= age < 65", "65 <= age < 80", "age >= 80"],
                "sex": ["female", "male"],
                "location_setting": "reported source categories; not interpreted as clinical center",
                "threshold": "per-seed validation-frozen overall SHD threshold",
                "test_subgroup_tuning": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    heterogeneity_rows = []
    for seed in args.seeds:
        seed_table = subgroup_table[subgroup_table.seed == seed]
        for family, prefix in (("age", "age_"), ("sex", "sex_"), ("setting", "setting_")):
            family_table = seed_table[seed_table.subgroup.str.startswith(prefix)]
            if len(family_table) >= 2:
                heterogeneity_rows.append(
                    {
                        "seed": seed,
                        "family": family,
                        "auroc_range": float(family_table.auroc.max() - family_table.auroc.min()),
                        "auprc_range": float(family_table.auprc.max() - family_table.auprc.min()),
                        "interpretation": "descriptive range; not an interaction p-value",
                    }
                )
    pd.DataFrame(heterogeneity_rows).to_csv(analysis_dir / "subgroup_heterogeneity.csv", index=False)

    plot_paired_forest(paired_ci, figures_dir / "paired_difference_forest")
    plot_calibration(calibration_bins, figures_dir / "calibration_curve")
    status = {
        "status": "complete",
        "input_dir": str(args.input_dir),
        "seeds": args.seeds,
        "n_test": int(len(labels)),
        "n_labels": int(labels.shape[1]),
        "one_test_row_per_patient": one_row_per_patient,
        "bootstrap_iterations": args.bootstrap_iters,
        "permutation_iterations": args.permutation_iters,
    }
    (analysis_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    sha256_outputs(args.output_dir)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
