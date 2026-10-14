#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate checkpoints with per-label metrics and paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_project.data.datasets import ECHONEXT_LABEL_COLS, ECHONEXT_PRIMARY, EchoNextNpyDataset
from ecg_project.models import ECGConvNeXt1D


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def load_model(checkpoint_path: Path) -> ECGConvNeXt1D:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_config" not in checkpoint or "model" not in checkpoint:
        raise ValueError(f"Checkpoint missing model_config/model: {checkpoint_path}")
    model = ECGConvNeXt1D(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(model: ECGConvNeXt1D, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    probabilities, labels = [], []
    for waveform, tabular, target in loader:
        logits = model(
            waveform.to(DEVICE, non_blocking=True),
            tabular.to(DEVICE, non_blocking=True),
        )
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.numpy())
    return np.vstack(probabilities), np.vstack(labels)


def compute_per_label(labels: np.ndarray, probs: np.ndarray, method: str) -> pd.DataFrame:
    rows = []
    for idx, label in enumerate(ECHONEXT_LABEL_COLS):
        rows.append(
            {
                "method": method,
                "label_index": idx,
                "label": label,
                "prevalence": float(np.mean(labels[:, idx])),
                "auroc": safe_auroc(labels[:, idx], probs[:, idx]),
                "auprc": safe_auprc(labels[:, idx], probs[:, idx]),
            }
        )
    return pd.DataFrame(rows)


def primary_and_macro(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    primary_idx = ECHONEXT_LABEL_COLS.index(ECHONEXT_PRIMARY)
    per_auc = [safe_auroc(labels[:, idx], probs[:, idx]) for idx in range(labels.shape[1])]
    per_pr = [safe_auprc(labels[:, idx], probs[:, idx]) for idx in range(labels.shape[1])]
    return {
        "primary_auroc": float(per_auc[primary_idx]),
        "primary_auprc": float(per_pr[primary_idx]),
        "macro_auroc": float(np.nanmean(per_auc)),
        "macro_auprc": float(np.nanmean(per_pr)),
    }


def bootstrap_metrics(
    labels: np.ndarray,
    probs_by_method: dict[str, np.ndarray],
    iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    methods = list(probs_by_method)
    metric_samples: dict[tuple[str, str], list[float]] = {
        (method, metric): []
        for method in methods
        for metric in ["primary_auroc", "primary_auprc", "macro_auroc", "macro_auprc"]
    }
    diff_samples: dict[str, list[float]] = {
        metric: [] for metric in ["primary_auroc", "primary_auprc", "macro_auroc", "macro_auprc"]
    }
    if len(methods) != 2:
        raise ValueError("Paired difference currently expects exactly two methods.")
    baseline_name, best_name = methods[0], methods[1]
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        sample_labels = labels[idx]
        sample_metrics = {}
        for method in methods:
            sample_metrics[method] = primary_and_macro(sample_labels, probs_by_method[method][idx])
            for metric, value in sample_metrics[method].items():
                if not np.isnan(value):
                    metric_samples[(method, metric)].append(value)
        for metric in diff_samples:
            base_value = sample_metrics[baseline_name][metric]
            best_value = sample_metrics[best_name][metric]
            if not np.isnan(base_value) and not np.isnan(best_value):
                diff_samples[metric].append(best_value - base_value)

    ci_rows = []
    for method in methods:
        point = primary_and_macro(labels, probs_by_method[method])
        for metric in ["primary_auroc", "primary_auprc", "macro_auroc", "macro_auprc"]:
            values = np.asarray(metric_samples[(method, metric)], dtype=float)
            ci_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "point": point[metric],
                    "ci_low": float(np.percentile(values, 2.5)),
                    "ci_high": float(np.percentile(values, 97.5)),
                    "bootstrap_n": int(len(values)),
                }
            )

    diff_rows = []
    full_base = primary_and_macro(labels, probs_by_method[baseline_name])
    full_best = primary_and_macro(labels, probs_by_method[best_name])
    for metric in ["primary_auroc", "primary_auprc", "macro_auroc", "macro_auprc"]:
        values = np.asarray(diff_samples[metric], dtype=float)
        diff_rows.append(
            {
                "comparison": f"{best_name} - {baseline_name}",
                "metric": metric,
                "point_difference": full_best[metric] - full_base[metric],
                "ci_low": float(np.percentile(values, 2.5)),
                "ci_high": float(np.percentile(values, 97.5)),
                "bootstrap_n": int(len(values)),
            }
        )
    return pd.DataFrame(ci_rows), pd.DataFrame(diff_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoint per-label metrics and paired bootstrap comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/echonext")
    parser.add_argument("--csv-name", default="echonext_metadata_100k.csv")
    parser.add_argument("--test-wave-name", default="EchoNext_test_waveforms.npy")
    parser.add_argument("--test-tab-name", default="EchoNext_test_tabular_features.npy")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--best-checkpoint", required=True)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--best-name", default="best")
    parser.add_argument("--output-dir", default="experiments/20260708_bootstrap_best_vs_baseline")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260708)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    dataset = EchoNextNpyDataset(
        data_dir / args.csv_name,
        data_dir / args.test_wave_name,
        data_dir / args.test_tab_name,
        "test",
        ECHONEXT_LABEL_COLS,
        None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    checkpoints = {
        args.baseline_name: Path(args.baseline_checkpoint),
        args.best_name: Path(args.best_checkpoint),
    }
    probs_by_method = {}
    labels_ref = None
    for name, checkpoint_path in checkpoints.items():
        model = load_model(checkpoint_path)
        probs, labels = predict(model, loader)
        if labels_ref is None:
            labels_ref = labels
        elif not np.array_equal(labels_ref, labels):
            raise RuntimeError("Label arrays differ across evaluations.")
        probs_by_method[name] = probs
        np.savez_compressed(output_dir / f"predictions_{name}.npz", labels=labels, probabilities=probs)
        print(f"[pred] {name}: {probs.shape} from {checkpoint_path}")

    assert labels_ref is not None
    per_label_frames = [compute_per_label(labels_ref, probs, name) for name, probs in probs_by_method.items()]
    per_label_long = pd.concat(per_label_frames, ignore_index=True)
    per_label_wide = (
        per_label_long.pivot(index=["label_index", "label", "prevalence"], columns="method", values=["auroc", "auprc"])
        .reset_index()
    )
    per_label_wide.columns = [
        "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
        for col in per_label_wide.columns
    ]
    if args.baseline_name in probs_by_method and args.best_name in probs_by_method:
        per_label_wide["delta_auroc"] = (
            per_label_wide[f"auroc_{args.best_name}"] - per_label_wide[f"auroc_{args.baseline_name}"]
        )
        per_label_wide["delta_auprc"] = (
            per_label_wide[f"auprc_{args.best_name}"] - per_label_wide[f"auprc_{args.baseline_name}"]
        )
    per_label_long.to_csv(output_dir / "per_label_metrics_long.csv", index=False)
    per_label_wide.to_csv(output_dir / "per_label_baseline_vs_best.csv", index=False)

    ordered = {args.baseline_name: probs_by_method[args.baseline_name], args.best_name: probs_by_method[args.best_name]}
    ci_df, diff_df = bootstrap_metrics(labels_ref, ordered, args.bootstrap_iters, args.bootstrap_seed)
    ci_df.to_csv(output_dir / "bootstrap_primary_ci.csv", index=False)
    diff_df.to_csv(output_dir / "paired_difference_bootstrap.csv", index=False)

    summary = {
        "output_dir": str(output_dir),
        "n_test": int(labels_ref.shape[0]),
        "n_labels": int(labels_ref.shape[1]),
        "bootstrap_iters": int(args.bootstrap_iters),
        "point_metrics": {name: primary_and_macro(labels_ref, probs) for name, probs in ordered.items()},
        "paired_difference": diff_df.to_dict("records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
