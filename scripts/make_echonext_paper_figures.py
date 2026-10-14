#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate paper figures for EchoNext-style ECG-SHD experiments.

The script is intentionally post-hoc: it does not change training code or model
weights. It reads the EchoNext test split, evaluates a saved downstream
checkpoint for the primary SHD label, and exports publication-ready figures.

The plots follow the presentation spirit of the public EchoNext Mini example:
show an ECG example, summarize label prevalence, show ROC/PR behavior, and
place the current method against the public benchmark and internal ablations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_project.models.convnext import ECGConvNeXt1D  # noqa: E402


LABEL_COLS = [
    "lvef_lte_45_flag",
    "lvwt_gte_13_flag",
    "aortic_stenosis_moderate_or_greater_flag",
    "aortic_regurgitation_moderate_or_greater_flag",
    "mitral_regurgitation_moderate_or_greater_flag",
    "tricuspid_regurgitation_moderate_or_greater_flag",
    "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_systolic_dysfunction_moderate_or_greater_flag",
    "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45_flag",
    "tr_max_gte_32_flag",
    "shd_moderate_or_greater_flag",
]

SHORT_LABELS = [
    "LVEF<=45",
    "LVWT>=13",
    "AS mod+",
    "AR mod+",
    "MR mod+",
    "TR mod+",
    "PR mod+",
    "RV dysf.",
    "Effusion",
    "PASP>=45",
    "TRmax>=3.2",
    "SHD",
]

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EchoNext paper figures")
    parser.add_argument("--data-dir", default="data/echonext")
    parser.add_argument("--csv-name", default="echonext_metadata_100k.csv")
    parser.add_argument("--test-wave-name", default="EchoNext_test_waveforms.npy")
    parser.add_argument("--test-tab-name", default="EchoNext_test_tabular_features.npy")
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/20260630_eca_drop_path_010/"
            "echonext_eca_drop_path_010_pretrained_bce_eca_drop_path_010_seed42.pth"
        ),
    )
    parser.add_argument("--best-metrics", default="experiments/20260630_eca_drop_path_010/metrics.csv")
    parser.add_argument("--baseline-metrics", default="experiments/20260623_fcmae_convnext_baseline/metrics.csv")
    parser.add_argument("--multiseed-metrics", default="experiments/20260707_eca_drop_path_010_multiseed/metrics.csv")
    parser.add_argument("--saliency-dir", default="experiments/20260707_interpretability_eca_drop_path_010_seed42")
    parser.add_argument("--output-dir", default="figures/paper/20260707_echonext_mini")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--public-benchmark-auroc", type=float, default=0.82)
    parser.add_argument("--strict-control-auroc", type=float, default=0.8502)
    return parser.parse_args()


def safe_torch_load(path: str | Path, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_metric(path: str | Path, metric: str) -> float | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or metric not in df.columns:
        return None
    return float(df.iloc[-1][metric])


def read_multiseed(path: str | Path) -> list[dict[str, float]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "seed": float(row["Seed"]),
                    "primary_auroc": float(row["Test_Primary_AUROC"]),
                    "auprc": float(row["Test_Primary_AUPRC"]),
                    "macro_auroc": float(row["Test_Macro_AUROC"]),
                }
            )
    return rows


def load_labels(csv_path: Path, split: str = "test") -> np.ndarray:
    df = pd.read_csv(csv_path)
    df = df[df["split"] == split].reset_index(drop=True)
    return df[LABEL_COLS].to_numpy(dtype=np.float32)


def normalize_wave_array(raw: np.ndarray) -> np.ndarray:
    """Return ECG waves as (N, 12, T) without loading more than needed."""
    array = np.asarray(raw)
    if array.ndim == 4 and array.shape[1] == 1:
        array = np.squeeze(array, axis=1)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D or squeezeable 4D ECG array, got {array.shape}")
    if array.shape[1] == 12:
        return np.ascontiguousarray(array)
    if array.shape[-1] == 12:
        return np.ascontiguousarray(array.transpose(0, 2, 1))
    raise ValueError(f"Cannot infer lead dimension from ECG array shape {array.shape}")


def get_wave_batch(waves_raw: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return normalize_wave_array(waves_raw[indices])


def load_model(checkpoint_path: Path, device: torch.device) -> ECGConvNeXt1D:
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    if "model" not in checkpoint or "model_config" not in checkpoint:
        raise ValueError(f"Checkpoint is not a downstream checkpoint: {checkpoint_path}")
    model = ECGConvNeXt1D(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_primary(
    model: ECGConvNeXt1D,
    waves_raw: np.ndarray,
    tabular: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    probs: list[np.ndarray] = []
    n = len(tabular)
    for start in range(0, n, batch_size):
        indices = np.arange(start, min(start + batch_size, n))
        wave = torch.tensor(get_wave_batch(waves_raw, indices), dtype=torch.float32, device=device)
        tab = torch.tensor(np.asarray(tabular[indices]), dtype=torch.float32, device=device)
        logits = model(wave, tab)
        probs.append(torch.sigmoid(logits)[:, -1].detach().cpu().numpy())
    return np.concatenate(probs)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ecg_example(waves_raw: np.ndarray, labels: np.ndarray, probs: np.ndarray, output_dir: Path, sampling_rate: float) -> int:
    positive = np.where(labels[:, -1] == 1)[0]
    if len(positive) > 0:
        sample_idx = int(positive[np.argmax(probs[positive])])
    else:
        sample_idx = int(np.argmax(probs))
    wave = get_wave_batch(waves_raw, np.asarray([sample_idx]))[0]
    centered = wave - wave.mean(axis=1, keepdims=True)
    scaled = centered / (wave.std(axis=1, keepdims=True) + 1e-6)
    time = np.arange(scaled.shape[1]) / sampling_rate if sampling_rate > 0 else np.arange(scaled.shape[1])

    fig, axes = plt.subplots(6, 2, figsize=(10.5, 8.2), sharex=True)
    axes = axes.ravel()
    for lead_idx, ax in enumerate(axes):
        ax.plot(time, scaled[lead_idx], color="#2F5597", linewidth=0.8)
        ax.axhline(0, color="#BFBFBF", linewidth=0.4)
        ax.set_title(LEAD_NAMES[lead_idx], loc="left", fontsize=9)
        ax.set_yticks([])
        ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.4)
    axes[-1].set_xlabel("Time (s)" if sampling_rate > 0 else "Sample index")
    fig.suptitle(f"Example 12-lead ECG from test split (idx={sample_idx}, SHD prob={probs[sample_idx]:.3f})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, output_dir, "fig_ecg_example_12lead")
    return sample_idx


def plot_label_prevalence(labels: np.ndarray, output_dir: Path) -> None:
    prevalence = labels.mean(axis=0)
    order = np.argsort(prevalence)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = ["#4477AA" if SHORT_LABELS[i] != "SHD" else "#CC6677" for i in order]
    ax.barh(np.asarray(SHORT_LABELS)[order], prevalence[order] * 100.0, color=colors)
    ax.set_xlabel("Positive prevalence in test split (%)")
    ax.set_title("EchoNext multi-label prevalence")
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.6)
    for y, value in enumerate(prevalence[order] * 100.0):
        ax.text(value + 0.15, y, f"{value:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_label_prevalence")


def plot_roc_pr(labels: np.ndarray, probs: np.ndarray, output_dir: Path) -> dict[str, float]:
    y_true = labels[:, -1]
    primary_auroc = float(roc_auc_score(y_true, probs))
    primary_auprc = float(average_precision_score(y_true, probs))
    fpr, tpr, _ = roc_curve(y_true, probs)
    precision, recall, _ = precision_recall_curve(y_true, probs)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(fpr, tpr, color="#4477AA", linewidth=2.0, label=f"AUROC={primary_auroc:.4f}")
    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("Primary SHD ROC")
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].grid(True, color="#E6E6E6", linewidth=0.6)

    axes[1].plot(recall, precision, color="#CC6677", linewidth=2.0, label=f"AUPRC={primary_auprc:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Primary SHD precision-recall")
    axes[1].legend(loc="lower left", frameon=False)
    axes[1].grid(True, color="#E6E6E6", linewidth=0.6)

    fig.tight_layout()
    save_figure(fig, output_dir, "fig_primary_roc_pr")
    return {"primary_auroc_recomputed": primary_auroc, "primary_auprc_recomputed": primary_auprc}


def plot_benchmark_comparison(args: argparse.Namespace, output_dir: Path) -> dict[str, float | int | None]:
    rows = [
        ("Public EchoNext-Mini benchmark", args.public_benchmark_auroc, "#999999"),
    ]
    baseline = read_metric(args.baseline_metrics, "Test_Primary_AUROC")
    if baseline is not None:
        rows.append(("Internal FCMAE ConvNeXt baseline", baseline, "#88CCEE"))
    rows.append(("Strict FCMAE(ECA)->ECA control", args.strict_control_auroc, "#44AA99"))
    best = read_metric(args.best_metrics, "Test_Primary_AUROC")
    if best is not None:
        rows.append(("Ours seed 42", best, "#CC6677"))
    multiseed = read_multiseed(args.multiseed_metrics)
    multiseed_values = []
    if best is not None:
        multiseed_values.append(best)
    multiseed_values.extend(row["primary_auroc"] for row in multiseed)
    multiseed_mean = float(np.mean(multiseed_values)) if multiseed_values else None
    multiseed_std = float(np.std(multiseed_values, ddof=1)) if len(multiseed_values) > 1 else None
    if multiseed_mean is not None:
        rows.append((f"Ours multi-seed mean (n={len(multiseed_values)})", multiseed_mean, "#AA4499"))

    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [row[2] for row in rows]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    x = np.arange(len(rows))
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Primary SHD AUROC")
    ax.set_ylim(0.78, max(values) + 0.015)
    ax.set_title("Public benchmark and internal model comparison")
    ax.grid(True, axis="y", color="#E6E6E6", linewidth=0.6)
    for i, value in enumerate(values):
        ax.text(i, value + 0.002, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_benchmark_comparison")

    return {
        "public_benchmark_auroc": args.public_benchmark_auroc,
        "internal_baseline_auroc": baseline,
        "strict_control_auroc": args.strict_control_auroc,
        "best_seed42_auroc": best,
        "multiseed_count_including_seed42": len(multiseed_values),
        "multiseed_mean_auroc": multiseed_mean,
        "multiseed_std_auroc": multiseed_std,
    }


def plot_optional_saliency(args: argparse.Namespace, output_dir: Path) -> bool:
    path = Path(args.saliency_dir) / "lead_importance.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    if "lead" not in df.columns or "normalized_importance" not in df.columns:
        return False
    values = df["normalized_importance"].astype(float).to_numpy() * 100.0
    leads = df["lead"].astype(str).to_numpy()
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.barh(leads[order], values[order], color="#DDCC77")
    ax.set_xlabel("Normalized saliency (%)")
    ax.set_title("Integrated gradients lead importance")
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.6)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_lead_saliency")
    return True


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    csv_path = data_dir / args.csv_name
    wave_path = data_dir / args.test_wave_name
    tab_path = data_dir / args.test_tab_name
    labels = load_labels(csv_path, split="test")
    waves_raw = np.load(wave_path, mmap_mode="r")
    tabular = np.load(tab_path, mmap_mode="r")
    if len(labels) != len(tabular):
        raise ValueError(f"Label/tabular length mismatch: labels={len(labels)} tabular={len(tabular)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(args.checkpoint), device)
    probs = predict_primary(model, waves_raw, tabular, args.batch_size, device)
    if len(probs) != len(labels):
        raise ValueError(f"Prediction/label length mismatch: probs={len(probs)} labels={len(labels)}")

    sample_idx = plot_ecg_example(waves_raw, labels, probs, output_dir, args.sampling_rate)
    plot_label_prevalence(labels, output_dir)
    curve_metrics = plot_roc_pr(labels, probs, output_dir)
    comparison_metrics = plot_benchmark_comparison(args, output_dir)
    saliency_available = plot_optional_saliency(args, output_dir)

    summary = {
        "output_dir": str(output_dir),
        "n_test": int(len(labels)),
        "positive_shd_rate": float(labels[:, -1].mean()),
        "selected_ecg_sample_index": int(sample_idx),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "curve_metrics": curve_metrics,
        "comparison_metrics": comparison_metrics,
        "saliency_figure_available": saliency_available,
    }
    (output_dir / "figure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
