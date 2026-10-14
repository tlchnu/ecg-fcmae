#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""All-label post-hoc saliency for EchoNext ECG-SHD checkpoints.

This script is a paper-figure utility. It does not modify training, model
weights, data splits, or evaluation definitions. It loads one trained
downstream checkpoint, computes probabilities for the EchoNext split, and then
generates saliency heatmaps for all 12 outputs: 11 SHD subtype labels plus the
aggregate SHD label.

Outputs:
  - per_label/<idx>_<name>/avg_saliency_heatmap.{png,pdf}
  - per_label/<idx>_<name>/lead_importance.csv
  - fig_all_labels_lead_importance_heatmap.{png,pdf}
  - fig_all_labels_temporal_saliency_heatmap.{png,pdf}
  - fig_per_label_metrics.{png,pdf}
  - all_label_summary.csv
  - summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_project.data.datasets import ECHONEXT_LABEL_COLS, EchoNextNpyDataset  # noqa: E402
from ecg_project.models.convnext import ECGConvNeXt1D  # noqa: E402


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="All-label EchoNext saliency analysis")
    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/20260630_eca_drop_path_010/"
            "echonext_eca_drop_path_010_pretrained_bce_eca_drop_path_010_seed42.pth"
        ),
        help="Downstream checkpoint with model and model_config keys.",
    )
    parser.add_argument("--data-dir", default="data/echonext")
    parser.add_argument("--output-dir", default="figures/paper/20260707_echonext_mini_all_labels")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--csv-name", default="echonext_metadata_100k.csv")
    parser.add_argument("--test-wave-name", default="EchoNext_test_waveforms.npy")
    parser.add_argument("--test-tab-name", default="EchoNext_test_tabular_features.npy")
    parser.add_argument("--val-wave-name", default="EchoNext_val_waveforms.npy")
    parser.add_argument("--val-tab-name", default="EchoNext_val_tabular_features.npy")
    parser.add_argument("--train-wave-name", default="EchoNext_train_waveforms.npy")
    parser.add_argument("--train-tab-name", default="EchoNext_train_tabular_features.npy")
    parser.add_argument("--num-samples-per-label", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=512)
    parser.add_argument("--saliency-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--method", choices=["grad_input", "integrated_gradients"], default="integrated_gradients")
    parser.add_argument("--ig-steps", type=int, default=8)
    parser.add_argument("--max-sample-plots-per-label", type=int, default=2)
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    parser.add_argument("--temporal-bins", type=int, default=500)
    return parser.parse_args()


def split_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_dir = Path(args.data_dir)
    if args.split == "test":
        wave_name, tab_name = args.test_wave_name, args.test_tab_name
    elif args.split == "val":
        wave_name, tab_name = args.val_wave_name, args.val_tab_name
    else:
        wave_name, tab_name = args.train_wave_name, args.train_tab_name
    return data_dir / args.csv_name, data_dir / wave_name, data_dir / tab_name


def safe_stem(text: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return stem or "label"


def torch_load(path: str | Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_model(checkpoint_path: str | Path, device: torch.device) -> ECGConvNeXt1D:
    checkpoint = torch_load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected downstream checkpoint with key 'model': {checkpoint_path}")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"Checkpoint is missing model_config: {checkpoint_path}")
    model = ECGConvNeXt1D(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def collect_all_probs(
    model: ECGConvNeXt1D,
    dataset: EchoNextNpyDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, labels = [], []
    for wave, tab, y in loader:
        logits = model(wave.to(device), tab.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def select_samples(probs: np.ndarray, labels: np.ndarray, num_samples: int) -> tuple[np.ndarray, list[str]]:
    """Select mostly true-positive high-score samples and hard negatives."""
    num_pos = int(np.ceil(num_samples * 0.75))
    num_neg = max(0, num_samples - num_pos)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]

    pos_sorted = pos[np.argsort(-probs[pos])] if len(pos) else np.array([], dtype=int)
    neg_sorted = neg[np.argsort(-probs[neg])] if len(neg) else np.array([], dtype=int)
    selected = list(pos_sorted[:num_pos]) + list(neg_sorted[:num_neg])
    groups = ["positive_high_score"] * min(num_pos, len(pos_sorted)) + [
        "negative_high_score"
    ] * min(num_neg, len(neg_sorted))

    if len(selected) < num_samples:
        chosen = set(selected)
        fallback = [idx for idx in np.argsort(-probs) if int(idx) not in chosen]
        selected.extend(fallback[: num_samples - len(selected)])
        groups.extend(["fallback_high_score"] * (num_samples - len(groups)))
    return np.asarray(selected[:num_samples], dtype=int), groups[:num_samples]


def saliency_grad_input(
    model: ECGConvNeXt1D,
    wave: torch.Tensor,
    tab: torch.Tensor,
    target_idx: int,
) -> torch.Tensor:
    wave = wave.detach().clone().requires_grad_(True)
    logits = model(wave, tab)[:, target_idx].sum()
    grad = torch.autograd.grad(logits, wave)[0]
    return (grad * wave).detach().abs()


def saliency_integrated_gradients(
    model: ECGConvNeXt1D,
    wave: torch.Tensor,
    tab: torch.Tensor,
    target_idx: int,
    steps: int,
) -> torch.Tensor:
    baseline = torch.zeros_like(wave)
    total_grad = torch.zeros_like(wave)
    delta = wave.detach() - baseline
    for step in range(1, steps + 1):
        alpha = float(step) / float(steps)
        scaled = (baseline + alpha * delta).detach().requires_grad_(True)
        logits = model(scaled, tab)[:, target_idx].sum()
        grad = torch.autograd.grad(logits, scaled)[0]
        total_grad += grad.detach()
    return (delta * total_grad / float(steps)).abs()


def compute_label_saliency(
    model: ECGConvNeXt1D,
    dataset: EchoNextNpyDataset,
    selected: np.ndarray,
    device: torch.device,
    batch_size: int,
    target_idx: int,
    method: str,
    ig_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    all_saliency, all_waves = [], []
    for start in range(0, len(selected), batch_size):
        batch_indices = selected[start : start + batch_size]
        batch = [dataset[int(i)] for i in batch_indices]
        wave = torch.stack([x[0] for x in batch]).to(device)
        tab = torch.stack([x[1] for x in batch]).to(device)
        if method == "integrated_gradients":
            saliency = saliency_integrated_gradients(model, wave, tab, target_idx, ig_steps)
        else:
            saliency = saliency_grad_input(model, wave, tab, target_idx)
        all_saliency.append(saliency.cpu().numpy())
        all_waves.append(wave.detach().cpu().numpy())
    return np.concatenate(all_saliency), np.concatenate(all_waves)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def time_axis(length: int, sampling_rate: float) -> np.ndarray:
    if sampling_rate <= 0:
        return np.arange(length)
    return np.arange(length) / sampling_rate


def downsample_time(values: np.ndarray, bins: int) -> np.ndarray:
    if values.shape[-1] <= bins:
        return values
    edges = np.linspace(0, values.shape[-1], bins + 1, dtype=int)
    return np.stack([values[..., edges[i] : edges[i + 1]].mean(axis=-1) for i in range(bins)], axis=-1)


def save_lead_importance(label_dir: Path, lead_values: np.ndarray) -> np.ndarray:
    total = float(lead_values.sum())
    normalized = lead_values / total if total > 0 else np.zeros_like(lead_values)
    with (label_dir / "lead_importance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lead", "mean_abs_saliency", "normalized_importance"])
        writer.writeheader()
        for lead, raw, norm in zip(LEAD_NAMES, lead_values, normalized):
            writer.writerow(
                {
                    "lead": lead,
                    "mean_abs_saliency": f"{float(raw):.8f}",
                    "normalized_importance": f"{float(norm):.8f}",
                }
            )
    return normalized


def plot_per_label_heatmap(
    label_dir: Path,
    avg_saliency: np.ndarray,
    short_label: str,
    label_col: str,
    sampling_rate: float,
) -> None:
    x = time_axis(avg_saliency.shape[1], sampling_rate)
    fig, ax = plt.subplots(figsize=(9.8, 4.3))
    image = ax.imshow(
        np.log1p(avg_saliency),
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[float(x[0]), float(x[-1]), -0.5, len(LEAD_NAMES) - 0.5],
    )
    ax.set_yticks(range(len(LEAD_NAMES)))
    ax.set_yticklabels(LEAD_NAMES)
    ax.set_xlabel("Time (s)" if sampling_rate > 0 else "Sample index")
    ax.set_title(f"{short_label}: average saliency ({label_col})")
    fig.colorbar(image, ax=ax, label="log(1 + abs saliency)")
    fig.tight_layout()
    save_figure(fig, label_dir, "avg_saliency_heatmap")

    lead_values = avg_saliency.mean(axis=1)
    order = np.argsort(lead_values)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(np.asarray(LEAD_NAMES)[order], lead_values[order], color="#4477AA")
    ax.set_xlabel("Mean absolute saliency")
    ax.set_ylabel("ECG lead")
    ax.set_title(f"{short_label}: lead contribution")
    fig.tight_layout()
    save_figure(fig, label_dir, "lead_importance")


def plot_sample_overlays(
    label_dir: Path,
    waves: np.ndarray,
    saliency: np.ndarray,
    selected: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    short_label: str,
    sampling_rate: float,
    max_plots: int,
) -> None:
    sample_dir = label_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    count = min(max_plots, len(selected))
    for rank in range(count):
        wave = waves[rank]
        sal = saliency[rank]
        wave_std = (wave - wave.mean(axis=1, keepdims=True)) / (wave.std(axis=1, keepdims=True) + 1e-6)
        x = time_axis(wave.shape[1], sampling_rate)
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.2), sharex=True)
        axes[0].imshow(
            np.clip(wave_std, -3, 3),
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            extent=[float(x[0]), float(x[-1]), -0.5, len(LEAD_NAMES) - 0.5],
            vmin=-3,
            vmax=3,
        )
        axes[0].set_yticks(range(len(LEAD_NAMES)))
        axes[0].set_yticklabels(LEAD_NAMES)
        axes[0].set_title(
            f"{short_label} sample idx={int(selected[rank])}, "
            f"target={int(labels[rank])}, prob={float(probs[rank]):.3f}"
        )
        image = axes[1].imshow(
            np.log1p(sal),
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=[float(x[0]), float(x[-1]), -0.5, len(LEAD_NAMES) - 0.5],
        )
        axes[1].set_yticks(range(len(LEAD_NAMES)))
        axes[1].set_yticklabels(LEAD_NAMES)
        axes[1].set_xlabel("Time (s)" if sampling_rate > 0 else "Sample index")
        axes[1].set_title("Input saliency")
        fig.colorbar(image, ax=axes[1], label="log(1 + abs saliency)")
        fig.tight_layout()
        save_figure(fig, sample_dir, f"sample_{rank:02d}_idx{int(selected[rank])}")


def plot_combined_lead_heatmap(output_dir: Path, lead_matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    image = ax.imshow(lead_matrix * 100.0, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(LEAD_NAMES)))
    ax.set_xticklabels(LEAD_NAMES)
    ax.set_yticks(range(len(SHORT_LABELS)))
    ax.set_yticklabels(SHORT_LABELS)
    ax.set_xlabel("ECG lead")
    ax.set_title("Normalized saliency by label and lead")
    fig.colorbar(image, ax=ax, label="Normalized saliency (%)")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_all_labels_lead_importance_heatmap")


def plot_combined_temporal_heatmap(
    output_dir: Path,
    saliency_stack: np.ndarray,
    sampling_rate: float,
    temporal_bins: int,
) -> None:
    temporal = saliency_stack.mean(axis=1)
    temporal = np.log1p(temporal)
    temporal = downsample_time(temporal, temporal_bins)
    row_min = temporal.min(axis=1, keepdims=True)
    row_max = temporal.max(axis=1, keepdims=True)
    temporal_norm = (temporal - row_min) / (row_max - row_min + 1e-8)
    total_seconds = saliency_stack.shape[-1] / sampling_rate if sampling_rate > 0 else saliency_stack.shape[-1]
    x_extent = [0.0, float(total_seconds)]
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    image = ax.imshow(
        temporal_norm,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[x_extent[0], x_extent[1], -0.5, len(SHORT_LABELS) - 0.5],
    )
    ax.set_yticks(range(len(SHORT_LABELS)))
    ax.set_yticklabels(SHORT_LABELS)
    ax.set_xlabel("Time (s)" if sampling_rate > 0 else "Sample index")
    ax.set_title("Label-wise temporal saliency aggregated over leads")
    fig.colorbar(image, ax=ax, label="Row-normalized log saliency")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_all_labels_temporal_saliency_heatmap")


def plot_metric_bars(output_dir: Path, aurocs: np.ndarray, auprcs: np.ndarray, prevalence: np.ndarray) -> None:
    x = np.arange(len(SHORT_LABELS))
    width = 0.28
    fig, ax1 = plt.subplots(figsize=(10.2, 4.6))
    ax1.bar(x - width / 2, aurocs, width=width, label="AUROC", color="#4477AA")
    ax1.bar(x + width / 2, auprcs, width=width, label="AUPRC", color="#CC6677")
    ax1.set_ylim(0.0, 1.0)
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(SHORT_LABELS, rotation=35, ha="right")
    ax1.grid(True, axis="y", color="#E6E6E6", linewidth=0.6)
    ax2 = ax1.twinx()
    ax2.plot(x, prevalence * 100.0, color="#999933", marker="o", linewidth=1.2, label="Prevalence")
    ax2.set_ylabel("Prevalence (%)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="lower left")
    ax1.set_title("Per-label test metrics and prevalence")
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_per_label_metrics")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    per_label_dir = output_dir / "per_label"
    per_label_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    csv_path, wave_path, tab_path = split_paths(args)
    dataset = EchoNextNpyDataset(
        csv_path=str(csv_path),
        wave_path=str(wave_path),
        tab_path=str(tab_path),
        split_name=args.split,
        label_cols=ECHONEXT_LABEL_COLS,
        limit=None,
    )
    probs, labels = collect_all_probs(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.score_batch_size,
        num_workers=args.num_workers,
    )

    lead_matrix = np.zeros((len(ECHONEXT_LABEL_COLS), len(LEAD_NAMES)), dtype=np.float64)
    saliency_stack = []
    summary_rows = []

    for label_idx, (label_col, short_label) in enumerate(zip(ECHONEXT_LABEL_COLS, SHORT_LABELS)):
        label_stem = f"{label_idx:02d}_{safe_stem(short_label)}"
        label_dir = per_label_dir / label_stem
        label_dir.mkdir(parents=True, exist_ok=True)

        label_probs = probs[:, label_idx]
        label_targets = labels[:, label_idx]
        selected, groups = select_samples(label_probs, label_targets, args.num_samples_per_label)
        selected_probs = label_probs[selected]
        selected_targets = label_targets[selected]

        saliency, waves = compute_label_saliency(
            model=model,
            dataset=dataset,
            selected=selected,
            device=device,
            batch_size=args.saliency_batch_size,
            target_idx=label_idx,
            method=args.method,
            ig_steps=args.ig_steps,
        )
        avg_saliency = saliency.mean(axis=0)
        np.save(label_dir / "avg_saliency.npy", avg_saliency)
        np.save(label_dir / "saliency_selected.npy", saliency)
        np.save(label_dir / "selected_indices.npy", selected)

        lead_values = avg_saliency.mean(axis=1)
        lead_matrix[label_idx] = save_lead_importance(label_dir, lead_values)
        saliency_stack.append(avg_saliency)
        plot_per_label_heatmap(label_dir, avg_saliency, short_label, label_col, args.sampling_rate)
        plot_sample_overlays(
            label_dir=label_dir,
            waves=waves,
            saliency=saliency,
            selected=selected,
            probs=selected_probs,
            labels=selected_targets,
            short_label=short_label,
            sampling_rate=args.sampling_rate,
            max_plots=args.max_sample_plots_per_label,
        )

        with (label_dir / "sample_scores.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["rank", "test_index", "group", "target_label", "target_probability"],
            )
            writer.writeheader()
            for rank, sample_idx in enumerate(selected):
                writer.writerow(
                    {
                        "rank": rank,
                        "test_index": int(sample_idx),
                        "group": groups[rank],
                        "target_label": int(selected_targets[rank]),
                        "target_probability": f"{float(selected_probs[rank]):.8f}",
                    }
                )

        row = {
            "label_index": label_idx,
            "short_label": short_label,
            "label_column": label_col,
            "prevalence": float(label_targets.mean()),
            "auroc": safe_auc(label_targets, label_probs),
            "auprc": safe_auprc(label_targets, label_probs),
            "num_selected": int(len(selected)),
            "selected_positive_count": int(selected_targets.sum()),
            "label_dir": str(label_dir.relative_to(output_dir)),
        }
        for lead, value in zip(LEAD_NAMES, lead_matrix[label_idx]):
            row[f"lead_{lead}_normalized"] = float(value)
        summary_rows.append(row)
        (label_dir / "summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(json.dumps({"label": short_label, "auroc": row["auroc"], "auprc": row["auprc"]}, indent=2))

    saliency_array = np.stack(saliency_stack, axis=0)
    np.save(output_dir / "all_labels_avg_saliency.npy", saliency_array)
    np.save(output_dir / "all_labels_lead_importance.npy", lead_matrix)

    with (output_dir / "all_label_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    aurocs = np.asarray([row["auroc"] for row in summary_rows], dtype=float)
    auprcs = np.asarray([row["auprc"] for row in summary_rows], dtype=float)
    prevalence = np.asarray([row["prevalence"] for row in summary_rows], dtype=float)
    plot_combined_lead_heatmap(output_dir, lead_matrix)
    plot_combined_temporal_heatmap(output_dir, saliency_array, args.sampling_rate, args.temporal_bins)
    plot_metric_bars(output_dir, aurocs, auprcs, prevalence)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "device": str(device),
        "method": args.method,
        "ig_steps": args.ig_steps if args.method == "integrated_gradients" else None,
        "num_labels": len(ECHONEXT_LABEL_COLS),
        "num_samples_per_label": args.num_samples_per_label,
        "outputs": [
            "all_label_summary.csv",
            "all_labels_avg_saliency.npy",
            "all_labels_lead_importance.npy",
            "fig_all_labels_lead_importance_heatmap.png",
            "fig_all_labels_temporal_saliency_heatmap.png",
            "fig_per_label_metrics.png",
            "per_label/",
        ],
        "interpretation_note": (
            "Gradient-based saliency is a post-hoc association analysis and should not be "
            "interpreted as causal clinical evidence."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
