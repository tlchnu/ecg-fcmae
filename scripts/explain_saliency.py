#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Post-hoc saliency analysis for EchoNext ECG ConvNeXt checkpoints.

This script does not change model training. It loads a trained downstream
checkpoint, selects high-confidence test samples for the target label, and
computes input-gradient or Integrated Gradients saliency over the 12-lead ECG.

Outputs are intended for paper-level qualitative analysis:
  - sample_scores.csv: selected test samples and target probabilities
  - lead_importance.csv: mean saliency per ECG lead
  - saliency_selected.npy: saliency tensor for selected samples
  - avg_saliency_heatmap.png/pdf and lead_importance.png/pdf
  - per-sample waveform/saliency heatmaps for the top selected examples
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
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ecg_project.data.datasets import (  # noqa: E402
    ECHONEXT_LABEL_COLS,
    ECHONEXT_PRIMARY,
    EchoNextNpyDataset,
)
from ecg_project.models.convnext import ECGConvNeXt1D  # noqa: E402


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoNext saliency analysis")
    parser.add_argument("--checkpoint", required=True, help="Downstream .pth checkpoint")
    parser.add_argument("--data-dir", default="data/echonext")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--csv-name", default="echonext_metadata_100k.csv")
    parser.add_argument("--test-wave-name", default="EchoNext_test_waveforms.npy")
    parser.add_argument("--test-tab-name", default="EchoNext_test_tabular_features.npy")
    parser.add_argument("--val-wave-name", default="EchoNext_val_waveforms.npy")
    parser.add_argument("--val-tab-name", default="EchoNext_val_tabular_features.npy")
    parser.add_argument("--train-wave-name", default="EchoNext_train_waveforms.npy")
    parser.add_argument("--train-tab-name", default="EchoNext_train_tabular_features.npy")
    parser.add_argument("--target-index", type=int, default=-1)
    parser.add_argument("--target-name", default="SHD")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--method", choices=["grad_input", "integrated_gradients"], default="grad_input")
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--max-plots", type=int, default=6)
    parser.add_argument("--sampling-rate", type=float, default=250.0)
    return parser.parse_args()


def split_paths(args: argparse.Namespace) -> tuple[str, str, str]:
    data_dir = Path(args.data_dir)
    if args.split == "test":
        wave_name, tab_name = args.test_wave_name, args.test_tab_name
    elif args.split == "val":
        wave_name, tab_name = args.val_wave_name, args.val_tab_name
    else:
        wave_name, tab_name = args.train_wave_name, args.train_tab_name
    return str(data_dir / args.csv_name), str(data_dir / wave_name), str(data_dir / tab_name)


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
def collect_scores(
    model: ECGConvNeXt1D,
    dataset: EchoNextNpyDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    target_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probs, labels = [], []
    for wave, tab, y in loader:
        logits = model(wave.to(device), tab.to(device))
        probs.append(torch.sigmoid(logits)[:, target_idx].cpu().numpy())
        labels.append(y[:, target_idx].numpy())
    return np.concatenate(probs), np.concatenate(labels)


def select_samples(probs: np.ndarray, labels: np.ndarray, num_samples: int) -> tuple[np.ndarray, list[str]]:
    """Select mostly true positive high-confidence samples plus hard negatives."""
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


def compute_saliency(
    model: ECGConvNeXt1D,
    dataset: EchoNextNpyDataset,
    selected: np.ndarray,
    device: torch.device,
    batch_size: int,
    target_idx: int,
    method: str,
    ig_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_saliency, all_waves, all_labels = [], [], []
    for start in range(0, len(selected), batch_size):
        batch_indices = selected[start : start + batch_size]
        batch = [dataset[int(i)] for i in batch_indices]
        wave = torch.stack([x[0] for x in batch]).to(device)
        tab = torch.stack([x[1] for x in batch]).to(device)
        labels = torch.stack([x[2] for x in batch]).numpy()
        if method == "integrated_gradients":
            saliency = saliency_integrated_gradients(model, wave, tab, target_idx, ig_steps)
        else:
            saliency = saliency_grad_input(model, wave, tab, target_idx)
        all_saliency.append(saliency.cpu().numpy())
        all_waves.append(wave.detach().cpu().numpy())
        all_labels.append(labels)
    return np.concatenate(all_saliency), np.concatenate(all_waves), np.concatenate(all_labels)


def time_axis(length: int, sampling_rate: float) -> np.ndarray:
    if sampling_rate <= 0:
        return np.arange(length)
    return np.arange(length) / sampling_rate


def save_lead_importance(output_dir: Path, avg_saliency: np.ndarray) -> None:
    lead_values = avg_saliency.mean(axis=1)
    total = float(lead_values.sum())
    with (output_dir / "lead_importance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["lead", "mean_abs_saliency", "normalized_importance"])
        writer.writeheader()
        for lead, value in zip(LEAD_NAMES, lead_values):
            writer.writerow(
                {
                    "lead": lead,
                    "mean_abs_saliency": f"{float(value):.8f}",
                    "normalized_importance": f"{float(value / total) if total > 0 else 0.0:.8f}",
                }
            )

    order = np.argsort(lead_values)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(np.asarray(LEAD_NAMES)[order], lead_values[order], color="#4477AA")
    ax.set_xlabel("Mean absolute saliency")
    ax.set_ylabel("ECG lead")
    ax.set_title("Lead-level contribution for SHD output")
    fig.tight_layout()
    fig.savefig(output_dir / "lead_importance.png", dpi=300)
    fig.savefig(output_dir / "lead_importance.pdf")
    plt.close(fig)


def save_avg_heatmap(output_dir: Path, avg_saliency: np.ndarray, sampling_rate: float) -> None:
    x = time_axis(avg_saliency.shape[1], sampling_rate)
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
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
    ax.set_title("Average SHD saliency over selected test samples")
    fig.colorbar(image, ax=ax, label="log(1 + abs saliency)")
    fig.tight_layout()
    fig.savefig(output_dir / "avg_saliency_heatmap.png", dpi=300)
    fig.savefig(output_dir / "avg_saliency_heatmap.pdf")
    plt.close(fig)


def save_sample_heatmaps(
    output_dir: Path,
    waves: np.ndarray,
    saliency: np.ndarray,
    selected: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    sampling_rate: float,
    max_plots: int,
) -> None:
    plot_dir = output_dir / "samples"
    plot_dir.mkdir(parents=True, exist_ok=True)
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
            f"Test index {int(selected[rank])}: target={int(labels[rank])}, "
            f"prob={float(probs[rank]):.3f}"
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
        axes[1].set_title("Gradient saliency")
        fig.colorbar(image, ax=axes[1], label="log(1 + abs saliency)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"sample_{rank:02d}_idx{int(selected[rank])}.png", dpi=300)
        fig.savefig(plot_dir / f"sample_{rank:02d}_idx{int(selected[rank])}.pdf")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_idx = args.target_index
    if target_idx < 0:
        target_idx = len(ECHONEXT_LABEL_COLS) + target_idx
    if not 0 <= target_idx < len(ECHONEXT_LABEL_COLS):
        raise ValueError(f"target-index out of range: {args.target_index}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    csv_path, wave_path, tab_path = split_paths(args)
    dataset = EchoNextNpyDataset(
        csv_path=csv_path,
        wave_path=wave_path,
        tab_path=tab_path,
        split_name=args.split,
        label_cols=ECHONEXT_LABEL_COLS,
        limit=None,
    )

    probs, labels = collect_scores(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_idx=target_idx,
    )
    selected, groups = select_samples(probs, labels, args.num_samples)
    selected_probs = probs[selected]
    selected_labels = labels[selected]

    saliency, waves, full_labels = compute_saliency(
        model=model,
        dataset=dataset,
        selected=selected,
        device=device,
        batch_size=max(1, min(args.batch_size, 8)),
        target_idx=target_idx,
        method=args.method,
        ig_steps=args.ig_steps,
    )

    np.save(output_dir / "saliency_selected.npy", saliency)
    np.save(output_dir / "waveforms_selected.npy", waves)
    avg_saliency = saliency.mean(axis=0)
    np.save(output_dir / "avg_saliency.npy", avg_saliency)

    with (output_dir / "sample_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "test_index", "group", "target_label", "target_probability"],
        )
        writer.writeheader()
        for rank, idx in enumerate(selected):
            writer.writerow(
                {
                    "rank": rank,
                    "test_index": int(idx),
                    "group": groups[rank],
                    "target_label": int(selected_labels[rank]),
                    "target_probability": f"{float(selected_probs[rank]):.8f}",
                }
            )

    save_lead_importance(output_dir, avg_saliency)
    save_avg_heatmap(output_dir, avg_saliency, args.sampling_rate)
    save_sample_heatmaps(
        output_dir=output_dir,
        waves=waves,
        saliency=saliency,
        selected=selected,
        probs=selected_probs,
        labels=selected_labels,
        sampling_rate=args.sampling_rate,
        max_plots=args.max_plots,
    )

    summary = {
        "checkpoint": str(Path(args.checkpoint)),
        "split": args.split,
        "target_index": target_idx,
        "target_column": ECHONEXT_LABEL_COLS[target_idx],
        "target_name": args.target_name,
        "method": args.method,
        "ig_steps": args.ig_steps if args.method == "integrated_gradients" else None,
        "num_selected": int(len(selected)),
        "positive_selected": int(selected_labels.sum()),
        "outputs": [
            "sample_scores.csv",
            "lead_importance.csv",
            "avg_saliency_heatmap.png",
            "lead_importance.png",
            "samples/",
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
