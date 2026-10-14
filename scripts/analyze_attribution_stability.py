#!/usr/bin/env python
"""Resumable cross-seed integrated-gradients stability audit for EchoNext."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ecg_project.data.datasets import ECHONEXT_LABEL_COLS, EchoNextNpyDataset
from scripts.evaluate_lead_robustness import LEAD_NAMES
from scripts.export_shd_predictions import load_model, sha256_file


def select_balanced_indices(
    labels: np.ndarray, samples_per_label: int, seed: int
) -> list[np.ndarray]:
    """Select fixed positive/negative-balanced rows independently of model scores."""
    if samples_per_label <= 0:
        raise ValueError("samples_per_label must be positive")
    selections = []
    for label_index in range(labels.shape[1]):
        rng = np.random.default_rng(seed + 1009 * label_index)
        positive = np.flatnonzero(labels[:, label_index] == 1)
        negative = np.flatnonzero(labels[:, label_index] == 0)
        positive_target = min(len(positive), samples_per_label // 2)
        negative_target = min(len(negative), samples_per_label - positive_target)
        selected_positive = rng.choice(positive, positive_target, replace=False) if positive_target else np.array([], dtype=int)
        selected_negative = rng.choice(negative, negative_target, replace=False) if negative_target else np.array([], dtype=int)
        selected = np.concatenate([selected_positive, selected_negative]).astype(np.int64)
        if len(selected) < samples_per_label:
            chosen = set(selected.tolist())
            remaining = np.asarray([index for index in range(len(labels)) if index not in chosen], dtype=np.int64)
            extra = rng.choice(remaining, min(samples_per_label - len(selected), len(remaining)), replace=False)
            selected = np.concatenate([selected, extra])
        selections.append(np.sort(selected))
    return selections


def downsample_mean(values: np.ndarray, bins: int) -> np.ndarray:
    """Average the last axis into a fixed number of temporal bins."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    edges = np.linspace(0, values.shape[-1], bins + 1, dtype=int)
    return np.asarray([values[..., edges[index] : edges[index + 1]].mean() for index in range(bins)])


def summarize_attribution(
    attribution: np.ndarray, temporal_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Summarize absolute attribution over samples into lead and time profiles."""
    absolute = np.abs(np.asarray(attribution, dtype=float))
    lead_importance = absolute.mean(axis=(0, 2))
    temporal_importance = absolute.mean(axis=(0, 1))
    return lead_importance, downsample_mean(temporal_importance, temporal_bins)


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    """Return Spearman correlation or NaN when either vector is constant."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(spearmanr(first, second).statistic)


def safe_pearson(first: np.ndarray, second: np.ndarray) -> float:
    """Return Pearson correlation or NaN when either vector is constant."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if np.all(first == first[0]) or np.all(second == second[0]):
        return float("nan")
    return float(pearsonr(first, second).statistic)


def parse_seed_checkpoint(values: list[str]) -> dict[int, Path]:
    """Parse repeated `seed=checkpoint` arguments."""
    checkpoints = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"Expected seed=checkpoint, got {value!r}")
        checkpoints[int(seed_text)] = Path(path_text)
    return checkpoints


def compute_train_lead_statistics(wave_path: Path, chunk_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Stream train waveforms and fit scalar per-lead mean/std using train only."""
    waves = np.load(wave_path, mmap_mode="r")
    sums = np.zeros(12, dtype=np.float64)
    squared_sums = np.zeros(12, dtype=np.float64)
    count = 0
    for start in range(0, len(waves), chunk_size):
        batch = np.asarray(waves[start : start + chunk_size], dtype=np.float64)
        batch = np.squeeze(batch, axis=1)
        sums += batch.sum(axis=(0, 1))
        squared_sums += np.square(batch).sum(axis=(0, 1))
        count += batch.shape[0] * batch.shape[1]
    mean = sums / count
    variance = np.maximum(squared_sums / count - np.square(mean), 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def fetch_batch(dataset: EchoNextNpyDataset, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack selected waveform/tabular rows on a target device."""
    rows = [dataset[int(index)] for index in indices]
    wave = torch.stack([row[0] for row in rows]).to(device)
    tabular = torch.stack([row[1] for row in rows]).to(device)
    return wave, tabular


def make_baseline(
    name: str,
    wave: torch.Tensor,
    lead_mean: np.ndarray,
    lead_std: np.ndarray,
    seed: int,
) -> torch.Tensor:
    """Construct zero, train-lead-mean, or deterministic small-noise baselines."""
    if name == "zero":
        return torch.zeros_like(wave)
    if name == "train_lead_mean":
        values = torch.as_tensor(lead_mean, dtype=wave.dtype, device=wave.device).view(1, 12, 1)
        return values.expand_as(wave).clone()
    if name.startswith("noise_"):
        generator = torch.Generator(device=wave.device)
        generator.manual_seed(seed)
        scale = torch.as_tensor(lead_std, dtype=wave.dtype, device=wave.device).view(1, 12, 1) * 0.05
        return torch.randn(wave.shape, generator=generator, device=wave.device, dtype=wave.dtype) * scale
    raise ValueError(f"Unknown baseline: {name}")


def integrated_gradients(
    model: torch.nn.Module,
    wave: torch.Tensor,
    tabular: torch.Tensor,
    label_index: int,
    baseline: torch.Tensor,
    steps: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute signed IG and per-sample completeness error in bounded batches."""
    all_attribution = []
    all_error = []
    for start in range(0, len(wave), batch_size):
        input_batch = wave[start : start + batch_size]
        tabular_batch = tabular[start : start + batch_size]
        baseline_batch = baseline[start : start + batch_size]
        delta = input_batch - baseline_batch
        total_gradient = torch.zeros_like(input_batch)
        for step in range(1, steps + 1):
            alpha = float(step) / float(steps)
            interpolated = (baseline_batch + alpha * delta).detach().requires_grad_(True)
            target_logit = model(interpolated, tabular_batch)[:, label_index].sum()
            gradient = torch.autograd.grad(target_logit, interpolated)[0]
            total_gradient += gradient.detach()
        attribution = delta * total_gradient / float(steps)
        with torch.no_grad():
            logit_difference = (
                model(input_batch, tabular_batch)[:, label_index]
                - model(baseline_batch, tabular_batch)[:, label_index]
            )
        approximation = attribution.flatten(1).sum(dim=1)
        denominator = torch.clamp(logit_difference.abs(), min=1e-6)
        relative_error = (approximation - logit_difference).abs() / denominator
        all_attribution.append(attribution.detach().cpu().numpy())
        all_error.append(relative_error.detach().cpu().numpy())
    return np.concatenate(all_attribution), np.concatenate(all_error)


@torch.inference_mode()
def occlusion_drops(
    model: torch.nn.Module,
    wave: torch.Tensor,
    tabular: torch.Tensor,
    label_index: int,
    temporal_windows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mean target-logit drops for each lead and fixed time window."""
    original = model(wave, tabular)[:, label_index]
    lead_drops = []
    for lead_index in range(12):
        occluded = wave.clone()
        occluded[:, lead_index] = 0.0
        lead_drops.append(float((original - model(occluded, tabular)[:, label_index]).mean().cpu()))
    time_drops = []
    edges = np.linspace(0, wave.shape[-1], temporal_windows + 1, dtype=int)
    for window_index in range(temporal_windows):
        occluded = wave.clone()
        occluded[:, :, edges[window_index] : edges[window_index + 1]] = 0.0
        time_drops.append(float((original - model(occluded, tabular)[:, label_index]).mean().cpu()))
    return np.asarray(lead_drops), np.asarray(time_drops)


def randomize_model_parameters(model: torch.nn.Module) -> None:
    """Reinitialize every leaf module that exposes `reset_parameters`."""
    for module in model.modules():
        if len(list(module.children())) == 0 and hasattr(module, "reset_parameters"):
            module.reset_parameters()


def save_heatmap(matrix: np.ndarray, rows: list[str], columns: list[str], output_stem: Path, title: str) -> None:
    """Save one academic-style stability heatmap as PNG and PDF."""
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
    fig, ax = plt.subplots(figsize=(max(7, len(columns) * 0.45), max(5, len(rows) * 0.35)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Mean absolute attribution")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-seed all-label integrated-gradients stability audit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", action="append", required=True, help="seed=checkpoint")
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=[42, 2024, 3407],
        help="Exact checkpoint seed set required by the formal analysis",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/echonext"))
    parser.add_argument("--csv-name", default="echonext_metadata_100k.csv")
    parser.add_argument("--test-wave-name", default="EchoNext_test_waveforms.npy")
    parser.add_argument("--test-tab-name", default="EchoNext_test_tabular_features.npy")
    parser.add_argument("--train-wave-name", default="EchoNext_train_waveforms.npy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-label", type=int, default=64)
    parser.add_argument("--selection-seed", type=int, default=20260715)
    parser.add_argument("--ig-steps", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--reference-steps", type=int, default=64)
    parser.add_argument("--noise-baselines", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temporal-bins", type=int, default=100)
    parser.add_argument("--occlusion-windows", type=int, default=20)
    parser.add_argument("--randomization-samples", type=int, default=16)
    parser.add_argument(
        "--shuffled-label-checkpoint",
        type=Path,
        default=None,
        help="Validation-selected checkpoint trained with deterministically shuffled training labels.",
    )
    parser.add_argument("--label-indices", type=int, nargs="+", default=list(range(12)))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = parse_seed_checkpoint(args.checkpoint)
    if sorted(checkpoints) != sorted(args.expected_seeds):
        raise ValueError(
            "Formal attribution stability seed mismatch: "
            f"expected {sorted(args.expected_seeds)}, got {sorted(checkpoints)}"
        )
    if args.reference_steps not in args.ig_steps:
        raise ValueError("reference_steps must be included in ig_steps")
    analysis_dir = args.output_dir / "analysis"
    figures_dir = args.output_dir / "figures"
    failures_dir = args.output_dir / "failures"
    raw_dir = args.output_dir / "raw"
    for directory in (analysis_dir, figures_dir, failures_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)
    config = {
        "seeds": sorted(checkpoints),
        "expected_seeds": sorted(args.expected_seeds),
        "samples_per_label": args.samples_per_label,
        "selection_seed": args.selection_seed,
        "selection_rule": "fixed label-balanced selection independent of model predictions",
        "ig_steps": args.ig_steps,
        "reference_steps": args.reference_steps,
        "baselines": ["zero", "train_lead_mean"] + [f"noise_{index}" for index in range(args.noise_baselines)],
        "parameter_randomization_samples_per_label": args.randomization_samples,
        "label_randomization_checkpoint": (
            str(args.shuffled_label_checkpoint) if args.shuffled_label_checkpoint else None
        ),
        "interpretation": "post-hoc model sensitivity audit, not physiological or causal localization",
    }
    (analysis_dir / "analysis_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    dataset = EchoNextNpyDataset(
        args.data_dir / args.csv_name,
        args.data_dir / args.test_wave_name,
        args.data_dir / args.test_tab_name,
        "test",
        ECHONEXT_LABEL_COLS,
        None,
    )
    labels = dataset.labels.numpy().astype(np.int64)
    selections = select_balanced_indices(labels, args.samples_per_label, args.selection_seed)
    sample_rows = []
    for label_index, indices in enumerate(selections):
        for row_index in indices:
            sample_rows.append(
                {
                    "label_index": label_index,
                    "label": ECHONEXT_LABEL_COLS[label_index],
                    "row_id": int(row_index),
                    "target": int(labels[row_index, label_index]),
                }
            )
    pd.DataFrame(sample_rows).to_csv(analysis_dir / "sample_manifest.csv", index=False)
    np.savez_compressed(
        analysis_dir / "sample_indices.npz",
        **{f"label_{index}": indices for index, indices in enumerate(selections)},
    )
    statistics_path = analysis_dir / "train_lead_statistics.npz"
    if statistics_path.exists():
        statistics = np.load(statistics_path)
        lead_mean, lead_std = statistics["mean"], statistics["std"]
    else:
        lead_mean, lead_std = compute_train_lead_statistics(args.data_dir / args.train_wave_name)
        np.savez_compressed(statistics_path, mean=lead_mean, std=lead_std, fitted_split="train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    convergence_rows = []
    baseline_rows = []
    occlusion_rows = []
    randomization_rows = []
    reference_lead: dict[int, dict[int, np.ndarray]] = {}
    reference_time: dict[int, dict[int, np.ndarray]] = {}
    for seed, checkpoint_path in sorted(checkpoints.items()):
        model = load_model(checkpoint_path, device)
        reference_lead[seed] = {}
        reference_time[seed] = {}
        seed_raw_dir = raw_dir / f"seed_{seed}"
        seed_raw_dir.mkdir(parents=True, exist_ok=True)
        for label_index in args.label_indices:
            indices = selections[label_index]
            wave, tabular = fetch_batch(dataset, indices, device)
            zero = make_baseline("zero", wave, lead_mean, lead_std, seed)
            step_summaries = {}
            reference_attribution = None
            for steps in args.ig_steps:
                attribution, completeness = integrated_gradients(
                    model, wave, tabular, label_index, zero, steps, args.batch_size
                )
                lead_profile, time_profile = summarize_attribution(attribution, args.temporal_bins)
                step_summaries[steps] = (lead_profile, time_profile)
                if steps == args.reference_steps:
                    reference_attribution = attribution
                    reference_lead[seed][label_index] = lead_profile
                    reference_time[seed][label_index] = time_profile
                    np.savez_compressed(
                        seed_raw_dir / f"label_{label_index:02d}_ig_zero_step{steps}.npz",
                        row_id=indices,
                        attribution=attribution.astype(np.float32),
                        completeness_error=completeness.astype(np.float32),
                        checkpoint_sha256=np.asarray(sha256_file(checkpoint_path)),
                    )
                convergence_rows.append(
                    {
                        "seed": seed,
                        "label_index": label_index,
                        "label": ECHONEXT_LABEL_COLS[label_index],
                        "steps": steps,
                        "mean_completeness_relative_error": float(np.mean(completeness)),
                        "median_completeness_relative_error": float(np.median(completeness)),
                    }
                )
            reference_lead_profile, reference_time_profile = step_summaries[max(args.ig_steps)]
            for steps, (lead_profile, time_profile) in step_summaries.items():
                row = convergence_rows[-len(args.ig_steps) + args.ig_steps.index(steps)]
                row["lead_spearman_vs_max_steps"] = safe_spearman(lead_profile, reference_lead_profile)
                row["time_pearson_vs_max_steps"] = safe_pearson(time_profile, reference_time_profile)
            assert reference_attribution is not None
            zero_lead, zero_time = summarize_attribution(reference_attribution, args.temporal_bins)
            baseline_names = ["train_lead_mean"] + [f"noise_{index}" for index in range(args.noise_baselines)]
            for baseline_offset, baseline_name in enumerate(baseline_names):
                baseline = make_baseline(
                    baseline_name,
                    wave,
                    lead_mean,
                    lead_std,
                    args.selection_seed + seed + 1009 * label_index + baseline_offset,
                )
                attribution, completeness = integrated_gradients(
                    model,
                    wave,
                    tabular,
                    label_index,
                    baseline,
                    args.reference_steps,
                    args.batch_size,
                )
                lead_profile, time_profile = summarize_attribution(attribution, args.temporal_bins)
                baseline_rows.append(
                    {
                        "seed": seed,
                        "label_index": label_index,
                        "label": ECHONEXT_LABEL_COLS[label_index],
                        "baseline": baseline_name,
                        "lead_spearman_vs_zero": safe_spearman(lead_profile, zero_lead),
                        "time_pearson_vs_zero": safe_pearson(time_profile, zero_time),
                        "mean_completeness_relative_error": float(np.mean(completeness)),
                    }
                )
            lead_drop, time_drop = occlusion_drops(
                model, wave, tabular, label_index, args.occlusion_windows
            )
            ig_time_for_occlusion = downsample_mean(np.abs(reference_attribution).mean(axis=(0, 1)), args.occlusion_windows)
            occlusion_rows.append(
                {
                    "seed": seed,
                    "label_index": label_index,
                    "label": ECHONEXT_LABEL_COLS[label_index],
                    "lead_spearman_ig_vs_logit_drop": safe_spearman(zero_lead, lead_drop),
                    "time_spearman_ig_vs_logit_drop": safe_spearman(ig_time_for_occlusion, time_drop),
                }
            )
            del wave, tabular, zero, reference_attribution

        randomize_model_parameters(model)
        model.eval()
        for label_index in args.label_indices:
            indices = selections[label_index][: args.randomization_samples]
            wave, tabular = fetch_batch(dataset, indices, device)
            zero = torch.zeros_like(wave)
            randomized, _ = integrated_gradients(
                model, wave, tabular, label_index, zero, 32, args.batch_size
            )
            randomized_lead, randomized_time = summarize_attribution(randomized, args.temporal_bins)
            original_archive = np.load(seed_raw_dir / f"label_{label_index:02d}_ig_zero_step{args.reference_steps}.npz")
            original = original_archive["attribution"][: args.randomization_samples]
            original_lead, original_time = summarize_attribution(original, args.temporal_bins)
            randomization_rows.append(
                {
                    "seed": seed,
                    "label_index": label_index,
                    "label": ECHONEXT_LABEL_COLS[label_index],
                    "check": "all_parameter_randomization",
                    "lead_spearman_original_vs_randomized": safe_spearman(original_lead, randomized_lead),
                    "time_pearson_original_vs_randomized": safe_pearson(original_time, randomized_time),
                    "status": "complete",
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cross_seed_rows = []
    for label_index in args.label_indices:
        for first_seed, second_seed in combinations(sorted(checkpoints), 2):
            cross_seed_rows.append(
                {
                    "label_index": label_index,
                    "label": ECHONEXT_LABEL_COLS[label_index],
                    "seed_pair": f"{first_seed}-{second_seed}",
                    "lead_spearman": safe_spearman(reference_lead[first_seed][label_index], reference_lead[second_seed][label_index]),
                    "time_pearson": safe_pearson(reference_time[first_seed][label_index], reference_time[second_seed][label_index]),
                }
            )
    pd.DataFrame(convergence_rows).to_csv(analysis_dir / "ig_convergence.csv", index=False)
    pd.DataFrame(
        cross_seed_rows,
        columns=["label_index", "label", "seed_pair", "lead_spearman", "time_pearson"],
    ).to_csv(analysis_dir / "cross_seed_stability.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(analysis_dir / "baseline_sensitivity.csv", index=False)
    pd.DataFrame(occlusion_rows).to_csv(analysis_dir / "occlusion_agreement.csv", index=False)
    shuffled_label_complete = args.shuffled_label_checkpoint is not None
    if shuffled_label_complete:
        shuffled_model = load_model(args.shuffled_label_checkpoint, device)
        shuffled_model.eval()
        reference_seed = 42
        for label_index in args.label_indices:
            indices = selections[label_index][: args.randomization_samples]
            wave, tabular = fetch_batch(dataset, indices, device)
            zero = torch.zeros_like(wave)
            shuffled_attribution, _ = integrated_gradients(
                shuffled_model,
                wave,
                tabular,
                label_index,
                zero,
                32,
                args.batch_size,
            )
            shuffled_lead, shuffled_time = summarize_attribution(
                shuffled_attribution, args.temporal_bins
            )
            original_archive = np.load(
                raw_dir
                / f"seed_{reference_seed}"
                / f"label_{label_index:02d}_ig_zero_step{args.reference_steps}.npz"
            )
            original = original_archive["attribution"][: args.randomization_samples]
            original_lead, original_time = summarize_attribution(original, args.temporal_bins)
            randomization_rows.append(
                {
                    "seed": reference_seed,
                    "label_index": label_index,
                    "label": ECHONEXT_LABEL_COLS[label_index],
                    "check": "shuffled_label_retrain",
                    "lead_spearman_original_vs_randomized": safe_spearman(
                        original_lead, shuffled_lead
                    ),
                    "time_pearson_original_vs_randomized": safe_pearson(
                        original_time, shuffled_time
                    ),
                    "status": "complete",
                }
            )
        del shuffled_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        randomization_rows.append(
            {
                "seed": "not_available",
                "label_index": "all",
                "label": "all",
                "check": "shuffled_label_retrain",
                "lead_spearman_original_vs_randomized": np.nan,
                "time_pearson_original_vs_randomized": np.nan,
                "status": "blocked_missing_retrained_shuffled_label_checkpoint",
            }
        )
    pd.DataFrame(randomization_rows).to_csv(analysis_dir / "randomization_checks.csv", index=False)
    shuffled_status = (
        {
            "status": "complete",
            "checkpoint": str(args.shuffled_label_checkpoint),
            "checkpoint_sha256": sha256_file(args.shuffled_label_checkpoint),
            "test_data_used_for_checkpoint_selection": False,
        }
        if shuffled_label_complete
        else {
            "status": "blocked",
            "reason": "No validation-selected checkpoint retrained on shuffled training labels is available.",
            "test_data_used": False,
            "required_next_action": "Train a separately named shuffled-label sanity model without test selection.",
        }
    )
    (failures_dir / "shuffled_label_retrain_status.json").write_text(
        json.dumps(shuffled_status, indent=2), encoding="utf-8"
    )
    lead_matrix = np.mean(
        np.stack([[reference_lead[seed][label] for label in args.label_indices] for seed in sorted(checkpoints)]),
        axis=0,
    )
    time_matrix = np.mean(
        np.stack([[reference_time[seed][label] for label in args.label_indices] for seed in sorted(checkpoints)]),
        axis=0,
    )
    row_names = [ECHONEXT_LABEL_COLS[index] for index in args.label_indices]
    save_heatmap(lead_matrix, row_names, LEAD_NAMES, figures_dir / "lead_stability_heatmap", "Cross-seed mean lead attribution")
    save_heatmap(time_matrix, row_names, [str(index) for index in range(args.temporal_bins)], figures_dir / "temporal_stability_heatmap", "Cross-seed mean temporal attribution")
    cross_seed_complete = len(checkpoints) >= 2
    completed = ["ig_convergence", "baseline_sensitivity", "parameter_randomization", "occlusion_agreement"]
    blocked = []
    if cross_seed_complete:
        completed.append("cross_seed_stability")
    else:
        blocked.append("cross_seed_stability_pending_additional_seeds")
    if shuffled_label_complete:
        completed.append("shuffled_label_retrain")
    else:
        blocked.append("shuffled_label_retrain")
    status = {
        "status": "complete" if not blocked else "partial",
        "completed": completed,
        "blocked": blocked,
        "seeds": sorted(checkpoints),
        "labels": args.label_indices,
        "samples_per_label": args.samples_per_label,
    }
    (analysis_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    hash_lines = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256sums.txt":
            hash_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(args.output_dir).as_posix()}")
    (analysis_dir / "sha256sums.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
