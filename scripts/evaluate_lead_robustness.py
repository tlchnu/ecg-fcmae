#!/usr/bin/env python
"""Evaluate frozen full-12 checkpoints under deterministic zero-filled lead masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from joblib import Parallel, delayed
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (PROJECT_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ecg_project.data.datasets import ECHONEXT_LABEL_COLS, ECHONEXT_PRIMARY, EchoNextNpyDataset
from scripts.analyze_shd_paper_results import (
    cluster_bootstrap_indices,
    multilabel_summary,
    percentile_interval,
    safe_metric,
)
from scripts.export_shd_predictions import load_model, sha256_file


# Some cloned servers predate the dataset-level lead constants. Keeping this
# immutable standard order local makes the analysis portable without changing
# any training or data-loading behavior.
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def build_lead_masks(random_seeds: list[int]) -> dict[str, np.ndarray]:
    """Return fixed keep masks for predefined, single-lead, and random-missing tests."""
    masks: dict[str, np.ndarray] = {}

    def keep(name: str, indices: list[int]) -> None:
        mask = np.zeros(12, dtype=bool)
        mask[indices] = True
        masks[name] = mask

    keep("full12", list(range(12)))
    keep("limb6", [0, 1, 2, 3, 4, 5])
    keep("precordial6", [6, 7, 8, 9, 10, 11])
    keep("anatomical4", [0, 1, 6, 10])
    for index, lead_name in enumerate(LEAD_NAMES):
        keep(f"single_{lead_name}", [index])
    for random_seed in random_seeds:
        rng = np.random.default_rng(random_seed)
        for missing_count in (1, 2, 4, 6):
            missing = rng.choice(12, size=missing_count, replace=False)
            mask = np.ones(12, dtype=bool)
            mask[missing] = False
            masks[f"random_missing_{missing_count}_seed{random_seed}"] = mask
    return masks


def apply_lead_mask(waveform: torch.Tensor, keep_mask: np.ndarray) -> torch.Tensor:
    """Clone a waveform batch and zero channels unavailable under a keep mask."""
    masked = waveform.clone()
    tensor_mask = torch.as_tensor(keep_mask, dtype=torch.bool, device=masked.device)
    masked[:, ~tensor_mask, :] = 0.0
    return masked


@torch.inference_mode()
def predict_masked(
    model: torch.nn.Module,
    loader: DataLoader,
    keep_mask: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict one deterministic lead condition in loader order."""
    probabilities = []
    labels = []
    for waveform, tabular, target in loader:
        waveform = apply_lead_mask(waveform.to(device, non_blocking=True), keep_mask)
        logits = model(waveform, tabular.to(device, non_blocking=True))
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(target.numpy())
    return np.vstack(probabilities), np.vstack(labels)


def condition_group(name: str) -> str:
    """Map a concrete mask to a plot/table family."""
    if name.startswith("random_missing_"):
        return "_".join(name.split("_")[:3])
    if name.startswith("single_"):
        return "single_lead"
    return name


def parse_seed_checkpoint(values: list[str]) -> dict[int, Path]:
    """Parse repeated `seed=checkpoint` arguments."""
    result = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"Expected seed=checkpoint, got {value!r}")
        result[int(seed_text)] = Path(path_text)
    return result


def load_prediction_archive(
    path: Path, expected_conditions: list[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a completed prediction archive only when condition order is exact."""
    archive = np.load(path)
    conditions = archive["condition_names"].astype(str).tolist()
    if conditions != expected_conditions:
        raise ValueError(f"condition mismatch in {path}")
    probabilities = archive["probabilities"]
    if len(probabilities) != len(conditions):
        raise ValueError(f"probability/condition count mismatch in {path}")
    return archive["labels"], {
        condition: probabilities[index]
        for index, condition in enumerate(conditions)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen 12-channel zero-fill lead robustness evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", action="append", required=True, help="seed=checkpoint; repeat for each seed")
    parser.add_argument("--data-dir", type=Path, default=Path("data/echonext"))
    parser.add_argument("--csv-name", default="echonext_metadata_100k.csv")
    parser.add_argument("--test-wave-name", default="EchoNext_test_waveforms.npy")
    parser.add_argument("--test-tab-name", default="EchoNext_test_tabular_features.npy")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-mask-seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = parse_seed_checkpoint(args.checkpoint)
    if sorted(checkpoints) != [42, 2024, 3407]:
        raise ValueError("Formal lead robustness requires seeds 42, 2024, and 3407")
    analysis_dir = args.output_dir / "analysis"
    figures_dir = args.output_dir / "figures"
    failures_dir = args.output_dir / "failures"
    for directory in (analysis_dir, figures_dir, failures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    masks = build_lead_masks(args.random_mask_seeds)
    manifest = {
        name: {
            "keep_indices": np.flatnonzero(mask).astype(int).tolist(),
            "keep_leads": [LEAD_NAMES[index] for index in np.flatnonzero(mask)],
            "missing_count": int(np.sum(~mask)),
            "evaluation_type": "frozen 12-channel checkpoint with unavailable leads zero-filled",
        }
        for name, mask in masks.items()
    }
    (analysis_dir / "lead_mask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (analysis_dir / "analysis_config.yaml").write_text(
        yaml.safe_dump(
            {
                "seeds": sorted(checkpoints),
                "random_mask_seeds": args.random_mask_seeds,
                "bootstrap_iterations": args.bootstrap_iters,
                "bootstrap_seed": args.bootstrap_seed,
                "claim_boundary": "zero-fill stress test; not true reduced-input device equivalence",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    dataset = EchoNextNpyDataset(
        args.data_dir / args.csv_name,
        args.data_dir / args.test_wave_name,
        args.data_dir / args.test_tab_name,
        "test",
        ECHONEXT_LABEL_COLS,
        None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    primary_index = ECHONEXT_LABEL_COLS.index(ECHONEXT_PRIMARY)
    predictions: dict[int, dict[str, np.ndarray]] = {}
    labels_reference = None
    metric_rows = []
    condition_names = list(masks)
    for seed, checkpoint_path in sorted(checkpoints.items()):
        archive_path = analysis_dir / f"predictions_seed_{seed}.npz"
        if archive_path.exists():
            labels, predictions[seed] = load_prediction_archive(
                archive_path, condition_names
            )
            if labels_reference is None:
                labels_reference = labels
            elif not np.array_equal(labels_reference, labels):
                raise RuntimeError("Test labels changed across prediction archives")
            print(f"[lead] seed={seed} resumed {len(condition_names)} conditions")
        else:
            model = load_model(checkpoint_path, device)
            predictions[seed] = {}
            for condition, keep_mask in masks.items():
                probabilities, labels = predict_masked(model, loader, keep_mask, device)
                if labels_reference is None:
                    labels_reference = labels
                elif not np.array_equal(labels_reference, labels):
                    raise RuntimeError("Test labels changed across conditions")
                predictions[seed][condition] = probabilities
                summary = multilabel_summary(labels, probabilities, primary_index)
                print(f"[lead] seed={seed} condition={condition} primary_auroc={summary['primary_auroc']:.4f}")
            np.savez_compressed(
                archive_path,
                labels=labels_reference.astype(np.float32),
                condition_names=np.asarray(condition_names, dtype="U96"),
                probabilities=np.stack([predictions[seed][name] for name in condition_names]).astype(np.float32),
                checkpoint_sha256=np.asarray(sha256_file(checkpoint_path)),
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for condition, keep_mask in masks.items():
            probabilities = predictions[seed][condition]
            summary = multilabel_summary(labels, probabilities, primary_index)
            metric_rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "condition_group": condition_group(condition),
                    "kept_leads": int(np.sum(keep_mask)),
                    "missing_leads": int(np.sum(~keep_mask)),
                    **summary,
                }
            )
    assert labels_reference is not None
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(analysis_dir / "lead_robustness_metrics.csv", index=False)

    clusters = np.arange(len(labels_reference), dtype=np.int64)
    rng = np.random.default_rng(args.bootstrap_seed)
    resamples = [cluster_bootstrap_indices(clusters, rng) for _ in range(args.bootstrap_iters)]
    def evaluate_difference(condition: str, metric: str) -> dict[str, object]:
        label_index = primary_index
        full_points = []
        condition_points = []
        for seed in sorted(checkpoints):
            y_true = labels_reference[:, label_index]
            full_points.append(safe_metric(y_true, predictions[seed]["full12"][:, label_index], metric))
            condition_points.append(safe_metric(y_true, predictions[seed][condition][:, label_index], metric))
        samples = []
        for indices in resamples:
            y_true = labels_reference[indices, label_index]
            seed_differences = []
            for seed in sorted(checkpoints):
                condition_value = safe_metric(y_true, predictions[seed][condition][indices, label_index], metric)
                full_value = safe_metric(y_true, predictions[seed]["full12"][indices, label_index], metric)
                seed_differences.append(condition_value - full_value)
            samples.append(float(np.nanmean(seed_differences)))
        low, high, effective = percentile_interval(samples)
        return {
            "condition": condition,
            "condition_group": condition_group(condition),
            "metric": f"primary_{metric}",
            "point_difference_vs_full12": float(np.mean(np.asarray(condition_points) - np.asarray(full_points))),
            "ci_low": low,
            "ci_high": high,
            "bootstrap_n": effective,
        }

    difference_rows = Parallel(n_jobs=args.n_jobs)(
        delayed(evaluate_difference)(condition, metric)
        for condition in masks
        if condition != "full12"
        for metric in ("auroc", "auprc")
    )
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(analysis_dir / "lead_difference_ci.csv", index=False)

    curve = (
        metrics[metrics.condition.str.startswith("random_missing_")]
        .groupby(["seed", "missing_leads"], as_index=False)["primary_auroc"]
        .mean()
        .groupby("missing_leads")["primary_auroc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    full_mean = float(metrics.loc[metrics.condition == "full12", "primary_auroc"].mean())
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(curve.missing_leads, curve["mean"], yerr=curve["std"], marker="o", color="#0072B2", capsize=4)
    ax.axhline(full_mean, color="black", linestyle="--", linewidth=1, label="Full-12")
    ax.set(xlabel="Number of zero-filled missing leads", ylabel="Primary AUROC", title="Frozen-model lead robustness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "lead_robustness_curve.png", dpi=300)
    fig.savefig(figures_dir / "lead_robustness_curve.pdf")
    plt.close(fig)
    status = {
        "status": "complete",
        "seeds": sorted(checkpoints),
        "conditions": len(masks),
        "bootstrap_iterations": args.bootstrap_iters,
        "claim_boundary": "frozen full-12 zero-fill stress test",
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
