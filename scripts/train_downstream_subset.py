#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the standard downstream trainer with a deterministic train-label subset.

This wrapper deliberately leaves ``scripts/train_downstream.py`` unchanged.
It adds only two arguments:

  * ``--train-fraction``: fraction of the original train split to keep.
  * ``--train-subset-seed``: RNG seed used to sample train indices.

Validation and test splits are never subsampled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_base_trainer():
    """Import the existing downstream trainer as a module."""
    trainer_path = PROJECT_ROOT / "scripts" / "train_downstream.py"
    spec = importlib.util.spec_from_file_location("train_downstream_base", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base trainer from {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_subset_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of the original train split to keep; validation/test stay unchanged.",
    )
    parser.add_argument(
        "--train-subset-seed",
        type=int,
        default=42,
        help="RNG seed for deterministic train subset sampling.",
    )
    parser.add_argument(
        "--subset-permutation-file",
        type=Path,
        default=None,
        help=(
            "Shared fixed train permutation (.npy). Fractions are prefixes of this "
            "permutation, so all low-label subsets are genuinely nested."
        ),
    )
    return parser.parse_known_args(argv)


def load_or_create_permutation(path: Path, n_total: int, seed: int) -> np.ndarray:
    """Load one fixed permutation or create it atomically for nested subsets."""
    path = Path(path)
    if path.exists():
        permutation = np.asarray(np.load(path), dtype=np.int64)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        permutation = np.random.default_rng(seed).permutation(n_total).astype(np.int64)
        np.save(path, permutation)
    if (
        permutation.ndim != 1
        or len(permutation) != n_total
        or not np.array_equal(np.sort(permutation), np.arange(n_total, dtype=np.int64))
    ):
        raise ValueError(
            f"Invalid train permutation in {path}: expected each index in [0, {n_total}) once."
        )
    return permutation


def yaml_serializable_optional_path(path: Path | str | None) -> str | None:
    """Normalize an argparse path before the shared YAML experiment archive."""
    return None if path is None else str(path)


def prefix_subset_indices(permutation: np.ndarray, fraction: float) -> np.ndarray:
    """Return sorted indices from a fraction-sized prefix of a fixed permutation."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError("--train-fraction must be in (0, 1].")
    n_total = len(permutation)
    n_keep = max(1, int(round(n_total * fraction)))
    return np.sort(np.asarray(permutation[:n_keep], dtype=np.int64))


def subset_train_set(
    train_set,
    fraction: float,
    seed: int,
    output_dir: Path,
    permutation_file: Path | None = None,
):
    """Return a nested deterministic subset and persist its complete audit trail."""
    n_total = len(train_set)
    output_dir.mkdir(parents=True, exist_ok=True)
    permutation_path = permutation_file or output_dir / "train_permutation.npy"
    permutation = load_or_create_permutation(permutation_path, n_total, seed)
    indices = prefix_subset_indices(permutation, fraction)
    indices_path = output_dir / f"train_subset_fraction_{fraction:g}_seed_{seed}.npy"
    np.save(indices_path, indices)
    permutation_sha256 = hashlib.sha256(Path(permutation_path).read_bytes()).hexdigest()
    indices_sha256 = hashlib.sha256(indices_path.read_bytes()).hexdigest()
    manifest = {
        "fraction": float(fraction),
        "n_total": int(n_total),
        "n_selected": int(len(indices)),
        "permutation_seed": int(seed),
        "permutation_file": str(Path(permutation_path).resolve()),
        "permutation_sha256": permutation_sha256,
        "indices_file": str(indices_path.resolve()),
        "indices_sha256": indices_sha256,
        "selection_rule": "sorted prefix of one fixed permutation",
    }
    (output_dir / "train_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if fraction >= 1.0:
        return train_set, indices
    return Subset(train_set, indices.tolist()), indices


def main() -> None:
    custom_args, remaining = parse_subset_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    base = load_base_trainer()
    args = base.parse_args()
    args.train_fraction = custom_args.train_fraction
    args.train_subset_seed = custom_args.train_subset_seed
    args.subset_permutation_file = yaml_serializable_optional_path(
        custom_args.subset_permutation_file
    )

    if not args.experiment_name:
        note = args.des or args.model
        args.experiment_name = f"subset_{args.dataset}_{note}"
    if not args.des:
        args.des = args.model

    experiment_dir = base.prepare_experiment_directory(
        project_root=base.PROJECT_ROOT,
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        args=args,
        source_config=args.config,
    )

    results_path = Path(args.results_csv)
    if not results_path.is_absolute():
        results_path = experiment_dir / results_path
    args.results_csv = str(results_path)

    configs = base.build_config_list(args)
    print(
        f"device={base.device} | dataset={args.dataset} | train_fraction={args.train_fraction} "
        f"| subset_seed={args.train_subset_seed} | seeds={args.seeds}"
    )
    print(f"experiment_dir={experiment_dir}")
    print(f"configs={[(c['name'], c['loss']) for c in configs]}")

    if not args.skip_train:
        ctx = base.build_data_context(args)
        subset, indices = subset_train_set(
            ctx["train_set"],
            args.train_fraction,
            args.train_subset_seed,
            Path(experiment_dir),
            args.subset_permutation_file,
        )
        ctx["train_set"] = subset
        print(f"[subset] original_train={len(indices) if args.train_fraction >= 1 else 'see below'}")
        print(f"[subset] effective_train={len(ctx['train_set'])}")
        ctx.update(
            {
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "deterministic": args.deterministic,
                "dataset": args.dataset,
                "des": args.des,
                "augmentations": args.augmentations,
                "use_ema": args.ema,
                "ema_decay": args.ema_decay,
                "use_swa": args.swa,
                "swa_start_frac": args.swa_start_frac,
                "patience": args.patience,
                "grad_clip": args.grad_clip,
                "base_lr": args.base_lr,
                "weight_decay": args.weight_decay,
                "layer_decay": args.layer_decay,
                "force_llrd": args.llrd,
                "shd_consistency_weight": getattr(args, "shd_consistency_weight", 0.0),
                "pretrained_encoder": args.pretrained,
                "experiment_dir": str(experiment_dir),
                "setting_fn": base.make_setting_fn(args),
            }
        )
        base.run_experiments(ctx, args.seeds, args.epochs, args.results_csv, configs)
    else:
        print("[INFO] --skip-train: only analyzing existing metrics.")

    if Path(args.results_csv).exists():
        baseline_name = (
            f"convnext_{args.des}"
            if args.architecture_ablation and args.des
            else (
                "convnext"
                if args.architecture_ablation
                else (f"{base.BASELINE_NAME}_{args.des}" if (args.ablation and args.des) else base.BASELINE_NAME)
            )
        )
        agg = base.analyze(args.results_csv, baseline=baseline_name)
        base.plot_chart(
            agg,
            baseline_msr=args.baseline_msr,
            output_stem=str(Path(experiment_dir) / "ablation_results"),
            show=args.show_plot,
        )
    else:
        print(f"[WARN] Missing metrics file: {args.results_csv}")


if __name__ == "__main__":
    main()
