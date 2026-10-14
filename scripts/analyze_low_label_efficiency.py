#!/usr/bin/env python
"""Aggregate auditable, truly nested low-label SSL and scratch experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SEEDS = (42, 2024, 3407)
FRACTION_TAGS = {0.01: "01", 0.05: "05", 0.10: "10", 0.30: "30"}
METRIC_COLUMNS = (
    "Test_Primary_AUROC",
    "Test_Primary_AUPRC",
    "Test_Macro_AUROC",
)


def sha256(path: Path) -> str:
    """Return a file SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_index_file(directory: Path) -> Path:
    """Find the sole archived subset-index file without trusting remote paths."""
    candidates = sorted(directory.glob("train_subset_fraction_*.npy"))
    if not candidates:
        candidates = sorted(directory.glob("subset_*.npy"))
    if len(candidates) != 1:
        raise ValueError(f"expected one subset index file in {directory}, found {len(candidates)}")
    return candidates[0]


def validate_nested_subsets(
    records: list[dict[str, object]],
    n_total: int,
) -> pd.DataFrame:
    """Validate paired identity and set nesting, then describe a full 100% set."""
    rows = []
    indices_by_fraction: dict[float, np.ndarray] = {}
    for record in records:
        fraction = float(record["fraction"])
        method = str(record["method"])
        directory = Path(record["directory"])
        manifest = json.loads((directory / "train_subset_manifest.json").read_text(encoding="utf-8"))
        index_path = local_index_file(directory)
        indices = np.load(index_path).astype(np.int64)
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"duplicate indices in {index_path}")
        if np.any(indices < 0) or np.any(indices >= n_total):
            raise ValueError(f"out-of-range indices in {index_path}")
        actual_digest = sha256(index_path)
        expected_digest = manifest.get("indices_sha256")
        if expected_digest and expected_digest != actual_digest:
            raise ValueError(f"index SHA-256 mismatch in {directory}")
        previous = indices_by_fraction.get(fraction)
        if previous is not None and not np.array_equal(np.sort(previous), np.sort(indices)):
            raise ValueError(f"SSL and scratch do not share identical indices at fraction {fraction}")
        indices_by_fraction[fraction] = indices
        rows.append(
            {
                "fraction": fraction,
                "method": method,
                "n_total": n_total,
                "n_selected": int(len(indices)),
                "indices_sha256": actual_digest,
                "permutation_sha256": manifest.get("permutation_sha256"),
                "source_directory": str(directory),
                "nested_with_next": True,
            }
        )
    ordered = sorted(indices_by_fraction)
    for smaller, larger in zip(ordered, ordered[1:]):
        if not set(indices_by_fraction[smaller]).issubset(set(indices_by_fraction[larger])):
            raise ValueError(f"fraction {smaller} is not nested within {larger}")
    full_indices = np.arange(n_total, dtype=np.int64)
    rows.append(
        {
            "fraction": 1.0,
            "method": "shared_full_set",
            "n_total": n_total,
            "n_selected": n_total,
            "indices_sha256": hashlib.sha256(full_indices.tobytes()).hexdigest(),
            "permutation_sha256": rows[0].get("permutation_sha256") if rows else None,
            "source_directory": "generated complete index set [0, n_total)",
            "nested_with_next": True,
        }
    )
    return pd.DataFrame(rows)


def read_metrics(path: Path, method: str, fraction: float) -> pd.DataFrame:
    """Read one metrics file and attach design fields without altering values."""
    frame = pd.read_csv(path)
    required = {"Seed", *METRIC_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    frame = frame[frame["Seed"].isin(SEEDS)].copy()
    frame.insert(0, "fraction", fraction)
    frame.insert(1, "method", method)
    frame["source_file"] = str(path)
    return frame


def full_fraction_metrics(experiments_root: Path) -> pd.DataFrame:
    """Collect the strict matched 100% SSL and scratch three-seed results."""
    ssl_seed42 = read_metrics(
        experiments_root / "20260630_eca_drop_path_010/metrics.csv", "ssl", 1.0
    )
    ssl_other = read_metrics(
        experiments_root / "20260707_eca_drop_path_010_multiseed/metrics.csv", "ssl", 1.0
    )
    scratch = read_metrics(
        experiments_root / "20260708_best_recipe_no_pretrain_3seed/metrics.csv",
        "scratch",
        1.0,
    )
    return pd.concat(
        [ssl_seed42[ssl_seed42.Seed == 42], ssl_other[ssl_other.Seed.isin((2024, 3407))], scratch],
        ignore_index=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/20260715_low_label_nested"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    experiments_root = project_root / "experiments"
    output_root = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    analysis_dir = output_root / "analysis"
    figures_dir = output_root / "figures"
    failures_dir = output_root / "failures"
    splits_dir = output_root / "splits"
    for directory in (analysis_dir, figures_dir, failures_dir, splits_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metric_frames = []
    subset_records = []
    for fraction, tag in FRACTION_TAGS.items():
        for method in ("ssl", "scratch"):
            directory = experiments_root / f"20260715_low_label_nested_{method}_{tag}pct"
            metric_frames.append(read_metrics(directory / "metrics.csv", method, fraction))
            subset_records.append({"fraction": fraction, "method": method, "directory": directory})
    metric_frames.append(full_fraction_metrics(experiments_root))
    metrics = pd.concat(metric_frames, ignore_index=True)
    counts = metrics.groupby(["fraction", "method"])["Seed"].nunique()
    if not (counts == len(SEEDS)).all():
        raise ValueError(f"every fraction/method requires seeds {SEEDS}; observed {counts.to_dict()}")
    metrics.sort_values(["fraction", "method", "Seed"]).to_csv(
        analysis_dir / "seed_metrics.csv", index=False
    )

    n_total = int(
        json.loads(
            (Path(subset_records[0]["directory"]) / "train_subset_manifest.json").read_text(encoding="utf-8")
        )["n_total"]
    )
    subset_manifest = validate_nested_subsets(subset_records, n_total)
    subset_manifest.to_csv(analysis_dir / "subset_manifest.csv", index=False)
    np.save(splits_dir / "train_subset_fraction_1.0_seed_20260715.npy", np.arange(n_total, dtype=np.int64))

    summary = metrics.groupby(["fraction", "method"], as_index=False).agg(
        n_seeds=("Seed", "nunique"),
        primary_auroc_mean=("Test_Primary_AUROC", "mean"),
        primary_auroc_std=("Test_Primary_AUROC", "std"),
        primary_auprc_mean=("Test_Primary_AUPRC", "mean"),
        primary_auprc_std=("Test_Primary_AUPRC", "std"),
        macro_auroc_mean=("Test_Macro_AUROC", "mean"),
        macro_auroc_std=("Test_Macro_AUROC", "std"),
    )
    summary.to_csv(analysis_dir / "label_efficiency_summary.csv", index=False)
    paired = metrics.pivot(index=["fraction", "Seed"], columns="method", values=list(METRIC_COLUMNS))
    paired.columns = [f"{metric}_{method}" for metric, method in paired.columns]
    paired = paired.reset_index()
    for metric in METRIC_COLUMNS:
        paired[f"{metric}_ssl_minus_scratch"] = paired[f"{metric}_ssl"] - paired[f"{metric}_scratch"]
    paired.to_csv(analysis_dir / "paired_differences.csv", index=False)

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for method, color in (("ssl", "#0072B2"), ("scratch", "#D55E00")):
        group = summary[summary.method == method].sort_values("fraction")
        ax.errorbar(
            group.fraction * 100,
            group.primary_auroc_mean,
            yerr=group.primary_auroc_std,
            marker="o",
            capsize=4,
            label=method.upper(),
            color=color,
        )
    ax.set(xlabel="Labeled training data (%)", ylabel="Primary AUROC", title="Nested label-efficiency analysis")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 30, 100], ["1", "5", "10", "30", "100"])
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "label_efficiency_curve.png", dpi=300)
    fig.savefig(figures_dir / "label_efficiency_curve.pdf")
    plt.close(fig)

    config = {
        "fractions": list(FRACTION_TAGS) + [1.0],
        "seeds": list(SEEDS),
        "paired_ssl_scratch": True,
        "shared_nested_indices": True,
        "test_used_for_selection": False,
    }
    (analysis_dir / "analysis_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    status = {"status": "complete", "seed_rows": int(len(metrics)), "nested_validation": "passed"}
    (analysis_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    hash_lines = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "sha256sums.txt":
            hash_lines.append(f"{sha256(path)}  {path.relative_to(output_root).as_posix()}")
    (analysis_dir / "sha256sums.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
