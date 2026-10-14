#!/usr/bin/env python
"""Build centralized manuscript tables from auditable ECG-SHD experiment files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_ALIASES = {
    "seed": ("seed", "Seed"),
    "primary_auroc": ("primary_auroc", "Test_Primary_AUROC"),
    "primary_auprc": ("primary_auprc", "Test_Primary_AUPRC", "Test_AUPRC"),
    "macro_auroc": ("macro_auroc", "Test_Macro_AUROC", "Macro_AUROC"),
    "macro_auprc": ("macro_auprc", "Test_Macro_AUPRC", "Macro_AUPRC"),
}


def canonical_seed_metrics(frame: pd.DataFrame, experiment: str, source: Path) -> pd.DataFrame:
    """Normalize heterogeneous historical metrics columns without changing values."""
    result = pd.DataFrame(index=frame.index)
    result["experiment"] = experiment
    result["source_file"] = str(source)
    for canonical, aliases in METRIC_ALIASES.items():
        result[canonical] = np.nan
        for alias in aliases:
            if alias in frame.columns:
                result[canonical] = frame[alias]
                break
    for column in ("Config", "Model", "Des", "Loss", "Dataset", "Time(min)"):
        if column in frame.columns:
            result[column.lower().replace("(", "_").replace(")", "")] = frame[column]
    return result


def summarize_seed_metrics(
    metrics: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    """Aggregate seed-level numeric metrics into mean, sample SD, and seed count."""
    metric_columns = [
        column
        for column in ("primary_auroc", "primary_auprc", "macro_auroc", "macro_auprc")
        if column in metrics.columns
    ]
    grouped = metrics.groupby(group_columns, dropna=False)
    rows = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else len(group)
        for metric in metric_columns:
            finite = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else float("nan")
            row[f"{metric}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def copy_if_exists(source: Path, destination: Path) -> bool:
    """Copy one compact table if present and return its availability."""
    if not source.exists():
        return False
    destination.write_bytes(source.read_bytes())
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Centralize ECG-SHD manuscript result tables",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/20260715_shd_paper_complete_summary"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(exist_ok=True)
    experiments_root = project_root / "experiments"
    metric_frames = []
    for metrics_path in sorted(experiments_root.glob("*/metrics.csv")):
        try:
            frame = pd.read_csv(metrics_path)
            if len(frame):
                metric_frames.append(canonical_seed_metrics(frame, metrics_path.parent.name, metrics_path))
        except Exception as error:
            (failures_dir / f"read_{metrics_path.parent.name}.json").write_text(
                json.dumps({"status": "failed", "source": str(metrics_path), "error": str(error)}, indent=2),
                encoding="utf-8",
            )
    all_metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    all_metrics.to_csv(output_dir / "all_experiment_seed_metrics.csv", index=False)

    main_experiments = {
        "20260623_fcmae_convnext_baseline": "FCMAE + ConvNeXt",
        "20260623_fcmae_convnext_eca": "FCMAE(ECA) + ConvNeXt(ECA)",
        "20260630_stmem_base_official_echonext_seed42": "ST-MEM official encoder",
        "20260715_stmem_reproduction_seed2024": "ST-MEM official encoder",
        "20260715_stmem_reproduction_seed3407": "ST-MEM official encoder",
        "20260630_eca_drop_path_010": "Best recipe seed 42",
        "20260707_eca_drop_path_010_multiseed": "Best recipe seeds 2024/3407",
        "20260708_best_recipe_no_pretrain_3seed": "Matched scratch",
        "20260708_anatomical_group_fcmae_eca_downstream_3seed": "Anatomical-group FCMAE",
    }
    main_rows = all_metrics[all_metrics.experiment.isin(main_experiments)].copy() if len(all_metrics) else pd.DataFrame()
    if len(main_rows):
        main_rows["method"] = main_rows.experiment.map(main_experiments)
        # Merge the seed-42 and seed-2024/3407 directories into one best-recipe row.
        main_rows.loc[main_rows.experiment.isin(["20260630_eca_drop_path_010", "20260707_eca_drop_path_010_multiseed"]), "method"] = "FCMAE(ECA) + ECA/FiLM/drop-path 0.10"
        summarize_seed_metrics(main_rows, ["method"]).to_csv(output_dir / "table_main_methods.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "table_main_methods.csv", index=False)

    paired_analysis = experiments_root / "20260715_fcmae_eca_vs_scratch_paired" / "analysis"
    paired_seed_path = paired_analysis / "seed_metrics.csv"
    if paired_seed_path.exists():
        paired_long = pd.read_csv(paired_seed_path)
        paired_wide = paired_long.pivot(index="seed", columns="metric", values=["ssl", "scratch", "difference"])
        paired_wide.columns = [f"{value}_{metric}" for value, metric in paired_wide.columns]
        paired_wide.reset_index().to_csv(output_dir / "table_ssl_vs_scratch.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "table_ssl_vs_scratch.csv", index=False)
    copy_if_exists(
        paired_analysis / "paired_difference_ci.csv",
        output_dir / "table_ssl_vs_scratch_paired_ci.csv",
    )
    copy_if_exists(
        paired_analysis / "labelwise_metrics_ci.csv",
        output_dir / "table_per_label_baseline_vs_best.csv",
    )
    copy_if_exists(
        paired_analysis / "labelwise_paired_tests_bh_fdr.csv",
        output_dir / "table_per_label_paired_tests_bh_fdr.csv",
    )

    formal_low_label = experiments_root / "20260715_low_label_nested" / "analysis" / "label_efficiency_summary.csv"
    low_label = all_metrics[all_metrics.experiment.str.startswith("20260715_low_label_nested_")].copy() if len(all_metrics) else pd.DataFrame()
    if formal_low_label.exists():
        pd.read_csv(formal_low_label).to_csv(output_dir / "table_low_label.csv", index=False)
    elif len(low_label):
        extracted = low_label.experiment.str.extract(r"20260715_low_label_nested_(ssl|scratch)_(\d+)pct")
        low_label["method"] = extracted[0]
        low_label["fraction_percent"] = pd.to_numeric(extracted[1])
        summarize_seed_metrics(low_label, ["fraction_percent", "method"]).sort_values(["fraction_percent", "method"]).to_csv(
            output_dir / "table_low_label.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "table_low_label.csv", index=False)

    modality_names = {
        "20260708_ecg_only_pretrained_3seed": "ECG-only",
        "20260708_tabular_only_3seed": "Tabular-only",
        "20260630_eca_drop_path_010": "ECG+tabular FiLM",
        "20260707_eca_drop_path_010_multiseed": "ECG+tabular FiLM",
    }
    modality = all_metrics[all_metrics.experiment.isin(modality_names)].copy() if len(all_metrics) else pd.DataFrame()
    if len(modality):
        modality["modality"] = modality.experiment.map(modality_names)
        summarize_seed_metrics(modality, ["modality"]).to_csv(output_dir / "table_modality.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "table_modality.csv", index=False)

    screening_available = copy_if_exists(paired_analysis / "screening_metrics.csv", output_dir / "table_screening_calibration.csv")
    if screening_available and (paired_analysis / "calibration_metrics.json").exists():
        screening = pd.read_csv(output_dir / "table_screening_calibration.csv")
        calibration = pd.read_json(paired_analysis / "calibration_metrics.json")
        screening.merge(calibration, on="seed", how="left").to_csv(output_dir / "table_screening_calibration.csv", index=False)
    subgroup_available = copy_if_exists(paired_analysis / "subgroup_metrics.csv", output_dir / "table_subgroups.csv")

    lead_analysis = experiments_root / "20260715_reduced_lead_robustness" / "analysis"
    lead_available = copy_if_exists(lead_analysis / "lead_robustness_metrics.csv", output_dir / "table_lead_robustness.csv")
    true_reduced_names = {
        "20260715_true_reduced_limb6_3seed": "Limb-6 true input",
        "20260715_true_reduced_precordial6_3seed": "Precordial-6 true input",
        "20260715_true_reduced_anatomical4_3seed": "Anatomical-4 true input",
    }
    true_reduced = all_metrics[all_metrics.experiment.isin(true_reduced_names)].copy() if len(all_metrics) else pd.DataFrame()
    if len(true_reduced):
        true_reduced["lead_setting"] = true_reduced.experiment.map(true_reduced_names)
        summarize_seed_metrics(true_reduced, ["lead_setting"]).to_csv(
            output_dir / "table_true_reduced_leads.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "table_true_reduced_leads.csv", index=False)
    attribution_analysis = experiments_root / "20260715_attribution_stability" / "analysis"
    attribution_available = False
    if (attribution_analysis / "cross_seed_stability.csv").exists():
        stability = pd.read_csv(attribution_analysis / "cross_seed_stability.csv")
        stability.groupby(["label_index", "label"], as_index=False).agg(
            lead_spearman_mean=("lead_spearman", "mean"),
            lead_spearman_std=("lead_spearman", "std"),
            time_pearson_mean=("time_pearson", "mean"),
            time_pearson_std=("time_pearson", "std"),
        ).to_csv(output_dir / "table_attribution_stability.csv", index=False)
        attribution_available = True
    else:
        pd.DataFrame().to_csv(output_dir / "table_attribution_stability.csv", index=False)

    stmem_experiments = {
        "20260630_stmem_base_official_echonext_seed42",
        "20260715_stmem_reproduction_seed2024",
        "20260715_stmem_reproduction_seed3407",
    }
    stmem = all_metrics[all_metrics.experiment.isin(stmem_experiments)].copy() if len(all_metrics) else pd.DataFrame()
    if len(stmem):
        stmem["method"] = "ST-MEM official encoder"
        summarize_seed_metrics(stmem, ["method"]).to_csv(output_dir / "table_stmem.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "table_stmem.csv", index=False)

    external_family = experiments_root / "20260715_external_echo_validation"
    for directory in (external_family / "analysis", external_family / "figures", external_family / "failures"):
        directory.mkdir(parents=True, exist_ok=True)
    external_failure = {
        "status": "blocked",
        "reason": "No authorized independent cohort with equivalent echo-confirmed 11-subtype/aggregate SHD endpoints and mapping window is available.",
        "ptbxl_used_as_external_shd_validation": False,
        "ptbxl_boundary": "representation transfer/probing only",
        "fabricated_or_substituted_metrics": False,
    }
    (external_family / "failures" / "external_echo_validation_status.json").write_text(
        json.dumps(external_failure, indent=2), encoding="utf-8"
    )
    (external_family / "analysis" / "status.json").write_text(json.dumps(external_failure, indent=2), encoding="utf-8")

    status = {
        "status": "complete_with_blocked_items" if screening_available and subgroup_available and lead_available else "partial",
        "seed_metric_rows": int(len(all_metrics)),
        "tables": {
            "screening_calibration": screening_available,
            "subgroups": subgroup_available,
            "lead_robustness": lead_available,
            "true_reduced_leads": bool(len(true_reduced)),
            "attribution_stability": attribution_available,
            "stmem": bool(len(stmem)),
        },
        "blocked": ["external_echo_validation"] + ([] if attribution_available else ["attribution_stability_pending"]),
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    readme = """# 20260715 ECG-SHD paper result summary

All tables in this directory are generated from experiment `metrics.csv` files or formal prediction-level analyses. Empty tables indicate a pending upstream queue, not a zero result. The external-validation family is explicitly blocked because no authorized independent cohort with equivalent echo-confirmed SHD endpoints is available; PTB-XL is not substituted for this endpoint.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    hash_lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256sums.txt":
            hash_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "sha256sums.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
