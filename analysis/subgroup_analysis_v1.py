#!/usr/bin/env python
"""Strict patient-level EchoNext subgroup analysis for Scratch versus FCMAE(ECA)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
for search_path in (ANALYSIS_ROOT, SRC_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from subgroup_utils_v1 import (
    PRIMARY_LABEL,
    align_prediction_bundle,
    assign_age_group,
    benjamini_hochberg,
    bootstrap_two_sided_pvalue,
    ensemble_probabilities,
    normalize_clinical_context,
    normalize_race_ethnicity,
    normalize_sex,
    paired_bootstrap_metrics,
    safe_binary_metric,
)


DISPLAY_NAMES = {"scratch": "Scratch", "ssl": "FCMAE(ECA)", "fcmae": "FCMAE(ECA)"}
CATEGORY_ORDER = ["Age", "Sex", "Race/Ethnicity", "Clinical context"]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz_bundle(path: Path) -> dict[str, np.ndarray]:
    """Load an NPZ prediction bundle without allowing pickled objects."""
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def build_subgroups(metadata: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """Build requested subgroup masks in stable paper-table order."""
    age = assign_age_group(metadata["age_at_ecg"])
    sex = normalize_sex(metadata["sex"])
    race = normalize_race_ethnicity(metadata["race_ethnicity"])
    context = normalize_clinical_context(metadata["location_setting"])
    definitions: list[tuple[str, str, np.ndarray]] = []
    for value in ("18-59", "60-69", "70-79", "80+"):
        definitions.append(("Age", value, age == value))
    for value in ("Female", "Male"):
        definitions.append(("Sex", value, sex == value))
    for value in ("Hispanic", "White", "Black", "Asian", "Other", "Unknown"):
        definitions.append(("Race/Ethnicity", value, race == value))
    context_values = ["Emergency", "Inpatient", "Outpatient", "Procedural"]
    if np.any(context == "Unknown"):
        context_values.append("Unknown")
    for value in context_values:
        definitions.append(("Clinical context", value, context == value))
    return definitions


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    """Return a 95% percentile interval or NaNs for no valid values."""
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(valid, [2.5, 97.5])
    return float(low), float(high)


def format_metric_ci(point: float, low: float, high: float) -> str:
    """Format a point estimate and CI for Markdown/LaTeX tables."""
    if not np.isfinite(point):
        return "NA"
    return f"{point:.3f} ({low:.3f}-{high:.3f})"


def analyze_one_subgroup(
    category: str,
    subgroup: str,
    labels: np.ndarray,
    scratch_probabilities: np.ndarray,
    fcmae_probabilities: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    """Compute ensemble metrics, paired bootstrap CIs, p-values, and raw draws."""
    y_true = np.asarray(labels, dtype=int)
    scratch = np.asarray(scratch_probabilities, dtype=float)
    fcmae = np.asarray(fcmae_probabilities, dtype=float)
    n = int(len(y_true))
    positive_n = int(y_true.sum())
    prevalence = float(y_true.mean()) if n else float("nan")
    point = {
        "scratch_auroc": safe_binary_metric(y_true, scratch, "auroc") if n else float("nan"),
        "scratch_auprc": safe_binary_metric(y_true, scratch, "auprc") if n else float("nan"),
        "fcmae_auroc": safe_binary_metric(y_true, fcmae, "auroc") if n else float("nan"),
        "fcmae_auprc": safe_binary_metric(y_true, fcmae, "auprc") if n else float("nan"),
    }
    if n and np.unique(y_true).size >= 2:
        raw = paired_bootstrap_metrics(y_true, scratch, fcmae, iterations, seed)
    else:
        raw = {name: np.asarray([], dtype=float) for name in (
            "scratch_auroc", "fcmae_auroc", "delta_auroc",
            "scratch_auprc", "fcmae_auprc", "delta_auprc",
        )}
    intervals = {name: percentile_ci(values) for name, values in raw.items()}
    paired_row = {
        "subgroup_category": category,
        "subgroup": subgroup,
        "n": n,
        "positive_n": positive_n,
        "prevalence": prevalence,
        "delta_auroc": point["fcmae_auroc"] - point["scratch_auroc"],
        "delta_auroc_ci_low": intervals["delta_auroc"][0],
        "delta_auroc_ci_high": intervals["delta_auroc"][1],
        "auroc_p": bootstrap_two_sided_pvalue(raw["delta_auroc"]),
        "delta_auprc": point["fcmae_auprc"] - point["scratch_auprc"],
        "delta_auprc_ci_low": intervals["delta_auprc"][0],
        "delta_auprc_ci_high": intervals["delta_auprc"][1],
        "auprc_p": bootstrap_two_sided_pvalue(raw["delta_auprc"]),
        "bootstrap_n": int(len(raw["delta_auroc"])),
        "reason": "single outcome class" if n and np.unique(y_true).size < 2 else ("empty subgroup" if n == 0 else ""),
    }
    return {"point": point, "intervals": intervals, "paired_row": paired_row, "raw": raw}


def latex_escape(value: object) -> str:
    """Escape minimal LaTeX special characters used by subgroup labels."""
    text = str(value)
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def bold_higher(first: str, second: str, first_point: float, second_point: float, latex: bool) -> tuple[str, str]:
    """Bold the higher finite point estimate without implying significance."""
    wrapper = (lambda value: rf"\textbf{{{value}}}") if latex else (lambda value: f"**{value}**")
    if np.isfinite(first_point) and np.isfinite(second_point):
        if first_point > second_point:
            first = wrapper(first)
        elif second_point > first_point:
            second = wrapper(second)
    return first, second


def write_main_tables(wide: pd.DataFrame, output_dir: Path) -> None:
    """Write the requested Markdown and booktabs LaTeX main tables."""
    headers = [
        "Subgroup category", "Subgroup", "n", "Positive n", "Prevalence",
        "Scratch AUROC (95% CI)", "Scratch AUPRC (95% CI)",
        "FCMAE(ECA) AUROC (95% CI)", "FCMAE(ECA) AUPRC (95% CI)",
    ]
    markdown = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    latex = [
        r"\begin{tabular}{llrrrllll}", r"\toprule",
        r"Category & Subgroup & $n$ & Positive & Prev. & Scratch AUROC & Scratch AUPRC & FCMAE AUROC & FCMAE AUPRC \\",
        r"\midrule",
    ]
    previous_category = None
    for _, row in wide.iterrows():
        scratch_auroc = format_metric_ci(row.scratch_auroc, row.scratch_auroc_ci_low, row.scratch_auroc_ci_high)
        fcmae_auroc = format_metric_ci(row.fcmae_auroc, row.fcmae_auroc_ci_low, row.fcmae_auroc_ci_high)
        scratch_auprc = format_metric_ci(row.scratch_auprc, row.scratch_auprc_ci_low, row.scratch_auprc_ci_high)
        fcmae_auprc = format_metric_ci(row.fcmae_auprc, row.fcmae_auprc_ci_low, row.fcmae_auprc_ci_high)
        md_sa, md_fa = bold_higher(scratch_auroc, fcmae_auroc, row.scratch_auroc, row.fcmae_auroc, False)
        md_sp, md_fp = bold_higher(scratch_auprc, fcmae_auprc, row.scratch_auprc, row.fcmae_auprc, False)
        markdown.append(
            f"| {row.subgroup_category} | {row.subgroup} | {int(row.n)} | {int(row.positive_n)} | "
            f"{row.prevalence:.3f} | {md_sa} | {md_sp} | {md_fa} | {md_fp} |"
        )
        if previous_category is not None and row.subgroup_category != previous_category:
            latex.append(r"\midrule")
        tx_sa, tx_fa = bold_higher(scratch_auroc, fcmae_auroc, row.scratch_auroc, row.fcmae_auroc, True)
        tx_sp, tx_fp = bold_higher(scratch_auprc, fcmae_auprc, row.scratch_auprc, row.fcmae_auprc, True)
        latex.append(
            f"{latex_escape(row.subgroup_category)} & {latex_escape(row.subgroup)} & {int(row.n)} & "
            f"{int(row.positive_n)} & {row.prevalence:.3f} & {tx_sa} & {tx_sp} & {tx_fa} & {tx_fp} \\\\"
        )
        previous_category = row.subgroup_category
    note = (
        "95% CIs use 2,000 patient-level bootstrap resamples with paired indices for both models. "
        "AUPRC depends on subgroup prevalence. Age and sex are also included among both matched models' tabular inputs. "
        "Bold indicates the higher point estimate only and does not necessarily indicate statistical significance."
    )
    markdown.extend(["", f"Note: {note}"])
    latex.extend([r"\bottomrule", r"\end{tabular}", "", f"% Note: {note}"])
    (output_dir / "subgroup_table_main.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (output_dir / "subgroup_table_main.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def write_paired_table(paired: pd.DataFrame, output_dir: Path) -> None:
    """Write the strict paired-difference booktabs table."""
    lines = [
        r"\begin{tabular}{llrrllllll}", r"\toprule",
        r"Category & Subgroup & $n$ & Prevalence & $\Delta$ AUROC (95\% CI) & $p$ & $q$ & $\Delta$ AUPRC (95\% CI) & $p$ & $q$ \\",
        r"\midrule",
    ]
    previous_category = None
    for _, row in paired.iterrows():
        if previous_category is not None and row.subgroup_category != previous_category:
            lines.append(r"\midrule")
        lines.append(
            f"{latex_escape(row.subgroup_category)} & {latex_escape(row.subgroup)} & {int(row.n)} & {row.prevalence:.3f} & "
            f"{format_metric_ci(row.delta_auroc, row.delta_auroc_ci_low, row.delta_auroc_ci_high)} & "
            f"{row.auroc_p:.4f} & {row.auroc_q:.4f} & "
            f"{format_metric_ci(row.delta_auprc, row.delta_auprc_ci_low, row.delta_auprc_ci_high)} & "
            f"{row.auprc_p:.4f} & {row.auprc_q:.4f} \\\\"
        )
        previous_category = row.subgroup_category
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "subgroup_table_paired.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_forests(wide: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
    """Create paper-ready model-AUROC and paired-delta forest plots."""
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
    labels = [f"{row.subgroup_category}: {row.subgroup}" for _, row in wide.iterrows()]
    y = np.arange(len(wide))[::-1]
    plot_rows = []
    fig, ax = plt.subplots(figsize=(7.2, max(5.5, 0.38 * len(wide))))
    colors = {"Scratch": "#7A7A7A", "FCMAE(ECA)": "#0072B2"}
    offsets = {"Scratch": -0.11, "FCMAE(ECA)": 0.11}
    for model, prefix in (("Scratch", "scratch"), ("FCMAE(ECA)", "fcmae")):
        point = wide[f"{prefix}_auroc"].to_numpy(float)
        low = wide[f"{prefix}_auroc_ci_low"].to_numpy(float)
        high = wide[f"{prefix}_auroc_ci_high"].to_numpy(float)
        ax.errorbar(point, y + offsets[model], xerr=np.vstack([point - low, high - point]), fmt="o", ms=4,
                    capsize=2, color=colors[model], label=model)
        for index, row in wide.iterrows():
            plot_rows.append({"subgroup_category": row.subgroup_category, "subgroup": row.subgroup,
                              "model": model, "auroc": point[index], "ci_low": low[index], "ci_high": high[index]})
    ax.set_yticks(y, labels)
    ax.set_xlabel("AUROC (95% CI)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "subgroup_auroc_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "subgroup_auroc_forest.pdf", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(plot_rows).to_csv(output_dir / "subgroup_auroc_forest_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, max(5.5, 0.38 * len(paired))))
    point = paired["delta_auroc"].to_numpy(float)
    low = paired["delta_auroc_ci_low"].to_numpy(float)
    high = paired["delta_auroc_ci_high"].to_numpy(float)
    ax.errorbar(point, y, xerr=np.vstack([point - low, high - point]), fmt="o", ms=4,
                capsize=2, color="#D55E00")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("FCMAE(ECA) - Scratch AUROC (95% CI)")
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "subgroup_delta_forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "subgroup_delta_forest.pdf", bbox_inches="tight")
    plt.close(fig)
    paired[["subgroup_category", "subgroup", "delta_auroc", "delta_auroc_ci_low", "delta_auroc_ci_high"]].to_csv(
        output_dir / "subgroup_delta_forest_data.csv", index=False
    )


def write_summary(wide: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
    """Write a cautious, data-derived interpretation of subgroup results."""
    valid = paired[np.isfinite(paired.delta_auroc)].copy()
    positive = int((valid.delta_auroc > 0).sum())
    largest = valid.loc[valid.delta_auroc.idxmax()]
    smallest = valid.loc[valid.delta_auroc.idxmin()]
    valid["ci_width"] = valid.delta_auroc_ci_high - valid.delta_auroc_ci_low
    widest = valid.loc[valid.ci_width.idxmax()]
    significant = valid[(valid.auroc_q < 0.05) | ((valid.delta_auroc_ci_low > 0) | (valid.delta_auroc_ci_high < 0))]
    age = wide[wide.subgroup_category == "Age"]
    lines = [
        "# Subgroup analysis v1 summary",
        "",
        f"FCMAE(ECA) had a positive AUROC difference in {positive}/{len(valid)} evaluated subgroups.",
        f"The largest AUROC difference was {largest.delta_auroc:.4f} in {largest.subgroup_category}/{largest.subgroup}; "
        f"the smallest was {smallest.delta_auroc:.4f} in {smallest.subgroup_category}/{smallest.subgroup}.",
        f"The widest paired AUROC interval occurred in {widest.subgroup_category}/{widest.subgroup} "
        f"(n={int(widest.n)}, positive_n={int(widest.positive_n)}), consistent with greater sampling uncertainty in smaller or outcome-sparse groups.",
        "",
        "## Age patterns",
        "",
        "Age-specific AUROC and AUPRC are descriptive. AUPRC changes must be interpreted alongside prevalence; a higher AUPRC does not by itself prove stronger discrimination.",
    ]
    for _, row in age.iterrows():
        lines.append(
            f"- {row.subgroup}: prevalence={row.prevalence:.3f}, FCMAE AUROC={row.fcmae_auroc:.3f}, FCMAE AUPRC={row.fcmae_auprc:.3f}."
        )
    lines.extend([
        "",
        "## Statistical and clinical interpretation",
        "",
        f"{len(significant)} subgroup AUROC comparisons met either a paired 95% interval excluding zero or AUROC q<0.05. "
        "Comparisons not meeting these criteria are described numerically rather than as significant improvements.",
        "Overlapping subgroup confidence intervals do not establish fairness, and subgroup stability does not establish clinical equivalence.",
        "This is a single-center fixed-test-set analysis and must not be presented as external validation or proof of external generalization.",
        "Age and sex are both subgroup variables and FiLM inputs to the two matched models, so differences are descriptive model-behavior results rather than causal demographic effects.",
    ])
    (output_dir / "subgroup_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_csv", type=Path, required=True)
    parser.add_argument("--label_csv", type=Path, required=True)
    parser.add_argument("--prediction_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260722)
    parser.add_argument("--models", nargs="+", default=["scratch", "ssl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407])
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from ecg_project.data.datasets import ECHONEXT_LABEL_COLS

    metadata_all = pd.read_csv(args.metadata_csv)
    label_all = pd.read_csv(args.label_csv)
    metadata = metadata_all.loc[metadata_all["split"] == "test"].reset_index(drop=True)
    label_frame = label_all.loc[label_all["split"] == "test"].reset_index(drop=True)
    required_metadata = {
        "patient_key", "ecg_key", "age_at_ecg", "sex", "race_ethnicity", "location_setting"
    }
    missing = required_metadata.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata missing required subgroup fields: {sorted(missing)}")
    if not metadata[["patient_key", "ecg_key"]].equals(label_frame[["patient_key", "ecg_key"]]):
        raise ValueError("Metadata and label CSV patient/ECG order differs")
    labels = label_frame[ECHONEXT_LABEL_COLS].to_numpy(dtype=np.float32)
    if np.isnan(labels).any():
        raise ValueError("Test labels contain missing values")

    model_names = list(args.models)
    if "scratch" not in model_names or not ({"ssl", "fcmae"} & set(model_names)):
        raise ValueError("Primary comparison requires models scratch and ssl/fcmae")
    fcmae_key = "ssl" if "ssl" in model_names else "fcmae"
    probabilities: dict[str, dict[int, np.ndarray]] = {model: {} for model in model_names}
    prediction_paths = []
    probability_ranges = {}
    for seed in args.seeds:
        for model in model_names:
            path = args.prediction_root / f"seed_{seed}" / f"{model}_test.npz"
            bundle = load_npz_bundle(path)
            aligned = align_prediction_bundle(metadata, labels, bundle, ECHONEXT_LABEL_COLS, PRIMARY_LABEL)
            probabilities[model][seed] = aligned["primary_probability"].to_numpy(float)
            prediction_paths.append(path)
            probability_ranges[f"{model}_seed_{seed}"] = [
                float(probabilities[model][seed].min()), float(probabilities[model][seed].max())
            ]
    reference_patients = metadata["patient_key"].astype(str)
    integrity_lines = [
        "# Data integrity report",
        "",
        f"- Label CSV: `{args.label_csv}`",
        f"- Metadata CSV: `{args.metadata_csv}`",
        "- Patient ID: `patient_key`",
        "- Record ID: `ecg_key`",
        f"- Test rows: {len(metadata)}",
        f"- Unique patients: {reference_patients.nunique()}",
        f"- Duplicate patient IDs: {int(reference_patients.duplicated().sum())}",
        f"- Missing label rows: {int(label_frame[ECHONEXT_LABEL_COLS].isna().any(axis=1).sum())}",
        f"- Primary label: `{PRIMARY_LABEL}` at output index 11",
        "- Alignment: all prediction rows rebound by audited `row_id`; labels and output order match the CSV.",
        "- Probability validation: all values are finite and within [0,1]; logits were not accepted.",
        f"- Expected EchoNext-Mini test size difference: {len(metadata) - 5442:+d}",
        "",
        "## Subgroup mappings",
        "",
        "- Age: 18-59, 60-69, 70-79, 80+ using left-inclusive/right-exclusive boundaries.",
        "- Sex: source `female`/`male` to Female/Male.",
        "- Race/ethnicity: Hispanic, White, Black, Asian, Other, Unknown; Unknown retained.",
        "- Clinical context: Emergency, Inpatient, Outpatient, Procedural; Unknown only if present.",
        "",
        "## Prediction files",
        "",
    ]
    integrity_lines.extend(f"- `{path}` (SHA-256 `{sha256_file(path)}`)" for path in prediction_paths)
    integrity_lines.extend(["", "## Probability ranges", "", "```json", json.dumps(probability_ranges, indent=2), "```"])
    (args.output_dir / "data_integrity_report.md").write_text("\n".join(integrity_lines) + "\n", encoding="utf-8")
    config_text = (
        f"bootstrap_iters: {args.bootstrap_iters}\nbootstrap_seed: {args.bootstrap_seed}\n"
        f"seeds: [{', '.join(map(str, args.seeds))}]\nmodels: [{', '.join(model_names)}]\n"
        "ci: percentile_95\nbootstrap_unit: patient_id\npaired_indices: true\n"
    )
    (args.output_dir / "analysis_config.yaml").write_text(config_text, encoding="utf-8")
    if args.dry_run:
        (args.output_dir / "status.json").write_text(
            json.dumps({"status": "dry_run_complete", "n_test": len(metadata)}, indent=2), encoding="utf-8"
        )
        print(json.dumps({"status": "dry_run_complete", "n_test": len(metadata)}, indent=2))
        return

    scratch_ensemble = ensemble_probabilities([probabilities["scratch"][seed] for seed in args.seeds])
    fcmae_ensemble = ensemble_probabilities([probabilities[fcmae_key][seed] for seed in args.seeds])
    primary_labels = labels[:, -1].astype(int)
    long_rows = []
    wide_rows = []
    paired_rows = []
    seed_rows = []
    raw_arrays = {}
    for group_index, (category, subgroup, mask) in enumerate(build_subgroups(metadata)):
        result = analyze_one_subgroup(
            category, subgroup, primary_labels[mask], scratch_ensemble[mask], fcmae_ensemble[mask],
            args.bootstrap_iters, args.bootstrap_seed + group_index,
        )
        point = result["point"]
        intervals = result["intervals"]
        paired_rows.append(result["paired_row"])
        slug = re.sub(r"[^a-z0-9]+", "_", f"{category}_{subgroup}".lower()).strip("_")
        for name, values in result["raw"].items():
            raw_arrays[f"{slug}__{name}"] = values
        base = {
            "subgroup_category": category,
            "subgroup": subgroup,
            "n": int(mask.sum()),
            "positive_n": int(primary_labels[mask].sum()),
            "prevalence": float(primary_labels[mask].mean()) if mask.any() else float("nan"),
        }
        wide_row = dict(base)
        for model, prefix in (("Scratch", "scratch"), ("FCMAE(ECA)", "fcmae")):
            for metric in ("auroc", "auprc"):
                key = f"{prefix}_{metric}"
                low, high = intervals[key]
                wide_row[key] = point[key]
                wide_row[f"{key}_ci_low"] = low
                wide_row[f"{key}_ci_high"] = high
                long_rows.append({**base, "model": model, "metric": metric, "value": point[key],
                                  "ci_low": low, "ci_high": high, "bootstrap_n": len(result["raw"][key])})
        wide_rows.append(wide_row)
        for model_key, model_display in (("scratch", "Scratch"), (fcmae_key, "FCMAE(ECA)")):
            per_seed_values = {"auroc": [], "auprc": []}
            for seed in args.seeds:
                probs = probabilities[model_key][seed][mask]
                auroc = safe_binary_metric(primary_labels[mask], probs, "auroc")
                auprc = safe_binary_metric(primary_labels[mask], probs, "auprc")
                per_seed_values["auroc"].append(auroc)
                per_seed_values["auprc"].append(auprc)
                seed_rows.append({**base, "model": model_display, "seed": seed, "auroc": auroc, "auprc": auprc})
            auroc_values = np.asarray(per_seed_values["auroc"], dtype=float)
            auprc_values = np.asarray(per_seed_values["auprc"], dtype=float)
            for row in seed_rows[-len(args.seeds):]:
                row["across_seed_mean_auroc"] = float(np.nanmean(auroc_values))
                row["across_seed_sd_auroc"] = float(np.nanstd(auroc_values, ddof=1))
                row["across_seed_mean_auprc"] = float(np.nanmean(auprc_values))
                row["across_seed_sd_auprc"] = float(np.nanstd(auprc_values, ddof=1))

    long_table = pd.DataFrame(long_rows)
    wide_table = pd.DataFrame(wide_rows)
    paired_table = pd.DataFrame(paired_rows)
    paired_table["auroc_q"] = benjamini_hochberg(paired_table["auroc_p"])
    paired_table["auprc_q"] = benjamini_hochberg(paired_table["auprc_p"])
    paired_table["auroc_significant"] = (
        ((paired_table.delta_auroc_ci_low > 0) | (paired_table.delta_auroc_ci_high < 0))
        | (paired_table.auroc_q < 0.05)
    )
    paired_table["auprc_significant"] = (
        ((paired_table.delta_auprc_ci_low > 0) | (paired_table.delta_auprc_ci_high < 0))
        | (paired_table.auprc_q < 0.05)
    )
    long_table.to_csv(args.output_dir / "subgroup_metrics_long.csv", index=False)
    wide_table.to_csv(args.output_dir / "subgroup_comparison_wide.csv", index=False)
    paired_table.to_csv(args.output_dir / "subgroup_paired_differences.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(args.output_dir / "subgroup_metrics_by_seed.csv", index=False)
    np.savez_compressed(args.output_dir / "subgroup_bootstrap_raw.npz", **raw_arrays)
    write_main_tables(wide_table, args.output_dir)
    write_paired_table(paired_table, args.output_dir)
    plot_forests(wide_table, paired_table, args.output_dir)
    write_summary(wide_table, paired_table, args.output_dir)
    status = {
        "status": "complete",
        "n_test": len(metadata),
        "unique_patients": int(reference_patients.nunique()),
        "subgroups": len(wide_table),
        "bootstrap_iterations": args.bootstrap_iters,
        "seeds": args.seeds,
        "models": model_names,
    }
    (args.output_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
