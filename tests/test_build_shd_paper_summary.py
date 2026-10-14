"""Tests for manuscript table aggregation from seed-level metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "build_shd_paper_summary.py"
SPEC = importlib.util.spec_from_file_location("build_shd_paper_summary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_seed_summary_reports_mean_std_and_count() -> None:
    metrics = pd.DataFrame(
        {
            "method": ["ssl", "ssl", "ssl", "scratch", "scratch", "scratch"],
            "seed": [42, 2024, 3407, 42, 2024, 3407],
            "primary_auroc": [0.85, 0.84, 0.86, 0.82, 0.83, 0.81],
        }
    )

    summary = MODULE.summarize_seed_metrics(metrics, group_columns=["method"])

    ssl = summary.loc[summary.method == "ssl"].iloc[0]
    assert ssl["n_seeds"] == 3
    assert ssl["primary_auroc_mean"] == 0.85
    assert round(ssl["primary_auroc_std"], 6) == 0.01

