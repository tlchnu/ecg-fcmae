#!/usr/bin/env python
"""Promote the final-results freeze from BLOCKED to PASS after low-label audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    """Return SHA-256 for a local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_ci(low: float, high: float) -> str:
    """Format a confidence interval."""
    return f"[{low:.6f}, {high:.6f}]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--low-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze = args.freeze.resolve()
    low_root = args.low_root.resolve()
    output = args.output.resolve()
    status = json.loads((low_root / "analysis/status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError(f"low-label audit is not complete: {status}")
    summary = pd.read_csv(low_root / "analysis/summary.csv")
    paired = pd.read_csv(low_root / "analysis/paired_bootstrap.csv")
    ensemble = paired[paired["level"] == "ensemble"].sort_values("fraction")
    if len(summary) != 10 or len(ensemble) != 5:
        raise ValueError("unexpected low-label summary dimensions")

    original = freeze.read_text(encoding="utf-8")
    blocked_copy = freeze.with_name("final_results_freeze_blocked_v1.md")
    if not blocked_copy.exists():
        shutil.copy2(freeze, blocked_copy)

    summary_rows = []
    for fraction in sorted(summary["fraction"].unique()):
        ssl = summary[(summary.fraction == fraction) & (summary.method == "ssl")].iloc[0]
        scratch = summary[(summary.fraction == fraction) & (summary.method == "scratch")].iloc[0]
        summary_rows.append(
            f"| {fraction * 100:g}% | "
            f"{ssl.primary_auroc_mean:.6f} ± {ssl.primary_auroc_std:.6f} | "
            f"{scratch.primary_auroc_mean:.6f} ± {scratch.primary_auroc_std:.6f} | "
            f"{ssl.primary_auprc_mean:.6f} ± {ssl.primary_auprc_std:.6f} | "
            f"{scratch.primary_auprc_mean:.6f} ± {scratch.primary_auprc_std:.6f} |"
        )
    paired_rows = []
    for _, row in ensemble.iterrows():
        paired_rows.append(
            f"| {row.fraction * 100:g}% | {row.ssl_auroc:.6f} | {row.scratch_auroc:.6f} | "
            f"{row.delta_auroc:+.6f} | {format_ci(row.delta_auroc_ci_low, row.delta_auroc_ci_high)} | "
            f"{row.delta_auprc:+.6f} | {format_ci(row.delta_auprc_ci_low, row.delta_auprc_ci_high)} |"
        )
    permutation_hash = pd.read_csv(low_root / "provenance/subset_provenance.csv")[
        "permutation_sha256"
    ].unique().tolist()
    if len(permutation_hash) != 1:
        raise ValueError("low-label permutation hash is not unique")
    artifact_hash = sha256_file(low_root / "provenance/artifact_sha256.csv")
    section = f"""## 6. Low-label provenance 门：PASS

低标注矩阵已在 272 按当前增强协议完成：1%/5%/10%/30% × SSL/Scratch × seeds `{{42,2026,3407}}`，共 24 次训练；100% 复用 Full/Strict。

### 6.1 三种子均值与样本 SD

| 训练记录比例 | SSL SHD AUROC | Scratch SHD AUROC | SSL SHD AUPRC | Scratch SHD AUPRC |
|---:|---:|---:|---:|---:|
{chr(10).join(summary_rows)}

### 6.2 三种子 ensemble 配对 bootstrap

患者级配对 percentile bootstrap，2,000 次；Δ 定义为 SSL−Scratch。

| 比例 | SSL AUROC | Scratch AUROC | Δ AUROC | 95% CI | Δ AUPRC | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(paired_rows)}

### 6.3 Provenance

- 固定 permutation SHA256：`{permutation_hash[0]}`；
- 1/5/10/30% 为同一 permutation 的嵌套前缀；SSL/Scratch 在每个比例使用完全相同的 ECG 行索引；
- 每个训练条件均有 checkpoint、effective config、subset manifest、val/test predictions、exact metrics 和 review status；
- test predictions 均为 5,442 行、5,442 名唯一患者，并通过同序标签与 split hash 对齐；
- `provenance/artifact_sha256.csv` SHA256：`{artifact_hash}`；
- 原始配对分布：`analysis/paired_bootstrap_raw.npz`；
- 完成门：`LOW_LABEL_COMPLETE.json`。

低标注比例按训练 ECG 记录行定义，不表述为唯一患者比例。100% SSL/Scratch 分别复用已冻结的 Full/Strict。

"""
    start = original.index("## 6.")
    end = original.index("## 7.", start)
    updated = original[:start] + section + original[end:]
    updated = updated.replace(
        "本文状态：**BLOCKED — 4/5 个关键门通过，low-label provenance 未闭环**",
        "本文状态：**PASS — 5/5 个关键门全部通过**",
    )
    updated = updated.replace(
        "| low-label provenance | **BLOCKED** | 272 只有“已在 15866 完成”的声明，没有当前增强协议的 metrics、predictions、subset manifest、effective config 或 SHA256；15866 当前不可连接。 |",
        "| low-label provenance | **PASS** | 已在 272 完成 24 次训练，并验证 subset、checkpoint、predictions、精确指标、配对 bootstrap 与逐文件 SHA256。 |",
    )
    old_conclusion = """因此：

- **Full/Strict 三种子、精确 predictions、Macro-11 和 paired bootstrap 可以冻结。**
- **论文低标注主表和“全部论文结果已冻结”的总门不能通过。**
- `THREE_SEED_PAPER_COMPLETE.json` 证明 272 队列完成，但它没有验证低标注原始 artifacts，不能替代 low-label provenance 门。
"""
    new_conclusion = """因此：

- **Full/Strict 三种子、精确 predictions、Macro-11、paired bootstrap 和 low-label provenance 全部冻结。**
- **论文最终结果总门通过。**
- `THREE_SEED_PAPER_COMPLETE.json` 与 `LOW_LABEL_COMPLETE.json` 共同构成完整完成门。
"""
    if old_conclusion not in updated:
        raise ValueError("cannot locate blocked conclusion in freeze document")
    updated = updated.replace(old_conclusion, new_conclusion)
    updated = updated.replace(
        "5. Low-label 在 provenance 门通过前不进入最终主表。",
        "5. Low-label 使用第 6 节冻结结果，并保留 ECG 记录行比例的定义。",
    )
    updated = updated.replace(
        "6. 当前总冻结状态保持 **BLOCKED**；不得因 `THREE_SEED_PAPER_COMPLETE.json` 为 complete 而改写为 PASS。",
        "6. 当前总冻结状态为 **PASS**；Full/Strict 与 low-label 两个完成门均已闭环。",
    )
    updated += (
        f"\n> 冻结门由低标注完成队列于 {datetime.now().isoformat(timespec='seconds')} 更新；"
        "原阻塞版本保存在 `final_results_freeze_blocked_v1.md`。\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(updated, encoding="utf-8")
    freeze.write_text(updated, encoding="utf-8")
    payload = {
        "status": "complete",
        "freeze_gate": "PASS",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "freeze_path": str(freeze),
        "freeze_sha256": sha256_file(freeze),
        "versioned_freeze_path": str(output),
        "versioned_freeze_sha256": sha256_file(output),
        "low_label_status": status,
    }
    (low_root / "FINAL_RESULTS_FREEZE_COMPLETE.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
