# Results and manuscript map

All paths below are relative to `results/20260804_aug_jitter005_mask010_full_paper_v1/`. Values are copied from existing server artifacts. The paper uses augmented training and seeds **42, 2026, 3407**; the earlier no-augmentation protocol with seed 2027 is not mixed into these tables.

| Manuscript topic | Evidence directory/files | Code entry |
|---|---|---|
| Full vs same-architecture random initialization | `01_core/{full,strict_no_pretraining}/exact_seed_metrics.csv`, `exact_summary.csv` | `scripts/train_downstream.py`, `analysis/review_experiments_v1.py` |
| Full vs Strict ensemble difference | `10_three_seed_paper_statistics_v1/full_vs_strict_ensemble_paired_95ci.csv` | `scripts/analyze_three_seed_paper_predictions.py` |
| Matched component ablations | `01_core/{no_eca,gap,stage4_only}/` | `scripts/train_downstream.py` |
| Modalities and fusion | `01_core/{ecg_only,concat,tabular_only}/` | ECG trainer and `scripts/train_tabular_only.py` |
| A–G cumulative configuration chain | `02_component_chain/component_{A,B,C,D}/` and `component_aliases_three_seed.json` | Paper YAMLs; E/F/G reuse ECG-only/Concat/Full |
| Label efficiency | `03_low_label_272_v1/analysis/{summary,seed_metrics,paired_bootstrap}.csv` | `scripts/train_downstream_subset.py`, `scripts/analyze_aug_low_label_272.py` |
| Low-label provenance | `03_low_label_272_v1/provenance/subset_provenance.csv` | Shared deterministic training permutation |
| Reduced leads | `04_reduced_leads/`, `05_single_leads/` | Lead slicing and pretrained stem selection |
| Twelve endpoints and ensemble intervals | `10_three_seed_paper_statistics_v1/full_per_label_seed_and_ensemble.csv`, `full_ensemble_per_label_95ci.csv` | `scripts/analyze_three_seed_paper_predictions.py` |
| Ensemble overall/macro intervals | `10_three_seed_paper_statistics_v1/full_ensemble_aggregate_95ci.csv` | `scripts/analyze_three_seed_paper_predictions.py` |
| Sixteen subgroups | `07_subgroups_three_seed_v1/` | `analysis/subgroup_analysis_v1.py` |
| Computational efficiency | `08_efficiency_seed42/` | `analysis/model_efficiency_v1.py` |

The public artifact includes aggregate metrics, configuration copies, completion states and hashes. Patient-level prediction arrays, bootstrap arrays and training subset indices remain in authorized storage. Hash lists may name those private artifacts; listing their hashes does not mean their contents are distributed.

## Interpretation

- Seed tables use mean ± sample SD (`ddof=1`); a three-seed probability ensemble is a separate estimator.
- Patient-level paired bootstrap uses 2,000 resamples. Most variant comparisons use **variant minus Full**. Full-vs-Strict and low-label comparisons have their own explicit direction; consult their column names and freeze notes.
- Some historical per-seed `delta_*` fields are bootstrap distribution means, not observed point differences. This discrepancy is retained and documented in [the original freeze audit](evidence/final_results_freeze.md).
- Macro-11 excludes Overall SHD; Macro-12 includes all outputs. Do not substitute them.
- E/F/G are aliases, not additional independent training runs. The 100% low-label points reuse the main Full/Strict runs.
- No ECA, GAP and Stage-4-only comparisons do not establish separate significant benefits or equivalence from their small point differences. The cumulative A–G chain does not isolate each component's marginal effect.
- Rare-label AUPRC depends on prevalence. Subgroup results describe this fixed single-center test cohort; they do not establish external validation or causal demographic effects.

The bundled manuscript's published-model comparison may include results reported in other papers. Those rows must retain their original citations and are not represented as newly reproduced runs by this repository. Do not infer that every manuscript table has a local checkpoint.
