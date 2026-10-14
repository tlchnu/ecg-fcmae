# Data integrity report

- Label CSV: `data/echonext/echonext_metadata_100k.csv`
- Metadata CSV: `data/echonext/echonext_metadata_100k.csv`
- Patient ID: `patient_key`
- Record ID: `ecg_key`
- Test rows: 5442
- Unique patients: 5442
- Duplicate patient IDs: 0
- Missing label rows: 0
- Primary label: `shd_moderate_or_greater_flag` at output index 11
- Alignment: all prediction rows rebound by audited `row_id`; labels and output order match the CSV.
- Probability validation: all values are finite and within [0,1]; logits were not accepted.
- Expected EchoNext-Mini test size difference: +0

## Subgroup mappings

- Age: 18-59, 60-69, 70-79, 80+ using left-inclusive/right-exclusive boundaries.
- Sex: source `female`/`male` to Female/Male.
- Race/ethnicity: Hispanic, White, Black, Asian, Other, Unknown; Unknown retained.
- Clinical context: Emergency, Inpatient, Outpatient, Procedural; Unknown only if present.

## Prediction files

- `/root/autodl-tmp/project_001_ecg_convnext_ssl/experiments/20260804_aug_jitter005_mask010_full_paper_v1/07_subgroups_three_seed_v1/prediction_input/seed_42/scratch_test.npz` (SHA-256 `fb98b7be55eed7310cb4defca1ae6eda30645d71b67e94308b7d4be5ededf8a0`)
- `/root/autodl-tmp/project_001_ecg_convnext_ssl/experiments/20260804_aug_jitter005_mask010_full_paper_v1/07_subgroups_three_seed_v1/prediction_input/seed_42/ssl_test.npz` (SHA-256 `419eafbced7a627411654c8bef816397c21c5c28ba00b5dcb2ae3bd3d5eeafab`)
- `/root/autodl-tmp/project_001_ecg_convnext_ssl/experiments/20260804_aug_jitter005_mask010_full_paper_v1/07_subgroups_three_seed_v1/prediction_input/seed_2026/scratch_test.npz` (SHA-256 `b194161da5f2dc154a2e5cddfa03dc2062f47d34f5d255ee0c43b2568679b83c`)
- `/root/autodl-tmp/project_001_ecg_convnext_ssl/experiments/20260804_aug_jitter005_mask010_full_paper_v1/07_subgroups_three_seed_v1/prediction_input/seed_2026/ssl_test.npz` (SHA-256 `e2974da3669bea36ea793b9d85faafa43da8ac1e1c383182d58e710dedd4231b`)
- `/root/autodl-tmp/project_001_ecg_convnext_ssl/experiments/20260804_aug_jitter005_mask010_full_paper_v1/07_subgroups_three_seed_v1/prediction_input/seed_3407/scratch_test.npz` (SHA-256 `7c4309b4c063aac01a137c84fdbfba9c7849dad33e451460a8b3b8b3e5301749`)
- `/root/autodl-tmp/project_001_ecg_convnext_ssl/experiments/20260804_aug_jitter005_mask010_full_paper_v1/07_subgroups_three_seed_v1/prediction_input/seed_3407/ssl_test.npz` (SHA-256 `e1f0d5cd4145df178e784917b928c2e6cd60eb35716d9605cc5d81fa5145840a`)

## Probability ranges

```json
{
  "scratch_seed_42": [
    0.0122694605961442,
    0.997096061706543
  ],
  "ssl_seed_42": [
    0.018195828422904015,
    0.9966912269592285
  ],
  "scratch_seed_2026": [
    0.01468849927186966,
    0.993128776550293
  ],
  "ssl_seed_2026": [
    0.015720708295702934,
    0.9980618357658386
  ],
  "scratch_seed_3407": [
    0.010551211424171925,
    0.9960267543792725
  ],
  "ssl_seed_3407": [
    0.016756154596805573,
    0.996150016784668
  ]
}
```
