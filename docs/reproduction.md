# Reproduction guide

Run all commands from the repository root. The public release contains aggregate evidence; complete training and patient-level bootstrap reproduction require authorized data and checkpoints.

## Environment and protocol

Use Python 3.10 and `requirements.txt`. Training records identify PyTorch 2.5.1+cu124, CUDA 12.4, cuDNN 90100 and an RTX 4090. `docs/provenance/server_environment.json` separately records installed package versions observed during packaging.

The paper protocol uses seeds **42, 2026, 3407**, downstream Gaussian jitter with SD 0.05, and random masking of 10% of time points. FCMAE pretraining uses seed 42, 100 epochs, batch size 64, LR 0.00015, weight decay 0.05, patch size 25, mask ratio 0.60, normalized targets and ECA. The original pretrained checkpoint was selected by reconstruction loss, without test data.

Downstream Full and Strict use batch size 64, at most 100 epochs, LR 0.001, weight decay 0.05, layer decay 0.75, EMA decay 0.999, gradient clipping 1.0 and validation Overall SHD AUROC early stopping with patience 10. `llrd: false` in a pretrained run is only the explicit CLI flag: the trainer enables LLRD when pretrained weights are present. Strict explicitly enables LLRD to match.

## Training and validation

Prepare the inputs described in [data/README.md](../data/README.md). Run:

```bash
python scripts/pretrain_fcmae.py --config configs/paper/pretrain_fcmae_eca.yaml
python scripts/train_downstream.py --config configs/paper/full.yaml --pretrained experiments/20260629_fcmae_convnext_eca_pretrain/fcmae_convnext_eca_best.pt
python scripts/train_downstream.py --config configs/paper/strict_no_pretraining.yaml
```

Each model is evaluated on validation data during training and on test data after loading the validation-selected checkpoint. Retain the generated effective config, environment JSON, metrics and checkpoints together. `results/` is the immutable historical evidence; new runs go to `experiments/`.

Other configurations are in `configs/paper/`: `no_eca`, `gap`, `stage4_only`, `ecg_only`, `concat`, `component_A` through `component_D`, and the five reduced-lead conditions. Component E/F/G are aliases of ECG-only/Concat/Full. `tabular_only.yaml` must be passed to `scripts/train_tabular_only.py`.

Low-label runs use the wrapper and one fixed training permutation shared across fractions and methods:

```bash
python scripts/train_downstream_subset.py --config configs/paper/full.yaml --train-fraction 0.01 --train-subset-seed 20260801 --subset-permutation-file experiments/low_label/train_permutation.npy --experiment-name ssl_01pct --output-root experiments/low_label
python scripts/train_downstream_subset.py --config configs/paper/strict_no_pretraining.yaml --train-fraction 0.01 --train-subset-seed 20260801 --subset-permutation-file experiments/low_label/train_permutation.npy --experiment-name scratch_01pct --output-root experiments/low_label
```

Repeat at 0.05, 0.10 and 0.30, changing the experiment name. The 100% rows reuse Full/Strict. Historical subset hashes are in `results/.../03_low_label_272_v1/provenance/subset_provenance.csv`; validate provenance before claiming an exact replication. Validation/test sets are never subsampled.

## Prediction export and analysis

Inspect the documented CLI arguments with `--help`:

```bash
python scripts/export_shd_predictions.py --help
python analysis/review_experiments_v1.py --help
python analysis/subgroup_analysis_v1.py --help
python scripts/analyze_aug_low_label_272.py --help
```

Use `export_shd_predictions.py` for the ECG model and `export_tabular_predictions.py` for the tabular baseline. Export validation and test separately for each checkpoint and seed. Keep these NPZs private: they contain ordered patient identifiers and labels needed to check paired alignment.

`analysis/review_experiments_v1.py` computes exact seed metrics and same-seed paired bootstrap results. `scripts/analyze_aug_low_label_272.py` verifies nested subsets and compares SSL/Scratch. `analysis/subgroup_analysis_v1.py` performs the age/sex/race-ethnicity/clinical-context analysis. `analysis/model_efficiency_v1.py` reports the measured computational scope. Original summary CSVs are already included, so reading the reported results requires no data download or training.

Macro-11 excludes output index 11 (Overall SHD); Macro-12 includes it. The older generic trainer's `Test_Macro_AUROC` is Macro-12. Do not label it Macro-11. Three-seed means, ensemble estimates and patient-bootstrap intervals are distinct estimands. See [results_map.md](results_map.md).

## Validation and preserved implementation behavior

```bash
python scripts/verify_release.py
python -m pytest tests -q
```

The verifier checks source/result hashes, syntax, seed sets, exact mean/sample-SD consistency, and manuscript asset references. The tests exercise model forward/backward, checkpoint lead adaptation, metrics, deterministic subsets and analysis logic. They do not retrain the paper or independently recompute patient bootstrap intervals without private predictions.

This release preserves the numerical implementation used by the paper. It does not retrofit historical runs with new AMP, accumulation or resume semantics. FCMAE enables AMP on CUDA in its original implementation. Existing checkpoints are sufficient for inference and encoder transfer but do not provide a complete optimizer/RNG resume state. No training job is launched by the release verifier. Use a separate output directory for new experiments.

The archival launch script is saved as `.sh.txt` only as provenance; it includes historical server shutdown logic and is not a supported execution entry point. Use the portable commands above.
