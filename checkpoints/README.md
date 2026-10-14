# Pretrained encoders

Weights remain in the original authorized research storage and are not part of this repository.

The paper's ECA-enabled FCMAE encoder has SHA-256:
`bc63726fdfe4ee28a59e4910f8e01450d84273dbb7b2fb3c56a700cf9b37fb72`.

Use `fcmae_eca_best.pt` for the Full and applicable ablation configs. Component B uses a distinct architecture-matched no-ECA checkpoint, referenced as `fcmae_no_eca_best.pt`; the no-ECA checkpoint is not interchangeable with the ECA checkpoint. See `docs/evidence/secondary_results_freeze.md` for the component chain.

To retrain the ECA encoder, use `configs/paper/pretrain_fcmae_eca.yaml`. The resulting file is under `experiments/20260629_fcmae_convnext_eca_pretrain/`. A new training run is not expected to reproduce the original checkpoint hash.

The recorded no-ECA checkpoint SHA-256 is `5be160de9719d0881931835fb16f855f3052d73cbf020142e69f0ced1e2f4116`, as recorded by the original three-seed launch manifest.
