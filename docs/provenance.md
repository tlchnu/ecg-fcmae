# Source provenance and packaging decisions

Release date: **2026-10-15**. This is a packaging and validation task; no scientific result was regenerated or altered.

The experiment launch script explicitly changes into `canonical_snapshots/20260731_noaug_v1` and asserts commit `6b8eaf7b553758b962b7fe483b711284d514e247`, tree `665b9031feb5c0392bc86e3142b2d353bf4ea236`. This snapshot supplies the released model/data/metric code and core training/analysis scripts. Its name does not override the augmented paper configuration.

The server's working model code has subsequently changed in four files. Publishing that mutable directory as if it generated the paper would be misleading. The full per-file comparison is in `provenance/server_environment.json`. Later low-label audit scripts and tabular prediction export are included separately, with individual source hashes.

`provenance/source_files.json` records the exact source-relative path, original SHA-256, released SHA-256 and modification status for every copied file. Downloaded files were independently hashed after transfer. Core model and metric files remain byte-identical to the canonical snapshot.

Packaging changes:

1. Grouped paper code, matched tests, aggregate results and manuscript assets into a standalone repository.
2. Derived portable `configs/paper/` from existing effective configs, changing input/output/checkpoint paths and expanding continuation-only seed lists to the full paper set {42, 2026, 3407}. Original effective configs remain under `results/`.
3. Corrected stale ECA comments and the misleading FCMAE completion message. Training computations were not changed.
4. Replaced outdated packaging dependency metadata with versions observed on the server during this task. Historical run environment files remain intact.
5. Added release integrity checks, a result-to-code index and reproduction documentation.
6. Omitted the SSH endpoint from the public copy of the secondary freeze audit; the private working copy retains it.

Paper assets come from `outputs/bspc_overleaf_20260917` in the original local research project. The copied English text includes the October 8 annotation revisions and subsequent table-width edits. No new PDF compilation is claimed by this packaging task.

The following stay outside the public Git repository: source datasets, patient-level predictions and metadata, subset index arrays, attribution sample manifests, checkpoints, caches, credentials, server connection files, training queue logs and unrelated historical drafts. The untouched downloaded snapshot and operational logs are retained in the original project, outside this release directory.

The archive `original_paper_launch.sh.txt` preserves the original execution provenance. It contains machine-specific paths and shutdown logic and is not an executable entry point. All supported new runs use `configs/paper/` and CLI paths.
