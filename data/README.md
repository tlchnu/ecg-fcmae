# EchoNext input data

Obtain the dataset under its original access conditions. No dataset files or patient-level predictions are distributed here. The paper uses EchoNext 1.1.0; PTB-XL support in the code does not imply a PTB-XL result in this paper release.

Place the authorized files in `data/echonext/`:

```text
echonext_metadata_100k.csv
EchoNext_train_waveforms.npy
EchoNext_val_waveforms.npy
EchoNext_test_waveforms.npy
EchoNext_train_tabular_features.npy
EchoNext_val_tabular_features.npy
EchoNext_test_tabular_features.npy
```

Waveform inputs resolve to `(N, 12, 2500)`; tabular features have 7 columns. The label names and ordering are defined in `src/ecg_project/data/datasets.py`. Preserve the original split and row ordering; do not regenerate splits or fit preprocessing on the test data.

The frozen training/validation/test counts are 72,475 / 4,626 / 5,442. Seven tabular fields are sex, age, ventricular rate, atrial rate, PR interval, QRS duration, and corrected QT interval. Follow the source dataset's preprocessing; no new universal filter is introduced by this release.

Expected prediction split hashes from the original export procedure:

- Validation: `cf882636c4fcd7f94710c66c983d1a83626ecf1c50955e84357da511f5b3d34f1`
- Test: `8e4fa6df7f30565ea8aee4e06cb417637648bbce6f9d2a8c415d12b91c2faaba`

These hashes are computed from ordered IDs and labels by `export_shd_predictions.py`, not from raw waveform-file bytes. Exported NPZ files contain patient/record identifiers and must stay outside public Git history. `.gitignore` excludes them.
