"""Tests for nested low-label result aggregation."""

import json
from pathlib import Path

import numpy as np
from scripts.analyze_low_label_efficiency import sha256, validate_nested_subsets


def write_subset(directory: Path, indices: list[int], fraction: float, digest: str) -> None:
    directory.mkdir(parents=True)
    path = directory / f"subset_{fraction}.npy"
    np.save(path, np.asarray(indices, dtype=np.int64))
    (directory / "train_subset_manifest.json").write_text(
        json.dumps(
            {
                "fraction": fraction,
                "n_total": 10,
                "n_selected": len(indices),
                "indices_file": str(path),
                "indices_sha256": sha256(path),
                "permutation_sha256": "shared",
            }
        ),
        encoding="utf-8",
    )


def test_validate_nested_subsets_accepts_shared_nested_ssl_scratch(tmp_path: Path) -> None:
    records = []
    for fraction, indices in ((0.1, [1]), (0.3, [1, 4, 8])):
        for method in ("ssl", "scratch"):
            directory = tmp_path / f"{method}_{fraction}"
            write_subset(directory, indices, fraction, f"hash-{fraction}")
            records.append({"fraction": fraction, "method": method, "directory": directory})

    manifest = validate_nested_subsets(records, n_total=10)

    assert len(manifest) == 5  # four recorded subsets plus generated 100%
    assert manifest.query("fraction == 1.0").iloc[0]["nested_with_next"]


def test_validate_nested_subsets_rejects_nonmatching_pair(tmp_path: Path) -> None:
    write_subset(tmp_path / "ssl", [1, 4], 0.2, "ssl-hash")
    write_subset(tmp_path / "scratch", [1, 5], 0.2, "scratch-hash")
    records = [
        {"fraction": 0.2, "method": "ssl", "directory": tmp_path / "ssl"},
        {"fraction": 0.2, "method": "scratch", "directory": tmp_path / "scratch"},
    ]

    try:
        validate_nested_subsets(records, n_total=10)
    except ValueError as error:
        assert "do not share identical indices" in str(error)
    else:
        raise AssertionError("mismatched paired subsets must fail")
