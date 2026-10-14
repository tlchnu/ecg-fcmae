"""Tests for the fixed-permutation low-label sampling contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts" / "train_downstream_subset.py"
SPEC = importlib.util.spec_from_file_location("train_downstream_subset", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prefix_subsets_are_strictly_nested() -> None:
    permutation = np.array([7, 1, 9, 0, 4, 8, 2, 6, 5, 3], dtype=np.int64)
    subsets = {
        fraction: MODULE.prefix_subset_indices(permutation, fraction)
        for fraction in (0.1, 0.3, 0.5, 1.0)
    }

    assert set(subsets[0.1]) <= set(subsets[0.3])
    assert set(subsets[0.3]) <= set(subsets[0.5])
    assert set(subsets[0.5]) <= set(subsets[1.0])
    assert np.array_equal(subsets[1.0], np.sort(permutation))


def test_load_or_create_permutation_is_reused(tmp_path: Path) -> None:
    permutation_path = tmp_path / "train_permutation.npy"
    first = MODULE.load_or_create_permutation(permutation_path, n_total=20, seed=42)
    second = MODULE.load_or_create_permutation(permutation_path, n_total=20, seed=999)

    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(20))


def test_invalid_saved_permutation_is_rejected(tmp_path: Path) -> None:
    permutation_path = tmp_path / "invalid.npy"
    np.save(permutation_path, np.array([0, 1, 1, 3], dtype=np.int64))

    with pytest.raises(ValueError, match="permutation"):
        MODULE.load_or_create_permutation(permutation_path, n_total=4, seed=42)


def test_optional_path_is_yaml_serializable_before_experiment_archive(tmp_path: Path) -> None:
    normalized = MODULE.yaml_serializable_optional_path(tmp_path / "permutation.npy")

    assert isinstance(normalized, str)
    assert "permutation.npy" in yaml.safe_dump({"subset_permutation_file": normalized})
