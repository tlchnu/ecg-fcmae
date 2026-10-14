"""Tests for exact-scope model efficiency measurement v1."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "analysis" / "model_efficiency_v1.py"
SPEC = importlib.util.spec_from_file_location("model_efficiency_v1", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OperationCountTests(unittest.TestCase):
    def test_grouped_conv1d_macs(self) -> None:
        layer = nn.Conv1d(4, 8, kernel_size=3, groups=2, bias=False)
        output = torch.zeros(2, 8, 10)
        expected = 2 * 10 * 8 * (4 // 2) * 3
        self.assertEqual(MODULE.conv1d_macs(layer, output), expected)

    def test_linear_macs_supports_batched_tokens(self) -> None:
        layer = nn.Linear(5, 7, bias=False)
        output = torch.zeros(2, 3, 7)
        self.assertEqual(MODULE.linear_macs(layer, output), 2 * 3 * 5 * 7)

    def test_parameter_counts_separate_trainable(self) -> None:
        model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2, bias=False))
        model[1].weight.requires_grad_(False)
        counts = MODULE.parameter_counts(model)
        self.assertEqual(counts["total_parameters"], 24)
        self.assertEqual(counts["trainable_parameters"], 16)


class ReportingTests(unittest.TestCase):
    def test_bytes_to_megabytes_uses_binary_units(self) -> None:
        self.assertEqual(MODULE.bytes_to_megabytes(1024 * 1024), 1.0)

    def test_latency_summary_reports_requested_statistics(self) -> None:
        summary = MODULE.summarize_latency_ms(np.asarray([1.0, 2.0, 3.0, 4.0]))
        self.assertAlmostEqual(summary["latency_median_ms"], 2.5)
        self.assertAlmostEqual(summary["latency_mean_ms"], 2.5)
        self.assertAlmostEqual(summary["latency_std_ms"], np.std([1, 2, 3, 4], ddof=1))
        self.assertAlmostEqual(summary["latency_p25_ms"], 1.75)
        self.assertAlmostEqual(summary["latency_p75_ms"], 3.25)
        self.assertAlmostEqual(summary["latency_p95_ms"], 3.85)


if __name__ == "__main__":
    unittest.main()
