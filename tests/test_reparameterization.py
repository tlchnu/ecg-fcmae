"""Exact-equivalence test for large-kernel branch fusion."""

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecg_project.models.mixers import DilatedReparamDWConv1D


class ReparameterizationTests(unittest.TestCase):
    def test_switch_to_deploy_is_equivalent(self):
        torch.manual_seed(42)
        module = DilatedReparamDWConv1D(channels=8, kernel_size=13).eval()
        waveform = torch.randn(3, 8, 257)
        expected = module(waveform)
        module.switch_to_deploy()
        actual = module(waveform)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
