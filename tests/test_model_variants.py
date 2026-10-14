"""Shape and backward tests for registered model variants."""

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecg_project.models import MODEL_PRESETS, build_model


class ModelVariantTests(unittest.TestCase):
    def test_all_presets_forward_and_backward(self):
        waveform = torch.randn(2, 12, 1000)
        tabular = torch.randn(2, 7)
        for preset in MODEL_PRESETS:
            with self.subTest(preset=preset):
                model = build_model(
                    preset,
                    num_classes=5,
                    tabular_features=7,
                    depths=(1, 1, 1, 1),
                    dims=(16, 32, 64, 128),
                )
                logits = model(waveform, tabular)
                self.assertEqual(tuple(logits.shape), (2, 5))
                logits.mean().backward()

    def test_encoder_contract(self):
        model = build_model(
            "reparam_lk13",
            depths=(1, 1, 1, 1),
            dims=(16, 32, 64, 128),
        )
        features = model.forward_encoder(torch.randn(2, 12, 1000))
        self.assertEqual(features.ndim, 3)
        self.assertEqual(features.shape[1], 128)

    def test_leadaware_preset_forward_and_backward(self):
        waveform = torch.randn(2, 12, 1000)
        tabular = torch.randn(2, 7)
        model = build_model(
            "leadaware_convnext",
            num_classes=5,
            tabular_features=7,
            depths=(1, 1, 1, 1),
            dims=(16, 32, 64, 128),
            lead_mixer_heads=2,
        )
        logits = model(waveform, tabular)
        self.assertEqual(tuple(logits.shape), (2, 5))
        logits.mean().backward()

    def test_leadaware_stem_preserves_encoder_contract(self):
        model = build_model(
            "leadaware_convnext",
            depths=(1, 1, 1, 1),
            dims=(16, 32, 64, 128),
            lead_mixer_heads=2,
        )
        features = model.forward_encoder(torch.randn(2, 12, 1000))
        self.assertEqual(features.ndim, 3)
        self.assertEqual(features.shape[1], 128)


if __name__ == "__main__":
    unittest.main()
