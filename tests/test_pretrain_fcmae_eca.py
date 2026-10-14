"""Tests for FCMAE pretraining ECA configuration."""

import argparse
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "pretrain_fcmae.py"
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_pretrain_module():
    spec = importlib.util.spec_from_file_location("pretrain_fcmae", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FCMAEPretrainECATests(unittest.TestCase):
    def test_parse_args_accepts_use_eca(self):
        module = load_pretrain_module()
        with patch.object(
            sys,
            "argv",
            [
                "pretrain_fcmae.py",
                "--dataset",
                "echonext",
                "--data-dir",
                "data/echonext",
                "--use-eca",
            ],
        ):
            args = module.parse_args()
        self.assertTrue(args.use_eca)

    def test_encoder_config_enables_eca_weights(self):
        module = load_pretrain_module()
        args = argparse.Namespace(
            model="convnext_tiny",
            block_type=None,
            stage_kernel_sizes=None,
            layer_scale_init=None,
            use_eca=True,
            lead_setting="full12",
            lead_input_mode="zero_fill",
        )
        config = module.build_encoder_config(args)
        config.update(depths=(1, 1, 1, 1), dims=(16, 32, 64, 128))
        model = module.ECGFCMAE1d(config, patch_size=25, mask_ratio=0.6)
        eca_keys = [
            name for name, _ in model.backbone.named_parameters()
            if "eca" in name.lower()
        ]
        self.assertTrue(config["use_eca"])
        self.assertGreater(len(eca_keys), 0)

    def test_anatomical_mask_forward_runs(self):
        module = load_pretrain_module()
        args = argparse.Namespace(
            model="convnext_tiny",
            block_type=None,
            stage_kernel_sizes=None,
            layer_scale_init=None,
            use_eca=True,
            lead_setting="full12",
            lead_input_mode="zero_fill",
        )
        config = module.build_encoder_config(args)
        config.update(depths=(1, 1, 1, 1), dims=(16, 32, 64, 128))
        model = module.ECGFCMAE1d(
            config,
            patch_size=25,
            mask_ratio=0.6,
            masking_strategy="anatomical",
        )
        loss, pred, mask = model(torch.randn(2, 12, 500))
        self.assertEqual(tuple(pred.shape), (2, 12, 500))
        self.assertEqual(tuple(mask.shape), (2, 12, 500))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(mask.sum()), 0.0)

    def test_independent_time_mask_varies_by_lead(self):
        module = load_pretrain_module()
        config = module.get_model_config(
            "convnext_tiny",
            in_channels=12,
            num_classes=12,
            tabular_features=0,
            drop_path_rate=0.0,
            use_eca=True,
            multi_scale_feat=True,
            depths=(1, 1, 1, 1),
            dims=(16, 32, 64, 128),
        )
        model = module.ECGFCMAE1d(
            config,
            patch_size=25,
            mask_ratio=0.6,
            masking_strategy="time_independent",
        )
        torch.manual_seed(7)
        mask = model.gen_mask(2, 12, 20, torch.device("cpu"))
        self.assertEqual(tuple(mask.shape), (2, 12, 20))
        self.assertFalse(torch.equal(mask[:, 0], mask[:, 1]))
        self.assertTrue(torch.all(mask.sum(dim=-1) == 12))

    def test_true_single_lead_encoder_uses_one_input_channel(self):
        module = load_pretrain_module()
        args = argparse.Namespace(
            model="convnext_tiny",
            block_type=None,
            stage_kernel_sizes=None,
            layer_scale_init=None,
            use_eca=True,
            lead_setting="lead_i",
            lead_input_mode="slice",
        )
        config = module.build_encoder_config(args)
        self.assertEqual(config["in_channels"], 1)


if __name__ == "__main__":
    unittest.main()
