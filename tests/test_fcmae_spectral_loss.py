import unittest

import torch

from scripts.pretrain_fcmae import ECGFCMAE1d


def tiny_encoder_config():
    return {
        "in_channels": 2,
        "num_classes": 1,
        "tabular_features": 0,
        "depths": (1,),
        "dims": (8,),
        "stage_kernel_sizes": (3,),
        "drop_path_rate": 0.0,
        "layer_scale_init": 0.0,
        "use_eca": False,
        "multi_scale_feat": False,
        "pool": "avg",
        "fusion": "none",
    }


class FCMAESpectralLossTest(unittest.TestCase):
    def test_spectral_loss_is_disabled_by_default(self):
        model = ECGFCMAE1d(
            tiny_encoder_config(),
            in_chans=2,
            patch_size=8,
            mask_ratio=0.5,
            decoder_dim=8,
            decoder_depth=1,
        )
        wave = torch.randn(2, 2, 64)

        loss, pred, mask = model(wave)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(pred.shape), tuple(wave.shape))
        self.assertEqual(tuple(mask.shape), (2, 1, 64))
        self.assertIn("time_loss", model.last_loss_components)
        self.assertIn("spectral_loss", model.last_loss_components)
        self.assertEqual(model.last_loss_components["spectral_loss"], 0.0)

    def test_spectral_loss_contributes_to_total_loss(self):
        model = ECGFCMAE1d(
            tiny_encoder_config(),
            in_chans=2,
            patch_size=8,
            mask_ratio=0.5,
            decoder_dim=8,
            decoder_depth=1,
            spectral_loss_weight=0.1,
            spectral_loss_type="smooth_l1",
        )
        wave = torch.randn(2, 2, 64)

        loss, _, _ = model(wave)
        components = model.last_loss_components

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(components["spectral_loss"], 0.0)
        expected_total = components["time_loss"] + 0.1 * components["spectral_loss"]
        self.assertAlmostEqual(float(loss.detach()), expected_total, places=5)


if __name__ == "__main__":
    unittest.main()
