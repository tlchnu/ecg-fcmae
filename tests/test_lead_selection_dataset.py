import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ecg_project.data.datasets import (  # noqa: E402
    LEAD_SETTINGS,
    LeadSelectionDataset,
    ShuffledLabelDataset,
)


class LeadSelectionDatasetTests(unittest.TestCase):
    def test_anatomical4_keeps_shape_and_zero_fills_missing_leads(self):
        waves = torch.ones(2, 12, 100)
        tabular = torch.zeros(2, 7)
        labels = torch.zeros(2, 12)
        dataset = LeadSelectionDataset(TensorDataset(waves, tabular, labels), "anatomical4")

        wave, tab, label = dataset[0]
        self.assertEqual(tuple(wave.shape), (12, 100))
        self.assertEqual(tuple(tab.shape), (7,))
        self.assertEqual(tuple(label.shape), (12,))

        keep = set(LEAD_SETTINGS["anatomical4"])
        for lead_idx in range(12):
            expected = 1.0 if lead_idx in keep else 0.0
            self.assertTrue(torch.allclose(wave[lead_idx], torch.full((100,), expected)))

    def test_anatomical4_slice_returns_true_four_channel_input(self):
        waves = torch.arange(12, dtype=torch.float32).view(1, 12, 1).expand(1, 12, 8)
        tabular = torch.zeros(1, 7)
        labels = torch.zeros(1, 12)
        dataset = LeadSelectionDataset(
            TensorDataset(waves, tabular, labels),
            "anatomical4",
            input_mode="slice",
        )

        wave, _, _ = dataset[0]

        self.assertEqual(tuple(wave.shape), (4, 8))
        self.assertEqual(wave[:, 0].tolist(), [0.0, 1.0, 6.0, 10.0])

    def test_shuffled_label_dataset_preserves_features_and_permutates_labels(self):
        waves = torch.arange(5, dtype=torch.float32).view(5, 1, 1)
        tabular = torch.arange(5, dtype=torch.float32).view(5, 1)
        labels = torch.arange(5, dtype=torch.float32).view(5, 1)
        dataset = ShuffledLabelDataset(TensorDataset(waves, tabular, labels), seed=17)

        observed_waves = []
        observed_labels = []
        for index in range(5):
            wave, tab, label = dataset[index]
            observed_waves.append(float(wave.item()))
            observed_labels.append(float(label.item()))
            self.assertEqual(float(tab.item()), float(index))

        self.assertEqual(observed_waves, list(map(float, range(5))))
        self.assertCountEqual(observed_labels, list(map(float, range(5))))
        self.assertNotEqual(observed_labels, list(map(float, range(5))))


if __name__ == "__main__":
    unittest.main()
