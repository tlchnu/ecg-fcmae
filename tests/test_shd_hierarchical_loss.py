import unittest

import torch

from scripts.train_downstream import (
    SHDConsistencyLoss,
    aggregate_shd_probability,
    make_criterion,
)


class SHDHierarchicalLossTest(unittest.TestCase):
    def test_noisy_or_aggregate_maps_subtype_logits_to_shd_probability(self):
        logits = torch.full((2, 12), -10.0)
        logits[0, 3] = 10.0

        shd_prob = aggregate_shd_probability(logits, shd_index=11)

        self.assertEqual(tuple(shd_prob.shape), (2,))
        self.assertGreater(float(shd_prob[0]), 0.99)
        self.assertLess(float(shd_prob[1]), 0.01)

    def test_shd_consistency_loss_penalizes_inconsistent_direct_shd_prediction(self):
        targets = torch.zeros(2, 12)
        targets[0, 3] = 1.0
        targets[0, 11] = 1.0

        consistent = torch.full((2, 12), -10.0)
        consistent[0, 3] = 10.0
        consistent[0, 11] = 10.0

        inconsistent = consistent.clone()
        inconsistent[0, 11] = -10.0

        loss = SHDConsistencyLoss(base_loss="bce", shd_index=11, consistency_weight=1.0)

        self.assertLess(float(loss(consistent, targets)), float(loss(inconsistent, targets)))

    def test_make_criterion_returns_hierarchical_loss_when_weight_is_positive(self):
        criterion = make_criterion("bce", shd_consistency_weight=0.2, shd_index=11)

        self.assertIsInstance(criterion, SHDConsistencyLoss)


if __name__ == "__main__":
    unittest.main()
