import unittest

import numpy as np
import torch

from models.CrossDomainSupConLoss import CrossDomainSupConLoss
from utils.domain_batch_sampler import IdentityDomainBatchSampler


class A1ChannelConsistencyTest(unittest.TestCase):
    def test_sampler_builds_required_identity_domain_cells(self):
        labels = np.repeat(np.arange(8), 12)
        domains = np.tile(np.repeat(np.arange(3), 4), 8)
        sampler = IdentityDomainBatchSampler(
            labels, domains, identities_per_batch=4,
            domains_per_identity=2, samples_per_domain=2,
            seed=2024, batches_per_epoch=1,
        )
        batch = next(iter(sampler))
        self.assertEqual(len(batch), 16)
        batch_labels = labels[batch]
        batch_domains = domains[batch]
        for identity in np.unique(batch_labels):
            mask = batch_labels == identity
            unique, counts = np.unique(batch_domains[mask], return_counts=True)
            self.assertEqual(len(unique), 2)
            self.assertTrue(np.array_equal(counts, np.array([2, 2])))

    def test_sampler_is_epoch_deterministic(self):
        labels = np.repeat(np.arange(8), 12)
        domains = np.tile(np.repeat(np.arange(3), 4), 8)
        first = IdentityDomainBatchSampler(labels, domains, 4, 2, 2, seed=2024,
                                           batches_per_epoch=2)
        second = IdentityDomainBatchSampler(labels, domains, 4, 2, 2, seed=2024,
                                            batches_per_epoch=2)
        first.set_epoch(3)
        second.set_epoch(3)
        self.assertEqual(list(first), list(second))

    def test_loss_is_finite_and_backpropagates(self):
        features = torch.randn(8, 16, requires_grad=True)
        identities = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        domains = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        loss = CrossDomainSupConLoss(temperature=0.1)(features, identities, domains)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_loss_rejects_anchor_without_cross_domain_positive(self):
        features = torch.randn(4, 8)
        identities = torch.tensor([0, 0, 1, 1])
        domains = torch.tensor([0, 0, 0, 1])
        with self.assertRaisesRegex(ValueError, "without cross-domain positives"):
            CrossDomainSupConLoss()(features, identities, domains)

    def test_sampler_rejects_unpairable_identity(self):
        labels = np.array([0, 0, 1, 1, 1, 1])
        domains = np.array([0, 0, 0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "cannot form"):
            IdentityDomainBatchSampler(
                labels, domains, identities_per_batch=1,
                domains_per_identity=2, samples_per_domain=2,
            )


if __name__ == "__main__":
    unittest.main()
