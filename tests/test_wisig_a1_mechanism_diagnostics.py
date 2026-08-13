import unittest

import numpy as np

from wisig_a1_mechanism_diagnostics import (
    balanced_cell_indices,
    cosine_distance_metrics,
    domain_disjoint_identity_probe,
    identity_disjoint_domain_probe,
)


class A1MechanismDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2024)
        identities, domains, features = [], [], []
        for identity in range(6):
            for domain in range(3):
                for _ in range(8):
                    identities.append(identity); domains.append(domain)
                    vector = np.zeros(12, dtype=np.float32)
                    vector[identity] = 3.0
                    vector[6 + domain] = 1.0
                    features.append(vector + rng.normal(0, 0.05, 12))
        self.identities = np.asarray(identities)
        self.domains = np.asarray(domains)
        self.features = np.asarray(features, dtype=np.float32)

    def test_balanced_indices_are_reproducible(self):
        first = balanced_cell_indices(self.identities, self.domains, 4, 2024)
        second = balanced_cell_indices(self.identities, self.domains, 4, 2024)
        self.assertTrue(np.array_equal(first, second))
        pairs, counts = np.unique(
            np.stack((self.identities[first], self.domains[first]), axis=1),
            axis=0, return_counts=True,
        )
        self.assertEqual(len(pairs), 18)
        self.assertTrue(np.array_equal(counts, np.full(18, 4)))

    def test_probes_use_disjoint_axes(self):
        domain = identity_disjoint_domain_probe(
            self.features, self.identities, self.domains, 2024
        )
        identity = domain_disjoint_identity_probe(
            self.features, self.identities, self.domains, 2024
        )
        self.assertGreater(domain["domain_accuracy"], 0.9)
        self.assertGreater(identity["identity_accuracy"], 0.9)
        self.assertTrue(set(identity["identity_probe_train_domains"]).isdisjoint(
            identity["identity_probe_test_domains"]
        ))

    def test_distance_metrics_are_finite_and_separated(self):
        metrics = cosine_distance_metrics(self.features, self.identities, self.domains)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertGreater(metrics["different_identity_cosine_distance"],
                           metrics["same_identity_cross_domain_cosine_distance"])
        self.assertGreater(metrics["separation_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
