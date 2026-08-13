import unittest

import numpy as np

from wisig_a1_validation_novel_diagnostics import (
    balanced_source_indices,
    balanced_target_indices,
    identity_disjoint_source_target_probe,
    source_target_distance_metrics,
    source_to_target_identity_probe,
)


class A1ValidationNovelDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2024)
        source_y, source_d, source_f, target_y, target_f = [], [], [], [], []
        for identity in range(6):
            for domain in range(3):
                for _ in range(8):
                    vector = np.zeros(16, dtype=np.float32)
                    vector[identity] = 4.0
                    vector[6 + domain] = 0.5
                    source_y.append(identity); source_d.append(domain)
                    source_f.append(vector + rng.normal(0, 0.03, 16))
            for _ in range(12):
                vector = np.zeros(16, dtype=np.float32)
                vector[identity] = 4.0
                vector[10] = 0.5
                target_y.append(identity)
                target_f.append(vector + rng.normal(0, 0.03, 16))
        self.source_y = np.asarray(source_y)
        self.source_d = np.asarray(source_d)
        self.source_f = np.asarray(source_f, dtype=np.float32)
        self.target_y = np.asarray(target_y)
        self.target_f = np.asarray(target_f, dtype=np.float32)

    def test_balanced_sampling_is_reproducible(self):
        first = balanced_source_indices(self.source_y, self.source_d, 4, 2024)
        second = balanced_source_indices(self.source_y, self.source_d, 4, 2024)
        self.assertTrue(np.array_equal(first, second))
        target = balanced_target_indices(self.target_y, 5, 2024)
        self.assertTrue(np.array_equal(
            np.bincount(self.target_y[target]), np.full(6, 5)
        ))

    def test_target_identity_probe_and_domain_probe(self):
        identity = source_to_target_identity_probe(
            self.source_f, self.source_y, self.target_f, self.target_y, 2024
        )
        domain = identity_disjoint_source_target_probe(
            self.source_f, self.source_y, self.target_f, self.target_y, 2024
        )
        self.assertGreater(identity["target_identity_accuracy"], 0.9)
        self.assertGreater(domain["source_target_accuracy"], 0.9)
        self.assertEqual(domain["domain_probe_train_identities"], 4)
        self.assertEqual(domain["domain_probe_test_identities"], 2)

    def test_source_target_distance_is_finite_and_separated(self):
        metrics = source_target_distance_metrics(
            self.source_f, self.source_y, self.target_f, self.target_y
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertGreater(
            metrics["different_identity_target_cosine_distance"],
            metrics["same_identity_source_target_cosine_distance"],
        )
        self.assertGreater(metrics["target_separation_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
