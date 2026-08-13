import unittest

import numpy as np

from utils.adaptive_distribution_calibration import (
    augment_support_features,
    compute_class_statistics,
    nearest_base_statistics,
    shot_gate,
)


class AdaptiveDistributionCalibrationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        labels = np.repeat(np.arange(4), 10)
        features = rng.normal(size=(40, 6)) + labels[:, None] * 3.0
        self.classes, self.means, self.variances = compute_class_statistics(features, labels)

    def test_statistics_are_finite_and_class_aligned(self):
        self.assertTrue(np.array_equal(self.classes, np.arange(4)))
        self.assertEqual(self.means.shape, (4, 6))
        self.assertTrue(np.isfinite(self.variances).all())
        self.assertTrue((self.variances > 0).all())

    def test_nearest_statistics_returns_requested_classes(self):
        mean, variance, indices = nearest_base_statistics(
            self.means[0], self.means, self.variances, top_m=2
        )
        self.assertEqual(len(indices), 2)
        self.assertEqual(mean.shape, (6,))
        self.assertTrue((variance > 0).all())

    def test_shot_gate_decreases(self):
        self.assertGreater(shot_gate(1), shot_gate(5))
        self.assertGreater(shot_gate(5), shot_gate(20))

    def test_all_methods_are_reproducible_and_balanced(self):
        support_labels = np.repeat(np.arange(3), 2)
        support = self.means[:3].repeat(2, axis=0)
        for method in ("lr", "mean", "diag", "gated"):
            first_x, first_y = augment_support_features(
                support, support_labels, self.means, self.variances, method,
                np.random.default_rng(2024), synthetic_per_class=5,
            )
            second_x, second_y = augment_support_features(
                support, support_labels, self.means, self.variances, method,
                np.random.default_rng(2024), synthetic_per_class=5,
            )
            self.assertTrue(np.array_equal(first_x, second_x))
            self.assertTrue(np.array_equal(first_y, second_y))
            self.assertTrue(np.isfinite(first_x).all())
            if method == "lr":
                self.assertEqual(len(first_y), 6)
            elif method == "gated":
                self.assertTrue(np.array_equal(np.bincount(first_y), np.full(3, 6)))
            else:
                self.assertTrue(np.array_equal(np.bincount(first_y), np.full(3, 7)))


if __name__ == "__main__":
    unittest.main()
