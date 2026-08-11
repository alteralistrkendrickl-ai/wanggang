import unittest
from types import SimpleNamespace

import numpy as np

from wisig_validate_lr import make_config, sample_support


class WiSigValidationLRTest(unittest.TestCase):
    def test_route_is_validation_novel_val_only(self):
        args = SimpleNamespace(
            protocol="cross-rx", support_batch_size=32, query_batch_size=64
        )
        config = make_config(args, seed=2024, shot=1, device="cpu")
        self.assertEqual(config["dataset"]["query_split"], "val")
        self.assertTrue(config["dataset"]["root"].endswith("validation_novel"))

    def test_support_sampling_is_balanced_and_reproducible(self):
        labels = np.repeat(np.arange(30), 4)
        samples = np.arange(120 * 2 * 8, dtype=np.float32).reshape(120, 2, 8)
        first_x, first_y = sample_support(
            samples, labels, shot=2, seed=2024, num_classes=30
        )
        second_x, second_y = sample_support(
            samples, labels, shot=2, seed=2024, num_classes=30
        )
        self.assertTrue(np.array_equal(first_x, second_x))
        self.assertTrue(np.array_equal(first_y, second_y))
        classes, counts = np.unique(first_y, return_counts=True)
        self.assertTrue(np.array_equal(classes, np.arange(30)))
        self.assertTrue(np.array_equal(counts, np.full(30, 2)))


if __name__ == "__main__":
    unittest.main()
