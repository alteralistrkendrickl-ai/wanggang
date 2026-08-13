import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from wisig_validate_lr import (
    evaluate_lr_jobs,
    make_config,
    recover_log_rows,
    sample_support,
)


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

    def test_resume_log_recovers_completed_iterations(self):
        args = SimpleNamespace(
            protocol="cross-rx",
            checkpoint="best",
            shots=[1, 5],
            iterations=100,
            base_seed=2024,
        )
        checkpoint_hash = "a" * 64
        content = (
            "PROTOCOL=cross-rx\n"
            f"CHECKPOINT_SHA256={checkpoint_hash}\n"
            "SHOT=1 ITERATION=1/100 SEED=2024 ACC=46.0000\n"
            "SHOT=5 ITERATION=2/100 SEED=2025 ACC=55.2500\n"
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "interrupted.log"
            path.write_text(content, encoding="utf-8")
            rows = recover_log_rows(path, args, checkpoint_hash)
        self.assertEqual(len(rows), 2)
        self.assertEqual((rows[0]["shot"], rows[0]["iteration"]), (1, 1))
        self.assertEqual(rows[1]["accuracy"], "55.25000000")

    def test_parallel_lr_matches_serial_lr(self):
        rng = np.random.default_rng(2024)
        query_features = rng.normal(size=(90, 8)).astype(np.float32)
        query_labels = np.repeat(np.arange(3), 30)
        jobs = []
        for iteration, seed in enumerate((2024, 2025, 2026), start=1):
            features = rng.normal(size=(30, 8)).astype(np.float32)
            labels = np.repeat(np.arange(3), 10)
            jobs.append(((1, iteration), features, labels, seed))
        serial = evaluate_lr_jobs(jobs, query_features, query_labels, workers=1)
        parallel = evaluate_lr_jobs(jobs, query_features, query_labels, workers=2)
        self.assertEqual(serial, parallel)


if __name__ == "__main__":
    unittest.main()
