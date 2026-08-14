import unittest
from pathlib import Path

from wisig_final_matrix_prepare import (
    CHECKPOINT_HASHES,
    ITERATIONS,
    PROTOCOL_VARIANTS,
    RUN_ENABLED,
    SHOTS,
    TRAIN_SEEDS,
    build_draft_manifest,
    canonical_hash,
    frozen_data_files,
    frozen_models,
)


class WiSigFinalMatrixPrepareTest(unittest.TestCase):
    def test_exact_paired_matrix_is_frozen(self):
        models = frozen_models()
        self.assertEqual(len(models), 20)
        self.assertEqual(len(CHECKPOINT_HASHES), 20)
        counts = {
            protocol: sum(row["protocol"] == protocol for row in models)
            for protocol in PROTOCOL_VARIANTS
        }
        self.assertEqual(counts, {"cross-rx": 10, "cross-day": 10})
        self.assertEqual(TRAIN_SEEDS, (2024, 2025, 2026, 2027, 2028))
        self.assertEqual(SHOTS, (1, 5, 10, 15, 20))
        self.assertEqual(ITERATIONS, 100)

    def test_final_file_grid_contains_both_protocols_and_splits(self):
        files = frozen_data_files()
        self.assertEqual(len(files), 8)
        keys = {(row["protocol"], row["split"], row["kind"]) for row in files}
        expected = {
            (protocol, split, kind)
            for protocol in PROTOCOL_VARIANTS
            for split in ("train", "test")
            for kind in ("X", "Y")
        }
        self.assertEqual(keys, expected)

    def test_draft_manifest_is_deterministic_and_not_runnable(self):
        first = build_draft_manifest(compute_hashes=False)
        second = build_draft_manifest(compute_hashes=False)
        self.assertEqual(canonical_hash(first), canonical_hash(second))
        self.assertFalse(first["run_enabled"])
        self.assertFalse(RUN_ENABLED)
        self.assertEqual(first["expected_rows_per_protocol"], 5000)
        self.assertEqual(first["primary_statistical_unit"], "train_seed")

    def test_preparation_source_has_no_array_loader(self):
        source = (Path(__file__).parents[1] / "wisig_final_matrix_prepare.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("np.load", source)
        self.assertNotIn("load_data(", source)
        self.assertIn("RUN_ENABLED = False", source)


if __name__ == "__main__":
    unittest.main()
