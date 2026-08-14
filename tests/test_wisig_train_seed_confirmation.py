import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wisig_train_seed_confirmation import (
    COMMON_ARGS,
    PROTOCOL_VARIANTS,
    SEEDS,
    acquire_manifest,
    build_command,
    canonical_hash,
    checkpoint_hashes,
    experiment_name,
    frozen_manifest,
    verified_completed_task,
)


class WiSigTrainSeedConfirmationTest(unittest.TestCase):
    def test_frozen_confirmation_grid(self):
        self.assertEqual(SEEDS, (2027, 2028))
        self.assertEqual(PROTOCOL_VARIANTS["cross-rx"], ("B0", "A1C"))
        self.assertEqual(PROTOCOL_VARIANTS["cross-day"], ("B0", "A1S"))
        for protocol in PROTOCOL_VARIANTS:
            manifest = frozen_manifest(Path("/project"), protocol)
            self.assertEqual(len(manifest["tasks"]), 4)
            self.assertFalse(manifest["final_test_accessed"])
            self.assertTrue(
                manifest["selection_rule"]["selected_before_confirmation_seeds"]
            )

    def test_commands_preserve_frozen_training_budget(self):
        root = Path("/project")
        rx_b0 = build_command(root, "cross-rx", "B0", 2027)
        rx_a1c = build_command(root, "cross-rx", "A1C", 2027)
        day_a1s = build_command(root, "cross-day", "A1S", 2028)
        self.assertNotIn("--a1_enable", rx_b0)
        self.assertIn("--a1_enable", rx_a1c)
        self.assertIn("--a1_sampler", day_a1s)
        self.assertIn("0.1", rx_a1c)
        self.assertTrue(all(value in rx_b0 for value in COMMON_ARGS))

    def test_names_are_isolated_and_unique(self):
        names = {
            experiment_name(protocol, variant, seed)
            for protocol, variants in PROTOCOL_VARIANTS.items()
            for variant in variants
            for seed in SEEDS
        }
        self.assertEqual(len(names), 8)
        self.assertTrue(all("fair10e_seed20" in name for name in names))

    def test_manifest_is_exclusive_and_resumable(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            manifest = frozen_manifest(Path("/project"), "cross-day")
            digest = acquire_manifest(state, manifest, resume=False)
            self.assertEqual(digest, canonical_hash(manifest))
            self.assertEqual(acquire_manifest(state, manifest, resume=True), digest)
            with self.assertRaises(RuntimeError):
                acquire_manifest(state, manifest, resume=False)

    def test_interrupted_outputs_are_not_mistaken_for_complete(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            task = {"output_dir": str(output), "experiment_name": "synthetic"}
            (output / "best_encoder.pth").write_bytes(b"best")
            (output / "final_encoder.pth").write_bytes(b"final")
            hashes = checkpoint_hashes(task)
            self.assertIsNotNone(hashes)
            self.assertIsNone(
                verified_completed_task(task, {"state": "running"})
            )
            self.assertEqual(
                verified_completed_task(
                    task, {"state": "complete", "checkpoints": hashes}
                ),
                hashes,
            )


if __name__ == "__main__":
    unittest.main()
