import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wisig_train_seed_robustness import (
    COMMON_ARGS, SEEDS, VARIANTS, acquire_manifest, build_command,
    canonical_hash, experiment_name, frozen_manifest,
)


class WiSigTrainSeedRobustnessTest(unittest.TestCase):
    def test_frozen_grid(self):
        self.assertEqual(SEEDS, (2025, 2026))
        self.assertEqual(VARIANTS, ("B0", "A1S", "A1C"))
        manifest = frozen_manifest(Path("/project"), "cross-rx")
        self.assertEqual(len(manifest["tasks"]), 6)
        self.assertFalse(manifest["final_test_accessed"])

    def test_commands_differ_only_by_frozen_variant_and_seed(self):
        root = Path("/project")
        b0 = build_command(root, "cross-day", "B0", 2025)
        a1s = build_command(root, "cross-day", "A1S", 2025)
        a1c = build_command(root, "cross-day", "A1C", 2025)
        self.assertNotIn("--a1_sampler", b0)
        self.assertIn("--a1_sampler", a1s)
        self.assertIn("--a1_enable", a1c)
        self.assertIn("0.1", a1c)
        self.assertTrue(all(value in b0 for value in COMMON_ARGS))

    def test_names_are_isolated(self):
        names = {
            experiment_name(protocol, variant, seed)
            for protocol in ("cross-rx", "cross-day")
            for variant in VARIANTS for seed in SEEDS
        }
        self.assertEqual(len(names), 12)
        self.assertTrue(all("seed20" in name for name in names))

    def test_manifest_is_exclusive_and_resumable(self):
        with TemporaryDirectory() as directory:
            state = Path(directory) / "state"
            manifest = frozen_manifest(Path("/project"), "cross-day")
            digest = acquire_manifest(state, manifest, resume=False)
            self.assertEqual(digest, canonical_hash(manifest))
            self.assertEqual(acquire_manifest(state, manifest, resume=True), digest)
            with self.assertRaises(RuntimeError):
                acquire_manifest(state, manifest, resume=False)


if __name__ == "__main__":
    unittest.main()
