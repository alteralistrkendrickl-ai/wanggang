import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wisig_final_evaluate import (
    BASE_SEED, CHECKPOINTS, ITERATIONS, ROUTES, SHOTS,
    acquire_or_resume, canonical_hash, frozen_manifest, load_existing_rows,
)


class WiSigFinalEvaluateTest(unittest.TestCase):
    def manifest(self, directory, protocol="cross-day"):
        return frozen_manifest(protocol, Path(directory), Path(directory) / "final-root")

    def test_frozen_routes_are_exact(self):
        self.assertEqual(tuple(ROUTES["cross-rx"]), SHOTS)
        self.assertEqual(set(ROUTES["cross-rx"].values()), {"A1C"})
        self.assertEqual(
            ROUTES["cross-day"], {1: "A1C", 5: "A1C", 10: "A1S", 15: "A1S", 20: "B0"}
        )
        self.assertEqual(ITERATIONS, 100)
        self.assertEqual(BASE_SEED, 2024)

    def test_manifest_hash_is_deterministic(self):
        with TemporaryDirectory() as directory:
            first = self.manifest(directory)
            second = self.manifest(directory)
            self.assertEqual(canonical_hash(first), canonical_hash(second))

    def test_exclusive_unseal_and_exact_resume(self):
        with TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            manifest = self.manifest(directory)
            digest = acquire_or_resume(run_dir, manifest, resume=False)
            self.assertEqual(digest, canonical_hash(manifest))
            self.assertEqual(acquire_or_resume(run_dir, manifest, resume=True), digest)
            with self.assertRaises(RuntimeError):
                acquire_or_resume(run_dir, manifest, resume=False)
            changed = json.loads(json.dumps(manifest)); changed["base_seed"] += 1
            with self.assertRaises(RuntimeError):
                acquire_or_resume(run_dir, changed, resume=True)

    def test_existing_csv_rejects_wrong_route(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "iterations.csv"
            path.write_text(
                "protocol,role,checkpoint,checkpoint_sha256,shot,iteration,seed,accuracy,source\n"
                f"cross-day,final,A1S,{CHECKPOINTS['cross-day']['A1S'][1]},1,1,2024,50.0,computed\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                load_existing_rows(path, self.manifest(directory))


if __name__ == "__main__":
    unittest.main()
