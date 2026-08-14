import json
import tempfile
import unittest
from pathlib import Path

import wisig_final_matrix_evaluate as final


class WiSigFinalMatrixEvaluateTest(unittest.TestCase):
    def synthetic_manifest(self):
        models = []
        for protocol, variants in {
            "cross-rx": ("B0", "A1C"), "cross-day": ("B0", "A1S")
        }.items():
            for train_seed in final.TRAIN_SEEDS:
                for variant in variants:
                    models.append({
                        "protocol": protocol, "variant": variant,
                        "train_seed": train_seed, "checkpoint": "/unused",
                        "checkpoint_sha256": f"{protocol}-{variant}-{train_seed}",
                    })
        return {
            "schema": "wisig-final-paired-matrix-v2",
            "protocol_variants": {
                "cross-rx": ["B0", "A1C"], "cross-day": ["B0", "A1S"]
            },
            "train_seeds": list(final.TRAIN_SEEDS),
            "shots": list(final.SHOTS), "iterations": final.ITERATIONS,
            "support_base_seed": final.SUPPORT_BASE_SEED,
            "expected_rows_per_protocol": final.EXPECTED_ROWS_PER_PROTOCOL,
            "models": models,
            "data_files": [
                {"protocol": protocol, "split": split, "kind": kind,
                 "path": "/unused", "size_bytes": 1, "sha256": "unused"}
                for protocol in final.PROTOCOLS
                for split in ("train", "test") for kind in ("X", "Y")
            ],
        }

    def test_exact_grid_has_5000_rows_per_protocol(self):
        manifest = self.synthetic_manifest()
        final.validate_frozen_structure(manifest)
        self.assertEqual(len(final.expected_row_keys("cross-rx", manifest)), 5000)
        self.assertEqual(len(final.expected_row_keys("cross-day", manifest)), 5000)

    def test_exclusive_unseal_and_exact_resume(self):
        manifest = self.synthetic_manifest()
        digest = final.canonical_hash(manifest)
        old = final.EXPECTED_FROZEN_MANIFEST_SHA256
        final.EXPECTED_FROZEN_MANIFEST_SHA256 = digest
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.assertEqual(final.acquire_or_resume(root, manifest, False), digest)
                self.assertEqual(final.acquire_or_resume(root, manifest, True), digest)
                with self.assertRaisesRegex(RuntimeError, "already unsealed"):
                    final.acquire_or_resume(root, manifest, False)
                changed = json.loads(json.dumps(manifest))
                changed["shots"] = [1]
                with self.assertRaises(RuntimeError):
                    final.acquire_or_resume(root, changed, True)
        finally:
            final.EXPECTED_FROZEN_MANIFEST_SHA256 = old

    def test_existing_rows_reject_wrong_role_and_checkpoint(self):
        manifest = self.synthetic_manifest()
        model = next(row for row in manifest["models"] if (
            row["protocol"], row["variant"], row["train_seed"]
        ) == ("cross-rx", "B0", 2024))
        base = {
            "protocol": "cross-rx", "role": "validation", "variant": "B0",
            "train_seed": 2024, "checkpoint_sha256": model["checkpoint_sha256"],
            "shot": 1, "iteration": 1, "support_seed": 2024,
            "accuracy": "50.0", "source": "computed",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iterations.csv"
            final.atomic_write_csv(path, final.DETAIL_FIELDS, [base])
            with self.assertRaisesRegex(RuntimeError, "wrong protocol or role"):
                final.load_existing_rows(path, "cross-rx", manifest)
            base["role"] = "final"
            base["checkpoint_sha256"] = "wrong"
            final.atomic_write_csv(path, final.DETAIL_FIELDS, [base])
            with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
                final.load_existing_rows(path, "cross-rx", manifest)

    def test_source_keeps_audit_before_array_loading(self):
        source = (Path(__file__).parents[1] / "wisig_final_matrix_evaluate.py").read_text(
            encoding="utf-8"
        )
        audit_position = source.index("audit_frozen_files(manifest, compute_hashes=True)")
        load_position = source.index("rows = run_all(")
        self.assertLess(audit_position, load_position)
        self.assertIn('if args.mode == "audit":', source)
        self.assertIn("READY_FOR_USER_AUTHORIZATION=True", source)


if __name__ == "__main__":
    unittest.main()
