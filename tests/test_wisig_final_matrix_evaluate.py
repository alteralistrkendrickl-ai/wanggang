import json
import math
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
            "runtime_contract": final.FROZEN_RUNTIME,
            "statistical_plan": final.STATISTICAL_PLAN,
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

    def test_frozen_manifest_paths_are_platform_independent_posix_text(self):
        paths = (
            final.DRAFT_PATH_TEXT,
            final.FROZEN_RUNTIME["output_dir"],
            final.FROZEN_RUNTIME["global_unseal_lock"],
        )
        self.assertTrue(all(path.startswith("/home/") for path in paths))
        self.assertTrue(all("\\" not in path for path in paths))

    def test_exclusive_unseal_and_exact_resume(self):
        manifest = self.synthetic_manifest()
        digest = final.canonical_hash(manifest)
        old = final.EXPECTED_FROZEN_MANIFEST_SHA256
        final.EXPECTED_FROZEN_MANIFEST_SHA256 = digest
        try:
            with tempfile.TemporaryDirectory() as directory:
                lock = Path(directory) / "global" / "UNSEAL_MANIFEST.json"
                self.assertEqual(
                    final.acquire_or_resume(
                        lock, manifest, final.FROZEN_RUNTIME, False
                    ),
                    digest,
                )
                self.assertEqual(
                    final.acquire_or_resume(
                        lock, manifest, final.FROZEN_RUNTIME, True
                    ),
                    digest,
                )
                with self.assertRaisesRegex(RuntimeError, "already unsealed"):
                    final.acquire_or_resume(
                        lock, manifest, final.FROZEN_RUNTIME, False
                    )
                changed = json.loads(json.dumps(manifest))
                changed["shots"] = [1]
                with self.assertRaises(RuntimeError):
                    final.acquire_or_resume(
                        lock, changed, final.FROZEN_RUNTIME, True
                    )
                changed_runtime = dict(final.FROZEN_RUNTIME)
                changed_runtime["batch_size"] = 128
                with self.assertRaisesRegex(RuntimeError, "Runtime contract"):
                    final.acquire_or_resume(
                        lock, manifest, changed_runtime, True
                    )
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
            base["checkpoint_sha256"] = model["checkpoint_sha256"]
            base["accuracy"] = "nan"
            final.atomic_write_csv(path, final.DETAIL_FIELDS, [base])
            with self.assertRaisesRegex(RuntimeError, "finite"):
                final.load_existing_rows(path, "cross-rx", manifest)

    def test_training_provenance_contract(self):
        config = {
            "random_seed": 2027, "epoch": 10, "threshold": 0,
            "dataset": {
                "name": "wisig-cross-rx", "type": "iq", "normalize": "power",
                "batch_size": 32, "signal_length": 256,
            },
            "augmentation": {"awgn_enable": False},
            "encoder": {
                "name": "CVTSLANet", "feature_dim": 1024,
                "TSLA_config": {"seq_len": 256, "patch_size": 32},
            },
            "lfdb": {"enabled": False},
            "a1": {
                "sampler_enabled": True, "loss_enabled": True,
                "domain_key": "RX", "weight": 0.1, "temperature": 0.1,
            },
        }
        self.assertEqual(
            final.validate_training_config(config, "cross-rx", "A1C", 2027),
            "explicit",
        )
        config["augmentation"]["awgn_enable"] = True
        with self.assertRaisesRegex(RuntimeError, "awgn_enable"):
            final.validate_training_config(config, "cross-rx", "A1C", 2027)

    def test_legacy_b0_without_a1_section_is_explicitly_audited(self):
        config = {
            "random_seed": 2024, "epoch": 10, "threshold": 0,
            "dataset": {
                "name": "wisig-cross-rx", "type": "iq", "normalize": "power",
                "batch_size": 32, "signal_length": 256,
            },
            "augmentation": {"awgn_enable": False},
            "encoder": {
                "name": "CVTSLANet", "feature_dim": 1024,
                "TSLA_config": {"seq_len": 256, "patch_size": 32},
            },
            "lfdb": {"enabled": False},
        }
        self.assertEqual(
            final.validate_training_config(config, "cross-rx", "B0", 2024),
            "legacy_absent_b0",
        )
        with self.assertRaisesRegex(RuntimeError, "explicit A1 configuration"):
            final.validate_training_config(config, "cross-rx", "A1C", 2024)

    def test_frozen_primary_and_holm_statistics(self):
        self.assertAlmostEqual(
            final._two_sided_t_pvalue(2.7764451051977987, 4), 0.05, places=8
        )
        manifest = self.synthetic_manifest()
        effects = []
        for protocol in final.PROTOCOLS:
            candidate = manifest["protocol_variants"][protocol][1]
            for train_seed in final.TRAIN_SEEDS:
                for shot in final.SHOTS:
                    effects.append({
                        "protocol": protocol, "candidate": candidate,
                        "train_seed": train_seed, "shot": shot,
                        "mean_difference": f"{train_seed - 2023:.8f}",
                    })
        shot = final.paired_summaries(effects, manifest)
        primary = final.primary_summaries(effects, manifest)
        self.assertEqual(len(shot), 10)
        self.assertEqual(len(primary), 2)
        self.assertTrue(all(math.isfinite(float(row["raw_p"])) for row in shot))
        self.assertTrue(all(row["holm_reject"] == "True" for row in primary))

    def test_result_evidence_hashes_every_frozen_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = (
                "cross-rx/iterations.csv", "cross-rx/model_summary.csv",
                "cross-day/iterations.csv", "cross-day/model_summary.csv",
                "train_seed_effects.csv", "paired_summary.csv", "primary_summary.csv",
            )
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            evidence_path, digest, evidence = final.build_result_evidence(root)
            self.assertEqual(len(evidence["files"]), 7)
            self.assertEqual(final.sha256(evidence_path), digest)

    def test_source_keeps_audit_before_array_loading(self):
        source = (Path(__file__).parents[1] / "wisig_final_matrix_evaluate.py").read_text(
            encoding="utf-8"
        )
        audit_position = source.index("audit_frozen_files(manifest, compute_hashes=True)")
        load_position = source.index("rows = run_all(")
        self.assertLess(audit_position, load_position)
        self.assertIn('if args.mode == "audit":', source)
        self.assertIn("READY_FOR_USER_AUTHORIZATION=True", source)
        self.assertNotIn('parser.add_argument("--output-dir"', source)
        self.assertIn("GLOBAL_UNSEAL_LOCK", source)
        self.assertIn("STATISTICAL_PLAN", source)


if __name__ == "__main__":
    unittest.main()
