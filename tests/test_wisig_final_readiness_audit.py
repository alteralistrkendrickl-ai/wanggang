import tempfile
import unittest
from pathlib import Path

from wisig_final_readiness_audit import (
    SHOTS,
    aggregate_differences,
    experiment_name,
    validate_rows,
)


class WiSigFinalReadinessAuditTest(unittest.TestCase):
    def test_frozen_experiment_names(self):
        self.assertEqual(
            experiment_name("cross-rx", "B0", 2024),
            "CVTSLANet_wisig-cross-rx_iq_powerNorm_fair10e",
        )
        self.assertEqual(
            experiment_name("cross-rx", "A1C", 2028),
            "CVTSLANet_wisig-cross-rx_iq_powerNorm_A1C_fair10e_seed2028",
        )
        self.assertEqual(
            experiment_name("cross-day", "A1S", 2027),
            "CVTSLANet_wisig-cross-day_iq_powerNorm_A1S_fair10e_seed2027",
        )

    def test_validation_grid_is_strict_and_complete(self):
        digest = "a" * 64
        rows = []
        for shot in SHOTS:
            for iteration in range(1, 101):
                rows.append(
                    {
                        "protocol": "cross-rx",
                        "role": "validation",
                        "checkpoint_sha256": digest,
                        "shot": str(shot),
                        "iteration": str(iteration),
                        "seed": str(2023 + iteration),
                        "accuracy": str(shot + iteration / 1000),
                    }
                )
        values = validate_rows(rows, "cross-rx", digest)
        self.assertEqual(len(values), 500)
        rows[-1]["role"] = "final"
        with self.assertRaisesRegex(RuntimeError, "Non-validation"):
            validate_rows(rows, "cross-rx", digest)

    def test_pairing_uses_train_seed_as_statistical_unit(self):
        results = {}
        variants = {"cross-rx": ("B0", "A1C"), "cross-day": ("B0", "A1S")}
        for protocol, (baseline, candidate) in variants.items():
            for train_seed in range(2024, 2029):
                base = {}
                method = {}
                for shot in SHOTS:
                    for iteration in range(1, 101):
                        base[(shot, iteration)] = 50.0
                        method[(shot, iteration)] = 51.0 + (train_seed - 2024)
                results[(protocol, baseline, train_seed)] = base
                results[(protocol, candidate, train_seed)] = method
        summaries = aggregate_differences(results)
        self.assertEqual(len(summaries), 10)
        self.assertEqual(summaries[0]["train_seed_differences"], [1, 2, 3, 4, 5])
        self.assertAlmostEqual(summaries[0]["mean_difference"], 3.0)

    def test_source_has_no_final_array_loader(self):
        source = (Path(__file__).parents[1] / "wisig_final_readiness_audit.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("np.load", source)
        self.assertNotIn("X_test_", source)
        self.assertNotIn("Y_test_", source)


if __name__ == "__main__":
    unittest.main()
