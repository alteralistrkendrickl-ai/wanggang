import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from models.FSSEILoss import FSSEISTCLoss
from models.FSSEISTCFeature import FSSEISTCEncoder
from wisig_train_fssei import WiSigMetricDataset, experiment_name, normalize_iq
from wisig_validate_lr import resolve_checkpoint


class WiSigFSSEIBaselineTest(unittest.TestCase):
    def test_encoder_accepts_256_iq_and_backpropagates(self):
        encoder = FSSEISTCEncoder(feature_dim=32)
        inputs = torch.randn(4, 2, 256, requires_grad=True)
        features = encoder(inputs)
        self.assertEqual(features.shape, (4, 32))
        features.mean().backward()
        self.assertIsNotNone(inputs.grad)

    def test_metric_objective_is_finite(self):
        objective = FSSEISTCLoss(num_classes=4, feature_dim=16)
        embeddings = torch.randn(8, 16, requires_grad=True)
        logits = torch.randn(8, 4, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        losses = objective(logits, embeddings, labels)
        self.assertTrue(all(torch.isfinite(loss) for loss in losses))
        losses[0].backward()
        self.assertIsNotNone(embeddings.grad)

    def test_single_identity_validation_batch_has_zero_triplet(self):
        objective = FSSEISTCLoss(num_classes=4, feature_dim=8)
        embeddings = torch.randn(3, 8)
        logits = torch.randn(3, 4)
        labels = torch.zeros(3, dtype=torch.long)
        _, _, triplet, _ = objective(logits, embeddings, labels)
        self.assertEqual(float(triplet), 0.0)

    def test_dataset_normalizes_and_transposes_iq(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            samples = np.zeros((8, 256, 2), dtype=np.float32)
            samples[:, :, 0] = 3.0
            samples[:, :, 1] = 4.0
            labels = np.repeat(np.arange(4), 2)
            np.save(root / "x.npy", samples)
            np.save(root / "y.npy", labels)
            dataset = WiSigMetricDataset(root / "x.npy", root / "y.npy", expected_classes=4)
            sample, label = dataset[0]
        self.assertEqual(sample.shape, (2, 256))
        self.assertAlmostEqual(float(torch.sqrt(sample[0, 0] ** 2 + sample[1, 0] ** 2)), 1.0)
        self.assertEqual(int(label), 0)

    def test_zero_sample_normalization_is_finite(self):
        normalized = normalize_iq(np.zeros((2, 256), dtype=np.float32))
        self.assertTrue(np.isfinite(normalized).all())

    def test_names_are_protocol_seed_and_budget_specific(self):
        self.assertEqual(
            experiment_name("cross-rx", 10, 2024),
            "FSSEI_STC_wisig-cross-rx_iq_powerNorm_10e_seed2024",
        )

    def test_fssei_validation_requires_explicit_checkpoint(self):
        args = SimpleNamespace(encoder="fssei-stc", checkpoint_path="")
        with self.assertRaises(ValueError):
            resolve_checkpoint(args)


if __name__ == "__main__":
    unittest.main()
