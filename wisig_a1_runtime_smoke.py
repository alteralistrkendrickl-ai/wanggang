"""One structured WiSig A1 batch through the real encoder and loss path."""

import argparse
import math
import os

import numpy as np
import torch

from models.AutomaticWeightedLoss import AutomaticWeightedLoss
from models.CrossDomainSupConLoss import CrossDomainSupConLoss
from models.CVTSLANetFeature import create_model as create_encoder
from models.LinearClassifier import LinearClassifier
from models.ManifoldMixupLoss import ManifoldMixupLoss
from pretext import run_step
from utils.config import dataset_path_dict
from utils.get_dataset import get_pretrain_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("cross-rx", "cross-day"), required=True)
    parser.add_argument("--variant", choices=("A1-S", "A1-C"), required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_key = f"wisig-{args.protocol}"
    root = os.path.expanduser(dataset_path_dict[dataset_key]["linux"])
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    loss_enabled = args.variant == "A1-C"
    domain_key = "RX" if args.protocol == "cross-rx" else "DAY"
    tsla = {
        "seq_len": 256, "patch_size": 32, "num_channels": 2,
        "emb_dim": 256, "depth": 3, "dropout_rate": 0.3,
    }
    config = {
        "random_seed": 2024,
        "device": device_name,
        "dataset": {
            "root": root, "type": "iq", "normalize": "power",
            "batch_size": 32, "ratio": 0.2, "num_classes": 90,
            "signal_length": 256,
        },
        "augmentation": {"awgn_enable": False, "awgn_snr_range": (0.0, 30.0)},
        "encoder": {"feature_dim": 1024, "TSLA_config": tsla},
        "rot_classifier": {"num_classes": 8},
        "mtl": {"item": ("rot_cls", "sei_cls", "mml"), "num": 3},
        "a1": {
            "sampler_enabled": True, "loss_enabled": loss_enabled,
            "weight": 0.1, "temperature": 0.1, "domain_key": domain_key,
            "identities_per_batch": 8, "domains_per_identity": 2,
            "samples_per_domain": 2,
        },
    }
    train_loader, _ = get_pretrain_dataloader(config)
    inputs = next(iter(train_loader))
    labels = inputs[2].numpy()
    domains = inputs[3].numpy()
    identities = np.unique(labels)
    if len(inputs[0]) != 32 or len(identities) != 8:
        raise AssertionError("A1 batch is not 8 identities x 2 domains x 2 samples")
    for identity in identities:
        unique, counts = np.unique(domains[labels == identity], return_counts=True)
        if len(unique) != 2 or not np.array_equal(counts, np.array([2, 2])):
            raise AssertionError(f"Malformed A1 cell for identity {identity}")

    encoder = create_encoder(feature_dim=1024, dtype="iq", **tsla).to(device)
    rot_classifier = LinearClassifier(1024, 8).to(device)
    mixed_classifier = LinearClassifier(1024, 90).to(device)
    mtl = AutomaticWeightedLoss(num=3).to(device)
    a1_loss = CrossDomainSupConLoss(0.1).to(device) if loss_enabled else None
    losses, _ = run_step(
        config, inputs, device, encoder, rot_classifier, mixed_classifier,
        torch.nn.CrossEntropyLoss(), ManifoldMixupLoss(beta=2.0), mtl,
        a1_loss_fn=a1_loss, training=True,
    )
    if not all(math.isfinite(float(loss.detach().cpu())) for loss in losses):
        raise FloatingPointError("Non-finite A1 smoke loss")
    losses[-1].backward()
    gradients = [
        torch.isfinite(parameter.grad).all().item()
        for parameter in encoder.parameters() if parameter.grad is not None
    ]
    if not gradients or not all(gradients):
        raise FloatingPointError("Missing or non-finite encoder gradients")
    print(f"PROTOCOL={args.protocol}")
    print(f"VARIANT={args.variant}")
    print(f"DEVICE={device_name}")
    print(f"DOMAIN_KEY={domain_key}")
    print(f"BATCH_SHAPE={tuple(inputs[0].shape)}")
    print(f"IDENTITIES={len(identities)}")
    print("LOSSES=" + ",".join(f"{float(x.detach().cpu()):.6f}" for x in losses))
    print("WISIG_A1_RUNTIME_SMOKE: PASS")


if __name__ == "__main__":
    main()
