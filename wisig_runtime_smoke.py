"""One-batch runtime gate for the strict WiSig P3MC protocols.

This script does not train an experiment or write checkpoints. It verifies that
the explicit held-out validation domain, 256-point CVTSLANet configuration, and
the three original P3MC pretext losses can complete one forward/backward pass.
"""

import argparse
import math
import os

import numpy as np
import torch

from models.AutomaticWeightedLoss import AutomaticWeightedLoss
from models.CVTSLANetFeature import create_model as create_encoder
from models.LinearClassifier import LinearClassifier
from models.ManifoldMixupLoss import ManifoldMixupLoss
from pretext import run_step
from utils.config import dataset_path_dict
from utils.get_dataset import get_pretrain_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("cross-rx", "cross-day"), required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--rot-num", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for the mixup smoke test")
    if args.rot_num < 2:
        raise ValueError("--rot-num must be at least 2")

    dataset_key = f"wisig-{args.protocol}"
    dataset_root = os.path.expanduser(dataset_path_dict[dataset_key]["linux"])
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Strict WiSig dataset directory not found: {dataset_root}")

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)

    feature_dim = 1024
    tsla_config = {
        "seq_len": 256,
        "patch_size": 32,
        "num_channels": 2,
        "emb_dim": 256,
        "depth": 3,
        "dropout_rate": 0.3,
    }
    config = {
        "random_seed": 2024,
        "device": device_name,
        "dataset": {
            "root": dataset_root,
            "type": "iq",
            "normalize": "power",
            "batch_size": args.batch_size,
            "ratio": 0.2,
            "num_classes": 90,
            "signal_length": 256,
        },
        "augmentation": {"awgn_enable": False, "awgn_snr_range": (0.0, 30.0)},
        "encoder": {"feature_dim": feature_dim, "TSLA_config": tsla_config},
        "rot_classifier": {"num_classes": args.rot_num},
        "mtl": {"item": ("rot_cls", "sei_cls", "mml"), "num": 3},
    }

    train_loader, val_loader = get_pretrain_dataloader(config)
    train_expected = len(np.load(
        os.path.join(dataset_root, "Y_train_90Class.npy"), mmap_mode="r"
    ))
    val_expected = len(np.load(
        os.path.join(dataset_root, "Y_val_90Class.npy"), mmap_mode="r"
    ))
    if len(train_loader.dataset) != train_expected or len(val_loader.dataset) != val_expected:
        raise AssertionError(
            "Pretraining loader did not use the full source train set and explicit held-out val set"
        )

    encoder = create_encoder(feature_dim=feature_dim, dtype="iq", **tsla_config).to(device)
    rot_classifier = LinearClassifier(feature_dim, args.rot_num).to(device)
    mixed_classifier = LinearClassifier(feature_dim, 90).to(device)
    mml = ManifoldMixupLoss(beta=2.0)
    mtl = AutomaticWeightedLoss(num=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    inputs = next(iter(train_loader))
    losses, metrics = run_step(
        config, inputs, device, encoder, rot_classifier, mixed_classifier,
        criterion, mml, mtl, lfdb=None, training=True
    )
    total_loss = losses[-1]
    if not all(math.isfinite(float(loss.detach().cpu())) for loss in losses):
        raise FloatingPointError(f"Non-finite smoke loss: {losses}")
    total_loss.backward()
    finite_gradients = [
        torch.isfinite(parameter.grad).all().item()
        for parameter in encoder.parameters()
        if parameter.grad is not None
    ]
    if not finite_gradients or not all(finite_gradients):
        raise FloatingPointError("Encoder gradients are absent or non-finite")

    print(f"PROTOCOL={args.protocol}")
    print(f"DATASET_ROOT={dataset_root}")
    print(f"DEVICE={device_name}")
    print(f"TRAIN_SAMPLES={len(train_loader.dataset)}")
    print(f"VAL_SAMPLES={len(val_loader.dataset)}")
    print(f"BATCH_SHAPE={tuple(inputs[0].shape)}")
    print("LOSSES=" + ",".join(f"{float(loss.detach().cpu()):.6f}" for loss in losses))
    metric_items = []
    for name, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        metric_items.append(f"{name}:{float(value):.4f}")
    print("METRICS=" + ",".join(metric_items))
    print("WISIG_RUNTIME_SMOKE: PASS")


if __name__ == "__main__":
    main()
