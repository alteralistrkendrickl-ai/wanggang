"""Short throughput benchmark for strict WiSig P3MC pretraining.

The benchmark runs only a small number of batches and writes no checkpoint. It
uses the intended CVTSLANet dimensions, all three original P3MC pretext losses,
and the explicit held-out validation domain to estimate a full epoch duration.
"""

import argparse
import math
import os
import statistics
import time

import torch
from torch.optim import AdamW

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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rot-num", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def summarize_times(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    args = parse_args()
    if args.batch_size < 2 or args.rot_num < 2:
        raise ValueError("batch-size and rot-num must both be at least 2")
    if args.warmup < 0 or args.steps < 1:
        raise ValueError("warmup must be non-negative and steps must be positive")

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
        "augmentation": {"awgn_enable": True, "awgn_snr_range": (0.0, 30.0)},
        "encoder": {"feature_dim": feature_dim, "TSLA_config": tsla_config},
        "rot_classifier": {"num_classes": args.rot_num},
        "mtl": {"item": ("rot_cls", "sei_cls", "mml"), "num": 3},
    }

    train_loader, val_loader = get_pretrain_dataloader(config)
    encoder = create_encoder(feature_dim=feature_dim, dtype="iq", **tsla_config).to(device)
    rot_classifier = LinearClassifier(feature_dim, args.rot_num).to(device)
    mixed_classifier = LinearClassifier(feature_dim, 90).to(device)
    mml = ManifoldMixupLoss(beta=2.0)
    mtl = AutomaticWeightedLoss(num=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    modules = (encoder, rot_classifier, mixed_classifier, mtl)
    optimizers = [AdamW(module.parameters(), lr=0.001, weight_decay=5e-4) for module in modules]

    train_iterator = iter(train_loader)
    for _ in range(args.warmup):
        inputs, train_iterator = next_batch(train_iterator, train_loader)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        losses, _ = run_step(
            config, inputs, device, encoder, rot_classifier, mixed_classifier,
            criterion, mml, mtl, lfdb=None, training=True
        )
        losses[-1].backward()
        for optimizer in optimizers:
            optimizer.step()
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_times = []
    for _ in range(args.steps):
        synchronize(device)
        started = time.perf_counter()
        inputs, train_iterator = next_batch(train_iterator, train_loader)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        losses, _ = run_step(
            config, inputs, device, encoder, rot_classifier, mixed_classifier,
            criterion, mml, mtl, lfdb=None, training=True
        )
        losses[-1].backward()
        for optimizer in optimizers:
            optimizer.step()
        synchronize(device)
        train_times.append(time.perf_counter() - started)

    for module in modules:
        module.eval()
    val_times = []
    val_iterator = iter(val_loader)
    with torch.no_grad():
        for _ in range(args.steps):
            synchronize(device)
            started = time.perf_counter()
            inputs, val_iterator = next_batch(val_iterator, val_loader)
            losses, _ = run_step(
                config, inputs, device, encoder, rot_classifier, mixed_classifier,
                criterion, mml, mtl, lfdb=None, training=False
            )
            if not torch.isfinite(losses[-1]):
                raise FloatingPointError("Non-finite validation loss")
            synchronize(device)
            val_times.append(time.perf_counter() - started)

    train_summary = summarize_times(train_times)
    val_summary = summarize_times(val_times)
    train_batches = math.ceil(len(train_loader.dataset) / args.batch_size)
    val_batches = math.ceil(len(val_loader.dataset) / args.batch_size)
    estimated_train_seconds = train_batches * train_summary["mean"]
    estimated_val_seconds = val_batches * val_summary["mean"]

    print(f"PROTOCOL={args.protocol}")
    print(f"DEVICE={device_name}")
    print(f"BATCH_SIZE={args.batch_size}")
    print(f"ROT_NUM={args.rot_num}")
    print(f"TRAIN_SAMPLES={len(train_loader.dataset)}")
    print(f"VAL_SAMPLES={len(val_loader.dataset)}")
    print(f"TRAIN_BATCHES={train_batches}")
    print(f"VAL_BATCHES={val_batches}")
    print("TRAIN_STEP_SECONDS=" + ",".join(f"{key}:{value:.6f}" for key, value in train_summary.items()))
    print("VAL_STEP_SECONDS=" + ",".join(f"{key}:{value:.6f}" for key, value in val_summary.items()))
    print(f"ESTIMATED_TRAIN_EPOCH_MINUTES={estimated_train_seconds / 60.0:.2f}")
    print(f"ESTIMATED_VAL_EPOCH_MINUTES={estimated_val_seconds / 60.0:.2f}")
    print(f"ESTIMATED_TOTAL_EPOCH_MINUTES={(estimated_train_seconds + estimated_val_seconds) / 60.0:.2f}")
    if device.type == "cuda":
        print(f"CUDA_PEAK_ALLOCATED_GB={torch.cuda.max_memory_allocated(device) / (1024 ** 3):.3f}")
        print(f"CUDA_PEAK_RESERVED_GB={torch.cuda.max_memory_reserved(device) / (1024 ** 3):.3f}")
        print(f"CUDA_DEVICE={torch.cuda.get_device_name(device)}")
    print("WISIG_RUNTIME_BENCHMARK: PASS")


if __name__ == "__main__":
    main()
