"""Validation-only few-shot LR evaluation for strict WiSig protocols.

This entry point intentionally has no final-test mode.  Model/checkpoint and
few-shot choices must be made on ``validation_novel/val`` before a separate,
one-time final evaluation is enabled.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.config import dataset_path_dict, model_path_dict
from utils.get_dataset import load_data, power_normalize_fn
from utils.utils import create_model, load_encoder_weights, set_seed


TSLA_CONFIG = {
    "seq_len": 256,
    "patch_size": 32,
    "num_channels": 2,
    "emb_dim": 256,
    "depth": 3,
    "dropout_rate": 0.3,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a P3MC encoder on strict WiSig validation identities."
    )
    parser.add_argument("--protocol", choices=("cross-rx", "cross-day"), required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--shots", type=int, nargs="+", default=(1, 5, 10, 15, 20))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2024)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--support-batch-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="runs/WiSig_validation_lr")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return requested


def resolve_checkpoint(args):
    if args.checkpoint_path:
        return Path(args.checkpoint_path).expanduser().resolve()
    base_key = f"wisig-{args.protocol}"
    pretrain_name = dataset_path_dict[base_key]["name"]
    experiment = f"CVTSLANet_{pretrain_name}_iq_powerNorm"
    return (
        Path(__file__).resolve().parent
        / "runs"
        / "Pretext_random_rot"
        / experiment
        / f"{args.checkpoint}_encoder.pth"
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_config(args, seed, shot, device):
    dataset_key = f"wisig-{args.protocol}-validation-30"
    entry = dataset_path_dict[dataset_key]
    root = os.path.expanduser(entry["linux"])
    if entry.get("query_split") != "val" or Path(root).name != "validation_novel":
        raise RuntimeError(
            f"Refusing non-validation route: root={root!r}, "
            f"query_split={entry.get('query_split')!r}"
        )
    return {
        "random_seed": seed,
        "device": device,
        "dataset": {
            "root": root,
            "type": "iq",
            "normalize": "power",
            "train_batch_size": args.support_batch_size,
            "test_batch_size": args.query_batch_size,
            "num_classes": 30,
            "signal_length": 256,
            "shot": shot,
            "snr": None,
            "query_split": "val",
        },
    }


def load_validation_arrays(config):
    dataset = config["dataset"]
    train_x, train_y = load_data(
        dataset["root"],
        dataset["num_classes"],
        "train",
        signal_length=dataset["signal_length"],
    )
    query_x, query_y = load_data(
        dataset["root"],
        dataset["num_classes"],
        dataset["query_split"],
        signal_length=dataset["signal_length"],
    )
    expected_labels = np.arange(dataset["num_classes"])
    if not np.array_equal(np.unique(train_y).astype(int), expected_labels):
        raise RuntimeError("Validation support labels are not exactly 0..29")
    if not np.array_equal(np.unique(query_y).astype(int), expected_labels):
        raise RuntimeError("Validation query labels are not exactly 0..29")
    return train_x, train_y, power_normalize_fn(query_x), query_y.astype(np.uint8)


def sample_support(train_x, train_y, shot, seed, num_classes):
    rng = random.Random(seed)
    selected = []
    for class_id in range(num_classes):
        candidates = np.flatnonzero(train_y == class_id).tolist()
        if len(candidates) < shot:
            raise ValueError(
                f"Class {class_id} contains {len(candidates)} samples, "
                f"fewer than requested {shot}-shot"
            )
        selected.extend(rng.sample(candidates, shot))
    support_x = power_normalize_fn(train_x[selected])
    support_y = train_y[selected].astype(np.uint8)
    return support_x, support_y


def make_loader(x, y, batch_size, device):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device == "cuda",
    )


def extract_features(encoder, dataloader, device, description):
    features = []
    labels = []
    encoder.eval()
    with torch.inference_mode():
        for inputs, targets in tqdm(dataloader, desc=description, leave=False):
            features.append(encoder(inputs.to(device, non_blocking=True)).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not args.shots or any(shot < 1 for shot in args.shots):
        raise ValueError("shots must contain positive integers")

    device = resolve_device(args.device)
    checkpoint_path = resolve_checkpoint(args)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    set_seed(args.base_seed)
    encoder = create_model(
        model_path_dict["CVTSLANet"],
        feature_dim=args.feature_dim,
        dtype="iq",
        **TSLA_CONFIG,
    )
    load_encoder_weights(encoder, str(checkpoint_path), device)
    encoder = encoder.to(device)

    # Load each validation array once. This also prevents an iteration loop from
    # repeatedly re-reading the fixed query file.
    query_config = make_config(args, args.base_seed, args.shots[0], device)
    train_x, train_y, query_x, query_y = load_validation_arrays(query_config)
    query_loader = make_loader(
        query_x, query_y, args.query_batch_size, device
    )
    query_features, query_labels = extract_features(
        encoder, query_loader, device, "Extracting fixed validation queries"
    )

    checkpoint_hash = sha256(checkpoint_path)
    output_root = Path(args.output_dir).expanduser().resolve()
    run_name = f"{args.protocol}_{args.checkpoint}_{checkpoint_hash[:12]}"
    run_dir = output_root / run_name
    detail_rows = []
    summary_rows = []

    print(f"PROTOCOL={args.protocol}")
    print("ROLE=validation")
    print(f"QUERY_SPLIT={query_config['dataset']['query_split']}")
    print(f"DATASET_ROOT={query_config['dataset']['root']}")
    print(f"CHECKPOINT={checkpoint_path}")
    print(f"CHECKPOINT_SHA256={checkpoint_hash}")
    print(f"DEVICE={device}")
    print(f"QUERY_SAMPLES={len(query_labels)}")

    for shot in args.shots:
        accuracies = []
        for iteration in range(args.iterations):
            seed = args.base_seed + iteration
            set_seed(seed)
            support_x, support_y = sample_support(
                train_x, train_y, shot, seed, num_classes=30
            )
            support_loader = make_loader(
                support_x, support_y, args.support_batch_size, device
            )
            support_features, support_labels = extract_features(
                encoder,
                support_loader,
                device,
                f"{shot}-shot iteration {iteration + 1}/{args.iterations}",
            )
            classifier = LogisticRegression(max_iter=1000, random_state=seed)
            classifier.fit(support_features, support_labels)
            accuracy = float(classifier.score(query_features, query_labels) * 100.0)
            accuracies.append(accuracy)
            detail_rows.append(
                {
                    "protocol": args.protocol,
                    "role": "validation",
                    "checkpoint": args.checkpoint,
                    "checkpoint_sha256": checkpoint_hash,
                    "shot": shot,
                    "iteration": iteration + 1,
                    "seed": seed,
                    "accuracy": f"{accuracy:.8f}",
                }
            )
            print(
                f"SHOT={shot} ITERATION={iteration + 1}/{args.iterations} "
                f"SEED={seed} ACC={accuracy:.4f}"
            )

        values = np.asarray(accuracies, dtype=np.float64)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(values))
        summary = {
            "protocol": args.protocol,
            "role": "validation",
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_hash,
            "shot": shot,
            "n": len(values),
            "mean_accuracy": f"{values.mean():.8f}",
            "std_accuracy": f"{std:.8f}",
            "ci95_half": f"{ci95:.8f}",
        }
        summary_rows.append(summary)
        print(
            f"SUMMARY SHOT={shot} N={len(values)} MEAN={values.mean():.4f} "
            f"STD={std:.4f} CI95_HALF={ci95:.4f}"
        )

    write_csv(
        run_dir / "iterations.csv",
        (
            "protocol",
            "role",
            "checkpoint",
            "checkpoint_sha256",
            "shot",
            "iteration",
            "seed",
            "accuracy",
        ),
        detail_rows,
    )
    write_csv(
        run_dir / "summary.csv",
        (
            "protocol",
            "role",
            "checkpoint",
            "checkpoint_sha256",
            "shot",
            "n",
            "mean_accuracy",
            "std_accuracy",
            "ci95_half",
        ),
        summary_rows,
    )
    metadata = {
        "protocol": args.protocol,
        "role": "validation",
        "query_split": "val",
        "dataset_root": query_config["dataset"]["root"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "shots": args.shots,
        "iterations": args.iterations,
        "base_seed": args.base_seed,
        "feature_dim": args.feature_dim,
        "tsla_config": TSLA_CONFIG,
        "device": device,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_DIR={run_dir}")
    print("WISIG_VALIDATION_LR: PASS")


if __name__ == "__main__":
    main()
