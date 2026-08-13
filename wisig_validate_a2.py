"""Validation-only paired A2 ablation for strict WiSig protocols."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.adaptive_distribution_calibration import METHODS, augment_support_features
from utils.config import dataset_path_dict, model_path_dict
from utils.get_dataset import load_data, power_normalize_fn
from utils.utils import create_model, load_encoder_weights, set_seed
from wisig_validate_lr import (
    TSLA_CONFIG,
    extract_features,
    load_validation_arrays,
    make_config,
    make_loader,
    resolve_device,
    sample_support,
    sha256,
)


class MMapIQDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if len(self.x) != len(self.y):
            raise ValueError("base X/Y length mismatch")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        sample = np.asarray(self.x[index], dtype=np.float32)
        if sample.shape[0] != 2:
            sample = sample.T
        maximum = np.max(np.square(sample[0]) + np.square(sample[1]))
        sample = sample / max(float(np.sqrt(maximum)), 1e-12)
        return torch.from_numpy(np.array(sample, copy=True)), int(self.y[index])


def parse_args():
    parser = argparse.ArgumentParser(description="Paired A2 validation ablation; final test unavailable.")
    parser.add_argument("--protocol", choices=("cross-rx", "cross-day"), required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--shots", type=int, nargs="+", default=(1, 5, 10, 15, 20))
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=2024)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--support-batch-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=METHODS)
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--mean-strength", type=float, default=0.2)
    parser.add_argument("--variance-strength", type=float, default=0.2)
    parser.add_argument("--gate-scale", type=float, default=5.0)
    parser.add_argument("--synthetic-per-class", type=int, default=50)
    parser.add_argument("--output-dir", default="runs/WiSig_validation_a2")
    parser.add_argument("--cache-dir", default="runs/WiSig_a2_base_stats")
    return parser.parse_args()


def atomic_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def base_paths(protocol):
    root = Path(os.path.expanduser(dataset_path_dict[f"wisig-{protocol}"]["linux"]))
    return root / "X_train_90Class.npy", root / "Y_train_90Class.npy"


def compute_or_load_base_statistics(args, encoder, device, checkpoint_hash):
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{args.protocol}_{checkpoint_hash[:12]}_diag_stats.npz"
    if cache_path.is_file():
        cached = np.load(cache_path)
        classes = cached["classes"]
        means = cached["means"]
        variances = cached["variances"]
        if not np.array_equal(classes, np.arange(90)):
            raise RuntimeError("cached base classes are not exactly 0..89")
        return classes, means, variances, cache_path, True

    x_path, y_path = base_paths(args.protocol)
    dataset = MMapIQDataset(x_path, y_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=device == "cuda")
    counts = np.zeros(90, dtype=np.int64)
    sums = np.zeros((90, args.feature_dim), dtype=np.float64)
    square_sums = np.zeros((90, args.feature_dim), dtype=np.float64)
    encoder.eval()
    with torch.inference_mode():
        for signals, labels in tqdm(loader, desc="Extracting base statistics"):
            features = encoder(signals.to(device, non_blocking=True)).cpu().numpy()
            labels = labels.numpy().astype(int)
            for class_id in np.unique(labels):
                current = features[labels == class_id].astype(np.float64)
                counts[class_id] += len(current)
                sums[class_id] += current.sum(axis=0)
                square_sums[class_id] += np.square(current).sum(axis=0)
    if np.any(counts < 2):
        raise RuntimeError(f"base classes with fewer than two samples: {np.flatnonzero(counts < 2)}")
    means = sums / counts[:, None]
    variances = (square_sums - counts[:, None] * np.square(means)) / (counts[:, None] - 1)
    variances = np.maximum(variances, 1e-4)
    temporary = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, classes=np.arange(90), means=means.astype(np.float32),
        variances=variances.astype(np.float32), counts=counts,
        checkpoint_sha256=checkpoint_hash,
    )
    temporary.replace(cache_path)
    return np.arange(90), means.astype(np.float32), variances.astype(np.float32), cache_path, False


def summarize(rows, methods, shots):
    summaries = []
    for method in methods:
        for shot in shots:
            values = np.asarray([
                float(row["accuracy"]) for row in rows
                if row["method"] == method and int(row["shot"]) == shot
            ])
            if not len(values):
                continue
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            summaries.append({
                "method": method, "shot": shot, "n": len(values),
                "mean_accuracy": f"{values.mean():.8f}",
                "std_accuracy": f"{std:.8f}",
                "ci95_half": f"{1.96 * std / math.sqrt(len(values)):.8f}",
            })
    return summaries


def main():
    args = parse_args()
    if args.iterations < 1 or any(shot < 1 for shot in args.shots):
        raise ValueError("iterations and shots must be positive")
    if not 0 <= args.mean_strength <= 1 or not 0 <= args.variance_strength <= 1:
        raise ValueError("calibration strengths must be in [0, 1]")
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = resolve_device(args.device)
    set_seed(args.base_seed)
    encoder = create_model(
        model_path_dict["CVTSLANet"], feature_dim=args.feature_dim,
        dtype="iq", **TSLA_CONFIG,
    )
    load_encoder_weights(encoder, str(checkpoint_path), device)
    encoder = encoder.to(device)
    checkpoint_hash = sha256(checkpoint_path)

    query_config = make_config(args, args.base_seed, args.shots[0], device)
    train_x, train_y, query_x, query_y = load_validation_arrays(query_config)
    query_features, query_labels = extract_features(
        encoder, make_loader(query_x, query_y, args.query_batch_size, device),
        device, "Extracting fixed validation queries",
    )
    _, base_means, base_variances, cache_path, cache_hit = compute_or_load_base_statistics(
        args, encoder, device, checkpoint_hash
    )

    calibration_config = {
        "methods": list(args.methods), "top_m": args.top_m,
        "mean_strength": args.mean_strength,
        "variance_strength": args.variance_strength,
        "gate_scale": args.gate_scale,
        "synthetic_per_class": args.synthetic_per_class,
    }
    calibration_hash = hashlib.sha256(
        json.dumps(calibration_config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    run_dir = Path(args.output_dir).expanduser().resolve() / (
        f"{args.protocol}_{checkpoint_hash[:12]}_{calibration_hash}"
    )
    detail_path = run_dir / "iterations.csv"
    rows = []
    if detail_path.is_file():
        with detail_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["checkpoint_sha256"] != checkpoint_hash or row["protocol"] != args.protocol:
                raise RuntimeError("existing A2 CSV does not match protocol/checkpoint")
    completed = {(row["method"], int(row["shot"]), int(row["iteration"])) for row in rows}
    fields = ("protocol", "role", "checkpoint_sha256", "method", "shot", "iteration", "seed", "accuracy")
    print(f"PROTOCOL={args.protocol}")
    print("ROLE=validation")
    print("QUERY_SPLIT=val")
    print(f"CHECKPOINT_SHA256={checkpoint_hash}")
    print(f"BASE_STATS_CACHE={cache_path}")
    print(f"BASE_STATS_CACHE_HIT={cache_hit}")
    print("METHODS=" + ",".join(args.methods))

    for shot in args.shots:
        for iteration in range(1, args.iterations + 1):
            seed = args.base_seed + iteration - 1
            missing = [method for method in args.methods if (method, shot, iteration) not in completed]
            if not missing:
                continue
            set_seed(seed)
            support_x, support_y = sample_support(train_x, train_y, shot, seed, num_classes=30)
            support_features, support_labels = extract_features(
                encoder, make_loader(support_x, support_y, args.support_batch_size, device),
                device, f"{shot}-shot iteration {iteration}/{args.iterations}",
            )
            for method in missing:
                classifier_x, classifier_y = augment_support_features(
                    support_features, support_labels, base_means, base_variances,
                    method, np.random.default_rng(seed + 10_000), top_m=args.top_m,
                    mean_strength=args.mean_strength,
                    variance_strength=args.variance_strength,
                    gate_scale=args.gate_scale,
                    synthetic_per_class=args.synthetic_per_class,
                )
                classifier = LogisticRegression(max_iter=1000, random_state=seed)
                classifier.fit(classifier_x, classifier_y)
                accuracy = float(classifier.score(query_features, query_labels) * 100.0)
                rows.append({
                    "protocol": args.protocol, "role": "validation",
                    "checkpoint_sha256": checkpoint_hash, "method": method,
                    "shot": shot, "iteration": iteration, "seed": seed,
                    "accuracy": f"{accuracy:.8f}",
                })
                completed.add((method, shot, iteration))
                print(f"METHOD={method} SHOT={shot} ITERATION={iteration}/{args.iterations} SEED={seed} ACC={accuracy:.4f}")
            rows.sort(key=lambda row: (row["method"], int(row["shot"]), int(row["iteration"])))
            atomic_csv(detail_path, fields, rows)
            atomic_csv(run_dir / "summary.csv", ("method", "shot", "n", "mean_accuracy", "std_accuracy", "ci95_half"), summarize(rows, args.methods, args.shots))

    summaries = summarize(rows, args.methods, args.shots)
    for row in summaries:
        if int(row["n"]) == args.iterations:
            print(
                f"SUMMARY METHOD={row['method']} SHOT={row['shot']} N={row['n']} "
                f"MEAN={float(row['mean_accuracy']):.4f} STD={float(row['std_accuracy']):.4f} "
                f"CI95_HALF={float(row['ci95_half']):.4f}"
            )
    metadata = vars(args).copy()
    metadata.update({
        "role": "validation", "query_split": "val",
        "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "base_stats_cache": str(cache_path),
        "calibration_config": calibration_config,
        "calibration_hash": calibration_hash,
    })
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"RESULT_DIR={run_dir}")
    print("WISIG_A2_VALIDATION: PASS")


if __name__ == "__main__":
    main()
