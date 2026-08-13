"""Mechanism diagnostics on validation-novel identities only.

The final test split is deliberately not addressable. Frozen encoders are
compared on identical source-train and held-out-validation samples.
"""

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.config import dataset_path_dict, model_path_dict
from utils.get_dataset import power_normalize_fn
from utils.utils import create_model, load_encoder_weights, set_seed


TSLA_CONFIG = {
    "seq_len": 256, "patch_size": 32, "num_channels": 2,
    "emb_dim": 256, "depth": 3, "dropout_rate": 0.3,
}


def parse_args():
    parser = argparse.ArgumentParser(description="A1 validation-novel diagnostics")
    parser.add_argument("--protocol", choices=("cross-rx", "cross-day"), required=True)
    parser.add_argument("--checkpoint", action="append", nargs=2,
                        metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--samples-per-source-cell", type=int, default=5)
    parser.add_argument("--target-samples-per-identity", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="runs/WiSig_A1_validation_novel_mechanism")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def balanced_source_indices(labels, domains, samples_per_cell, seed):
    if samples_per_cell < 1:
        raise ValueError("samples-per-source-cell must be positive")
    rng = random.Random(seed)
    selected = []
    for identity in sorted(np.unique(labels).astype(int)):
        identity_domains = sorted(np.unique(domains[labels == identity]).astype(int))
        if not identity_domains:
            raise ValueError(f"Identity {identity} has no source domain")
        for domain in identity_domains:
            candidates = np.flatnonzero((labels == identity) & (domains == domain)).tolist()
            if len(candidates) < samples_per_cell:
                raise ValueError(
                    f"Identity/domain cell ({identity}, {domain}) has {len(candidates)} samples"
                )
            selected.extend(rng.sample(candidates, samples_per_cell))
    return np.asarray(selected, dtype=np.int64)


def balanced_target_indices(labels, samples_per_identity, seed):
    if samples_per_identity < 1:
        raise ValueError("target-samples-per-identity must be positive")
    rng = random.Random(seed + 1)
    selected = []
    for identity in sorted(np.unique(labels).astype(int)):
        candidates = np.flatnonzero(labels == identity).tolist()
        if len(candidates) < samples_per_identity:
            raise ValueError(
                f"Target identity {identity} has {len(candidates)} samples"
            )
        selected.extend(rng.sample(candidates, samples_per_identity))
    return np.asarray(selected, dtype=np.int64)


def source_to_target_identity_probe(source_features, source_labels,
                                    target_features, target_labels, seed):
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
    )
    model.fit(source_features, source_labels)
    pred = model.predict(target_features)
    return {
        "target_identity_accuracy": float(accuracy_score(target_labels, pred)),
        "target_identity_balanced_accuracy": float(
            balanced_accuracy_score(target_labels, pred)
        ),
        "identity_chance": 1.0 / len(np.unique(target_labels)),
    }


def identity_disjoint_source_target_probe(source_features, source_labels,
                                          target_features, target_labels, seed):
    identities = np.unique(source_labels)
    if not np.array_equal(identities, np.unique(target_labels)):
        raise ValueError("Source and target identity sets differ")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(identities)
    cut = max(1, int(round(0.7 * len(shuffled))))
    train_ids, test_ids = shuffled[:cut], shuffled[cut:]

    def balanced_side(features, labels, allowed_ids):
        parts, ys = [], []
        for identity in allowed_ids:
            rows = features[labels == identity]
            parts.append(rows)
            ys.append(np.full(len(rows), identity, dtype=int))
        return np.concatenate(parts), np.concatenate(ys)

    source_train, source_train_y = balanced_side(
        source_features, source_labels, train_ids
    )
    target_train, target_train_y = balanced_side(
        target_features, target_labels, train_ids
    )
    per_identity = min(
        min(np.sum(source_train_y == identity) for identity in train_ids),
        min(np.sum(target_train_y == identity) for identity in train_ids),
    )
    train_x, train_domain = [], []
    local_rng = np.random.default_rng(seed + 17)
    for features, labels, domain_label in (
        (source_train, source_train_y, 0), (target_train, target_train_y, 1)
    ):
        for identity in train_ids:
            rows = features[labels == identity]
            chosen = local_rng.choice(len(rows), size=per_identity, replace=False)
            train_x.append(rows[chosen])
            train_domain.append(np.full(per_identity, domain_label, dtype=int))
    train_x = np.concatenate(train_x)
    train_domain = np.concatenate(train_domain)

    test_x, test_domain = [], []
    for identity in test_ids:
        source_rows = source_features[source_labels == identity]
        target_rows = target_features[target_labels == identity]
        count = min(len(source_rows), len(target_rows))
        source_choice = local_rng.choice(len(source_rows), size=count, replace=False)
        target_choice = local_rng.choice(len(target_rows), size=count, replace=False)
        test_x.extend((source_rows[source_choice], target_rows[target_choice]))
        test_domain.extend((np.zeros(count, dtype=int), np.ones(count, dtype=int)))
    test_x = np.concatenate(test_x)
    test_domain = np.concatenate(test_domain)

    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
    )
    model.fit(train_x, train_domain)
    pred = model.predict(test_x)
    return {
        "source_target_accuracy": float(accuracy_score(test_domain, pred)),
        "source_target_balanced_accuracy": float(
            balanced_accuracy_score(test_domain, pred)
        ),
        "source_target_chance": 0.5,
        "domain_probe_train_identities": int(len(train_ids)),
        "domain_probe_test_identities": int(len(test_ids)),
    }


def source_target_distance_metrics(source_features, source_labels,
                                   target_features, target_labels):
    def normalized_centroid(rows):
        rows = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        centroid = rows.mean(axis=0)
        return centroid / max(np.linalg.norm(centroid), 1e-12)

    identities = sorted(np.unique(source_labels).astype(int))
    source_centroids, target_centroids, within = [], [], []
    for identity in identities:
        source_centroid = normalized_centroid(source_features[source_labels == identity])
        target_centroid = normalized_centroid(target_features[target_labels == identity])
        source_centroids.append(source_centroid)
        target_centroids.append(target_centroid)
        within.append(1.0 - float(np.dot(source_centroid, target_centroid)))
    target_centroids = np.asarray(target_centroids)
    between = []
    for i in range(len(target_centroids)):
        for j in range(i + 1, len(target_centroids)):
            between.append(1.0 - float(np.dot(target_centroids[i], target_centroids[j])))
    within_mean = float(np.mean(within))
    between_mean = float(np.mean(between))
    return {
        "same_identity_source_target_cosine_distance": within_mean,
        "different_identity_target_cosine_distance": between_mean,
        "target_separation_ratio": between_mean / max(within_mean, 1e-12),
    }


def extract_features(checkpoint, x, batch_size, device):
    encoder = create_model(
        model_path_dict["CVTSLANet"], feature_dim=1024, dtype="iq", **TSLA_CONFIG
    )
    load_encoder_weights(encoder, str(checkpoint), device)
    encoder = encoder.to(device).eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False, pin_memory=device == "cuda",
    )
    parts = []
    with torch.inference_mode():
        for (inputs,) in tqdm(loader, desc=f"Features {checkpoint.name}", leave=False):
            parts.append(encoder(inputs.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(parts)


def load_split(root, split, domain_key):
    suffix = "30Class.npy"
    x = np.load(root / f"X_{split}_{suffix}", mmap_mode="r")
    y = np.load(root / f"Y_{split}_{suffix}", mmap_mode="r").astype(int)
    d = np.load(root / f"{domain_key}_{split}_{suffix}", mmap_mode="r").astype(int)
    if not (len(x) == len(y) == len(d)):
        raise RuntimeError(f"{split} X/identity/domain length mismatch")
    return x, y, d


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    dataset_key = f"wisig-{args.protocol}-validation-30"
    entry = dataset_path_dict[dataset_key]
    root = Path(os.path.expanduser(entry["linux"])).resolve()
    if root.name != "validation_novel" or entry.get("query_split") != "val":
        raise RuntimeError(f"Refusing non-validation-novel route: {root}")
    domain_key = "RX" if args.protocol == "cross-rx" else "DAY"
    source_x, source_y, source_d = load_split(root, "train", domain_key)
    target_x, target_y, target_d = load_split(root, "val", domain_key)
    expected = np.arange(30)
    if not np.array_equal(np.unique(source_y), expected):
        raise RuntimeError("Validation-novel source labels are not exactly 0..29")
    if not np.array_equal(np.unique(target_y), expected):
        raise RuntimeError("Validation-novel target labels are not exactly 0..29")
    if np.intersect1d(np.unique(source_d), np.unique(target_d)).size:
        raise RuntimeError("Source and validation domains overlap")

    source_indices = balanced_source_indices(
        source_y, source_d, args.samples_per_source_cell, args.seed
    )
    target_count = min(
        args.target_samples_per_identity,
        min(int(np.sum(target_y == identity)) for identity in expected),
    )
    target_indices = balanced_target_indices(target_y, target_count, args.seed)
    selected_source_x = power_normalize_fn(np.asarray(source_x[source_indices]))
    selected_target_x = power_normalize_fn(np.asarray(target_x[target_indices]))
    selected_source_y = np.asarray(source_y[source_indices])
    selected_target_y = np.asarray(target_y[target_indices])

    names = [name for name, _ in args.checkpoint]
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint names must be unique")
    rows, checkpoint_meta = [], {}
    for name, raw_path in args.checkpoint:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_hash = sha256(path)
        source_features = extract_features(
            path, selected_source_x, args.batch_size, device
        )
        target_features = extract_features(
            path, selected_target_x, args.batch_size, device
        )
        metrics = {}
        metrics.update(source_to_target_identity_probe(
            source_features, selected_source_y, target_features, selected_target_y, args.seed
        ))
        metrics.update(identity_disjoint_source_target_probe(
            source_features, selected_source_y, target_features, selected_target_y, args.seed
        ))
        metrics.update(source_target_distance_metrics(
            source_features, selected_source_y, target_features, selected_target_y
        ))
        rows.append({"protocol": args.protocol, "checkpoint_name": name,
                     "checkpoint_sha256": checkpoint_hash, **metrics})
        checkpoint_meta[name] = {"path": str(path), "sha256": checkpoint_hash}
        print("MODEL=" + name + " " + " ".join(
            f"{key}={value:.6f}" for key, value in metrics.items()
            if isinstance(value, float)
        ))

    output_dir = Path(args.output_dir).expanduser().resolve() / args.protocol
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    metadata = {
        "protocol": args.protocol,
        "role": "validation_novel_train_to_val_only",
        "dataset_root": str(root), "domain_key": domain_key,
        "source_domains": sorted(np.unique(source_d).astype(int).tolist()),
        "validation_domains": sorted(np.unique(target_d).astype(int).tolist()),
        "samples_per_source_cell": args.samples_per_source_cell,
        "target_samples_per_identity": target_count,
        "selected_source_samples": int(len(source_indices)),
        "selected_target_samples": int(len(target_indices)),
        "seed": args.seed, "checkpoints": checkpoint_meta,
        "validation_novel_accessed": True,
        "validation_query_labels_used_for_diagnostics": True,
        "test_novel_accessed": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"RESULT_DIR={output_dir}")
    print("WISIG_A1_VALIDATION_NOVEL_DIAGNOSTICS: PASS")


if __name__ == "__main__":
    main()
