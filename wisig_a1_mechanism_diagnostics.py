"""Mechanism diagnostics for frozen WiSig B0/A1 encoders.

Only the 90 base-pretraining training identities and their training-domain
labels are read.  Validation-novel and test-novel arrays are intentionally not
addressable by this entry point.
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
    parser = argparse.ArgumentParser(description="A1 base-training mechanism diagnostics")
    parser.add_argument("--protocol", choices=("cross-rx", "cross-day"), required=True)
    parser.add_argument("--checkpoint", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--samples-per-cell", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="runs/WiSig_A1_mechanism")
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


def balanced_cell_indices(labels, domains, samples_per_cell, seed):
    if samples_per_cell < 1:
        raise ValueError("samples-per-cell must be positive")
    rng = random.Random(seed)
    selected = []
    for identity in sorted(np.unique(labels).astype(int)):
        identity_domains = sorted(np.unique(domains[labels == identity]).astype(int))
        if len(identity_domains) < 2:
            raise ValueError(f"Identity {identity} has fewer than two domains")
        for domain in identity_domains:
            candidates = np.flatnonzero((labels == identity) & (domains == domain)).tolist()
            if len(candidates) < samples_per_cell:
                raise ValueError(
                    f"Identity/domain cell ({identity}, {domain}) has {len(candidates)} samples"
                )
            selected.extend(rng.sample(candidates, samples_per_cell))
    return np.asarray(selected, dtype=np.int64)


def identity_disjoint_domain_probe(features, identities, domains, seed):
    unique_ids = np.unique(identities)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    cut = max(1, int(round(0.7 * len(shuffled))))
    train_ids, test_ids = shuffled[:cut], shuffled[cut:]
    train = np.isin(identities, train_ids)
    test = np.isin(identities, test_ids)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
    )
    model.fit(features[train], domains[train])
    pred = model.predict(features[test])
    return {
        "domain_accuracy": float(accuracy_score(domains[test], pred)),
        "domain_balanced_accuracy": float(balanced_accuracy_score(domains[test], pred)),
        "domain_chance": 1.0 / len(np.unique(domains)),
        "domain_probe_train_identities": int(len(train_ids)),
        "domain_probe_test_identities": int(len(test_ids)),
    }


def domain_disjoint_identity_probe(features, identities, domains, seed):
    unique_domains = np.unique(domains)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_domains)
    cut = max(1, len(shuffled) // 2)
    train_domains, test_domains = shuffled[:cut], shuffled[cut:]
    train = np.isin(domains, train_domains)
    test = np.isin(domains, test_domains)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed)
    )
    model.fit(features[train], identities[train])
    pred = model.predict(features[test])
    return {
        "identity_accuracy": float(accuracy_score(identities[test], pred)),
        "identity_balanced_accuracy": float(balanced_accuracy_score(identities[test], pred)),
        "identity_chance": 1.0 / len(np.unique(identities)),
        "identity_probe_train_domains": [int(x) for x in sorted(train_domains)],
        "identity_probe_test_domains": [int(x) for x in sorted(test_domains)],
    }


def cosine_distance_metrics(features, identities, domains):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-12)
    identity_centroids = []
    within = []
    for identity in sorted(np.unique(identities).astype(int)):
        mask = identities == identity
        identity_centroid = normalized[mask].mean(axis=0)
        identity_centroid /= max(np.linalg.norm(identity_centroid), 1e-12)
        identity_centroids.append(identity_centroid)
        domain_centroids = []
        for domain in sorted(np.unique(domains[mask]).astype(int)):
            centroid = normalized[mask & (domains == domain)].mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            domain_centroids.append(centroid)
        for i in range(len(domain_centroids)):
            for j in range(i + 1, len(domain_centroids)):
                within.append(1.0 - float(np.dot(domain_centroids[i], domain_centroids[j])))
    identity_centroids = np.asarray(identity_centroids)
    between = []
    for i in range(len(identity_centroids)):
        for j in range(i + 1, len(identity_centroids)):
            between.append(1.0 - float(np.dot(identity_centroids[i], identity_centroids[j])))
    within_mean = float(np.mean(within))
    between_mean = float(np.mean(between))
    return {
        "same_identity_cross_domain_cosine_distance": within_mean,
        "different_identity_cosine_distance": between_mean,
        "separation_ratio": between_mean / max(within_mean, 1e-12),
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


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    dataset_key = f"wisig-{args.protocol}"
    root = Path(os.path.expanduser(dataset_path_dict[dataset_key]["linux"])).resolve()
    domain_key = "RX" if args.protocol == "cross-rx" else "DAY"
    x = np.load(root / "X_train_90Class.npy", mmap_mode="r")
    labels = np.load(root / "Y_train_90Class.npy", mmap_mode="r").astype(int)
    domains = np.load(root / f"{domain_key}_train_90Class.npy", mmap_mode="r").astype(int)
    if not (len(x) == len(labels) == len(domains)):
        raise RuntimeError("X/identity/domain length mismatch")
    if not np.array_equal(np.unique(labels), np.arange(90)):
        raise RuntimeError("Base identities are not exactly 0..89")
    indices = balanced_cell_indices(labels, domains, args.samples_per_cell, args.seed)
    selected_x = power_normalize_fn(np.asarray(x[indices]))
    selected_y = np.asarray(labels[indices])
    selected_d = np.asarray(domains[indices])

    rows = []
    checkpoint_meta = {}
    names = [name for name, _ in args.checkpoint]
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint names must be unique")
    for name, raw_path in args.checkpoint:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_hash = sha256(path)
        features = extract_features(path, selected_x, args.batch_size, device)
        metrics = {}
        metrics.update(identity_disjoint_domain_probe(features, selected_y, selected_d, args.seed))
        metrics.update(domain_disjoint_identity_probe(features, selected_y, selected_d, args.seed))
        metrics.update(cosine_distance_metrics(features, selected_y, selected_d))
        row = {"protocol": args.protocol, "checkpoint_name": name,
               "checkpoint_sha256": checkpoint_hash, **metrics}
        rows.append(row)
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
        "protocol": args.protocol, "role": "base_pretraining_train_only",
        "dataset_root": str(root), "domain_key": domain_key,
        "samples_per_cell": args.samples_per_cell, "selected_samples": int(len(indices)),
        "seed": args.seed, "checkpoints": checkpoint_meta,
        "validation_novel_accessed": False, "test_novel_accessed": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"RESULT_DIR={output_dir}")
    print("WISIG_A1_MECHANISM_DIAGNOSTICS: PASS")


if __name__ == "__main__":
    main()
