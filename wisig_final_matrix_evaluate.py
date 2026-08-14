"""Guarded one-time paired-matrix final evaluation for strict WiSig.

Audit mode verifies hashes only and never loads an array.  Run mode executes the
entire frozen matrix for both protocols; partial runs may only resume the exact
same exclusive manifest.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from pathlib import Path


DRAFT_PATH = Path(
    "/home/yuanlong/yl/wanggang_wisig_a1_fair/runs/"
    "WiSig_final_matrix_v2_preparation/FROZEN_MANIFEST_DRAFT.json"
)
DRAFT_FILE_SHA256 = "542900a28a80ef20bdf07583f4a76932c5f36889dfd8d50ea3a571fa750c4c4f"
DRAFT_CANONICAL_SHA256 = "f763179fb4fcdf6569bdbc3f620bd7c239a1520d8892572b9415f3889d89c5df"
EXPECTED_FROZEN_MANIFEST_SHA256 = "44fa7befb480889e99d0b9c8d41366f0853b452ff56460e60863010b49776639"
CONFIRM_TOKEN = "UNSEAL-WISIG-FINAL-PAIRED-MATRIX-V2"
PROTOCOLS = ("cross-rx", "cross-day")
SHOTS = (1, 5, 10, 15, 20)
TRAIN_SEEDS = (2024, 2025, 2026, 2027, 2028)
ITERATIONS = 100
SUPPORT_BASE_SEED = 2024
NUM_CLASSES = 30
EXPECTED_ROWS_PER_PROTOCOL = 5000
T_CRITICAL_DF4_95 = 2.7764451051977987

DETAIL_FIELDS = (
    "protocol", "role", "variant", "train_seed", "checkpoint_sha256",
    "shot", "iteration", "support_seed", "accuracy", "source",
)
MODEL_SUMMARY_FIELDS = (
    "protocol", "role", "variant", "train_seed", "checkpoint_sha256",
    "shot", "n", "mean_accuracy", "std_accuracy", "ci95_half",
)
PAIRED_SUMMARY_FIELDS = (
    "protocol", "candidate", "shot", "train_seeds", "mean_difference",
    "ci95_low", "ci95_high", "positive_train_seeds",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen paired WiSig final matrix")
    parser.add_argument("--mode", choices=("audit", "run"), default="audit")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-workers", type=int, default=3)
    parser.add_argument("--output-dir", default="runs/WiSig_final_paired_matrix_v2")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_and_freeze_draft(path=DRAFT_PATH, verify_file=True):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if verify_file:
        actual_file_hash = sha256(path)
        if actual_file_hash != DRAFT_FILE_SHA256:
            raise RuntimeError(
                f"Draft file hash mismatch: expected {DRAFT_FILE_SHA256}, got {actual_file_hash}"
            )
    draft = json.loads(path.read_text(encoding="utf-8"))
    actual_canonical = canonical_hash(draft)
    if verify_file and actual_canonical != DRAFT_CANONICAL_SHA256:
        raise RuntimeError(
            "Draft canonical hash mismatch: "
            f"expected {DRAFT_CANONICAL_SHA256}, got {actual_canonical}"
        )
    if draft.get("schema") != "wisig-final-paired-matrix-v2-draft":
        raise RuntimeError("Unexpected draft schema")
    if draft.get("run_enabled") is not False:
        raise RuntimeError("Preparation draft unexpectedly enabled execution")
    if draft.get("final_arrays_loaded") is not False:
        raise RuntimeError("Preparation draft unexpectedly loaded final arrays")
    manifest = json.loads(json.dumps(draft))
    manifest["schema"] = "wisig-final-paired-matrix-v2"
    manifest["run_enabled"] = True
    manifest["source_draft"] = {
        "path": str(DRAFT_PATH),
        "file_sha256": DRAFT_FILE_SHA256,
        "canonical_sha256": DRAFT_CANONICAL_SHA256,
    }
    return manifest


def validate_frozen_structure(manifest):
    if manifest.get("protocol_variants") != {
        "cross-rx": ["B0", "A1C"], "cross-day": ["B0", "A1S"]
    }:
        raise RuntimeError("Frozen protocol/variant matrix changed")
    if tuple(manifest.get("train_seeds", ())) != TRAIN_SEEDS:
        raise RuntimeError("Frozen training seeds changed")
    if tuple(manifest.get("shots", ())) != SHOTS:
        raise RuntimeError("Frozen shots changed")
    if manifest.get("iterations") != ITERATIONS:
        raise RuntimeError("Frozen iteration count changed")
    if manifest.get("support_base_seed") != SUPPORT_BASE_SEED:
        raise RuntimeError("Frozen support seed changed")
    if manifest.get("expected_rows_per_protocol") != EXPECTED_ROWS_PER_PROTOCOL:
        raise RuntimeError("Frozen row count changed")
    if len(manifest.get("models", ())) != 20:
        raise RuntimeError("Frozen manifest must contain 20 models")
    if len(manifest.get("data_files", ())) != 8:
        raise RuntimeError("Frozen manifest must contain 8 data files")
    model_keys = [
        (row["protocol"], row["variant"], int(row["train_seed"]))
        for row in manifest["models"]
    ]
    if len(model_keys) != len(set(model_keys)):
        raise RuntimeError("Duplicate frozen model key")
    expected_keys = {
        (protocol, variant, seed)
        for protocol, variants in manifest["protocol_variants"].items()
        for variant in variants
        for seed in TRAIN_SEEDS
    }
    if set(model_keys) != expected_keys:
        raise RuntimeError("Frozen model grid is incomplete")


def audit_frozen_files(manifest, compute_hashes=True):
    validate_frozen_structure(manifest)
    for row in manifest["models"]:
        path = Path(row["checkpoint"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if compute_hashes and sha256(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {path}")
    for row in manifest["data_files"]:
        path = Path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Final file size mismatch: {path}")
        if compute_hashes and sha256(path) != row["sha256"]:
            raise RuntimeError(f"Final file hash mismatch: {path}")


def acquire_or_resume(output_dir, manifest, resume):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "UNSEAL_MANIFEST.json"
    manifest_hash = canonical_hash(manifest)
    if manifest_hash != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen manifest hash mismatch: expected {EXPECTED_FROZEN_MANIFEST_SHA256}, "
            f"got {manifest_hash}"
        )
    record = {"manifest_sha256": manifest_hash, "manifest": manifest}
    if resume:
        if not lock_path.is_file():
            raise RuntimeError("Cannot resume: UNSEAL_MANIFEST.json does not exist")
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != record:
            raise RuntimeError("Cannot resume: frozen unseal manifest differs")
        return manifest_hash
    try:
        descriptor = os.open(
            str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
    except FileExistsError as error:
        raise RuntimeError(
            "Final matrix was already unsealed; only exact --resume is allowed"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    return manifest_hash


def expected_row_keys(protocol, manifest):
    variants = manifest["protocol_variants"][protocol]
    return {
        (variant, train_seed, shot, iteration)
        for variant in variants
        for train_seed in TRAIN_SEEDS
        for shot in SHOTS
        for iteration in range(1, ITERATIONS + 1)
    }


def model_lookup(manifest):
    return {
        (row["protocol"], row["variant"], int(row["train_seed"])): row
        for row in manifest["models"]
    }


def load_existing_rows(path, protocol, manifest):
    if not path.is_file():
        return []
    models = model_lookup(manifest)
    expected_keys = expected_row_keys(protocol, manifest)
    rows, seen = [], set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["variant"], int(row["train_seed"]),
                int(row["shot"]), int(row["iteration"]),
            )
            if key in seen:
                raise RuntimeError(f"Duplicate final row: {key}")
            if key not in expected_keys:
                raise RuntimeError(f"Out-of-grid final row: {key}")
            if row["protocol"] != protocol or row["role"] != "final":
                raise RuntimeError("Existing row has wrong protocol or role")
            expected_seed = SUPPORT_BASE_SEED + int(row["iteration"]) - 1
            if int(row["support_seed"]) != expected_seed:
                raise RuntimeError(f"Existing support seed mismatch: {key}")
            model = models[(protocol, row["variant"], int(row["train_seed"]))]
            if row["checkpoint_sha256"] != model["checkpoint_sha256"]:
                raise RuntimeError(f"Existing checkpoint mismatch: {key}")
            rows.append(row)
            seen.add(key)
    return rows


def atomic_write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def model_summaries(rows):
    summaries = []
    groups = {}
    for row in rows:
        key = (
            row["protocol"], row["variant"], int(row["train_seed"]),
            row["checkpoint_sha256"], int(row["shot"]),
        )
        groups.setdefault(key, []).append(float(row["accuracy"]))
    for key, values in sorted(groups.items()):
        if len(values) != ITERATIONS:
            continue
        protocol, variant, train_seed, checkpoint_hash, shot = key
        std = statistics.stdev(values)
        summaries.append({
            "protocol": protocol, "role": "final", "variant": variant,
            "train_seed": train_seed, "checkpoint_sha256": checkpoint_hash,
            "shot": shot, "n": len(values),
            "mean_accuracy": f"{statistics.mean(values):.8f}",
            "std_accuracy": f"{std:.8f}",
            "ci95_half": f"{1.96 * std / math.sqrt(len(values)):.8f}",
        })
    return summaries


def paired_summaries(rows, manifest):
    by_key = {
        (row["protocol"], row["variant"], int(row["train_seed"]),
         int(row["shot"]), int(row["iteration"])): float(row["accuracy"])
        for row in rows
    }
    output = []
    for protocol, variants in manifest["protocol_variants"].items():
        baseline, candidate = variants
        for shot in SHOTS:
            differences = []
            for train_seed in TRAIN_SEEDS:
                paired = [
                    by_key[(protocol, candidate, train_seed, shot, iteration)]
                    - by_key[(protocol, baseline, train_seed, shot, iteration)]
                    for iteration in range(1, ITERATIONS + 1)
                ]
                differences.append(statistics.mean(paired))
            mean = statistics.mean(differences)
            half = T_CRITICAL_DF4_95 * statistics.stdev(differences) / math.sqrt(5)
            output.append({
                "protocol": protocol, "candidate": candidate, "shot": shot,
                "train_seeds": 5, "mean_difference": f"{mean:.8f}",
                "ci95_low": f"{mean - half:.8f}",
                "ci95_high": f"{mean + half:.8f}",
                "positive_train_seeds": sum(value > 0 for value in differences),
            })
    return output


def sample_feature_indices(labels, shot, seed, num_classes=NUM_CLASSES):
    import numpy as np

    rng = random.Random(seed)
    selected = []
    for class_id in range(num_classes):
        candidates = np.flatnonzero(labels == class_id).tolist()
        if len(candidates) < shot:
            raise RuntimeError(
                f"Class {class_id} has {len(candidates)} support samples, needs {shot}"
            )
        selected.extend(rng.sample(candidates, shot))
    return selected


def resolve_device(requested):
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def protocol_data_files(manifest, protocol):
    return {
        (row["split"], row["kind"]): row["path"]
        for row in manifest["data_files"] if row["protocol"] == protocol
    }


def load_final_arrays(manifest, protocol):
    import numpy as np
    from utils.get_dataset import power_normalize_fn

    files = protocol_data_files(manifest, protocol)
    support_x = np.load(files[("train", "X")])
    support_y = np.load(files[("train", "Y")])
    query_x = np.load(files[("test", "X")])
    query_y = np.load(files[("test", "Y")])
    if support_x.ndim == 3 and support_x.shape[1] != 2:
        support_x = support_x.transpose((0, 2, 1))
    if query_x.ndim == 3 and query_x.shape[1] != 2:
        query_x = query_x.transpose((0, 2, 1))
    if support_x.shape[1:] != (2, 256) or query_x.shape[1:] != (2, 256):
        raise RuntimeError("Frozen final IQ shape is not (N, 2, 256)")
    expected = np.arange(NUM_CLASSES)
    if not np.array_equal(np.unique(support_y).astype(int), expected):
        raise RuntimeError("Final support labels are not exactly 0..29")
    if not np.array_equal(np.unique(query_y).astype(int), expected):
        raise RuntimeError("Final query labels are not exactly 0..29")
    return (
        power_normalize_fn(support_x), support_y.astype(np.uint8),
        power_normalize_fn(query_x), query_y.astype(np.uint8),
    )


def run_protocol(protocol, manifest, output_dir, device, batch_size, lr_workers):
    import torch
    from utils.config import model_path_dict
    from utils.utils import create_model, load_encoder_weights
    from wisig_validate_lr import (
        TSLA_CONFIG, evaluate_lr_jobs, extract_features, make_loader,
    )

    csv_path = output_dir / protocol / "iterations.csv"
    rows = load_existing_rows(csv_path, protocol, manifest)
    completed = {
        (row["variant"], int(row["train_seed"]), int(row["shot"]),
         int(row["iteration"])) for row in rows
    }
    support_x, support_y, query_x, query_y = load_final_arrays(manifest, protocol)
    support_loader = make_loader(support_x, support_y, batch_size, device)
    query_loader = make_loader(query_x, query_y, batch_size, device)
    models = [row for row in manifest["models"] if row["protocol"] == protocol]
    for model in models:
        variant, train_seed = model["variant"], int(model["train_seed"])
        model_keys = {
            (variant, train_seed, shot, iteration)
            for shot in SHOTS for iteration in range(1, ITERATIONS + 1)
        }
        if model_keys.issubset(completed):
            print(f"SKIP_COMPLETED={protocol}_{variant}_seed{train_seed}")
            continue
        encoder = create_model(
            model_path_dict["CVTSLANet"], feature_dim=1024,
            dtype="iq", **TSLA_CONFIG
        )
        load_encoder_weights(encoder, model["checkpoint"], device)
        encoder = encoder.to(device)
        support_features, support_labels = extract_features(
            encoder, support_loader, device,
            f"Final support pool {protocol} {variant} seed{train_seed}",
        )
        query_features, query_labels = extract_features(
            encoder, query_loader, device,
            f"Final queries {protocol} {variant} seed{train_seed}",
        )
        for shot in SHOTS:
            jobs = []
            for iteration in range(1, ITERATIONS + 1):
                key = (variant, train_seed, shot, iteration)
                if key in completed:
                    continue
                support_seed = SUPPORT_BASE_SEED + iteration - 1
                indices = sample_feature_indices(support_labels, shot, support_seed)
                jobs.append((
                    key, support_features[indices], support_labels[indices], support_seed
                ))
            scores = (
                evaluate_lr_jobs(jobs, query_features, query_labels, lr_workers)
                if jobs else {}
            )
            for key in sorted(scores):
                _, _, _, iteration = key
                support_seed = SUPPORT_BASE_SEED + iteration - 1
                accuracy = scores[key]
                rows.append({
                    "protocol": protocol, "role": "final", "variant": variant,
                    "train_seed": train_seed,
                    "checkpoint_sha256": model["checkpoint_sha256"],
                    "shot": shot, "iteration": iteration,
                    "support_seed": support_seed,
                    "accuracy": f"{accuracy:.8f}", "source": "computed",
                })
                completed.add(key)
            rows.sort(key=lambda row: (
                row["variant"], int(row["train_seed"]),
                int(row["shot"]), int(row["iteration"])
            ))
            atomic_write_csv(csv_path, DETAIL_FIELDS, rows)
            atomic_write_csv(
                output_dir / protocol / "model_summary.csv",
                MODEL_SUMMARY_FIELDS, model_summaries(rows),
            )
            print(
                f"COMPLETE_SHOT={protocol}_{variant}_seed{train_seed}_shot{shot} "
                f"ROWS={len(rows)}/{EXPECTED_ROWS_PER_PROTOCOL}"
            )
        del encoder, support_features, query_features
        if device == "cuda":
            torch.cuda.empty_cache()
    if len(rows) != EXPECTED_ROWS_PER_PROTOCOL:
        raise RuntimeError(
            f"{protocol} has {len(rows)} rows, expected {EXPECTED_ROWS_PER_PROTOCOL}"
        )
    if {(
        row["variant"], int(row["train_seed"]), int(row["shot"]),
        int(row["iteration"])
    ) for row in rows} != expected_row_keys(protocol, manifest):
        raise RuntimeError(f"{protocol} final grid is not exact")
    return rows


def run_all(manifest, output_dir, device, batch_size, lr_workers):
    all_rows = []
    for protocol in PROTOCOLS:
        print(f"START_PROTOCOL={protocol}")
        rows = run_protocol(
            protocol, manifest, output_dir, device, batch_size, lr_workers
        )
        all_rows.extend(rows)
        print(f"COMPLETE_PROTOCOL={protocol} ROWS={len(rows)}")
    atomic_write_csv(
        output_dir / "paired_summary.csv", PAIRED_SUMMARY_FIELDS,
        paired_summaries(all_rows, manifest),
    )
    return all_rows


def main():
    args = parse_args()
    manifest = load_and_freeze_draft()
    validate_frozen_structure(manifest)
    manifest_hash = canonical_hash(manifest)
    if manifest_hash != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise RuntimeError(
            f"Code-frozen manifest mismatch: expected {EXPECTED_FROZEN_MANIFEST_SHA256}, "
            f"got {manifest_hash}"
        )
    audit_frozen_files(manifest, compute_hashes=True)
    print(f"MODE={args.mode}")
    print(f"FROZEN_MANIFEST_SHA256={manifest_hash}")
    print("AUDITED_CHECKPOINTS=20/20")
    print("AUDITED_FINAL_FILES=8/8")
    print("EXPECTED_ROWS_CROSS_RX=5000")
    print("EXPECTED_ROWS_CROSS_DAY=5000")
    if args.mode == "audit":
        print("FINAL_ARRAYS_LOADED=False")
        print("READY_FOR_USER_AUTHORIZATION=True")
        print("WISIG_FINAL_MATRIX_AUDIT: PASS")
        return
    if args.confirm != CONFIRM_TOKEN:
        raise RuntimeError(f"Run mode requires --confirm {CONFIRM_TOKEN}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    acquired = acquire_or_resume(output_dir, manifest, args.resume)
    device = resolve_device(args.device)
    print(f"UNSEAL_MANIFEST_SHA256={acquired}")
    print(f"RESUME={args.resume}")
    print(f"DEVICE={device}")
    print(f"LR_WORKERS={args.lr_workers}")
    print("FINAL_ARRAYS_LOADED=ABOUT_TO_LOAD")
    rows = run_all(manifest, output_dir, device, args.batch_size, args.lr_workers)
    completion = {
        "manifest_sha256": acquired,
        "rows": len(rows),
        "expected_rows": 10000,
        "status": "complete",
        "final_test_accessed": True,
    }
    (output_dir / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(f"RESULT_DIR={output_dir}")
    print("WISIG_FINAL_PAIRED_MATRIX: PASS")


if __name__ == "__main__":
    main()
