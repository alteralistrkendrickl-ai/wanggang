"""One-time, pre-registered final evaluation for strict WiSig protocols.

Audit mode never loads any NPY array. Run mode is guarded by an exclusive
manifest and may only resume the exact same frozen evaluation.
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

from utils.config import dataset_path_dict, model_path_dict
from utils.get_dataset import load_data, power_normalize_fn
from utils.utils import create_model, load_encoder_weights, set_seed
from wisig_validate_lr import (
    DETAIL_FIELDS, SUMMARY_FIELDS, TSLA_CONFIG, evaluate_lr_jobs,
    extract_features, make_loader, sample_support, sha256, write_csv,
)


CONFIRM_TOKEN = "UNSEAL-WISIG-FINAL-ONCE"
SHOTS = (1, 5, 10, 15, 20)
ITERATIONS = 100
BASE_SEED = 2024
NUM_CLASSES = 30
CHECKPOINTS = {
    "cross-rx": {
        "A1C": (
            "runs/Pretext_random_rot/"
            "CVTSLANet_wisig-cross-rx_iq_powerNorm_A1C_fair10e/best_encoder.pth",
            "e3597d2a17afc75122e9a36bbefdefedeb9caa17c804c166a3f2d55647938442",
        ),
    },
    "cross-day": {
        "B0": (
            "/home/yuanlong/yl/wanggang_wisig_audit/runs/Pretext_random_rot/"
            "CVTSLANet_wisig-cross-day_iq_powerNorm/best_encoder.pth",
            "ff49ec9a41b49166b2d579ba78be5fae2b68f5990d08eb44daa7aceb12072afd",
        ),
        "A1S": (
            "runs/Pretext_random_rot/"
            "CVTSLANet_wisig-cross-day_iq_powerNorm_A1S_fair10e/best_encoder.pth",
            "545a0e2770169c8909841ed675b3a40bce6785f0f2ec66bb2e37e3e9eb4bcdde",
        ),
        "A1C": (
            "runs/Pretext_random_rot/"
            "CVTSLANet_wisig-cross-day_iq_powerNorm_A1C_fair10e/best_encoder.pth",
            "8d208cdd4e84c98d056319b424c2f50fcf694e88340ec0e908ad9d393410918c",
        ),
    },
}
ROUTES = {
    "cross-rx": {1: "A1C", 5: "A1C", 10: "A1C", 15: "A1C", 20: "A1C"},
    "cross-day": {1: "A1C", 5: "A1C", 10: "A1S", 15: "A1S", 20: "B0"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen one-time WiSig final evaluation")
    parser.add_argument("--protocol", choices=("cross-rx", "cross-day"), required=True)
    parser.add_argument("--mode", choices=("audit", "run"), default="audit")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-workers", type=int, default=3)
    parser.add_argument("--output-dir", default="runs/WiSig_final_frozen_v1")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def resolve_path(project_root, raw):
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def frozen_manifest(protocol, project_root, dataset_root):
    checkpoints = {}
    for name, (raw_path, expected_hash) in CHECKPOINTS[protocol].items():
        checkpoints[name] = {
            "path": str(resolve_path(project_root, raw_path)),
            "sha256": expected_hash,
        }
    return {
        "schema": "wisig-final-frozen-v1",
        "protocol": protocol,
        "role": "final_test_novel",
        "dataset_root": str(dataset_root),
        "support_split": "train",
        "query_split": "test",
        "num_classes": NUM_CLASSES,
        "shots": list(SHOTS),
        "iterations": ITERATIONS,
        "base_seed": BASE_SEED,
        "routes": {str(k): v for k, v in ROUTES[protocol].items()},
        "checkpoints": checkpoints,
        "lr": {"class": "LogisticRegression", "max_iter": 1000,
               "random_state": "per_iteration_seed"},
        "normalization": "power",
        "encoder": {"name": "CVTSLANet", "feature_dim": 1024,
                    "tsla_config": TSLA_CONFIG},
    }


def canonical_hash(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def audit_files(manifest, compute_hashes=True):
    root = Path(manifest["dataset_root"])
    if root.name == "validation_novel":
        raise RuntimeError("Final route unexpectedly points to validation_novel")
    required = [
        root / "X_train_30Class.npy", root / "Y_train_30Class.npy",
        root / "X_test_30Class.npy", root / "Y_test_30Class.npy",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name, item in manifest["checkpoints"].items():
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if compute_hashes:
            actual = sha256(path)
            if actual != item["sha256"]:
                raise RuntimeError(
                    f"Checkpoint hash mismatch for {name}: expected {item['sha256']}, got {actual}"
                )
    return required


def acquire_or_resume(run_dir, manifest, resume):
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / "UNSEAL_MANIFEST.json"
    manifest_hash = canonical_hash(manifest)
    record = {"manifest_sha256": manifest_hash, "manifest": manifest}
    if resume:
        if not lock_path.is_file():
            raise RuntimeError("Cannot resume: unseal manifest does not exist")
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != record:
            raise RuntimeError("Cannot resume: frozen manifest differs")
        return manifest_hash
    try:
        descriptor = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise RuntimeError(
            "Final evaluation was already unsealed; use --resume only for the same run"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    return manifest_hash


def load_existing_rows(path, manifest):
    if not path.is_file():
        return []
    rows, seen = [], set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            shot, iteration, seed = int(row["shot"]), int(row["iteration"]), int(row["seed"])
            key = (shot, iteration)
            if key in seen:
                raise RuntimeError(f"Duplicate existing final row: {key}")
            if shot not in SHOTS or not 1 <= iteration <= ITERATIONS:
                raise RuntimeError(f"Out-of-scope existing final row: {key}")
            if seed != BASE_SEED + iteration - 1:
                raise RuntimeError(f"Existing seed mismatch: {key}")
            if row["protocol"] != manifest["protocol"] or row["role"] != "final":
                raise RuntimeError("Existing final CSV role/protocol mismatch")
            expected_name = ROUTES[manifest["protocol"]][shot]
            expected_hash = manifest["checkpoints"][expected_name]["sha256"]
            if row["checkpoint"] != expected_name or row["checkpoint_sha256"] != expected_hash:
                raise RuntimeError(f"Existing route/checkpoint mismatch: {key}")
            rows.append(row); seen.add(key)
    return rows


def summary_rows(rows, protocol):
    summaries = []
    for shot in SHOTS:
        values = np.asarray([float(r["accuracy"]) for r in rows if int(r["shot"]) == shot])
        if len(values) != ITERATIONS:
            continue
        std = float(values.std(ddof=1))
        name = ROUTES[protocol][shot]
        summaries.append({
            "protocol": protocol, "role": "final", "checkpoint": name,
            "checkpoint_sha256": CHECKPOINTS[protocol][name][1], "shot": shot,
            "n": len(values), "mean_accuracy": f"{values.mean():.8f}",
            "std_accuracy": f"{std:.8f}",
            "ci95_half": f"{1.96 * std / math.sqrt(len(values)):.8f}",
        })
    return summaries


def load_final_arrays(dataset_root):
    support_x, support_y = load_data(
        str(dataset_root), NUM_CLASSES, "train", signal_length=256
    )
    query_x, query_y = load_data(
        str(dataset_root), NUM_CLASSES, "test", signal_length=256
    )
    expected = np.arange(NUM_CLASSES)
    if not np.array_equal(np.unique(support_y).astype(int), expected):
        raise RuntimeError("Final support labels are not exactly 0..29")
    if not np.array_equal(np.unique(query_y).astype(int), expected):
        raise RuntimeError("Final query labels are not exactly 0..29")
    return support_x, support_y, power_normalize_fn(query_x), query_y.astype(np.uint8)


def run_final(args, manifest, run_dir, device):
    rows = load_existing_rows(run_dir / "iterations.csv", manifest)
    completed = {(int(r["shot"]), int(r["iteration"])) for r in rows}
    support_x, support_y, query_x, query_y = load_final_arrays(Path(manifest["dataset_root"]))
    needed_models = sorted(set(ROUTES[args.protocol].values()))
    for model_name in needed_models:
        checkpoint = Path(manifest["checkpoints"][model_name]["path"])
        encoder = create_model(
            model_path_dict["CVTSLANet"], feature_dim=1024, dtype="iq", **TSLA_CONFIG
        )
        load_encoder_weights(encoder, str(checkpoint), device)
        encoder = encoder.to(device)
        routed_shots = [s for s in SHOTS if ROUTES[args.protocol][s] == model_name]
        query_loader = make_loader(query_x, query_y, args.batch_size, device)
        query_features, query_labels = extract_features(
            encoder, query_loader, device, f"Final queries {model_name}"
        )
        for shot in routed_shots:
            jobs = []
            for iteration in range(1, ITERATIONS + 1):
                if (shot, iteration) in completed:
                    continue
                seed = BASE_SEED + iteration - 1
                set_seed(seed)
                sampled_x, sampled_y = sample_support(
                    support_x, support_y, shot, seed, NUM_CLASSES
                )
                loader = make_loader(sampled_x, sampled_y, args.batch_size, device)
                features, labels = extract_features(
                    encoder, loader, device, f"Final {model_name} {shot}-shot {iteration}/100"
                )
                jobs.append(((shot, iteration), features, labels, seed))
            scores = evaluate_lr_jobs(jobs, query_features, query_labels, args.lr_workers) if jobs else {}
            for key in sorted(scores):
                _, iteration = key
                seed = BASE_SEED + iteration - 1
                accuracy = scores[key]
                rows.append({
                    "protocol": args.protocol, "role": "final", "checkpoint": model_name,
                    "checkpoint_sha256": manifest["checkpoints"][model_name]["sha256"],
                    "shot": shot, "iteration": iteration, "seed": seed,
                    "accuracy": f"{accuracy:.8f}", "source": "computed",
                })
                completed.add(key)
                rows.sort(key=lambda r: (int(r["shot"]), int(r["iteration"])))
                write_csv(run_dir / "iterations.csv", DETAIL_FIELDS, rows)
                write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, summary_rows(rows, args.protocol))
                print(f"MODEL={model_name} SHOT={shot} ITERATION={iteration}/100 SEED={seed} ACC={accuracy:.4f}")
            current = [r for r in rows if int(r["shot"]) == shot]
            if len(current) != ITERATIONS:
                raise RuntimeError(f"Final shot {shot} has only {len(current)} rows")
            summary = next(r for r in summary_rows(rows, args.protocol) if int(r["shot"]) == shot)
            print(f"SUMMARY MODEL={model_name} SHOT={shot} N=100 "
                  f"MEAN={float(summary['mean_accuracy']):.4f} "
                  f"STD={float(summary['std_accuracy']):.4f} "
                  f"CI95_HALF={float(summary['ci95_half']):.4f}")
        del encoder
        if device == "cuda":
            torch.cuda.empty_cache()
    if len(rows) != len(SHOTS) * ITERATIONS:
        raise RuntimeError(f"Final evaluation has {len(rows)} rows, expected 500")
    return rows


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    entry = dataset_path_dict[f"wisig-{args.protocol}-30"]
    if entry.get("query_split") != "test":
        raise RuntimeError("Configured final dataset query split is not test")
    dataset_root = Path(os.path.expanduser(entry["linux"])).resolve()
    manifest = frozen_manifest(args.protocol, project_root, dataset_root)
    required = audit_files(manifest, compute_hashes=True)
    manifest_hash = canonical_hash(manifest)
    print(f"PROTOCOL={args.protocol}")
    print(f"MODE={args.mode}")
    print("ROLE=final_test_novel")
    print(f"DATASET_ROOT={dataset_root}")
    print(f"MANIFEST_SHA256={manifest_hash}")
    print("ROUTES=" + json.dumps(manifest["routes"], sort_keys=True))
    for name, item in manifest["checkpoints"].items():
        print(f"CHECKPOINT={name} SHA256={item['sha256']} PATH={item['path']}")
    for path in required:
        print(f"FINAL_FILE_PRESENT={path}")
    if args.mode == "audit":
        print("FINAL_ARRAYS_LOADED=False")
        print("WISIG_FINAL_AUDIT: PASS")
        return
    if args.confirm != CONFIRM_TOKEN:
        raise RuntimeError(f"Run mode requires --confirm {CONFIRM_TOKEN}")
    device = resolve_device(args.device)
    run_dir = Path(args.output_dir).expanduser().resolve() / args.protocol
    acquired_hash = acquire_or_resume(run_dir, manifest, args.resume)
    print(f"UNSEAL_MANIFEST_SHA256={acquired_hash}")
    print(f"RESUME={args.resume}")
    print(f"DEVICE={device}")
    print(f"LR_WORKERS={args.lr_workers}")
    rows = run_final(args, manifest, run_dir, device)
    completion = {
        "manifest_sha256": acquired_hash, "rows": len(rows),
        "status": "complete", "test_novel_accessed": True,
    }
    (run_dir / "COMPLETED.json").write_text(
        json.dumps(completion, indent=2), encoding="utf-8"
    )
    print(f"RESULT_DIR={run_dir}")
    print("WISIG_FINAL_EVALUATION: PASS")


if __name__ == "__main__":
    main()
