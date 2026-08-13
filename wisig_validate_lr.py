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
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    parser.add_argument(
        "--lr-workers", type=int, default=1,
        help="Parallel CPU workers for independent LR fits; 1 preserves serial execution.",
    )
    parser.add_argument("--output-dir", default="runs/WiSig_validation_lr")
    parser.add_argument(
        "--resume-log",
        default="",
        help="Recover completed SHOT/ITERATION/SEED/ACC records from a prior log.",
    )
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


_WORKER_QUERY_FEATURES = None
_WORKER_QUERY_LABELS = None


def _init_lr_worker(query_features, query_labels):
    global _WORKER_QUERY_FEATURES, _WORKER_QUERY_LABELS
    _WORKER_QUERY_FEATURES = query_features
    _WORKER_QUERY_LABELS = query_labels


def fit_and_score_lr(support_features, support_labels, seed):
    classifier = LogisticRegression(max_iter=1000, random_state=seed)
    classifier.fit(support_features, support_labels)
    return float(classifier.score(_WORKER_QUERY_FEATURES, _WORKER_QUERY_LABELS) * 100.0)


def evaluate_lr_jobs(jobs, query_features, query_labels, workers):
    """Evaluate independent LR jobs without changing classifier semantics."""
    if workers == 1:
        _init_lr_worker(query_features, query_labels)
        return {
            key: fit_and_score_lr(features, labels, seed)
            for key, features, labels, seed in jobs
        }
    results = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_lr_worker,
        initargs=(query_features, query_labels),
    ) as executor:
        futures = {
            executor.submit(fit_and_score_lr, features, labels, seed): key
            for key, features, labels, seed in jobs
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


DETAIL_FIELDS = (
    "protocol",
    "role",
    "checkpoint",
    "checkpoint_sha256",
    "shot",
    "iteration",
    "seed",
    "accuracy",
    "source",
)

SUMMARY_FIELDS = (
    "protocol",
    "role",
    "checkpoint",
    "checkpoint_sha256",
    "shot",
    "n",
    "mean_accuracy",
    "std_accuracy",
    "ci95_half",
)

RESULT_PATTERN = re.compile(
    r"SHOT=(?P<shot>\d+)\s+ITERATION=(?P<iteration>\d+)/(?:\d+)\s+"
    r"SEED=(?P<seed>\d+)\s+ACC=(?P<accuracy>\d+(?:\.\d+)?)"
)


def recover_log_rows(path, args, checkpoint_hash):
    if not path:
        return []
    log_path = Path(path).expanduser().resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"Resume log not found: {log_path}")
    content = log_path.read_text(encoding="utf-8", errors="replace")
    protocol_match = re.search(r"^PROTOCOL=(\S+)", content, re.MULTILINE)
    hash_match = re.search(r"^CHECKPOINT_SHA256=([0-9a-f]{64})", content, re.MULTILINE)
    if not protocol_match or protocol_match.group(1) != args.protocol:
        raise RuntimeError("Resume log protocol does not match this run")
    if not hash_match or hash_match.group(1) != checkpoint_hash:
        raise RuntimeError("Resume log checkpoint SHA256 does not match this run")

    allowed_shots = set(args.shots)
    rows_by_key = {}
    for match in RESULT_PATTERN.finditer(content):
        shot = int(match.group("shot"))
        iteration = int(match.group("iteration"))
        seed = int(match.group("seed"))
        accuracy = float(match.group("accuracy"))
        if shot not in allowed_shots or not 1 <= iteration <= args.iterations:
            continue
        expected_seed = args.base_seed + iteration - 1
        if seed != expected_seed:
            raise RuntimeError(
                f"Resume log seed mismatch for shot={shot}, iteration={iteration}: "
                f"expected {expected_seed}, got {seed}"
            )
        key = (shot, iteration)
        previous = rows_by_key.get(key)
        if previous is not None and previous["accuracy"] != f"{accuracy:.8f}":
            raise RuntimeError(f"Conflicting duplicate result in resume log: {key}")
        rows_by_key[key] = {
            "protocol": args.protocol,
            "role": "validation",
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_hash,
            "shot": shot,
            "iteration": iteration,
            "seed": seed,
            "accuracy": f"{accuracy:.8f}",
            "source": f"recovered:{log_path.name}",
        }
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def recover_csv_rows(path, args, checkpoint_hash):
    if not path.is_file():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            shot = int(row["shot"])
            iteration = int(row["iteration"])
            seed = int(row["seed"])
            if row["protocol"] != args.protocol:
                raise RuntimeError("Existing CSV protocol does not match this run")
            if row["checkpoint_sha256"] != checkpoint_hash:
                raise RuntimeError("Existing CSV checkpoint SHA256 does not match this run")
            if seed != args.base_seed + iteration - 1:
                raise RuntimeError("Existing CSV seed schedule does not match this run")
            if shot in args.shots and 1 <= iteration <= args.iterations:
                row["source"] = row.get("source") or "existing_csv"
                rows.append(row)
    return rows


def merge_detail_rows(*row_groups):
    merged = {}
    for rows in row_groups:
        for row in rows:
            key = (int(row["shot"]), int(row["iteration"]))
            previous = merged.get(key)
            if previous is not None and not math.isclose(
                float(previous["accuracy"]), float(row["accuracy"]), abs_tol=1e-8
            ):
                raise RuntimeError(f"Conflicting recovered results for {key}")
            if previous is None:
                merged[key] = row
    return [merged[key] for key in sorted(merged)]


def summarize_rows(rows, args, checkpoint_hash):
    summaries = []
    for shot in args.shots:
        values = np.asarray(
            [float(row["accuracy"]) for row in rows if int(row["shot"]) == shot],
            dtype=np.float64,
        )
        if len(values) != args.iterations:
            continue
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(values))
        summaries.append(
            {
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
        )
    return summaries


def main():
    args = parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    if not args.shots or any(shot < 1 for shot in args.shots):
        raise ValueError("shots must contain positive integers")
    if args.lr_workers < 1:
        raise ValueError("lr-workers must be positive")

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
    detail_rows = merge_detail_rows(
        recover_csv_rows(run_dir / "iterations.csv", args, checkpoint_hash),
        recover_log_rows(args.resume_log, args, checkpoint_hash),
    )
    completed = {(int(row["shot"]), int(row["iteration"])) for row in detail_rows}
    run_dir.mkdir(parents=True, exist_ok=True)
    if detail_rows:
        write_csv(run_dir / "iterations.csv", DETAIL_FIELDS, detail_rows)
        write_csv(
            run_dir / "summary.csv",
            SUMMARY_FIELDS,
            summarize_rows(detail_rows, args, checkpoint_hash),
        )

    print(f"PROTOCOL={args.protocol}")
    print("ROLE=validation")
    print(f"QUERY_SPLIT={query_config['dataset']['query_split']}")
    print(f"DATASET_ROOT={query_config['dataset']['root']}")
    print(f"CHECKPOINT={checkpoint_path}")
    print(f"CHECKPOINT_SHA256={checkpoint_hash}")
    print(f"DEVICE={device}")
    print(f"QUERY_SAMPLES={len(query_labels)}")
    print(f"RECOVERED_ITERATIONS={len(detail_rows)}")
    print(f"LR_WORKERS={args.lr_workers}")

    for shot in args.shots:
        jobs = []
        for iteration in range(args.iterations):
            iteration_number = iteration + 1
            if (shot, iteration_number) in completed:
                continue
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
            jobs.append(((shot, iteration_number), support_features, support_labels, seed))

        accuracies = evaluate_lr_jobs(
            jobs, query_features, query_labels, args.lr_workers
        ) if jobs else {}
        for shot_iteration in sorted(accuracies):
            _, iteration_number = shot_iteration
            seed = args.base_seed + iteration_number - 1
            accuracy = accuracies[shot_iteration]
            detail_rows.append(
                {
                    "protocol": args.protocol,
                    "role": "validation",
                    "checkpoint": args.checkpoint,
                    "checkpoint_sha256": checkpoint_hash,
                    "shot": shot,
                    "iteration": iteration_number,
                    "seed": seed,
                    "accuracy": f"{accuracy:.8f}",
                    "source": "computed",
                }
            )
            completed.add((shot, iteration_number))
            detail_rows.sort(key=lambda row: (int(row["shot"]), int(row["iteration"])))
            write_csv(run_dir / "iterations.csv", DETAIL_FIELDS, detail_rows)
            write_csv(
                run_dir / "summary.csv",
                SUMMARY_FIELDS,
                summarize_rows(detail_rows, args, checkpoint_hash),
            )
            print(
                f"SHOT={shot} ITERATION={iteration_number}/{args.iterations} "
                f"SEED={seed} ACC={accuracy:.4f}"
            )
        shot_rows = [row for row in detail_rows if int(row["shot"]) == shot]
        if len(shot_rows) != args.iterations:
            raise RuntimeError(f"Shot {shot} finished with only {len(shot_rows)} rows")
        summary = next(
            row for row in summarize_rows(detail_rows, args, checkpoint_hash)
            if int(row["shot"]) == shot
        )
        print(
            f"SUMMARY SHOT={shot} N={summary['n']} "
            f"MEAN={float(summary['mean_accuracy']):.4f} "
            f"STD={float(summary['std_accuracy']):.4f} "
            f"CI95_HALF={float(summary['ci95_half']):.4f}"
        )

    summary_rows = summarize_rows(detail_rows, args, checkpoint_hash)
    write_csv(run_dir / "iterations.csv", DETAIL_FIELDS, detail_rows)
    write_csv(run_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
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
        "resume_log": str(Path(args.resume_log).expanduser().resolve()) if args.resume_log else "",
        "recovered_iterations": sum(row["source"].startswith("recovered:") for row in detail_rows),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_DIR={run_dir}")
    print("WISIG_VALIDATION_LR: PASS")


if __name__ == "__main__":
    main()
