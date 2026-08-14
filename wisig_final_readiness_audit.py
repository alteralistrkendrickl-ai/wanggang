"""Read-only readiness audit for the frozen WiSig validation evidence.

This command deliberately audits validation artifacts only.  It never opens final
test arrays and it never authorizes or launches the one-time final evaluation.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


PROTOCOL_VARIANTS = {
    "cross-rx": ("B0", "A1C"),
    "cross-day": ("B0", "A1S"),
}
TRAIN_SEEDS = (2024, 2025, 2026, 2027, 2028)
SHOTS = (1, 5, 10, 15, 20)
ITERATIONS = 100
BASE_SEED = 2024
T_CRITICAL_DF4_95 = 2.7764451051977987
DEFAULT_VALIDATION_DIRS = (
    "runs/WiSig_validation_A1_fair10e_smoke",
    "runs/WiSig_validation_train_seed_robustness_v1",
    "runs/WiSig_validation_train_seed_confirmation_v1",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit frozen WiSig validation evidence without reading final data"
    )
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--baseline-2024-root",
        default="",
        help="Root containing the frozen seed-2024 B0 checkpoints and validation CSVs",
    )
    parser.add_argument(
        "--validation-dir", action="append", default=[],
        help="Validation result root; repeat to override the three frozen defaults",
    )
    parser.add_argument(
        "--output",
        default="runs/WiSig_final_readiness_audit_v1/READINESS_AUDIT.json",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experiment_name(protocol, variant, train_seed):
    suffix = {"B0": "", "A1S": "_A1S", "A1C": "_A1C"}[variant]
    seed_suffix = "" if train_seed == 2024 else f"_seed{train_seed}"
    return (
        f"CVTSLANet_wisig-{protocol}_iq_powerNorm{suffix}_fair10e{seed_suffix}"
    )


def expected_models(project_root, baseline_2024_root):
    models = []
    for protocol, variants in PROTOCOL_VARIANTS.items():
        for train_seed in TRAIN_SEEDS:
            for variant in variants:
                name = experiment_name(protocol, variant, train_seed)
                checkpoint_root = project_root
                if train_seed == 2024 and variant == "B0":
                    checkpoint_root = baseline_2024_root
                    name = f"CVTSLANet_wisig-{protocol}_iq_powerNorm"
                models.append(
                    {
                        "protocol": protocol,
                        "variant": variant,
                        "train_seed": train_seed,
                        "experiment_name": name,
                        "checkpoint": checkpoint_root / "runs" /
                        "Pretext_random_rot" / name / "best_encoder.pth",
                    }
                )
    return models


def discover_csvs(validation_dirs):
    paths = []
    for root in validation_dirs:
        if not root.is_dir():
            raise FileNotFoundError(f"Validation directory missing: {root}")
        paths.extend(root.rglob("iterations.csv"))
    return sorted(set(path.resolve() for path in paths))


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_rows(rows, protocol, checkpoint_hash):
    expected_keys = {(shot, iteration) for shot in SHOTS for iteration in range(1, 101)}
    keys = []
    values = {}
    for row in rows:
        if row.get("protocol") != protocol:
            raise RuntimeError(f"Protocol mismatch in validation CSV: {row.get('protocol')}")
        if row.get("role") != "validation":
            raise RuntimeError(f"Non-validation row encountered: {row.get('role')}")
        if row.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError("Checkpoint hash mismatch in validation CSV")
        shot = int(row["shot"])
        iteration = int(row["iteration"])
        seed = int(row["seed"])
        if seed != BASE_SEED + iteration - 1:
            raise RuntimeError(f"Support seed mismatch at {(shot, iteration)}")
        key = (shot, iteration)
        keys.append(key)
        values[key] = float(row["accuracy"])
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if len(rows) != len(expected_keys) or set(keys) != expected_keys or duplicates:
        raise RuntimeError(
            f"Validation grid mismatch: rows={len(rows)}, duplicates={duplicates[:3]}"
        )
    return values


def mean_ci95(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    half = T_CRITICAL_DF4_95 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def aggregate_differences(results):
    summaries = []
    for protocol, variants in PROTOCOL_VARIANTS.items():
        baseline, candidate = variants
        for shot in SHOTS:
            per_train_seed = []
            for train_seed in TRAIN_SEEDS:
                b0 = results[(protocol, baseline, train_seed)]
                method = results[(protocol, candidate, train_seed)]
                paired = [
                    method[(shot, iteration)] - b0[(shot, iteration)]
                    for iteration in range(1, ITERATIONS + 1)
                ]
                per_train_seed.append(statistics.mean(paired))
            mean, half = mean_ci95(per_train_seed)
            summaries.append(
                {
                    "protocol": protocol,
                    "candidate": candidate,
                    "shot": shot,
                    "train_seed_differences": per_train_seed,
                    "mean_difference": mean,
                    "train_seed_ci95_low": mean - half,
                    "train_seed_ci95_high": mean + half,
                }
            )
    return summaries


def audit(project_root, baseline_2024_root, validation_dirs):
    csv_paths = discover_csvs(validation_dirs)
    csv_by_key = defaultdict(list)
    for path in csv_paths:
        rows = read_csv(path)
        if not rows:
            continue
        key = (rows[0].get("protocol"), rows[0].get("checkpoint_sha256"))
        csv_by_key[key].append((path, rows))

    model_records = []
    results = {}
    for model in expected_models(project_root, baseline_2024_root):
        checkpoint = model["checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
        checkpoint_hash = sha256(checkpoint)
        matches = csv_by_key.get((model["protocol"], checkpoint_hash), [])
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one validation CSV for {model['experiment_name']}; "
                f"found {len(matches)}"
            )
        csv_path, rows = matches[0]
        values = validate_rows(rows, model["protocol"], checkpoint_hash)
        key = (model["protocol"], model["variant"], model["train_seed"])
        results[key] = values
        model_records.append(
            {
                **{k: v for k, v in model.items() if k != "checkpoint"},
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_hash,
                "validation_csv": str(csv_path),
                "validation_rows": len(rows),
            }
        )

    return {
        "schema": "wisig-final-readiness-audit-v1",
        "final_arrays_loaded": False,
        "final_entrypoint_status": "STALE_DO_NOT_RUN",
        "expected_model_count": 20,
        "audited_model_count": len(model_records),
        "validation_grid": {
            "shots": list(SHOTS),
            "iterations": ITERATIONS,
            "support_seeds": [BASE_SEED, BASE_SEED + ITERATIONS - 1],
            "train_seeds": list(TRAIN_SEEDS),
        },
        "models": model_records,
        "paired_validation_summary": aggregate_differences(results),
        "external_baseline": {
            "name": "FS-SEI repository adapted STC-CVCNN",
            "status": "screening_only_not_faithful_dmel_reproduction",
            "included_in_final_matrix": False,
        },
        "ready_for_final_unseal": False,
        "blocking_reasons": [
            "The existing final entrypoint still implements the obsolete per-shot route.",
            "A new paired-matrix final manifest and runner have not been implemented or audited.",
            "The external FS-SEI result is an adapted screening baseline, not a faithful reproduction.",
        ],
    }


def main():
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    if args.baseline_2024_root:
        baseline_2024_root = Path(args.baseline_2024_root).expanduser().resolve()
    else:
        baseline_2024_root = project_root.parent / "wanggang_wisig_audit"
    if args.validation_dir:
        validation_dirs = [Path(path).expanduser().resolve() for path in args.validation_dir]
    else:
        validation_dirs = [project_root / path for path in DEFAULT_VALIDATION_DIRS]
        validation_dirs.append(
            baseline_2024_root / "runs" / "WiSig_validation_lr_100runs"
        )
    report = audit(project_root, baseline_2024_root, validation_dirs)
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = project_root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"AUDITED_MODELS={report['audited_model_count']}/20")
    print("FINAL_ARRAYS_LOADED=False")
    print("FINAL_ENTRYPOINT_STATUS=STALE_DO_NOT_RUN")
    for row in report["paired_validation_summary"]:
        print(
            f"SUMMARY PROTOCOL={row['protocol']} CANDIDATE={row['candidate']} "
            f"SHOT={row['shot']} TRAIN_SEEDS=5 DIFF={row['mean_difference']:+.4f} "
            f"CI95=[{row['train_seed_ci95_low']:+.4f},"
            f"{row['train_seed_ci95_high']:+.4f}]"
        )
    print(f"REPORT={output.resolve()}")
    print("READY_FOR_FINAL_UNSEAL=False")
    print("WISIG_FINAL_READINESS_AUDIT: PASS")


if __name__ == "__main__":
    main()
