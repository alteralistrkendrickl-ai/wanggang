"""Frozen confirmatory WiSig training-seed runs for the simplified A1 route."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


SEEDS = (2027, 2028)
PROTOCOL_VARIANTS = {
    "cross-rx": ("B0", "A1C"),
    "cross-day": ("B0", "A1S"),
}
CONFIRM_TOKEN = "RUN-WISIG-CONFIRM-SEEDS-2027-2028"
COMMON_ARGS = (
    "--encoder", "CVTSLANet",
    "--TSLA_len", "256",
    "--TSLA_patch", "32",
    "--batch_size", "32",
    "--rot_num", "8",
    "--feature_dim", "1024",
    "--epoch", "10",
    "--threshold", "0",
    "--save_freq", "1",
    "--no_awgn",
    "--no_lfdb",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen confirmatory WiSig training-seed runs"
    )
    parser.add_argument(
        "--protocol", choices=tuple(PROTOCOL_VARIANTS), required=True
    )
    parser.add_argument("--mode", choices=("audit", "run"), default="audit")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--state-dir", default="runs/WiSig_train_seed_confirmation_v1"
    )
    return parser.parse_args()


def run_tag(seed):
    return f"fair10e_seed{seed}"


def experiment_name(protocol, variant, seed):
    suffix = {"B0": "", "A1S": "_A1S", "A1C": "_A1C"}[variant]
    return f"CVTSLANet_wisig-{protocol}_iq_powerNorm{suffix}_{run_tag(seed)}"


def variant_args(variant):
    if variant == "B0":
        return ()
    if variant == "A1S":
        return ("--a1_sampler",)
    if variant == "A1C":
        return ("--a1_enable", "--a1_weight", "0.1", "--a1_temperature", "0.1")
    raise ValueError(variant)


def build_command(project_root, protocol, variant, seed):
    return [
        sys.executable,
        str(project_root / "pretext.py"),
        "--dataset",
        f"wisig-{protocol}",
        *COMMON_ARGS,
        *variant_args(variant),
        "--run_tag",
        run_tag(seed),
        "--random_seed",
        str(seed),
    ]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_manifest(project_root, protocol):
    variants = PROTOCOL_VARIANTS[protocol]
    tasks = []
    for seed in SEEDS:
        for variant in variants:
            name = experiment_name(protocol, variant, seed)
            output = project_root / "runs" / "Pretext_random_rot" / name
            tasks.append(
                {
                    "protocol": protocol,
                    "variant": variant,
                    "seed": seed,
                    "run_tag": run_tag(seed),
                    "experiment_name": name,
                    "output_dir": str(output.resolve()),
                    "command": build_command(
                        project_root, protocol, variant, seed
                    ),
                }
            )
    return {
        "schema": "wisig-train-seed-confirmation-v1",
        "protocol": protocol,
        "seeds": list(SEEDS),
        "variants": list(variants),
        "common_args": list(COMMON_ARGS),
        "tasks": tasks,
        "selection_rule": {
            "cross-rx": "A1C",
            "cross-day": "A1S",
            "selected_before_confirmation_seeds": True,
        },
        "final_test_accessed": False,
    }


def canonical_hash(manifest):
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_base_data(protocol):
    root = Path(
        os.path.expanduser(
            f"~/Datasets/WiSig_P3MC_strict_v1/ManyTx_{protocol.replace('-', '_')}"
        )
    )
    required = (
        root / "X_train_90Class.npy",
        root / "Y_train_90Class.npy",
        root / "X_val_90Class.npy",
        root / "Y_val_90Class.npy",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    return required


def acquire_manifest(state_dir, manifest, resume):
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "TRAINING_MANIFEST.json"
    record = {"manifest_sha256": canonical_hash(manifest), "manifest": manifest}
    if resume:
        if not path.is_file():
            raise RuntimeError("Cannot resume: training manifest does not exist")
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise RuntimeError("Cannot resume: frozen training manifest differs")
        return record["manifest_sha256"]
    try:
        descriptor = os.open(
            str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
        )
    except FileExistsError as error:
        raise RuntimeError("Training manifest already exists; use --resume") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    return record["manifest_sha256"]


def checkpoint_hashes(task):
    output = Path(task["output_dir"])
    best = output / "best_encoder.pth"
    final = output / "final_encoder.pth"
    if not (best.is_file() and final.is_file()):
        return None
    return {"best_sha256": sha256(best), "final_sha256": sha256(final)}


def verified_completed_task(task, recorded_entry):
    """Only a status-recorded completion may be skipped after interruption."""
    if not recorded_entry or recorded_entry.get("state") != "complete":
        return None
    current = checkpoint_hashes(task)
    if current is None:
        raise RuntimeError(
            f"Recorded completion is missing checkpoints: {task['experiment_name']}"
        )
    if recorded_entry.get("checkpoints") != current:
        raise RuntimeError(
            f"Completed checkpoint hash changed for {task['experiment_name']}"
        )
    return current


def write_status(path, status):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def audit(manifest, allow_existing):
    required = check_base_data(manifest["protocol"])
    project_root = Path(__file__).resolve().parent
    if not (project_root / "pretext.py").is_file():
        raise FileNotFoundError(project_root / "pretext.py")
    for task in manifest["tasks"]:
        output = Path(task["output_dir"])
        if output.exists() and not allow_existing:
            raise RuntimeError(f"Initial output path already exists: {output}")
    return required


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    manifest = frozen_manifest(project_root, args.protocol)
    state_dir = Path(args.state_dir).expanduser().resolve() / args.protocol
    required = audit(manifest, allow_existing=args.resume)
    print(f"PROTOCOL={args.protocol}")
    print(f"MODE={args.mode}")
    print(f"MANIFEST_SHA256={canonical_hash(manifest)}")
    print("SEEDS=2027,2028")
    print(f"VARIANTS={','.join(PROTOCOL_VARIANTS[args.protocol])}")
    print("FINAL_TEST_ACCESSED=False")
    for path in required:
        print(f"BASE_FILE_PRESENT={path}")
    for task in manifest["tasks"]:
        print(
            f"TASK={task['variant']} SEED={task['seed']} "
            f"OUTPUT={task['output_dir']}"
        )
    if args.mode == "audit":
        print("WISIG_TRAIN_SEED_CONFIRMATION_AUDIT: PASS")
        return
    if args.confirm != CONFIRM_TOKEN:
        raise RuntimeError(f"Run mode requires --confirm {CONFIRM_TOKEN}")

    manifest_hash = acquire_manifest(state_dir, manifest, args.resume)
    status_path = state_dir / "status.json"
    status = {"manifest_sha256": manifest_hash, "tasks": {}}
    if args.resume and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("Status manifest hash differs")

    for task in manifest["tasks"]:
        key = f"{task['variant']}_seed{task['seed']}"
        recorded_entry = status["tasks"].get(key)
        complete = verified_completed_task(task, recorded_entry)
        if complete is not None:
            print(f"SKIP_COMPLETED={key}")
            continue

        status["tasks"][key] = {"state": "running"}
        write_status(status_path, status)
        print(f"START_TASK={key}")
        subprocess.run(task["command"], cwd=project_root, check=True)
        complete = checkpoint_hashes(task)
        if complete is None:
            raise RuntimeError(f"Task returned without complete checkpoints: {key}")
        status["tasks"][key] = {
            "state": "complete",
            "checkpoints": complete,
        }
        write_status(status_path, status)
        print(f"COMPLETE_TASK={key} BEST_SHA256={complete['best_sha256']}")

    print(f"STATE_DIR={state_dir}")
    print("WISIG_TRAIN_SEED_CONFIRMATION: PASS")


if __name__ == "__main__":
    main()
