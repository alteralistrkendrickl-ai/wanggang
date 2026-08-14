"""Prepare the frozen paired-matrix WiSig final manifest without loading arrays.

The command hashes final files as opaque bytes and verifies all 20 checkpoints.
Run mode is intentionally unavailable in this preparation stage.
"""

import argparse
import hashlib
import json
from pathlib import Path


READINESS_AUDIT_SHA256 = (
    "ba3c4fcbadab7aee26f213f34c03dcb72d2f1f2cf76123c50a0f69c36b870aa1"
)
READINESS_AUDIT_PATH = Path(
    "/home/yuanlong/yl/wanggang_wisig_a1_fair/runs/"
    "WiSig_final_readiness_audit_v1/READINESS_AUDIT.json"
)
PROTOCOL_VARIANTS = {
    "cross-rx": ("B0", "A1C"),
    "cross-day": ("B0", "A1S"),
}
TRAIN_SEEDS = (2024, 2025, 2026, 2027, 2028)
SHOTS = (1, 5, 10, 15, 20)
ITERATIONS = 100
SUPPORT_BASE_SEED = 2024
NUM_CLASSES = 30
RUN_ENABLED = False

DATASET_ROOTS = {
    "cross-rx": Path(
        "/home/yuanlong/Datasets/WiSig_P3MC_strict_v1/ManyTx_cross_rx"
    ),
    "cross-day": Path(
        "/home/yuanlong/Datasets/WiSig_P3MC_strict_v1/ManyTx_cross_day"
    ),
}

CHECKPOINT_HASHES = {
    ("cross-rx", "B0", 2024): "8d8c83e5600a19b0e47b99d37f9e8a07685c48b3b96c1808e3d98c9a2e3a2950",
    ("cross-rx", "A1C", 2024): "e3597d2a17afc75122e9a36bbefdefedeb9caa17c804c166a3f2d55647938442",
    ("cross-rx", "B0", 2025): "34c1567a7958afe7463d5dd20707fa4af301c5f489a707bff445d76a7c3033fe",
    ("cross-rx", "A1C", 2025): "e2f88383f9a59683ca8a75499f2e25efd0a152d1702f087fb11c8e275355d168",
    ("cross-rx", "B0", 2026): "0201af582afdff8668fcbf21ead2a2891b6bfd173b2f18230763d0fb915c4a73",
    ("cross-rx", "A1C", 2026): "63f21f021cbeec87c706c074b2b5e86b7caf0c2d27b582fc9631e333377b8e98",
    ("cross-rx", "B0", 2027): "826b2b4b936f5fc6f2425252cbac0d2ec1d5e75f8182a1863457f55739366e8f",
    ("cross-rx", "A1C", 2027): "0f71991b8728074b29a43033a4a8fe36c8b62f0e89bda112909f7789b154edf3",
    ("cross-rx", "B0", 2028): "6f748f45125243aaaa6b8b33c59e9c207da2fd4bf0919d557a39129813db851b",
    ("cross-rx", "A1C", 2028): "300e54a18362313d0ccbbd18e72c0e9d3f8d9833b541772e9dea55f8b71fcdbe",
    ("cross-day", "B0", 2024): "ff49ec9a41b49166b2d579ba78be5fae2b68f5990d08eb44daa7aceb12072afd",
    ("cross-day", "A1S", 2024): "545a0e2770169c8909841ed675b3a40bce6785f0f2ec66bb2e37e3e9eb4bcdde",
    ("cross-day", "B0", 2025): "dbb2d004f5295cbcffaa5b509012441ae351c0ba38698b07ff98213aef98b761",
    ("cross-day", "A1S", 2025): "db2cbb16273a0e8b593a0ef3e9fe4ed9f064b8720020cb833fea7057f9dc9059",
    ("cross-day", "B0", 2026): "3d8e0e6ed58db9f0bcb498b06f3832b582e37dd81e256f2354546607be8d3716",
    ("cross-day", "A1S", 2026): "0bf70ec9d80d469ae92e18cb0f06825cebaf33243004874737d92715105e5f6a",
    ("cross-day", "B0", 2027): "4d8362c2cd0a029c41898ac86c58d215ba4dde82c204c36ddae4eccb1c2965ba",
    ("cross-day", "A1S", 2027): "6d5004e50cb201cb96032332125d576f0ca5324b973a5fec84c714b07de1019f",
    ("cross-day", "B0", 2028): "39f026bf8aebe826925008281600104ff132ef3d01793f2e970dc5cf6480694a",
    ("cross-day", "A1S", 2028): "0c03bcbee53df9ee4ca3a9124415e3d5c80c97bde3785f09500f83c61a93a104",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare frozen WiSig final matrix")
    parser.add_argument("--mode", choices=("audit", "run"), default="audit")
    parser.add_argument(
        "--output-dir", default="runs/WiSig_final_matrix_v2_preparation"
    )
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


def checkpoint_path(protocol, variant, train_seed):
    if train_seed == 2024 and variant == "B0":
        root = Path("/home/yuanlong/yl/wanggang_wisig_audit")
        name = f"CVTSLANet_wisig-{protocol}_iq_powerNorm"
    else:
        root = Path("/home/yuanlong/yl/wanggang_wisig_a1_fair")
        suffix = {"B0": "", "A1S": "_A1S", "A1C": "_A1C"}[variant]
        seed_suffix = "" if train_seed == 2024 else f"_seed{train_seed}"
        name = (
            f"CVTSLANet_wisig-{protocol}_iq_powerNorm{suffix}_fair10e"
            f"{seed_suffix}"
        )
    return root / "runs" / "Pretext_random_rot" / name / "best_encoder.pth"


def frozen_models():
    models = []
    for protocol, variants in PROTOCOL_VARIANTS.items():
        for train_seed in TRAIN_SEEDS:
            for variant in variants:
                key = (protocol, variant, train_seed)
                models.append(
                    {
                        "protocol": protocol,
                        "variant": variant,
                        "train_seed": train_seed,
                        "checkpoint": str(checkpoint_path(*key)),
                        "checkpoint_sha256": CHECKPOINT_HASHES[key],
                    }
                )
    return models


def frozen_data_files():
    files = []
    for protocol, root in DATASET_ROOTS.items():
        for split in ("train", "test"):
            for kind in ("X", "Y"):
                files.append(
                    {
                        "protocol": protocol,
                        "split": split,
                        "kind": kind,
                        "path": str(root / f"{kind}_{split}_30Class.npy"),
                    }
                )
    return files


def validate_readiness_report():
    if not READINESS_AUDIT_PATH.is_file():
        raise FileNotFoundError(READINESS_AUDIT_PATH)
    actual = sha256(READINESS_AUDIT_PATH)
    if actual != READINESS_AUDIT_SHA256:
        raise RuntimeError(
            f"Readiness report hash mismatch: expected {READINESS_AUDIT_SHA256}, got {actual}"
        )
    report = json.loads(READINESS_AUDIT_PATH.read_text(encoding="utf-8"))
    if report.get("audited_model_count") != 20:
        raise RuntimeError("Readiness report does not contain 20 audited models")
    if report.get("final_arrays_loaded") is not False:
        raise RuntimeError("Readiness report unexpectedly accessed final arrays")
    report_models = {
        (row["protocol"], row["variant"], int(row["train_seed"])):
        (row["checkpoint"], row["checkpoint_sha256"])
        for row in report["models"]
    }
    expected = {
        (row["protocol"], row["variant"], row["train_seed"]):
        (row["checkpoint"], row["checkpoint_sha256"])
        for row in frozen_models()
    }
    if report_models != expected:
        raise RuntimeError("Embedded final matrix differs from readiness evidence")


def build_draft_manifest(compute_hashes=True):
    models = frozen_models()
    data_files = frozen_data_files()
    if compute_hashes:
        validate_readiness_report()
        for model in models:
            path = Path(model["checkpoint"])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = sha256(path)
            if actual != model["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint hash mismatch: {path}")
        for item in data_files:
            path = Path(item["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            item["sha256"] = sha256(path)
            item["size_bytes"] = path.stat().st_size
    return {
        "schema": "wisig-final-paired-matrix-v2-draft",
        "readiness_audit": {
            "path": str(READINESS_AUDIT_PATH),
            "sha256": READINESS_AUDIT_SHA256,
        },
        "final_arrays_loaded": False,
        "protocol_variants": {
            protocol: list(variants)
            for protocol, variants in PROTOCOL_VARIANTS.items()
        },
        "train_seeds": list(TRAIN_SEEDS),
        "shots": list(SHOTS),
        "iterations": ITERATIONS,
        "support_base_seed": SUPPORT_BASE_SEED,
        "num_classes": NUM_CLASSES,
        "expected_rows_per_protocol": 5000,
        "pairing": ["protocol", "train_seed", "shot", "support_seed"],
        "primary_statistical_unit": "train_seed",
        "selection": "all frozen B0-candidate pairs; no per-shot routing",
        "models": models,
        "data_files": data_files,
        "evaluation": {
            "encoder": "CVTSLANet",
            "feature_dim": 1024,
            "signal_length": 256,
            "normalization": "power",
            "classifier": "LogisticRegression",
            "max_iter": 1000,
            "random_state": "support_seed",
        },
        "run_enabled": False,
    }


def main():
    args = parse_args()
    if args.mode == "run" or RUN_ENABLED:
        raise RuntimeError(
            "Final run is disabled in the preparation stage; no arrays were loaded"
        )
    manifest = build_draft_manifest(compute_hashes=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "FROZEN_MANIFEST_DRAFT.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_hash = canonical_hash(manifest)
    print("AUDITED_CHECKPOINTS=20/20")
    print("AUDITED_FINAL_FILES=8/8")
    print("EXPECTED_ROWS_CROSS_RX=5000")
    print("EXPECTED_ROWS_CROSS_DAY=5000")
    print("FINAL_ARRAYS_LOADED=False")
    print("RUN_ENABLED=False")
    print(f"DRAFT_MANIFEST_SHA256={manifest_hash}")
    print(f"DRAFT_MANIFEST={output}")
    print("WISIG_FINAL_MATRIX_PREPARATION: PASS")


if __name__ == "__main__":
    main()
