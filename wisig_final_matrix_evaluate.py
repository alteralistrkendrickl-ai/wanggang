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


DRAFT_PATH_TEXT = (
    "/home/yuanlong/yl/wanggang_wisig_a1_fair/runs/"
    "WiSig_final_matrix_v2_preparation/FROZEN_MANIFEST_DRAFT.json"
)
DRAFT_PATH = Path(DRAFT_PATH_TEXT)
DRAFT_FILE_SHA256 = "542900a28a80ef20bdf07583f4a76932c5f36889dfd8d50ea3a571fa750c4c4f"
DRAFT_CANONICAL_SHA256 = "f763179fb4fcdf6569bdbc3f620bd7c239a1520d8892572b9415f3889d89c5df"
EXPECTED_FROZEN_MANIFEST_SHA256 = "5b6e1ea34f15bafe1ebdcf29f716ef15518c537d6b0503fd7963e25244c63471"
EXPECTED_TRAINING_PROVENANCE_SHA256 = (
    "ae177a24e8a2eee02d17c1a485304839143781d87870c1284ff1ecee120605cd"
)
CONFIRM_TOKEN = "UNSEAL-WISIG-FINAL-PAIRED-MATRIX-V2"
PROTOCOLS = ("cross-rx", "cross-day")
SHOTS = (1, 5, 10, 15, 20)
TRAIN_SEEDS = (2024, 2025, 2026, 2027, 2028)
ITERATIONS = 100
SUPPORT_BASE_SEED = 2024
NUM_CLASSES = 30
EXPECTED_ROWS_PER_PROTOCOL = 5000
T_CRITICAL_DF4_95 = 2.7764451051977987
FROZEN_OUTPUT_DIR_TEXT = (
    "/home/yuanlong/yl/wanggang_wisig_final_readiness/runs/"
    "WiSig_final_paired_matrix_v2"
)
FROZEN_OUTPUT_DIR = Path(FROZEN_OUTPUT_DIR_TEXT)
GLOBAL_UNSEAL_LOCK_TEXT = (
    "/home/yuanlong/yl/wanggang_wisig_final_readiness/runs/"
    "WiSig_final_paired_matrix_v2_UNSEAL_MANIFEST.json"
)
GLOBAL_UNSEAL_LOCK = Path(GLOBAL_UNSEAL_LOCK_TEXT)
FROZEN_RUNTIME = {
    "device": "cuda",
    "batch_size": 256,
    "lr_workers": 3,
    "output_dir": FROZEN_OUTPUT_DIR_TEXT,
    "global_unseal_lock": GLOBAL_UNSEAL_LOCK_TEXT,
    "protocol_order": list(PROTOCOLS),
}
STATISTICAL_PLAN = {
    "primary_estimand": "macro_average_candidate_minus_b0_over_five_shots",
    "primary_unit": "train_seed",
    "primary_family": "two_protocol_endpoints_holm_two_sided_alpha_0.05",
    "secondary_estimand": "candidate_minus_b0_for_each_shot",
    "secondary_families": "five_shots_within_each_protocol_holm_two_sided_alpha_0.05",
    "support_seed_pairing": True,
    "report_all_protocols_and_shots": True,
}

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
    "ci95_low", "ci95_high", "raw_p", "holm_adjusted_p", "holm_reject",
    "positive_train_seeds",
)
PRIMARY_SUMMARY_FIELDS = (
    "protocol", "candidate", "estimand", "train_seeds", "mean_difference",
    "ci95_low", "ci95_high", "raw_p", "holm_adjusted_p", "holm_reject",
    "positive_train_seeds",
)
TRAIN_SEED_EFFECT_FIELDS = (
    "protocol", "candidate", "train_seed", "shot", "mean_difference",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Frozen paired WiSig final matrix")
    parser.add_argument("--mode", choices=("audit", "run"), default="audit")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--resume", action="store_true")
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


def validate_training_provenance_hash(provenance):
    actual = canonical_hash(provenance)
    if actual != EXPECTED_TRAINING_PROVENANCE_SHA256:
        raise RuntimeError(
            "Code-frozen training provenance mismatch: expected "
            f"{EXPECTED_TRAINING_PROVENANCE_SHA256}, got {actual}"
        )
    return actual


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
        "path": DRAFT_PATH_TEXT,
        "file_sha256": DRAFT_FILE_SHA256,
        "canonical_sha256": DRAFT_CANONICAL_SHA256,
    }
    manifest["runtime_contract"] = FROZEN_RUNTIME
    manifest["statistical_plan"] = STATISTICAL_PLAN
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
    if manifest.get("runtime_contract") != FROZEN_RUNTIME:
        raise RuntimeError("Frozen runtime contract changed")
    if manifest.get("statistical_plan") != STATISTICAL_PLAN:
        raise RuntimeError("Frozen statistical plan changed")
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
    data_keys = [
        (row["protocol"], row["split"], row["kind"])
        for row in manifest["data_files"]
    ]
    expected_data_keys = {
        (protocol, split, kind)
        for protocol in PROTOCOLS
        for split in ("train", "test")
        for kind in ("X", "Y")
    }
    if len(data_keys) != len(set(data_keys)) or set(data_keys) != expected_data_keys:
        raise RuntimeError("Frozen final data-file grid is not exact")


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


def _require_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(
            f"Training provenance mismatch for {label}: "
            f"expected {expected!r}, got {actual!r}"
        )


def validate_training_config(config, protocol, variant, train_seed):
    expected_domain = {"cross-rx": "RX", "cross-day": "DAY"}[protocol]
    expected_a1 = {
        "B0": (False, False),
        "A1S": (True, False),
        "A1C": (True, True),
    }[variant]
    a1_config = config.get("a1")
    legacy_absent_b0 = variant == "B0" and a1_config is None
    if not legacy_absent_b0 and not isinstance(a1_config, dict):
        raise RuntimeError(
            f"Training provenance mismatch for {protocol}/{variant}/seed{train_seed}/a1: "
            "expected an explicit A1 configuration dictionary"
        )
    checks = {
        "random_seed": (config.get("random_seed"), train_seed),
        "epoch": (config.get("epoch"), 10),
        "threshold": (config.get("threshold"), 0),
        "dataset.name": (config.get("dataset", {}).get("name"), f"wisig-{protocol}"),
        "dataset.type": (config.get("dataset", {}).get("type"), "iq"),
        "dataset.normalize": (config.get("dataset", {}).get("normalize"), "power"),
        "dataset.batch_size": (config.get("dataset", {}).get("batch_size"), 32),
        "dataset.signal_length": (
            config.get("dataset", {}).get("signal_length"), 256
        ),
        "augmentation.awgn_enable": (
            config.get("augmentation", {}).get("awgn_enable"), False
        ),
        "encoder.name": (config.get("encoder", {}).get("name"), "CVTSLANet"),
        "encoder.feature_dim": (config.get("encoder", {}).get("feature_dim"), 1024),
        "encoder.seq_len": (
            config.get("encoder", {}).get("TSLA_config", {}).get("seq_len"), 256
        ),
        "encoder.patch_size": (
            config.get("encoder", {}).get("TSLA_config", {}).get("patch_size"), 32
        ),
        "lfdb.enabled": (config.get("lfdb", {}).get("enabled"), False),
    }
    if not legacy_absent_b0:
        checks.update({
            "a1.sampler_enabled": (
                a1_config.get("sampler_enabled"), expected_a1[0]
            ),
            "a1.loss_enabled": (a1_config.get("loss_enabled"), expected_a1[1]),
            "a1.domain_key": (a1_config.get("domain_key"), expected_domain),
        })
    if variant == "A1C":
        checks.update({
            "a1.weight": (a1_config.get("weight"), 0.1),
            "a1.temperature": (a1_config.get("temperature"), 0.1),
        })
    for label, (actual, expected) in checks.items():
        _require_equal(actual, expected, f"{protocol}/{variant}/seed{train_seed}/{label}")
    return "legacy_absent_b0" if legacy_absent_b0 else "explicit"


def audit_training_provenance(manifest):
    import torch

    evidence = []
    for model in manifest["models"]:
        protocol = model["protocol"]
        variant = model["variant"]
        train_seed = int(model["train_seed"])
        checkpoint_path = Path(model["checkpoint"]).with_name("checkpoint.pth")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Training provenance checkpoint is missing: {checkpoint_path}"
            )
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
        if not isinstance(config, dict):
            raise RuntimeError(
                f"Training checkpoint has no auditable config: {checkpoint_path}"
            )
        a1_config_status = validate_training_config(
            config, protocol, variant, train_seed
        )
        evidence.append({
            "protocol": protocol,
            "variant": variant,
            "train_seed": train_seed,
            "training_checkpoint": str(checkpoint_path),
            "training_checkpoint_sha256": sha256(checkpoint_path),
            "a1_config_status": a1_config_status,
        })
        del checkpoint
    return evidence


def acquire_or_resume(lock_path, manifest, runtime_contract, resume):
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = canonical_hash(manifest)
    if manifest_hash != EXPECTED_FROZEN_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen manifest hash mismatch: expected {EXPECTED_FROZEN_MANIFEST_SHA256}, "
            f"got {manifest_hash}"
        )
    if runtime_contract != FROZEN_RUNTIME:
        raise RuntimeError("Runtime contract differs from the code-frozen contract")
    record = {
        "manifest_sha256": manifest_hash,
        "manifest": manifest,
        "runtime_contract": runtime_contract,
    }
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


def validate_accuracy(value, label="accuracy"):
    try:
        accuracy = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 100.0:
        raise RuntimeError(f"{label} is outside finite [0, 100]: {value!r}")
    return accuracy


def load_existing_rows(path, protocol, manifest):
    if not path.is_file():
        return []
    models = model_lookup(manifest)
    expected_keys = expected_row_keys(protocol, manifest)
    rows, seen = [], set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DETAIL_FIELDS:
            raise RuntimeError("Existing final CSV schema differs from the frozen schema")
        for row in reader:
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
            validate_accuracy(row["accuracy"], f"Existing accuracy for {key}")
            if row["source"] != "computed":
                raise RuntimeError(f"Existing row has unexpected source: {key}")
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


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
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


def train_seed_effects(rows, manifest):
    by_key = {
        (row["protocol"], row["variant"], int(row["train_seed"]),
         int(row["shot"]), int(row["iteration"])): float(row["accuracy"])
        for row in rows
    }
    effects = []
    for protocol, variants in manifest["protocol_variants"].items():
        baseline, candidate = variants
        for train_seed in TRAIN_SEEDS:
            for shot in SHOTS:
                paired = [
                    by_key[(protocol, candidate, train_seed, shot, iteration)]
                    - by_key[(protocol, baseline, train_seed, shot, iteration)]
                    for iteration in range(1, ITERATIONS + 1)
                ]
                effects.append({
                    "protocol": protocol,
                    "candidate": candidate,
                    "train_seed": train_seed,
                    "shot": shot,
                    "mean_difference": f"{statistics.mean(paired):.8f}",
                })
    return effects


def _regularized_incomplete_beta(x, a, b):
    """Numerically stable regularized incomplete beta without SciPy."""
    if not 0.0 <= x <= 1.0 or a <= 0.0 or b <= 0.0:
        raise ValueError("Invalid incomplete-beta arguments")
    if x in (0.0, 1.0):
        return x

    def continued_fraction(aa, bb, xx):
        max_iterations, epsilon, floor = 200, 3.0e-14, 1.0e-300
        qab, qap, qam = aa + bb, aa + 1.0, aa - 1.0
        c = 1.0
        d = 1.0 - qab * xx / qap
        d = 1.0 / (floor if abs(d) < floor else d)
        result = d
        for iteration in range(1, max_iterations + 1):
            twice = 2 * iteration
            coefficient = (
                iteration * (bb - iteration) * xx
                / ((qam + twice) * (aa + twice))
            )
            d = 1.0 + coefficient * d
            d = floor if abs(d) < floor else d
            c = 1.0 + coefficient / c
            c = floor if abs(c) < floor else c
            d = 1.0 / d
            result *= d * c
            coefficient = -(
                (aa + iteration) * (qab + iteration) * xx
                / ((aa + twice) * (qap + twice))
            )
            d = 1.0 + coefficient * d
            d = floor if abs(d) < floor else d
            c = 1.0 + coefficient / c
            c = floor if abs(c) < floor else c
            d = 1.0 / d
            delta = d * c
            result *= delta
            if abs(delta - 1.0) <= epsilon:
                return result
        raise RuntimeError("Incomplete-beta continued fraction did not converge")

    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * continued_fraction(a, b, x) / a
    return 1.0 - front * continued_fraction(b, a, 1.0 - x) / b


def _two_sided_t_pvalue(t_statistic, degrees_of_freedom):
    if degrees_of_freedom <= 0 or not math.isfinite(t_statistic):
        if math.isinf(t_statistic) and degrees_of_freedom > 0:
            return 0.0
        raise ValueError("Invalid t-test arguments")
    x = degrees_of_freedom / (degrees_of_freedom + t_statistic ** 2)
    return _regularized_incomplete_beta(x, degrees_of_freedom / 2.0, 0.5)


def _effect_summary(values):

    values = [float(value) for value in values]
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    half = T_CRITICAL_DF4_95 * std / math.sqrt(len(values))
    if std == 0.0:
        raw_p = 1.0 if mean == 0.0 else 0.0
    else:
        t_statistic = mean / (std / math.sqrt(len(values)))
        raw_p = _two_sided_t_pvalue(t_statistic, len(values) - 1)
    if not math.isfinite(raw_p) or not 0.0 <= raw_p <= 1.0:
        raise RuntimeError(f"Invalid paired t-test p-value: {raw_p}")
    return mean, mean - half, mean + half, raw_p


def holm_adjust(raw_p_by_key, alpha=0.05):
    ordered = sorted(raw_p_by_key.items(), key=lambda item: (item[1], str(item[0])))
    count = len(ordered)
    adjusted, reject = {}, {}
    running_adjusted = 0.0
    still_rejecting = True
    for rank, (key, raw_p) in enumerate(ordered):
        multiplier = count - rank
        running_adjusted = max(running_adjusted, min(1.0, multiplier * raw_p))
        adjusted[key] = running_adjusted
        if still_rejecting and raw_p <= alpha / multiplier:
            reject[key] = True
        else:
            still_rejecting = False
            reject[key] = False
    return adjusted, reject


def paired_summaries(effect_rows, manifest):
    grouped = {}
    for row in effect_rows:
        key = (row["protocol"], int(row["shot"]))
        grouped.setdefault(key, []).append(float(row["mean_difference"]))
    raw = {key: _effect_summary(values) for key, values in grouped.items()}
    output = []
    for protocol, variants in manifest["protocol_variants"].items():
        candidate = variants[1]
        family = {
            shot: raw[(protocol, shot)][3]
            for shot in SHOTS
        }
        adjusted, reject = holm_adjust(family)
        for shot in SHOTS:
            values = grouped[(protocol, shot)]
            mean, low, high, raw_p = raw[(protocol, shot)]
            output.append({
                "protocol": protocol, "candidate": candidate, "shot": shot,
                "train_seeds": len(values), "mean_difference": f"{mean:.8f}",
                "ci95_low": f"{low:.8f}", "ci95_high": f"{high:.8f}",
                "raw_p": f"{raw_p:.10f}",
                "holm_adjusted_p": f"{adjusted[shot]:.10f}",
                "holm_reject": str(reject[shot]),
                "positive_train_seeds": sum(value > 0 for value in values),
            })
    return output


def primary_summaries(effect_rows, manifest):
    by_protocol_seed = {}
    for row in effect_rows:
        key = (row["protocol"], int(row["train_seed"]))
        by_protocol_seed.setdefault(key, []).append(float(row["mean_difference"]))
    protocol_values = {
        protocol: [
            statistics.mean(by_protocol_seed[(protocol, train_seed)])
            for train_seed in TRAIN_SEEDS
        ]
        for protocol in PROTOCOLS
    }
    raw = {
        protocol: _effect_summary(values)
        for protocol, values in protocol_values.items()
    }
    adjusted, reject = holm_adjust({
        protocol: summary[3] for protocol, summary in raw.items()
    })
    output = []
    for protocol in PROTOCOLS:
        candidate = manifest["protocol_variants"][protocol][1]
        values = protocol_values[protocol]
        mean, low, high, raw_p = raw[protocol]
        output.append({
            "protocol": protocol,
            "candidate": candidate,
            "estimand": STATISTICAL_PLAN["primary_estimand"],
            "train_seeds": len(values),
            "mean_difference": f"{mean:.8f}",
            "ci95_low": f"{low:.8f}",
            "ci95_high": f"{high:.8f}",
            "raw_p": f"{raw_p:.10f}",
            "holm_adjusted_p": f"{adjusted[protocol]:.10f}",
            "holm_reject": str(reject[protocol]),
            "positive_train_seeds": sum(value > 0 for value in values),
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
    support_x = np.load(files[("train", "X")], allow_pickle=False)
    support_y = np.load(files[("train", "Y")], allow_pickle=False)
    query_x = np.load(files[("test", "X")], allow_pickle=False)
    query_y = np.load(files[("test", "Y")], allow_pickle=False)
    if support_x.ndim == 3 and support_x.shape[1] != 2:
        support_x = support_x.transpose((0, 2, 1))
    if query_x.ndim == 3 and query_x.shape[1] != 2:
        query_x = query_x.transpose((0, 2, 1))
    if support_x.shape[1:] != (2, 256) or query_x.shape[1:] != (2, 256):
        raise RuntimeError("Frozen final IQ shape is not (N, 2, 256)")
    if support_y.ndim != 1 or query_y.ndim != 1:
        raise RuntimeError("Frozen final labels must be one-dimensional")
    if len(support_x) != len(support_y) or len(query_x) != len(query_y):
        raise RuntimeError("Frozen final X/Y lengths differ")
    for name, values in (
        ("support IQ", support_x), ("query IQ", query_x),
        ("support labels", support_y), ("query labels", query_y),
    ):
        if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
            raise RuntimeError(f"Frozen final {name} contains non-finite data")
    for name, labels in (("support", support_y), ("query", query_y)):
        if not np.equal(labels, np.floor(labels)).all():
            raise RuntimeError(f"Frozen final {name} labels are not exact integers")
    expected = np.arange(NUM_CLASSES)
    if not np.array_equal(np.unique(support_y).astype(int), expected):
        raise RuntimeError("Final support labels are not exactly 0..29")
    if not np.array_equal(np.unique(query_y).astype(int), expected):
        raise RuntimeError("Final query labels are not exactly 0..29")
    support_x = power_normalize_fn(support_x)
    query_x = power_normalize_fn(query_x)
    if not np.isfinite(support_x).all() or not np.isfinite(query_x).all():
        raise RuntimeError("Power-normalized final IQ contains non-finite data")
    return (
        support_x, support_y.astype(np.uint8),
        query_x, query_y.astype(np.uint8),
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
                accuracy = validate_accuracy(
                    scores[key], f"Computed accuracy for {protocol}/{key}"
                )
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
    effects = train_seed_effects(all_rows, manifest)
    atomic_write_csv(
        output_dir / "train_seed_effects.csv", TRAIN_SEED_EFFECT_FIELDS, effects,
    )
    atomic_write_csv(
        output_dir / "paired_summary.csv", PAIRED_SUMMARY_FIELDS,
        paired_summaries(effects, manifest),
    )
    atomic_write_csv(
        output_dir / "primary_summary.csv", PRIMARY_SUMMARY_FIELDS,
        primary_summaries(effects, manifest),
    )
    return all_rows


def build_result_evidence(output_dir):
    relative_paths = [
        "cross-rx/iterations.csv",
        "cross-rx/model_summary.csv",
        "cross-day/iterations.csv",
        "cross-day/model_summary.csv",
        "train_seed_effects.csv",
        "paired_summary.csv",
        "primary_summary.csv",
    ]
    files = []
    for relative in relative_paths:
        path = output_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Expected final result is missing: {path}")
        files.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    evidence = {
        "schema": "wisig-final-result-evidence-v2",
        "files": files,
    }
    evidence_path = output_dir / "RESULT_EVIDENCE.json"
    atomic_write_json(evidence_path, evidence)
    return evidence_path, sha256(evidence_path), evidence


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
    provenance = audit_training_provenance(manifest)
    provenance_hash = validate_training_provenance_hash(provenance)
    print(f"MODE={args.mode}")
    print(f"FROZEN_MANIFEST_SHA256={manifest_hash}")
    print("AUDITED_CHECKPOINTS=20/20")
    print("AUDITED_TRAINING_CONFIGS=20/20")
    print(f"TRAINING_PROVENANCE_SHA256={provenance_hash}")
    print("AUDITED_FINAL_FILES=8/8")
    print("EXPECTED_ROWS_CROSS_RX=5000")
    print("EXPECTED_ROWS_CROSS_DAY=5000")
    if args.mode == "audit":
        print("FINAL_ARRAYS_LOADED=False")
        print(f"FROZEN_OUTPUT_DIR={FROZEN_OUTPUT_DIR}")
        print(f"GLOBAL_UNSEAL_LOCK={GLOBAL_UNSEAL_LOCK}")
        print(f"PRIMARY_ESTIMAND={STATISTICAL_PLAN['primary_estimand']}")
        print("READY_FOR_USER_AUTHORIZATION=True")
        print("WISIG_FINAL_MATRIX_AUDIT: PASS")
        return
    if args.confirm != CONFIRM_TOKEN:
        raise RuntimeError(f"Run mode requires --confirm {CONFIRM_TOKEN}")
    output_dir = FROZEN_OUTPUT_DIR
    completed_path = output_dir / "COMPLETED.json"
    if args.resume and completed_path.is_file():
        raise RuntimeError("Final matrix is already complete; resume is forbidden")
    acquired = acquire_or_resume(
        GLOBAL_UNSEAL_LOCK, manifest, FROZEN_RUNTIME, args.resume
    )
    device = resolve_device(FROZEN_RUNTIME["device"])
    print(f"UNSEAL_MANIFEST_SHA256={acquired}")
    print(f"RESUME={args.resume}")
    print(f"DEVICE={device}")
    print(f"BATCH_SIZE={FROZEN_RUNTIME['batch_size']}")
    print(f"LR_WORKERS={FROZEN_RUNTIME['lr_workers']}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"GLOBAL_UNSEAL_LOCK={GLOBAL_UNSEAL_LOCK}")
    print("FINAL_ARRAYS_LOADED=ABOUT_TO_LOAD")
    rows = run_all(
        manifest, output_dir, device,
        FROZEN_RUNTIME["batch_size"], FROZEN_RUNTIME["lr_workers"],
    )
    evidence_path, evidence_hash, evidence = build_result_evidence(output_dir)
    completion = {
        "manifest_sha256": acquired,
        "runtime_contract": FROZEN_RUNTIME,
        "statistical_plan": STATISTICAL_PLAN,
        "training_provenance_sha256": provenance_hash,
        "rows": len(rows),
        "expected_rows": 10000,
        "result_evidence": evidence,
        "result_evidence_file": str(evidence_path),
        "result_evidence_sha256": evidence_hash,
        "status": "complete",
        "final_test_accessed": True,
    }
    atomic_write_json(completed_path, completion)
    print(f"RESULT_DIR={output_dir}")
    print(f"RESULT_EVIDENCE_SHA256={evidence_hash}")
    print(f"COMPLETED_SHA256={sha256(completed_path)}")
    print("WISIG_FINAL_PAIRED_MATRIX: PASS")


if __name__ == "__main__":
    main()
