import argparse
import json
import os

import numpy as np


SPLITS = ("train", "val", "test")


def _load(root, prefix, split, class_count, mmap_mode="r"):
    path = os.path.expanduser(
        os.path.join(root, f"{prefix}_{split}_{class_count}Class.npy")
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode=mmap_mode)


def verify_protocol(root, class_count):
    root = os.path.expanduser(root)
    manifest_path = os.path.join(root, f"protocol_{class_count}Class.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)

    protocol = manifest["protocol"]
    domain_prefix = "RX" if protocol == "cross_rx" else "DAY" if protocol == "cross_day" else None
    report = {"root": os.path.abspath(root), "protocol": protocol, "splits": {}}
    domains = {}

    for split in SPLITS:
        x = _load(root, "X", split, class_count)
        y = _load(root, "Y", split, class_count)
        rx = _load(root, "RX", split, class_count)
        day = _load(root, "DAY", split, class_count)
        global_tx = _load(root, "TX_GLOBAL", split, class_count)
        lengths = {len(x), len(y), len(rx), len(day), len(global_tx)}
        if len(lengths) != 1:
            raise ValueError(f"Mismatched array lengths in {split}: {lengths}")
        labels = set(np.unique(y).astype(int).tolist())
        expected_labels = set(range(class_count))
        if labels != expected_labels:
            missing = sorted(expected_labels - labels)
            extra = sorted(labels - expected_labels)
            raise ValueError(
                f"Invalid labels in {split}; missing={missing}, extra={extra}"
            )
        report["splits"][split] = {
            "samples": int(len(y)),
            "rx": sorted(np.unique(rx).astype(int).tolist()),
            "day": sorted(np.unique(day).astype(int).tolist()),
            "global_tx": sorted(np.unique(global_tx).astype(int).tolist()),
        }
        for label in expected_labels:
            mapped = np.unique(global_tx[np.asarray(y) == label]).astype(int).tolist()
            if len(mapped) != 1:
                raise ValueError(
                    f"Local label {label} maps to global transmitters {mapped} in {split}."
                )
        if domain_prefix is not None:
            domain_array = rx if domain_prefix == "RX" else day
            domains[split] = set(np.unique(domain_array).astype(int).tolist())

    if domain_prefix is not None:
        overlaps = {
            "train_val": sorted(domains["train"] & domains["val"]),
            "train_test": sorted(domains["train"] & domains["test"]),
            "val_test": sorted(domains["val"] & domains["test"]),
        }
        if any(overlaps.values()):
            raise ValueError(
                f"{protocol} domain leakage detected: {overlaps}"
            )
        report["domain_leakage"] = False
        report["held_out_domains"] = manifest.get("held_out_domains", {})

    return report


def verify_strict_root(output_root):
    output_root = os.path.abspath(os.path.expanduser(output_root))
    manifest_path = os.path.join(output_root, "strict_p3mc_protocol.json")
    with open(manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    groups = {
        name: set(int(value) for value in values)
        for name, values in manifest["identity_groups"].items()
    }
    names = list(groups)
    identity_overlaps = {
        f"{names[i]}__{names[j]}": sorted(groups[names[i]] & groups[names[j]])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    }
    if any(identity_overlaps.values()):
        raise ValueError(f"Identity leakage detected: {identity_overlaps}")

    reports = {}
    base_count = len(groups["base_pretraining"])
    for protocol_name, spec in manifest["protocols"].items():
        root = os.path.join(output_root, spec["directory"])
        protocol_reports = {}
        checks = [
            (root, base_count, groups["base_pretraining"], "base_pretraining"),
        ]
        checks.extend(
            (
                root,
                int(class_count),
                set(int(value) for value in global_indices),
                f"test_novel_{class_count}",
            )
            for class_count, global_indices in manifest["target_tasks"].items()
        )
        checks.append(
            (
                os.path.join(root, "validation_novel"),
                len(groups["validation_novel"]),
                groups["validation_novel"],
                "validation_novel",
            )
        )
        for check_root, class_count, expected_global, label in checks:
            report = verify_protocol(check_root, class_count)
            observed = {
                split: set(report["splits"][split]["global_tx"])
                for split in SPLITS
            }
            mismatched = {
                split: {
                    "missing": sorted(expected_global - values),
                    "extra": sorted(values - expected_global),
                }
                for split, values in observed.items()
                if values != expected_global
            }
            if mismatched:
                raise ValueError(f"{label} identity mismatch: {mismatched}")
            protocol_reports[label] = report
        reports[protocol_name] = protocol_reports

    return {
        "root": output_root,
        "protocol_version": manifest["protocol_version"],
        "identity_leakage": False,
        "identity_groups": {name: sorted(values) for name, values in groups.items()},
        "protocols": reports,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify fixed ManyTx IID/cross-domain data protocols."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--classes", type=int, default=90)
    parser.add_argument(
        "--strict-root",
        action="store_true",
        help="verify every identity and domain split described by strict_p3mc_protocol.json",
    )
    args = parser.parse_args()
    report = verify_strict_root(args.root) if args.strict_root else verify_protocol(args.root, args.classes)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Protocol verification: PASS")


if __name__ == "__main__":
    main()
