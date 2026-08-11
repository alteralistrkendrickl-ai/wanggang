import argparse
import gc
import json
import os
import pickle

import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def _as_iq_samples(block):
    samples = np.asarray(block, dtype=np.float32)
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError(f"Unexpected sample shape: {samples.shape}")
    return samples.transpose(0, 2, 1).astype(np.float32, copy=False)


def _collect_class_records(
    tx_data,
    tx_index,
    equalized_index,
    global_tx_index=None,
):
    if global_tx_index is None:
        global_tx_index = tx_index
    sample_parts = []
    rx_parts = []
    day_parts = []
    for rx_index, rx_data in enumerate(tx_data):
        for day_index, date_data in enumerate(rx_data):
            block = date_data[equalized_index]
            if not len(block):
                continue
            samples = _as_iq_samples(block)
            sample_parts.append(samples)
            rx_parts.append(np.full(len(samples), rx_index, dtype=np.int16))
            day_parts.append(np.full(len(samples), day_index, dtype=np.int16))

    if not sample_parts:
        return {
            "x": np.empty((0, 2, 256), dtype=np.float32),
            "tx": np.empty(0, dtype=np.int16),
            "global_tx": np.empty(0, dtype=np.int16),
            "rx": np.empty(0, dtype=np.int16),
            "day": np.empty(0, dtype=np.int16),
        }

    x = np.concatenate(sample_parts, axis=0)
    return {
        "x": x,
        "tx": np.full(len(x), tx_index, dtype=np.int16),
        "global_tx": np.full(len(x), global_tx_index, dtype=np.int16),
        "rx": np.concatenate(rx_parts),
        "day": np.concatenate(day_parts),
    }


def _split_iid_indices(count, test_ratio, val_ratio, rng):
    indices = np.arange(count)
    rng.shuffle(indices)
    test_count = max(1, int(round(count * test_ratio)))
    remaining = count - test_count
    val_count = max(1, int(round(remaining * val_ratio)))
    if remaining - val_count < 1:
        raise ValueError(
            f"Class with {count} samples is too small for train/val/test splitting."
        )
    return {
        "test": indices[:test_count],
        "val": indices[test_count:test_count + val_count],
        "train": indices[test_count + val_count:],
    }


def _empty_split_parts():
    return {
        split: {key: [] for key in ("x", "tx", "global_tx", "rx", "day")}
        for split in SPLIT_NAMES
    }


def _append_records(parts, split, records, indices):
    for key in ("x", "tx", "global_tx", "rx", "day"):
        parts[split][key].append(records[key][indices])


def _write_split_files(parts, output_dir, class_count):
    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    for split in SPLIT_NAMES:
        if not parts[split]["x"]:
            raise ValueError(f"Protocol produced an empty {split} split.")
        arrays = {
            key: np.concatenate(parts[split][key], axis=0)
            for key in ("x", "tx", "global_tx", "rx", "day")
        }
        np.save(
            os.path.join(output_dir, f"X_{split}_{class_count}Class.npy"),
            arrays["x"],
        )
        np.save(
            os.path.join(output_dir, f"Y_{split}_{class_count}Class.npy"),
            arrays["tx"].astype(np.int64),
        )
        np.save(
            os.path.join(output_dir, f"TX_GLOBAL_{split}_{class_count}Class.npy"),
            arrays["global_tx"],
        )
        np.save(
            os.path.join(output_dir, f"RX_{split}_{class_count}Class.npy"),
            arrays["rx"],
        )
        np.save(
            os.path.join(output_dir, f"DAY_{split}_{class_count}Class.npy"),
            arrays["day"],
        )
        summary[split] = {
            "samples": int(len(arrays["x"])),
            "tx": sorted(np.unique(arrays["tx"]).astype(int).tolist()),
            "global_tx": sorted(
                np.unique(arrays["global_tx"]).astype(int).tolist()
            ),
            "rx": sorted(np.unique(arrays["rx"]).astype(int).tolist()),
            "day": sorted(np.unique(arrays["day"]).astype(int).tolist()),
        }
    return summary


def _metadata_names(values):
    if values is None:
        return []
    return [str(value) for value in values]


def _write_protocol_manifest(output_dir, protocol, class_count, summary, **details):
    manifest_path = os.path.join(output_dir, "protocol.json")
    common_keys = {
        "source",
        "equalized_index",
        "rx_names",
        "day_names",
    }
    common = {key: details[key] for key in common_keys if key in details}
    class_details = {
        key: value for key, value in details.items() if key not in common_keys
    }
    manifest = {"protocol": protocol, **common, "classes": {}}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as file:
            existing = json.load(file)
        if existing.get("protocol") == protocol:
            manifest = existing
            manifest.update(common)
            manifest.setdefault("classes", {})
    class_manifest = {
        "class_count": int(class_count),
        "splits": summary,
        **class_details,
    }
    manifest["classes"][str(class_count)] = class_manifest
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    with open(
        os.path.join(output_dir, f"protocol_{class_count}Class.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump({"protocol": protocol, **common, **class_manifest}, file, indent=2, ensure_ascii=False)


def _write_iid_protocol(
    class_records,
    class_count,
    output_dir,
    seed,
    test_ratio,
    val_ratio,
    metadata,
):
    rng = np.random.default_rng(seed)
    parts = _empty_split_parts()
    for label in range(class_count):
        records = class_records[label]
        indices = _split_iid_indices(len(records["x"]), test_ratio, val_ratio, rng)
        for split in SPLIT_NAMES:
            _append_records(parts, split, records, indices[split])

    summary = _write_split_files(parts, output_dir, class_count)
    _write_protocol_manifest(
        output_dir,
        "iid",
        class_count,
        summary,
        seed=int(seed),
        test_ratio=float(test_ratio),
        validation_ratio_within_training=float(val_ratio),
        **metadata,
    )
    return summary


def _all_domain_ids(class_records, class_count, domain_key):
    domains = set()
    for records in class_records[:class_count]:
        domains.update(np.unique(records[domain_key]).astype(int).tolist())
    return sorted(domains)


def _parse_domain_ids(values):
    if not values:
        return None
    return sorted({int(value) for value in values})


def _resolve_domain_split(
    class_records,
    class_count,
    domain_key,
    val_domains=None,
    test_domains=None,
    val_count=1,
    test_count=1,
):
    all_domains = _all_domain_ids(class_records, class_count, domain_key)
    val_domains = _parse_domain_ids(val_domains)
    test_domains = _parse_domain_ids(test_domains)
    if (val_domains is None) != (test_domains is None):
        raise ValueError("Validation and test domain IDs must be specified together.")
    if val_domains is None:
        if len(all_domains) <= val_count + test_count:
            raise ValueError(
                f"Not enough {domain_key} domains for separate train/val/test splits: "
                f"{all_domains}"
            )
        test_domains = all_domains[-test_count:]
        val_domains = all_domains[-(test_count + val_count):-test_count]

    unknown = (set(val_domains) | set(test_domains)) - set(all_domains)
    if unknown:
        raise ValueError(f"Unknown {domain_key} domain IDs: {sorted(unknown)}")
    if set(val_domains) & set(test_domains):
        raise ValueError("Validation and test domains must be disjoint.")
    train_domains = sorted(set(all_domains) - set(val_domains) - set(test_domains))
    if not train_domains:
        raise ValueError("No training domains remain after domain holdout.")
    return train_domains, val_domains, test_domains


def _write_domain_protocol(
    class_records,
    class_count,
    output_dir,
    domain_key,
    val_domains,
    test_domains,
    metadata,
    split_caps=None,
    seed=2024,
):
    split_caps = split_caps or {}
    train_domains, val_domains, test_domains = _resolve_domain_split(
        class_records,
        class_count,
        domain_key,
        val_domains=val_domains,
        test_domains=test_domains,
        val_count=2 if domain_key == "rx" else 1,
        test_count=2 if domain_key == "rx" else 1,
    )
    domains_by_split = {
        "train": np.asarray(train_domains),
        "val": np.asarray(val_domains),
        "test": np.asarray(test_domains),
    }
    parts = _empty_split_parts()
    for label in range(class_count):
        records = class_records[label]
        for split, domains in domains_by_split.items():
            indices = np.flatnonzero(np.isin(records[domain_key], domains))
            if len(indices) == 0:
                raise ValueError(
                    f"Transmitter {label} has no samples in {split} {domain_key} "
                    f"domains {domains.tolist()}. Choose different held-out domains."
                )
            cap = split_caps.get(split)
            if cap is not None:
                cap = int(cap)
                if len(indices) < cap:
                    raise ValueError(
                        f"Transmitter {label} has {len(indices)} samples in {split}, "
                        f"fewer than the required balanced cap {cap}."
                    )
                rng = np.random.default_rng(
                    int(seed) + label * 1009 + SPLIT_NAMES.index(split) * 100_003
                )
                indices = np.sort(rng.choice(indices, size=cap, replace=False))
            _append_records(parts, split, records, indices)

    summary = _write_split_files(parts, output_dir, class_count)
    protocol_name = "cross_rx" if domain_key == "rx" else "cross_day"
    _write_protocol_manifest(
        output_dir,
        protocol_name,
        class_count,
        summary,
        held_out_domains={
            "train": train_domains,
            "val": val_domains,
            "test": test_domains,
        },
        split_caps={
            split: (None if split_caps.get(split) is None else int(split_caps[split]))
            for split in SPLIT_NAMES
        },
        seed=int(seed),
        **metadata,
    )
    return summary


def _print_summary(output_dir, class_count, summary):
    split_text = " ".join(
        f"{split}={summary[split]['samples']}" for split in SPLIT_NAMES
    )
    print(f"{output_dir}: {class_count}Class {split_text}", flush=True)


def _collect_records_for_indices(data, global_tx_indices, equalized_index, role):
    records = []
    for local_label, global_tx_index in enumerate(global_tx_indices):
        if local_label % 10 == 0:
            print(
                f"Collecting {role}: {local_label + 1}/{len(global_tx_indices)}",
                flush=True,
            )
        record = _collect_class_records(
            data["data"][global_tx_index],
            local_label,
            equalized_index,
            global_tx_index=global_tx_index,
        )
        if len(record["x"]) == 0:
            raise ValueError(
                f"Global transmitter {global_tx_index} is empty for "
                f"equalized_index={equalized_index}."
            )
        records.append(record)
    return records


def _strict_group_metadata(data, path, equalized_index, indices, role):
    tx_names = _metadata_names(data.get("tx_list"))
    return {
        "source": os.path.abspath(os.path.expanduser(path)),
        "equalized_index": int(equalized_index),
        "equalized_value": str(data["equalized_list"][equalized_index]),
        "rx_names": _metadata_names(data.get("rx_list")),
        "day_names": _metadata_names(data.get("capture_date_list")),
        "tx_names": [tx_names[index] for index in indices],
        "global_tx_indices": [int(index) for index in indices],
        "identity_role": role,
        "identity_disjoint": True,
    }


def _strict_protocol_specs(val_rx, test_rx, val_day, test_day):
    return {
        "cross_rx": {
            "directory": "ManyTx_cross_rx",
            "domain_key": "rx",
            "val_domains": val_rx,
            "test_domains": test_rx,
            "split_caps": {"train": None, "val": 20, "test": 50},
        },
        "cross_day": {
            "directory": "ManyTx_cross_day",
            "domain_key": "day",
            "val_domains": val_day,
            "test_domains": test_day,
            "split_caps": {"train": None, "val": 200, "test": 200},
        },
    }


def convert_p3mc_strict(
    path,
    output_root,
    equalized_index=0,
    seed=2024,
    protocols=("cross_rx", "cross_day"),
    val_rx=(6,),
    test_rx=(10,),
    val_day=(2,),
    test_day=(3,),
    base_indices=range(0, 90),
    validation_indices=range(90, 120),
    test_indices=range(120, 150),
    target_class_counts=(10, 20, 30),
):
    path = os.path.expanduser(path)
    output_root = os.path.abspath(os.path.expanduser(output_root))
    protocols = tuple(protocols)
    unknown = set(protocols) - {"cross_rx", "cross_day"}
    if unknown:
        raise ValueError(f"Strict P3MC mode does not support protocols: {sorted(unknown)}")

    base_indices = list(base_indices)
    validation_indices = list(validation_indices)
    test_indices = list(test_indices)
    identity_sets = [set(base_indices), set(validation_indices), set(test_indices)]
    if any(identity_sets[i] & identity_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Base, validation-novel, and test-novel identities must be disjoint.")
    if max(target_class_counts) > len(test_indices):
        raise ValueError("Target class count exceeds the test-novel identity pool.")

    specs = _strict_protocol_specs(val_rx, test_rx, val_day, test_day)
    output_dirs = [
        os.path.join(output_root, specs[name]["directory"])
        for name in protocols
    ]
    output_dirs += [os.path.join(directory, "validation_novel") for directory in output_dirs]
    occupied = [
        directory
        for directory in output_dirs
        if os.path.isdir(directory) and os.listdir(directory)
    ]
    if occupied:
        raise FileExistsError(
            "Refusing to overwrite non-empty protocol directories: " + ", ".join(occupied)
        )

    print(f"Loading {path}", flush=True)
    with open(path, "rb") as file:
        data = pickle.load(file)
    total_tx = len(data["data"])
    all_indices = base_indices + validation_indices + test_indices
    if min(all_indices) < 0 or max(all_indices) >= total_tx:
        raise ValueError(
            f"Identity indices must be within [0, {total_tx - 1}], got "
            f"[{min(all_indices)}, {max(all_indices)}]."
        )
    if equalized_index < 0 or equalized_index >= len(data["equalized_list"]):
        raise ValueError(f"Invalid equalized_index={equalized_index}")

    def write_group(records, indices, role, class_counts, validation_subdir=False):
        metadata = _strict_group_metadata(data, path, equalized_index, indices, role)
        for protocol_name in protocols:
            spec = specs[protocol_name]
            directory = os.path.join(output_root, spec["directory"])
            if validation_subdir:
                directory = os.path.join(directory, "validation_novel")
            for class_count in class_counts:
                class_metadata = dict(metadata)
                class_metadata["tx_names"] = metadata["tx_names"][:class_count]
                class_metadata["global_tx_indices"] = metadata["global_tx_indices"][:class_count]
                summary = _write_domain_protocol(
                    records,
                    class_count,
                    directory,
                    spec["domain_key"],
                    spec["val_domains"],
                    spec["test_domains"],
                    class_metadata,
                    split_caps=spec["split_caps"],
                    seed=seed,
                )
                _print_summary(directory, class_count, summary)

    base_records = _collect_records_for_indices(
        data, base_indices, equalized_index, "base_pretraining"
    )
    write_group(base_records, base_indices, "base_pretraining", (len(base_indices),))
    del base_records
    gc.collect()

    test_records = _collect_records_for_indices(
        data, test_indices, equalized_index, "test_novel"
    )
    write_group(
        test_records,
        test_indices,
        "test_novel",
        tuple(sorted(set(int(value) for value in target_class_counts))),
    )
    del test_records
    gc.collect()

    validation_records = _collect_records_for_indices(
        data, validation_indices, equalized_index, "validation_novel"
    )
    write_group(
        validation_records,
        validation_indices,
        "validation_novel",
        (len(validation_indices),),
        validation_subdir=True,
    )
    del validation_records
    gc.collect()

    protocol_manifest = {
        "protocol_version": "strict_p3mc_base_to_novel_v1",
        "source": os.path.abspath(path),
        "seed": int(seed),
        "equalized_index": int(equalized_index),
        "equalized_value": str(data["equalized_list"][equalized_index]),
        "identity_groups": {
            "base_pretraining": base_indices,
            "validation_novel": validation_indices,
            "test_novel": test_indices,
        },
        "target_tasks": {
            str(count): test_indices[:count]
            for count in sorted(set(int(value) for value in target_class_counts))
        },
        "protocols": {
            name: {
                "directory": specs[name]["directory"],
                "domain_key": specs[name]["domain_key"],
                "val_domains": list(specs[name]["val_domains"]),
                "test_domains": list(specs[name]["test_domains"]),
                "split_caps": specs[name]["split_caps"],
            }
            for name in protocols
        },
    }
    os.makedirs(output_root, exist_ok=True)
    manifest_path = os.path.join(output_root, "strict_p3mc_protocol.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(protocol_manifest, file, indent=2, ensure_ascii=False)
    print(f"Strict protocol manifest: {manifest_path}", flush=True)
    del data
    gc.collect()


def convert_pkl(
    path,
    output_dir,
    class_counts,
    equalized_index=0,
    seed=2024,
    test_ratio=0.2,
    val_ratio=0.2,
    protocols=("iid",),
    val_rx=None,
    test_rx=None,
    val_day=None,
    test_day=None,
):
    output_dir = os.path.expanduser(output_dir)
    print(f"Loading {path}", flush=True)
    with open(os.path.expanduser(path), "rb") as file:
        data = pickle.load(file)

    max_class_count = max(class_counts)
    if max_class_count > len(data["data"]):
        raise ValueError(
            f"Requested {max_class_count} classes, but dataset contains "
            f"{len(data['data'])}."
        )

    print(f"Collecting first {max_class_count} transmitters", flush=True)
    class_records = [
        _collect_class_records(data["data"][index], index, equalized_index)
        for index in range(max_class_count)
    ]
    empty = [
        index for index, records in enumerate(class_records)
        if len(records["x"]) == 0
    ]
    if empty:
        raise ValueError(f"Empty transmitter classes found: {empty[:20]}")

    metadata = {
        "source": os.path.abspath(os.path.expanduser(path)),
        "equalized_index": int(equalized_index),
        "tx_names": _metadata_names(data.get("tx_list"))[:max_class_count],
        "rx_names": _metadata_names(data.get("rx_list")),
        "day_names": _metadata_names(data.get("capture_date_list")),
    }
    output_parent = os.path.dirname(output_dir)
    dataset_name = os.path.basename(output_dir)
    for class_count in class_counts:
        if "iid" in protocols:
            summary = _write_iid_protocol(
                class_records,
                class_count,
                output_dir,
                seed,
                test_ratio,
                val_ratio,
                metadata,
            )
            _print_summary(output_dir, class_count, summary)
        if "cross_rx" in protocols:
            cross_rx_dir = os.path.join(output_parent, f"{dataset_name}_cross_rx")
            summary = _write_domain_protocol(
                class_records,
                class_count,
                cross_rx_dir,
                "rx",
                val_rx,
                test_rx,
                metadata,
            )
            _print_summary(cross_rx_dir, class_count, summary)
        if "cross_day" in protocols:
            cross_day_dir = os.path.join(output_parent, f"{dataset_name}_cross_day")
            summary = _write_domain_protocol(
                class_records,
                class_count,
                cross_day_dir,
                "day",
                val_day,
                test_day,
                metadata,
            )
            _print_summary(cross_day_dir, class_count, summary)


def main():
    parser = argparse.ArgumentParser(
        description="Convert compact WiSig PKL files and preserve receiver/day domains."
    )
    parser.add_argument("--single", default=r"E:\Single\SingleDay.pkl")
    parser.add_argument("--manytx", default=r"E:\ManyTx\ManyTx.pkl")
    parser.add_argument("--output-root", default="Datasets")
    parser.add_argument("--equalized-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--manytx-protocols",
        nargs="+",
        choices=["iid", "cross_rx", "cross_day"],
        default=["iid", "cross_rx", "cross_day"],
    )
    parser.add_argument("--val-rx", type=int, nargs="+")
    parser.add_argument("--test-rx", type=int, nargs="+")
    parser.add_argument("--val-day", type=int, nargs="+")
    parser.add_argument("--test-day", type=int, nargs="+")
    parser.add_argument(
        "--manytx-class-counts",
        type=int,
        nargs="+",
        default=[90, 30, 20, 10],
    )
    parser.add_argument(
        "--target-class-counts",
        type=int,
        nargs="+",
        default=[10, 20, 30],
        help="nested test-novel task sizes used only with --strict-p3mc",
    )
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-manytx", action="store_true")
    parser.add_argument(
        "--strict-p3mc",
        action="store_true",
        help=(
            "create identity-disjoint base/validation-novel/test-novel WiSig "
            "protocols instead of overlapping first-N class files"
        ),
    )
    parser.add_argument("--base-tx-start", type=int, default=0)
    parser.add_argument("--base-tx-count", type=int, default=90)
    parser.add_argument("--validation-tx-start", type=int, default=90)
    parser.add_argument("--validation-tx-count", type=int, default=30)
    parser.add_argument("--test-tx-start", type=int, default=120)
    parser.add_argument("--test-tx-count", type=int, default=30)
    args = parser.parse_args()

    if not args.skip_single:
        convert_pkl(
            args.single,
            os.path.join(args.output_root, "Single"),
            class_counts=[28, 20, 10],
            equalized_index=args.equalized_index,
            seed=args.seed,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            protocols=("iid",),
        )
    if not args.skip_manytx:
        if args.strict_p3mc:
            convert_p3mc_strict(
                args.manytx,
                args.output_root,
                equalized_index=args.equalized_index,
                seed=args.seed,
                protocols=tuple(args.manytx_protocols),
                val_rx=tuple(args.val_rx or [6]),
                test_rx=tuple(args.test_rx or [10]),
                val_day=tuple(args.val_day or [2]),
                test_day=tuple(args.test_day or [3]),
                base_indices=range(
                    args.base_tx_start,
                    args.base_tx_start + args.base_tx_count,
                ),
                validation_indices=range(
                    args.validation_tx_start,
                    args.validation_tx_start + args.validation_tx_count,
                ),
                test_indices=range(
                    args.test_tx_start,
                    args.test_tx_start + args.test_tx_count,
                ),
                target_class_counts=tuple(args.target_class_counts),
            )
        else:
            convert_pkl(
                args.manytx,
                os.path.join(args.output_root, "ManyTx"),
                class_counts=args.manytx_class_counts,
                equalized_index=args.equalized_index,
                seed=args.seed,
                test_ratio=args.test_ratio,
                val_ratio=args.val_ratio,
                protocols=tuple(args.manytx_protocols),
                val_rx=args.val_rx,
                test_rx=args.test_rx,
                val_day=args.val_day,
                test_day=args.test_day,
            )


if __name__ == "__main__":
    main()
