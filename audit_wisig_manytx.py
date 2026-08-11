#!/usr/bin/env python3
"""Read-only structural audit for the official WiSig ManyTx.pkl dataset."""

import argparse
import gc
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXPECTED_SIZE = 4_182_065_863
EXPECTED_SHA256 = "c0319174d40eb64bc49f201743941ebedc5cc0ced284c655cab798b2bdd44275"


def available_memory_bytes():
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def safe_len(value):
    try:
        return len(value)
    except (TypeError, AttributeError):
        return None


def axis_summary(counter, axis_size):
    values = np.asarray([counter.get(index, 0) for index in range(axis_size)])
    if values.size == 0:
        return {"size": 0, "nonzero": 0, "min": 0, "median": 0.0, "max": 0, "total": 0}
    return {
        "size": int(values.size),
        "nonzero": int(np.count_nonzero(values)),
        "min": int(values.min()),
        "median": float(np.median(values)),
        "max": int(values.max()),
        "total": int(values.sum()),
    }


def audit_dataset(dataset, sample_block_limit=32, sample_element_limit=1_000_000):
    required_keys = [
        "data",
        "tx_list",
        "rx_list",
        "capture_date_list",
        "equalized_list",
    ]
    missing = [key for key in required_keys if key not in dataset]
    if missing:
        raise KeyError(f"Missing expected top-level keys: {missing}")

    data = dataset["data"]
    tx_list = list(dataset["tx_list"])
    rx_list = list(dataset["rx_list"])
    date_list = list(dataset["capture_date_list"])
    equalized_list = list(dataset["equalized_list"])

    metadata_sizes = {
        "tx": len(tx_list),
        "rx": len(rx_list),
        "date": len(date_list),
        "equalized": len(equalized_list),
    }
    observed_outer_tx = safe_len(data)
    warnings = []
    if observed_outer_tx != len(tx_list):
        warnings.append(
            f"data outer length {observed_outer_tx} != tx_list length {len(tx_list)}"
        )

    total_by_tx = Counter()
    total_by_rx = Counter()
    total_by_date = Counter()
    total_by_equalized = Counter()
    tx_domain_pairs = defaultdict(set)
    nonempty_blocks = 0
    empty_blocks = 0
    none_blocks = 0
    total_signals = 0
    shape_counts = Counter()
    dtype_counts = Counter()
    sampled_blocks = []
    sampled_elements = 0
    sampled_nonfinite = 0
    sampled_abs_max = 0.0
    observed_rx_lengths = Counter()
    observed_date_lengths = Counter()
    observed_equalized_lengths = Counter()

    for tx_index, tx_data in enumerate(data):
        rx_length = safe_len(tx_data)
        observed_rx_lengths[str(rx_length)] += 1
        if rx_length is None:
            warnings.append(f"tx index {tx_index} is not iterable")
            continue
        for rx_index, rx_data in enumerate(tx_data):
            date_length = safe_len(rx_data)
            observed_date_lengths[str(date_length)] += 1
            if date_length is None:
                warnings.append(f"tx={tx_index}, rx={rx_index} is not iterable")
                continue
            for date_index, date_data in enumerate(rx_data):
                equalized_length = safe_len(date_data)
                observed_equalized_lengths[str(equalized_length)] += 1
                if equalized_length is None:
                    warnings.append(
                        f"tx={tx_index}, rx={rx_index}, date={date_index} is not iterable"
                    )
                    continue
                for equalized_index, block in enumerate(date_data):
                    if block is None:
                        none_blocks += 1
                        continue
                    block_length = safe_len(block)
                    if not block_length:
                        empty_blocks += 1
                        continue

                    array = np.asarray(block)
                    signal_count = int(array.shape[0]) if array.ndim else 1
                    total_signals += signal_count
                    nonempty_blocks += 1
                    total_by_tx[tx_index] += signal_count
                    total_by_rx[rx_index] += signal_count
                    total_by_date[date_index] += signal_count
                    total_by_equalized[equalized_index] += signal_count
                    tx_domain_pairs[tx_index].add((rx_index, date_index))
                    shape_counts[str(tuple(int(x) for x in array.shape))] += 1
                    dtype_counts[str(array.dtype)] += 1

                    if len(sampled_blocks) < sample_block_limit:
                        sample = array.reshape(-1)
                        remaining = max(sample_element_limit - sampled_elements, 0)
                        sample = sample[:remaining]
                        if sample.size and np.issubdtype(sample.dtype, np.number):
                            finite = np.isfinite(sample)
                            sampled_nonfinite += int(sample.size - np.count_nonzero(finite))
                            if np.any(finite):
                                sampled_abs_max = max(
                                    sampled_abs_max,
                                    float(np.max(np.abs(sample[finite]))),
                                )
                        sampled_elements += int(sample.size)
                        sampled_blocks.append(
                            {
                                "tx_index": tx_index,
                                "rx_index": rx_index,
                                "date_index": date_index,
                                "equalized_index": equalized_index,
                                "shape": list(array.shape),
                                "dtype": str(array.dtype),
                                "signals": signal_count,
                            }
                        )

    tx_domain_counts = np.asarray(
        [len(tx_domain_pairs.get(index, set())) for index in range(len(tx_list))]
    )
    eligible_two_domains = int(np.count_nonzero(tx_domain_counts >= 2))
    eligible_four_domains = int(np.count_nonzero(tx_domain_counts >= 4))

    return {
        "top_level_keys": sorted(str(key) for key in dataset.keys()),
        "metadata_sizes": metadata_sizes,
        "metadata_examples": {
            "tx": [str(item) for item in tx_list[:5]],
            "rx": [str(item) for item in rx_list[:5]],
            "date": [str(item) for item in date_list[:5]],
            "equalized": [str(item) for item in equalized_list[:5]],
        },
        "observed_nested_lengths": {
            "outer_tx": observed_outer_tx,
            "rx_lengths": dict(observed_rx_lengths),
            "date_lengths": dict(observed_date_lengths),
            "equalized_lengths": dict(observed_equalized_lengths),
        },
        "blocks": {
            "nonempty": nonempty_blocks,
            "empty": empty_blocks,
            "none": none_blocks,
            "shape_counts": dict(shape_counts.most_common()),
            "dtype_counts": dict(dtype_counts.most_common()),
        },
        "signals": {
            "total": total_signals,
            "by_tx": axis_summary(total_by_tx, len(tx_list)),
            "by_rx": axis_summary(total_by_rx, len(rx_list)),
            "by_date": axis_summary(total_by_date, len(date_list)),
            "by_equalized": axis_summary(total_by_equalized, len(equalized_list)),
        },
        "cross_domain_readiness": {
            "tx_with_at_least_2_rx_day_pairs": eligible_two_domains,
            "tx_with_at_least_4_rx_day_pairs": eligible_four_domains,
            "rx_day_pairs_per_tx_min": int(tx_domain_counts.min()) if tx_domain_counts.size else 0,
            "rx_day_pairs_per_tx_median": float(np.median(tx_domain_counts)) if tx_domain_counts.size else 0.0,
            "rx_day_pairs_per_tx_max": int(tx_domain_counts.max()) if tx_domain_counts.size else 0,
        },
        "sampled_numeric_check": {
            "blocks": len(sampled_blocks),
            "elements": sampled_elements,
            "nonfinite": sampled_nonfinite,
            "abs_max": sampled_abs_max,
            "block_examples": sampled_blocks,
        },
        "warnings": warnings[:100],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="/home/yuanlong/Datasets/WiSig_original/ManyTx.pkl",
    )
    parser.add_argument(
        "--output",
        default="wisig_manytx_audit_summary.json",
        help="JSON summary path; dataset is never modified",
    )
    parser.add_argument(
        "--force-low-memory",
        action="store_true",
        help="attempt pickle.load even when available memory is below the safety threshold",
    )
    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    file_size = path.stat().st_size
    print(f"Dataset: {path}", flush=True)
    print(f"File size: {file_size} bytes", flush=True)
    print(f"Expected size: {EXPECTED_SIZE} bytes", flush=True)
    if file_size != EXPECTED_SIZE:
        raise RuntimeError("File size does not match the verified local/server copy")

    available = available_memory_bytes()
    safety_threshold = int(file_size * 1.5)
    if available is not None:
        print(f"MemAvailable: {available / (1024 ** 3):.2f} GiB", flush=True)
        print(f"Safety threshold: {safety_threshold / (1024 ** 3):.2f} GiB", flush=True)
        if available < safety_threshold and not args.force_low_memory:
            raise MemoryError(
                "Available memory is below 1.5x the pickle size; rerun only after "
                "freeing memory, or use --force-low-memory after reviewing the risk"
            )

    print("Loading pickle read-only; this may take several minutes...", flush=True)
    with path.open("rb") as handle:
        dataset = pickle.load(handle)
    print(f"Loaded top-level object: {type(dataset)!r}", flush=True)
    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dict, got {type(dataset)!r}")

    summary = audit_dataset(dataset)
    summary["file"] = {
        "path": str(path),
        "size": file_size,
        "verified_sha256": EXPECTED_SHA256,
    }
    output = Path(args.output).expanduser().resolve()
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("===== AUDIT SUMMARY =====", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Summary written to: {output}", flush=True)
    del dataset
    gc.collect()


if __name__ == "__main__":
    main()
