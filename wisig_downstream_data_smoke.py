"""Verify strict validation/final query routing before downstream experiments."""

import argparse
import os

from utils.config import dataset_path_dict
from utils.get_dataset import get_finetune_dataloader


EXPECTED_QUERY_COUNTS = {
    ("cross-rx", "validation"): 600,
    ("cross-rx", "final"): 1500,
    ("cross-day", "validation"): 6000,
    ("cross-day", "final"): 6000,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("cross-rx", "cross-day"), required=True)
    return parser.parse_args()


def check_route(protocol, role):
    if role == "validation":
        dataset_key = f"wisig-{protocol}-validation-30"
    else:
        dataset_key = f"wisig-{protocol}-30"
    entry = dataset_path_dict[dataset_key]
    root = os.path.expanduser(entry["linux"])
    query_split = entry["query_split"]
    config = {
        "random_seed": 2024,
        "device": "cpu",
        "dataset": {
            "root": root,
            "type": "iq",
            "normalize": "power",
            "train_batch_size": 30,
            "test_batch_size": 128,
            "num_classes": 30,
            "signal_length": 256,
            "shot": 1,
            "snr": None,
            "query_split": query_split,
        },
    }
    support_loader, query_loader = get_finetune_dataloader(config)
    support_count = len(support_loader.dataset)
    query_count = len(query_loader.dataset)
    expected_query_count = EXPECTED_QUERY_COUNTS[(protocol, role)]
    expected_split = "val" if role == "validation" else "test"
    if support_count != 30:
        raise AssertionError(f"Expected 30 one-shot support samples, got {support_count}")
    if query_count != expected_query_count:
        raise AssertionError(
            f"Expected {expected_query_count} {role} queries, got {query_count}"
        )
    if query_split != expected_split:
        raise AssertionError(
            f"Expected query split {expected_split!r}, got {query_split!r}"
        )
    print(
        f"ROLE={role} DATASET={dataset_key} ROOT={root} "
        f"QUERY_SPLIT={query_split} SUPPORT={support_count} QUERY={query_count}"
    )


def main():
    args = parse_args()
    check_route(args.protocol, "validation")
    check_route(args.protocol, "final")
    print(f"WISIG_DOWNSTREAM_DATA_SMOKE_{args.protocol.upper().replace('-', '_')}: PASS")


if __name__ == "__main__":
    main()
