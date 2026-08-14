"""One-batch real-data runtime smoke for the adapted FS-SEI baseline."""

import argparse

import torch

from models.FSSEILoss import FSSEISTCLoss
from wisig_train_fssei import (
    FSSEIModel,
    WiSigMetricDataset,
    make_loader,
    resolve_base_paths,
    resolve_device,
)
from utils.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocols",
        choices=("cross-rx", "cross-day"),
        nargs="+",
        default=("cross-rx", "cross-day"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def run_protocol(protocol, args, device):
    root, paths = resolve_base_paths(protocol)
    dataset = WiSigMetricDataset(paths["train_x"], paths["train_y"], expected_classes=90)
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        workers=0,
        pin_memory=device == "cuda",
    )
    inputs, labels = next(iter(loader))
    inputs = inputs.to(device)
    labels = labels.to(device)
    model = FSSEIModel(feature_dim=args.feature_dim, num_classes=90).to(device)
    objective = FSSEISTCLoss(num_classes=90, feature_dim=args.feature_dim).to(device)
    embeddings, logits = model(inputs)
    total, ce, triplet, center = objective(logits, embeddings, labels)
    total.backward()
    if not torch.isfinite(total):
        raise RuntimeError(f"Non-finite loss for {protocol}")

    print(f"PROTOCOL={protocol}")
    print(f"DATASET_ROOT={root}")
    print(f"TRAIN_SAMPLES={len(dataset)}")
    print(f"DEVICE={device}")
    print(f"INPUT_SHAPE={tuple(inputs.shape)}")
    print(f"EMBEDDING_SHAPE={tuple(embeddings.shape)}")
    print(f"LOGIT_SHAPE={tuple(logits.shape)}")
    print(
        "LOSSES="
        f"total:{total.item():.6f},"
        f"ce:{ce.item():.6f},"
        f"triplet:{triplet.item():.6f},"
        f"center:{center.item():.6f}"
    )
    print("FINITE=True")
    print("FINAL_TEST_ACCESSED=False")


def main():
    args = parse_args()
    if args.batch_size < 2 or args.feature_dim < 1:
        raise ValueError("batch-size must be at least 2 and feature-dim positive")
    device = resolve_device(args.device)
    set_seed(args.seed)
    for protocol in args.protocols:
        run_protocol(protocol, args, device)
    print("WISIG_FSSEI_RUNTIME_SMOKE: PASS")


if __name__ == "__main__":
    main()
