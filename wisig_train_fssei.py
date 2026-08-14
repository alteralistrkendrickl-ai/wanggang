"""Train the adapted FS-SEI STC-CVCNN baseline on strict WiSig source data.

Only the 90-class base ``train`` and ``val`` arrays are opened.  Novel-class
validation and final-test arrays are deliberately outside this training entry
point.  The downstream few-shot evaluation remains validation-only through
``wisig_validate_lr.py``.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.FSSEILoss import FSSEISTCLoss
from models.FSSEISTCFeature import FSSEISTCEncoder
from utils.config import dataset_path_dict
from utils.utils import set_seed


SOURCE_REPOSITORY = "https://github.com/BeechburgPieStar/FS-SEI"


def parse_args():
    parser = argparse.ArgumentParser(description="Train adapted FS-SEI on strict WiSig")
    parser.add_argument("--protocol", choices=("cross-rx", "cross-day"), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", default="runs/Pretext_fssei")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def experiment_name(protocol, epochs, seed):
    return f"FSSEI_STC_wisig-{protocol}_iq_powerNorm_{epochs}e_seed{seed}"


def resolve_base_paths(protocol):
    key = f"wisig-{protocol}"
    entry = dataset_path_dict[key]
    platform = "windows" if os.name == "nt" else "linux"
    root = Path(os.path.expanduser(entry[platform])).resolve()
    paths = {
        "train_x": root / "X_train_90Class.npy",
        "train_y": root / "Y_train_90Class.npy",
        "val_x": root / "X_val_90Class.npy",
        "val_y": root / "Y_val_90Class.npy",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing strict base file {name}: {path}")
    forbidden = [path for path in paths.values() if "test" in path.name.lower()]
    if forbidden:
        raise RuntimeError(f"Training route unexpectedly contains final-test files: {forbidden}")
    return root, paths


def normalize_iq(sample):
    sample = np.asarray(sample, dtype=np.float32).copy()
    if sample.shape[0] != 2 and sample.shape[-1] == 2:
        sample = sample.T
    if sample.ndim != 2 or sample.shape[0] != 2:
        raise ValueError(f"Expected IQ sample (2, length), got {sample.shape}")
    power = np.square(sample[0]) + np.square(sample[1])
    sample /= max(float(np.sqrt(power.max())), 1e-12)
    return sample


class WiSigMetricDataset(Dataset):
    def __init__(self, x_path, y_path, expected_classes=90):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        if len(self.x) != len(self.y):
            raise ValueError(f"X/Y length mismatch: {len(self.x)} != {len(self.y)}")
        labels = np.unique(self.y).astype(int)
        if not np.array_equal(labels, np.arange(expected_classes)):
            raise ValueError(f"Expected labels 0..{expected_classes - 1}, got {labels.tolist()}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (
            torch.from_numpy(normalize_iq(self.x[index])),
            torch.tensor(int(self.y[index]), dtype=torch.long),
        )


class FSSEIModel(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=90):
        super().__init__()
        self.encoder = FSSEISTCEncoder(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, inputs):
        embeddings = self.encoder(inputs)
        return embeddings, self.classifier(embeddings)


def make_loader(dataset, batch_size, shuffle, seed, workers, pin_memory):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=shuffle,
    )


def run_epoch(model, objective, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = np.zeros(5, dtype=np.float64)
    seen = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            embeddings, logits = model(inputs)
            loss, ce, triplet, center = objective(logits, embeddings, labels)
            if training:
                loss.backward()
                optimizer.step()
            count = labels.numel()
            correct = logits.argmax(dim=1).eq(labels).sum().item()
            totals += np.asarray(
                [loss.item() * count, ce.item() * count, triplet.item() * count,
                 center.item() * count, correct],
                dtype=np.float64,
            )
            seen += count
    if seen == 0:
        raise RuntimeError("Empty dataloader")
    return {
        "loss": totals[0] / seen,
        "ce": totals[1] / seen,
        "triplet": totals[2] / seen,
        "center": totals[3] / seen,
        "accuracy": 100.0 * totals[4] / seen,
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.feature_dim < 1:
        raise ValueError("epochs, feature-dim and batch-size must be positive")
    device = resolve_device(args.device)
    set_seed(args.seed)
    root, paths = resolve_base_paths(args.protocol)
    run_dir = (
        Path(args.output_dir).expanduser().resolve()
        / experiment_name(args.protocol, args.epochs, args.seed)
    )
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite explicitly: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = WiSigMetricDataset(paths["train_x"], paths["train_y"])
    val_dataset = WiSigMetricDataset(paths["val_x"], paths["val_y"])
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.seed, args.num_workers, device == "cuda"
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.seed, args.num_workers, device == "cuda"
    )
    model = FSSEIModel(args.feature_dim, 90).to(device)
    objective = FSSEISTCLoss(num_classes=90, feature_dim=args.feature_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(objective.center.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    manifest = {
        "baseline": "FS-SEI repository adapted STC-CVCNN",
        "source_repository": SOURCE_REPOSITORY,
        "protocol": args.protocol,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "feature_dim": args.feature_dim,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "dataset_root": str(root),
        "opened_splits": ["base_train", "base_val"],
        "final_test_accessed": False,
        "adaptation": "nine complex-convolution blocks with global average pooling for 256 IQ samples",
    }
    atomic_json(run_dir / "manifest.json", manifest)

    print(f"PROTOCOL={args.protocol}")
    print("BASELINE=FS-SEI repository adapted STC-CVCNN")
    print(f"DEVICE={device}")
    print(f"TRAIN_SAMPLES={len(train_dataset)}")
    print(f"VAL_SAMPLES={len(val_dataset)}")
    print("FINAL_TEST_ACCESSED=False")
    print(f"RESULT_DIR={run_dir}")

    best_loss = float("inf")
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, objective, train_loader, device, optimizer)
        val_metrics = run_epoch(model, objective, val_loader, device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(
            f"EPOCH={epoch}/{args.epochs} "
            f"TRAIN_LOSS={train_metrics['loss']:.6f} TRAIN_ACC={train_metrics['accuracy']:.4f} "
            f"VAL_LOSS={val_metrics['loss']:.6f} VAL_ACC={val_metrics['accuracy']:.4f}"
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(model.encoder.state_dict(), run_dir / "best_encoder.pth")
            print(f"BEST_ENCODER_SAVED={epoch}")
        atomic_json(run_dir / "history.json", history)

    torch.save(model.encoder.state_dict(), run_dir / "final_encoder.pth")
    manifest.update(
        {
            "elapsed_seconds": time.time() - started,
            "best_encoder_sha256": sha256(run_dir / "best_encoder.pth"),
            "final_encoder_sha256": sha256(run_dir / "final_encoder.pth"),
            "state": "complete",
        }
    )
    atomic_json(run_dir / "manifest.json", manifest)
    print(f"BEST_ENCODER_SHA256={manifest['best_encoder_sha256']}")
    print(f"FINAL_ENCODER_SHA256={manifest['final_encoder_sha256']}")
    print("WISIG_FSSEI_TRAIN: PASS")


if __name__ == "__main__":
    main()
