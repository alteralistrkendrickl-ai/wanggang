import os
import shutil
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from models.lfdb import LightweightLFDB
from utils.channel_aug import add_random_awgn
from utils.config import pretrain_config
from utils.get_dataset import get_pretrain_dataloader
from utils.utils import (
    ListApply,
    RecordTime,
    accuracy,
    create_model,
    get_logger_and_writer,
    set_seed,
)


def _forward_features(encoder, inputs, labels=None, mix_lambda=None):
    if labels is None:
        return encoder(inputs)
    return encoder(inputs, labels, mix_lambda)


def run_step(config, inputs, device, encoder, rot_classifier, mixed_classifier,
             cls, mml, mtl, lfdb=None, training=False):
    """Run all configured pretext tasks for one batch."""
    signals, rot_labels, device_labels = inputs
    batch_size, rot_num = signals.shape[:2]

    rot_inputs = signals.reshape(batch_size * rot_num, *signals.shape[2:]).to(device)
    if training and config["augmentation"]["awgn_enable"]:
        rot_inputs = add_random_awgn(rot_inputs, config["augmentation"]["awgn_snr_range"])

    rot_labels = rot_labels.reshape(-1).long().to(device)
    device_labels = device_labels.long().to(device)
    expanded_device_labels = device_labels.repeat_interleave(rot_num)
    mixed_inputs = signals[:, 0].to(device)

    base_features = None
    fingerprint_features = None
    lfdb_mask = None
    adversarial_logits = None
    losses = {}
    metrics = {"rot_acc": 0.0, "sei_acc": 0.0, "mixed_acc": 0.0}

    def get_base_features():
        nonlocal base_features
        if base_features is None:
            base_features = _forward_features(encoder, rot_inputs)
        return base_features

    def get_fingerprint_features():
        nonlocal fingerprint_features, lfdb_mask, adversarial_logits
        features = get_base_features()
        if lfdb is None:
            return features
        if fingerprint_features is None:
            fingerprint_features, lfdb_mask, adversarial_logits = lfdb(features)
        return fingerprint_features

    for loss_name in config["mtl"]["item"]:
        if loss_name == "rot_cls":
            predictions = rot_classifier(get_base_features())
            losses[loss_name] = cls(predictions, rot_labels)
            metrics["rot_acc"] = accuracy(predictions, rot_labels)

        elif loss_name == "sei_cls":
            predictions = mixed_classifier(get_fingerprint_features())
            losses[loss_name] = cls(predictions, expanded_device_labels)
            metrics["sei_acc"] = accuracy(predictions, expanded_device_labels)

        elif loss_name == "mml":
            mix_lambda = mml.get_lamda()
            mixed_features, labels_a, labels_b = _forward_features(
                encoder, mixed_inputs, device_labels, mix_lambda
            )
            if lfdb is not None:
                mixed_features, _, _ = lfdb(mixed_features)
            predictions = mixed_classifier(mixed_features)
            losses[loss_name] = mml(predictions, labels_a, labels_b)
            metrics["mixed_acc"] = (
                mix_lambda * accuracy(predictions, labels_a)
                + (1.0 - mix_lambda) * accuracy(predictions, labels_b)
            )

        elif loss_name == "decouple":
            get_fingerprint_features()
            adversarial_loss = cls(adversarial_logits, expanded_device_labels)
            # SEI loss prevents the all-zero mask; this term encourages a compact fingerprint.
            mask_regularization = lfdb_mask.mean()
            losses[loss_name] = config["lfdb"]["weight"] * (
                adversarial_loss + 0.01 * mask_regularization
            )
        else:
            raise ValueError(f"Unknown pretext loss: {loss_name}")

    ordered_losses = [losses[name] for name in config["mtl"]["item"]]
    total_loss = mtl(*ordered_losses)
    return ordered_losses + [total_loss], metrics


def _run_epoch(logger, writer, config, epoch, dataloader, device, encoder,
               rot_classifier, mixed_classifier, cls, mml, mtl, lfdb=None,
               optimizers=None, schedulers=None):
    training = optimizers is not None
    modules = [encoder, rot_classifier, mixed_classifier, mtl]
    if lfdb is not None:
        modules.append(lfdb)
    for module in modules:
        module.train(training)

    metric_sums = {"rot_acc": 0.0, "sei_acc": 0.0, "mixed_acc": 0.0}
    loss_sums = [0.0] * (config["mtl"]["num"] + 1)
    split_name = "Train" if training else "Val"

    if training:
        logger.info(f"==> lr = {optimizers[0].param_groups[0]['lr']}")

    progress = tqdm(dataloader, desc=f"{split_name} epoch {epoch + 1}/{config['epoch']}")
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for inputs in progress:
            loss_items, metrics = run_step(
                config, inputs, device, encoder, rot_classifier, mixed_classifier,
                cls, mml, mtl, lfdb, training=training
            )
            if training:
                optimizers.zero_grad()
                loss_items[-1].backward()
                optimizers.step()

            for name, value in metrics.items():
                metric_sums[name] += value
            for index, loss in enumerate(loss_items):
                loss_sums[index] += loss.item()

    if training:
        schedulers.step()

    count = max(len(dataloader), 1)
    metrics = {name: value / count for name, value in metric_sums.items()}
    losses = [value / count for value in loss_sums]
    loss_names = [name.upper() for name in config["mtl"]["item"]] + ["TOTAL"]
    loss_text = ", ".join(f"{name}: {value:.6f}" for name, value in zip(loss_names, losses))
    logger.info(
        f"==> {split_name} Set: Rot-Acc: {metrics['rot_acc']:.2f}%, "
        f"SEI-Acc: {metrics['sei_acc']:.2f}%, Mixed-Acc: {metrics['mixed_acc']:.2f}%, "
        f"{loss_text}"
    )

    for name, value in metrics.items():
        writer.add_scalar(f"{split_name}/{name}", value, epoch)
    for name, value in zip(loss_names, losses):
        writer.add_scalar(f"{split_name}/loss_{name.lower()}", value, epoch)
    return metrics, losses


def _save_checkpoint(path, epoch, config, best_record, encoder, rot_classifier,
                     mixed_classifier, mtl, lfdb, optimizers, schedulers):
    torch.save({
        "epoch": epoch,
        "config": config,
        "best_record": best_record,
        "encoder": encoder.state_dict(),
        "rot_classifier": rot_classifier.state_dict(),
        "mixed_classifier": mixed_classifier.state_dict(),
        "mtl": mtl.state_dict(),
        "lfdb": lfdb.state_dict() if lfdb is not None else None,
        "optimizers": optimizers.state_dict(),
        "schedulers": schedulers.state_dict(),
    }, path)


def _load_checkpoint(path, device, encoder, rot_classifier, mixed_classifier,
                     mtl, lfdb, optimizers, schedulers):
    checkpoint = torch.load(path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    rot_classifier.load_state_dict(checkpoint["rot_classifier"])
    mixed_classifier.load_state_dict(checkpoint["mixed_classifier"])
    mtl.load_state_dict(checkpoint["mtl"])
    if lfdb is not None and checkpoint.get("lfdb") is not None:
        lfdb.load_state_dict(checkpoint["lfdb"])
    if checkpoint.get("optimizers") is not None:
        optimizers.load_state_dict(checkpoint["optimizers"])
    if checkpoint.get("schedulers") is not None:
        schedulers.load_state_dict(checkpoint["schedulers"])
    return checkpoint


def train_and_val(record_time, logger, writer, config, train_dl, val_dl, device,
                  encoder, rot_classifier, mixed_classifier, cls, mml, mtl,
                  lfdb, optimizers, schedulers, checkpoint=None):
    best_record = (
        deepcopy(checkpoint["best_record"])
        if checkpoint is not None and "best_record" in checkpoint
        else {"epoch": -1, "metrics": {}, "loss": [float("inf")] * (config["mtl"]["num"] + 1)}
    )

    for epoch in range(config["start_epoch"], config["epoch"]):
        logger.info("--------------------------------------------")
        logger.info(f"Epoch {epoch + 1}/{config['epoch']}")
        record_time.start()

        _run_epoch(
            logger, writer, config, epoch, train_dl, device, encoder,
            rot_classifier, mixed_classifier, cls, mml, mtl, lfdb,
            optimizers, schedulers
        )
        metrics, losses = _run_epoch(
            logger, writer, config, epoch, val_dl, device, encoder,
            rot_classifier, mixed_classifier, cls, mml, mtl, lfdb
        )

        if sum(losses[:-1]) < sum(best_record["loss"][:-1]):
            best_record = {"epoch": epoch, "metrics": metrics, "loss": losses}
            torch.save(encoder.state_dict(), os.path.join(config["exp_path"], "best_encoder.pth"))
            logger.info(f"==> Best encoder saved at epoch {epoch + 1}.")

        torch.save(encoder.state_dict(), os.path.join(config["exp_path"], "final_encoder.pth"))
        if epoch % config["save_freq"] == 0 or epoch == config["epoch"] - 1:
            checkpoint_path = os.path.join(config["exp_path"], "checkpoint.pth")
            _save_checkpoint(
                checkpoint_path, epoch, config, best_record, encoder,
                rot_classifier, mixed_classifier, mtl, lfdb, optimizers, schedulers
            )
            shutil.copy2(checkpoint_path, os.path.join(config["save_path"], "checkpoint.pth"))

        logger.info(
            "==> Time spent (current/mean/total/remain): {}/{}/{}/{}"
            .format(*record_time.step())
        )

    for filename in ("best_encoder.pth", "final_encoder.pth"):
        source = os.path.join(config["exp_path"], filename)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(config["save_path"], filename))
    logger.info("--------------------------------------------")
    logger.info(f"End. Best Record: {best_record}")


def pretext(config=None):
    config = pretrain_config() if config is None else config
    set_seed(config["random_seed"])
    config["exp_type"] += "_random_rot"
    run_root = os.path.join("runs", config["exp_type"], config["exp_name"])
    logger, writer, exp_path, save_path = get_logger_and_writer(run_root)
    config["exp_path"] = exp_path
    config["save_path"] = save_path

    logger.info(f"==> Running file: {os.path.abspath(__file__)}")
    logger.info(f"==> Config: {config}")
    train_dataloader, val_dataloader = get_pretrain_dataloader(config)
    device = torch.device(config["device"])

    encoder_config = config["encoder"]
    encoder_kwargs = {
        "feature_dim": encoder_config["feature_dim"],
        "dtype": config["dataset"]["type"],
    }
    if "TSLA" in encoder_config["name"]:
        encoder_kwargs.update(encoder_config["TSLA_config"])
    encoder = create_model(encoder_config["root"], **encoder_kwargs).to(device)
    rot_classifier = create_model(
        config["rot_classifier"]["root"],
        in_dim=config["rot_classifier"]["in_dim"],
        num_classes=config["rot_classifier"]["num_classes"],
    ).to(device)
    mixed_classifier = create_model(
        config["mixed_classifier"]["root"],
        in_dim=config["mixed_classifier"]["in_dim"],
        num_classes=config["mixed_classifier"]["num_classes"],
    ).to(device)

    lfdb = None
    if config["lfdb"]["enabled"]:
        lfdb = LightweightLFDB(
            feat_dim=config["encoder"]["feature_dim"],
            num_classes=config["lfdb"]["num_classes"],
        ).to(device)

    cls = torch.nn.CrossEntropyLoss()
    mml = create_model(config["mml"]["root"], beta=config["mml"]["beta"])
    mtl = create_model(config["mtl"]["root"], num=config["mtl"]["num"]).to(device)

    trainable_modules = [encoder, rot_classifier, mixed_classifier, mtl]
    if lfdb is not None:
        trainable_modules.append(lfdb)
    optimizer_config = config["optimizer"]
    optimizers = ListApply([
        AdamW(
            module.parameters(),
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config["weight_decay"],
        )
        for module in trainable_modules
    ])
    schedulers = ListApply([
        StepLR(
            optimizer,
            step_size=optimizer_config["step_size"],
            gamma=optimizer_config["gamma"],
        )
        for optimizer in optimizers
    ])

    checkpoint = None
    resume_path = config.get("resume", "")
    if resume_path:
        if os.path.isdir(resume_path):
            resume_path = os.path.join(resume_path, "checkpoint.pth")
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        checkpoint = _load_checkpoint(
            resume_path, device, encoder, rot_classifier, mixed_classifier,
            mtl, lfdb, optimizers, schedulers
        )
        config["start_epoch"] = checkpoint["epoch"] + 1
        logger.info(f"==> Resumed checkpoint: {resume_path}")
    else:
        config["start_epoch"] = 0

    record_time = RecordTime(max(config["epoch"] - config["start_epoch"], 0))
    train_and_val(
        record_time, logger, writer, config, train_dataloader, val_dataloader,
        device, encoder, rot_classifier, mixed_classifier, cls, mml, mtl,
        lfdb, optimizers, schedulers, checkpoint
    )
    writer.close()


if __name__ == "__main__":
    pretext(pretrain_config(
        encoder_name="CVTSLANet",
        classifiar_name="Linear",
        dataset_name="ads-b",
        input_type="iq",
        rot_num=8,
        feature_dim=1024,
        max_epoch=5,
        save_freq=1,
    ))
