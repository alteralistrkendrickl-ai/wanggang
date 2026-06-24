import os
import shutil

import pandas as pd
import torch
from tqdm import tqdm

from utils.config import finetune_config
from utils.get_dataset import get_finetune_dataloader
from utils.utils import (
    SummaryWriter,
    accuracy,
    create_model,
    get_logger_and_writer,
    load_encoder_weights,
    set_seed,
)


def _build_encoder(config, device):
    encoder_config = config["encoder"]
    kwargs = {
        "feature_dim": encoder_config["feature_dim"],
        "dtype": config["dataset"]["type"],
    }
    if "TSLA" in encoder_config["name"]:
        kwargs.update(encoder_config["TSLA_config"])
    encoder = create_model(encoder_config["root"], **kwargs)
    load_encoder_weights(encoder, encoder_config["pretrain_path"], device)
    encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def _evaluate(encoder, classifier, dataloader, device):
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.long().to(device)
            predictions = classifier(encoder(inputs))
            total_correct += (predictions.argmax(dim=1) == labels).sum().item()
            total_samples += labels.numel()
    return 100.0 * total_correct / max(total_samples, 1)


def finetune(config, logger, writer=None):
    set_seed(config["random_seed"])
    train_loader, test_loader = get_finetune_dataloader(config)
    device = torch.device(config["device"])
    encoder = _build_encoder(config, device)
    classifier = create_model(
        config["classifier"]["root"],
        in_dim=config["classifier"]["in_dim"],
        num_classes=config["classifier"]["num_classes"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config["epoch"]):
        classifier.train()
        loss_sum = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.long().to(device)
            with torch.no_grad():
                features = encoder(inputs)
            predictions = classifier(features)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        if writer is not None:
            writer.add_scalar("Train/loss", loss_sum / max(len(train_loader), 1), epoch)

    test_accuracy = _evaluate(encoder, classifier, test_loader, device)
    logger.info(f"Acc: {test_accuracy:.2f}%")
    return test_accuracy


def downstream(config=None):
    config = finetune_config(classifier_name="Linear") if config is None else config
    if isinstance(config["dataset"]["shot"], list):
        if len(config["dataset"]["shot"]) != 1:
            raise ValueError("downstream_linear.downstream expects one shot value per run")
        config["dataset"]["shot"] = config["dataset"]["shot"][0]
    if isinstance(config["dataset"]["snr"], list):
        if len(config["dataset"]["snr"]) != 1:
            raise ValueError("downstream_linear.downstream expects one SNR value per run")
        config["dataset"]["snr"] = config["dataset"]["snr"][0]
    snr = config["dataset"]["snr"]
    if snr is None:
        result_root = os.path.join(
            "runs", config["exp_type"], config["exp_name"], f"{config['dataset']['shot']}Shot"
        )
    else:
        result_root = os.path.join(
            "runs", config["exp_type"], config["exp_name"],
            f"{snr}dB", f"{config['dataset']['shot']}Shot"
        )
    logger, _, exp_path, save_path = get_logger_and_writer(result_root, create_writer=False)
    config["exp_path"] = exp_path
    config["save_path"] = os.path.dirname(save_path)
    logger.info(f"==> Running file: {os.path.abspath(__file__)}")
    logger.info(f"==> Config: {config}")

    accuracies = []
    base_seed = config["random_seed"]
    for iteration in range(config["iteration"]):
        logger.info("--------------------------------------------")
        logger.info(f"Iteration {iteration + 1}/{config['iteration']}")
        config["random_seed"] = base_seed + iteration
        writer = SummaryWriter(os.path.join(exp_path, "writer", str(iteration)))
        accuracies.append(finetune(config, logger, writer))
        writer.close()

    result_path = os.path.join(exp_path, "test_result.xlsx")
    pd.DataFrame({"accuracy": accuracies}).to_excel(result_path, index=False)
    shutil.copy2(
        result_path,
        os.path.join(
            config["save_path"],
            f"{config['exp_name']}_{config['dataset']['shot']}Shot.xlsx",
        ),
    )


if __name__ == "__main__":
    downstream()
