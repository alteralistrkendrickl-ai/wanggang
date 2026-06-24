"""Unified downstream entry point for linear, LR, KNN and SVM evaluation."""

from copy import deepcopy

from downstream_knn import downstream as downstream_knn
from downstream_linear import downstream as downstream_linear
from downstream_lr import downstream as downstream_lr
from downstream_svm import downstream as downstream_svm
from utils.config import finetune_config


RUNNERS = {
    "linear": downstream_linear,
    "lr": downstream_lr,
    "knn": downstream_knn,
    "svm": downstream_svm,
}


def main():
    config = finetune_config(
        encoder_name="CVTSLANet",
        classifier_name="Linear",
        dataset_name="ads-b",
        input_type="iq",
        feature_dim=1024,
        shot=[5],
        max_iteration=3,
        max_epoch=100,
        snr=[0],
        snr_enable=False,
    )
    config["exp_type"] = "Downstream_random_rot"
    classifier_name = config["classifier"]["name"]
    runner = RUNNERS[classifier_name]

    suffix = classifier_name.upper()
    config["exp_name"] = config["exp_name"].replace(
        config["encoder"]["name"], f"{config['encoder']['name']}_{suffix}", 1
    )
    for snr in config["dataset"]["snr"]:
        for shot in config["dataset"]["shot"]:
            run_config = deepcopy(config)
            run_config["dataset"]["snr"] = snr
            run_config["dataset"]["shot"] = shot
            runner(run_config)


if __name__ == "__main__":
    main()
