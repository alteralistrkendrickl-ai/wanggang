import argparse
import os
import sys
import torch
from copy import deepcopy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path_dict = {
    "ads-b": {
        "name": "ads-b",
        "linux": "~/Datasets/ADS-B",
        "windows": os.path.join(PROJECT_ROOT, "Datasets", "ADS-B"),
        "pt_class": 90,
        "ft_class": 30
    },
    "ads-b-20": {
        "name": "ads-b",
        "linux": "~/Datasets/ADS-B",
        "windows": os.path.join(PROJECT_ROOT, "Datasets", "ADS-B"),
        "pt_class": 90,
        "ft_class": 20
    },
    "ads-b-10": {
        "name": "ads-b",
        "linux": "~/Datasets/ADS-B",
        "windows": os.path.join(PROJECT_ROOT, "Datasets", "ADS-B"),
        "pt_class": 90,
        "ft_class": 10
    },
    "lora": {
        "name": "lora",
        "linux": "~/Datasets/LoRa",
        "windows": "E:\\Datasets\\LoRa",
        "pt_class": 30,
        "ft_class": 10
    },
    "wifi": {
        "name": "wifi",
        "linux": "~/Datasets/WiFi_ft62",
        "windows": "E:\\Datasets\\WiFi_ft62",
        "pt_class": 10,
        "ft_class": 6
    }
}

_wisig_linux_root = os.environ.get(
    "WISIG_P3MC_ROOT", "~/Datasets/WiSig_P3MC_strict_v1"
)
_wisig_windows_root = os.environ.get(
    "WISIG_P3MC_ROOT_WINDOWS", r"E:\Datasets\WiSig_P3MC_strict_v1"
)
for protocol, directory in (
    ("cross-rx", "ManyTx_cross_rx"),
    ("cross-day", "ManyTx_cross_day"),
):
    base_config = {
        "name": f"wisig-{protocol}",
        "linux": os.path.join(_wisig_linux_root, directory),
        "windows": os.path.join(_wisig_windows_root, directory),
        "pt_class": 90,
        "ft_class": 30,
        "query_split": "test",
    }
    dataset_path_dict[f"wisig-{protocol}"] = base_config
    for ft_class in (10, 20, 30):
        alias = deepcopy(base_config)
        alias["ft_class"] = ft_class
        dataset_path_dict[f"wisig-{protocol}-{ft_class}"] = alias
    validation_config = deepcopy(base_config)
    validation_config["linux"] = os.path.join(base_config["linux"], "validation_novel")
    validation_config["windows"] = os.path.join(base_config["windows"], "validation_novel")
    validation_config["query_split"] = "val"
    dataset_path_dict[f"wisig-{protocol}-validation-30"] = validation_config

for ft_class in [30, 20, 10]:
    dataset_path_dict[f"ads-b{ft_class}"] = deepcopy(dataset_path_dict["ads-b"])
    dataset_path_dict[f"ads-b{ft_class}"]["ft_class"] = ft_class
    dataset_path_dict[f"ads-b-{ft_class}"] = deepcopy(dataset_path_dict["ads-b"])
    dataset_path_dict[f"ads-b-{ft_class}"]["ft_class"] = ft_class

model_path_dict = {
    "ResNet18": os.path.join(PROJECT_ROOT, "models", "ResNet18Feature.py"),
    "CVTSLANet": os.path.join(PROJECT_ROOT, "models", "CVTSLANetFeature.py"),
    "CVTSLANet-Shallow": os.path.join(PROJECT_ROOT, "models", "SCVTSLANet.py"),
    "CVTSLANet-Deep": os.path.join(PROJECT_ROOT, "models", "DCVTSLANet.py"),
    "CVCM": os.path.join(PROJECT_ROOT, "models", "CVCMFeature.py"),
    "STC": os.path.join(PROJECT_ROOT, "models", "STCFeature.py"),
    "LinearClassifier": os.path.join(PROJECT_ROOT, "models", "LinearClassifier.py"),
    "NCE": os.path.join(PROJECT_ROOT, "models", "NCELoss.py"),
    "MML": os.path.join(PROJECT_ROOT, "models", "ManifoldMixupLoss.py"),
    "MTL": os.path.join(PROJECT_ROOT, "models", "AutomaticWeightedLoss.py"),
    "Rotator": os.path.join(PROJECT_ROOT, "models", "Rotator.py")
}


def _validate_choice(value, choices, name):
    if value not in choices:
        raise ValueError(f"Unsupported {name} '{value}'. Available values: {', '.join(sorted(choices))}")


# The input param 'feature_dim' must be a valid integer and a positive even number.
def feature_dim_type(value):
    try:
        int_value = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer")

    if int_value <= 0 or int_value % 2 != 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive even number")

    return int_value


# The input param 'epoch_threshold' must be a valid float or integer.
def epoch_threshold_type(value):
    try:
        f_value = float(value)
        if f_value.is_integer():
            return int(f_value)
        return f_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid threshold value: {value}")


def TSLA_add_args(parser, seq_len=4800, patch_size=32, num_channels=2, emb_dim=256, depth=3, dropout_rate=0.3):
    parser.add_argument("--TSLA_len", type=int, default=seq_len)
    parser.add_argument("--TSLA_patch", type=int, default=patch_size)
    parser.add_argument("--TSLA_channels", type=int, default=num_channels)
    parser.add_argument("--TSLA_emb", type=feature_dim_type, default=emb_dim)
    parser.add_argument("--TSLA_depth", type=int, default=depth)
    parser.add_argument("--TSLA_dropout", type=float, default=dropout_rate)
    return parser


def TSLA_parse_args(opt):
    return {
        "seq_len": opt.TSLA_len,
        "patch_size": opt.TSLA_patch,
        "num_channels": opt.TSLA_channels,
        "emb_dim": opt.TSLA_emb,
        "depth": opt.TSLA_depth,
        "dropout_rate": opt.TSLA_dropout
    }


def pretrain_config(encoder_name="ResNet18", classifiar_name="Linear", dataset_name="ads-b", input_type="iq", rot_num=4, input_class=-1,
                    normalize_fn="power", batch_size=32, max_epoch=300, epoch_threshold=0.5, lr=0.001, lr_step=50, lr_gamma=0.1, momentum=0.9,
                    weight_decay=5e-4, feature_dim=2048, tsla_conf=None, mml_b=2.0, save_freq=50, RANDOM_SEED=2024, ablate="", extra_info="",
                    resume="", use_lfdb=False, lfdb_weight=1.0, awgn_enable=True, awgn_min=0.0, awgn_max=30.0,
                    a1_sampler=False, a1_enable=False, a1_weight=0.1, a1_temperature=0.1):
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", "-e", type=str, default=encoder_name)
    parser.add_argument("--classifiar", "-c", type=str, default=classifiar_name)
    parser.add_argument("--dataset", "-d", type=str, default=dataset_name)
    parser.add_argument("--input_type", "-t", type=str, default=input_type)
    parser.add_argument("--rot_num", type=int, default=rot_num)
    parser.add_argument("--input_class", type=int, default=input_class)
    parser.add_argument("--normalize_fn", "-n", type=str, default=normalize_fn)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--epoch", type=int, default=max_epoch)
    parser.add_argument("--threshold", type=epoch_threshold_type, default=epoch_threshold)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--lr_step", type=int, default=lr_step)
    parser.add_argument("--lr_gamma", type=float, default=lr_gamma)
    parser.add_argument("--momentum", type=float, default=momentum)
    parser.add_argument("--weight_decay", type=float, default=weight_decay)
    parser.add_argument("--feature_dim", type=feature_dim_type, default=feature_dim)
    parser.add_argument("--mml_b", type=float, default=mml_b)
    parser.add_argument("--save_freq", type=int, default=save_freq)
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--ablate", "-a", type=str, nargs='+', default=ablate)
    parser.add_argument("--extra_info", type=str, default=extra_info)
    parser.add_argument("--resume", "-r", type=str, default=resume)
    parser.add_argument("--use_lfdb", action="store_true", default=use_lfdb)
    parser.add_argument("--no_lfdb", action="store_false", dest="use_lfdb")
    parser.add_argument("--lfdb_weight", type=float, default=lfdb_weight)
    parser.add_argument("--awgn_enable", action="store_true", default=awgn_enable)
    parser.add_argument("--no_awgn", action="store_false", dest="awgn_enable")
    parser.add_argument("--awgn_min", type=float, default=awgn_min)
    parser.add_argument("--awgn_max", type=float, default=awgn_max)
    parser.add_argument("--a1_sampler", action="store_true", default=a1_sampler)
    parser.add_argument("--a1_enable", action="store_true", default=a1_enable)
    parser.add_argument("--a1_weight", type=float, default=a1_weight)
    parser.add_argument("--a1_temperature", type=float, default=a1_temperature)
    parser.add_argument("--a1_identities_per_batch", type=int, default=8)
    parser.add_argument("--a1_domains_per_identity", type=int, default=2)
    parser.add_argument("--a1_samples_per_domain", type=int, default=2)
    parser.add_argument(
        "--run_tag", type=str, default="",
        help="Optional filesystem-safe suffix used to isolate an experiment run.",
    )
    parser = TSLA_add_args(parser, **({} if tsla_conf is None else tsla_conf))
    opt = parser.parse_args()
    if opt.classifiar.lower() == "linear":
        opt.classifiar = "Linear"

    tsla_conf = TSLA_parse_args(opt)

    encoder_choices = {k for k in model_path_dict if not k.endswith("Classifier") and k not in {"NCE", "MML", "MTL", "Rotator"}}
    _validate_choice(opt.encoder, encoder_choices, "encoder")
    _validate_choice(opt.classifiar, {"Linear"}, "classifier")
    _validate_choice(opt.dataset, dataset_path_dict, "dataset")
    if opt.rot_num < 2:
        raise ValueError("rot_num must be at least 2")
    if "TSLA" in opt.encoder:
        if tsla_conf["seq_len"] <= 0:
            raise ValueError("TSLA_len must be positive")
        if tsla_conf["patch_size"] < 2 or tsla_conf["patch_size"] > tsla_conf["seq_len"]:
            raise ValueError("TSLA_patch must be between 2 and TSLA_len")
    if opt.awgn_min > opt.awgn_max:
        raise ValueError("awgn_min cannot be greater than awgn_max")
    if opt.a1_enable:
        opt.a1_sampler = True
    a1_batch_size = (opt.a1_identities_per_batch * opt.a1_domains_per_identity
                     * opt.a1_samples_per_domain)
    if opt.a1_sampler:
        if opt.dataset not in {"wisig-cross-rx", "wisig-cross-day"}:
            raise ValueError("A1 is restricted to strict WiSig cross-domain datasets")
        if a1_batch_size != opt.batch_size:
            raise ValueError(
                f"A1 structured batch size {a1_batch_size} must equal --batch_size {opt.batch_size}"
            )
        if opt.a1_domains_per_identity < 2:
            raise ValueError("A1 requires at least two domains per identity")
    if opt.a1_weight < 0 or opt.a1_temperature <= 0:
        raise ValueError("A1 weight must be non-negative and temperature positive")
    if opt.run_tag and (not opt.run_tag.replace("-", "").replace("_", "").isalnum()):
        raise ValueError("run_tag may contain only letters, digits, hyphens, and underscores")

    loss_item = ["rot_cls", "sei_cls", "mml"]
    if opt.use_lfdb:
        loss_item.append("decouple")
    ablate_item = ([opt.ablate] if not isinstance(opt.ablate, list) else opt.ablate) if opt.ablate else []
    exp_suffix = ""
    for item in ablate_item:
        if item in loss_item:
            loss_item.remove(item)
            exp_suffix += f"_{item}Ablate"
    if opt.a1_enable:
        # Keep the original 0.1/0.1 pilot path stable, while ensuring any
        # hyperparameter variant cannot overwrite its checkpoints.
        if opt.a1_weight == 0.1 and opt.a1_temperature == 0.1:
            a1_suffix = "_A1C"
        else:
            weight_tag = format(opt.a1_weight, ".8g").replace("-", "m").replace(".", "p")
            temperature_tag = format(opt.a1_temperature, ".8g").replace("-", "m").replace(".", "p")
            a1_suffix = f"_A1C_w{weight_tag}_t{temperature_tag}"
    else:
        a1_suffix = "_A1S" if opt.a1_sampler else ""
    run_suffix = f"_{opt.run_tag}" if opt.run_tag else ""
    exp = f"{opt.encoder}_{dataset_path_dict[opt.dataset]['name']}_{opt.input_type}_{opt.normalize_fn}Norm{exp_suffix}{a1_suffix}{run_suffix}"
    platform = "windows" if sys.platform.startswith("win") else "linux"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_item = tuple(loss_item)
    num_classes = dataset_path_dict[opt.dataset]["pt_class"] if opt.input_class == -1 else opt.input_class
    dataset_norm = opt.normalize_fn
    dataset_root = dataset_path_dict[opt.dataset][platform]
    dataset_root = dataset_root if opt.input_type == "iq" else os.path.join(dataset_root, opt.input_type)
    dataset_root = dataset_root if os.path.exists(os.path.expanduser(dataset_root)) else dataset_root.replace("~", "~/xulai")

    return {
        "exp_name": exp,
        "exp_type": "Pretext",
        "random_seed": opt.random_seed,
        "platform": platform,
        "device": device,
        "start_epoch": 0,
        "epoch": opt.epoch,
        "threshold": min(int(opt.threshold * opt.epoch), opt.epoch) if isinstance(opt.threshold, float) else opt.threshold,
        "save_freq": opt.save_freq,
        "resume": opt.resume,
        "dataset": {
            "name": opt.dataset,
            "root": dataset_root,
            "type": opt.input_type,
            "normalize": dataset_norm,
            "batch_size": opt.batch_size,
            "ratio": 0.2,
            "num_classes": num_classes,
            "signal_length": tsla_conf["seq_len"] if "TSLA" in opt.encoder else None,
        },
        "augmentation": {
            "awgn_enable": opt.awgn_enable,
            "awgn_snr_range": (opt.awgn_min, opt.awgn_max)
        },
        "a1": {
            "sampler_enabled": opt.a1_sampler,
            "loss_enabled": opt.a1_enable,
            "weight": opt.a1_weight,
            "temperature": opt.a1_temperature,
            "domain_key": "RX" if opt.dataset == "wisig-cross-rx" else (
                "DAY" if opt.dataset == "wisig-cross-day" else None
            ),
            "identities_per_batch": opt.a1_identities_per_batch,
            "domains_per_identity": opt.a1_domains_per_identity,
            "samples_per_domain": opt.a1_samples_per_domain,
        },
        "encoder": {
            "name": opt.encoder,
            "root": model_path_dict[opt.encoder],
            "feature_dim": opt.feature_dim,
            "TSLA_config": tsla_conf
        },
        "rot_classifier": {
            "root": model_path_dict[f"{opt.classifiar}Classifier"],
            "in_dim": opt.feature_dim,
            "num_classes": opt.rot_num
        },
        "mixed_classifier": {
            "root": model_path_dict[f"{opt.classifiar}Classifier"],
            "in_dim": opt.feature_dim,
            "num_classes": num_classes
        },
        "optimizer": {
            "lr": opt.lr,
            "momentum": opt.momentum,
            "weight_decay": opt.weight_decay,
            "step_size": opt.lr_step,
            "gamma": opt.lr_gamma
        },
        "mml": {
            "root": model_path_dict["MML"],
            "beta": opt.mml_b
        },
        "mtl": {
            "root": model_path_dict["MTL"],
            "item": loss_item,
            "num": len(loss_item)
        },
        "lfdb": {
            "enabled": opt.use_lfdb,
            "weight": opt.lfdb_weight,
            "num_classes": num_classes
        },
        "extra_info": opt.extra_info
    }


def finetune_config(encoder_name="ResNet18", classifier_name="Linear", dataset_name="ads-b", input_type="iq", input_class=-1, normalize_fn="power",
                    train_batch_size=30, test_batch_size=30, shot=5, max_epoch=50, max_iteration=100, lr=0.001, weight_decay=0, feature_dim=2048,
                    tsla_conf=None, RANDOM_SEED=2024, pretrain_normalize_fn="same", pretrain_batch_size=32, pretrain_epoch=250, ablate="",
                    extra_info="", pretrain_date="", snr_enable=False, snr=0):
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", "-e", type=str, default=encoder_name)
    parser.add_argument("--classifier", "-c", type=str, default=classifier_name)
    parser.add_argument("--dataset", "-d", type=str, default=dataset_name)
    parser.add_argument("--input_type", "-t", type=str, default=input_type)
    parser.add_argument("--input_class", type=int, default=input_class)
    parser.add_argument("--normalize_fn", "-n", type=str, default=normalize_fn)
    parser.add_argument("--train_batch_size", type=int, default=train_batch_size)
    parser.add_argument("--test_batch_size", type=int, default=test_batch_size)
    parser.add_argument("--shot", "-s", type=int, nargs='+', default=shot)
    parser.add_argument("--epoch", type=int, default=max_epoch)
    parser.add_argument("--iteration", "-i", type=int, default=max_iteration)
    parser.add_argument("--lr", type=float, default=lr)
    parser.add_argument("--weight_decay", type=float, default=weight_decay)
    parser.add_argument("--feature_dim", type=feature_dim_type, default=feature_dim)
    parser.add_argument("--random_seed", "-r", type=int, default=RANDOM_SEED)
    parser.add_argument("--pretrain_normalize_fn", type=str, default=normalize_fn if pretrain_normalize_fn == "same" else pretrain_normalize_fn)
    parser.add_argument("--pretrain_batch_size", type=int, default=pretrain_batch_size)
    parser.add_argument("--pretrain_epoch", type=int, default=pretrain_epoch)
    parser.add_argument("--pretrain_date", type=str, default=pretrain_date)
    parser.add_argument("--ablate", "-a", type=str, nargs='+', default=ablate)
    parser.add_argument("--extra_info", type=str, default=extra_info)
    parser.add_argument("--snr_enable", action="store_true", default=snr_enable)
    parser.add_argument("--snr", type=int, nargs='+', default=snr)
    parser = TSLA_add_args(parser, **({} if tsla_conf is None else tsla_conf))
    opt = parser.parse_args()
    classifier_aliases = {
        "dl": "Linear",
        "linear": "Linear",
        "lr": "lr",
        "knn": "knn",
        "svm": "svm",
    }
    opt.classifier = classifier_aliases.get(opt.classifier.lower(), opt.classifier)
    encoder_choices = {k for k in model_path_dict if not k.endswith("Classifier") and k not in {"NCE", "MML", "MTL", "Rotator"}}
    _validate_choice(opt.encoder, encoder_choices, "encoder")
    _validate_choice(opt.classifier.lower(), {"linear", "lr", "knn", "svm"}, "classifier")
    _validate_choice(opt.dataset, dataset_path_dict, "dataset")

    tsla_conf = TSLA_parse_args(opt)
    if "TSLA" in opt.encoder:
        if tsla_conf["seq_len"] <= 0:
            raise ValueError("TSLA_len must be positive")
        if tsla_conf["patch_size"] < 2 or tsla_conf["patch_size"] > tsla_conf["seq_len"]:
            raise ValueError("TSLA_patch must be between 2 and TSLA_len")

    loss_item = ["rot_cls", "sei_cls", "mml"]
    ablate_item = ([opt.ablate] if not isinstance(opt.ablate, list) else opt.ablate) if opt.ablate else []
    exp_suffix = ""
    for item in ablate_item:
        if item in loss_item:
            exp_suffix += f"_{item}Ablate"
    dataset_fullname = opt.dataset + "_" + opt.input_type + ("_snr" if opt.snr_enable else "")
    exp = f"{opt.encoder}_{dataset_fullname}_PT_{opt.pretrain_normalize_fn}Norm_FT_{opt.normalize_fn}Norm{exp_suffix}"
    pretrain_exp = f"{opt.encoder}_{dataset_path_dict[opt.dataset]['name']}_{opt.input_type}_{opt.pretrain_normalize_fn}Norm{exp_suffix}"
    platform = "windows" if sys.platform.startswith("win") else "linux"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = dataset_path_dict[opt.dataset]["ft_class"] if opt.input_class == -1 else opt.input_class
    dataset_norm = opt.normalize_fn
    dataset_root = dataset_path_dict[opt.dataset][platform]
    dataset_root = dataset_root if opt.input_type == "iq" else os.path.join(dataset_root, opt.input_type)
    dataset_root = dataset_root if os.path.exists(os.path.expanduser(dataset_root)) else dataset_root.replace("~", "~/xulai")

    return {
        "exp_name": exp,
        "exp_type": "Downstream" if opt.pretrain_epoch else 'FineZero',
        "random_seed": opt.random_seed,
        "platform": platform,
        "device": device,
        "iteration": opt.iteration,
        "epoch": opt.epoch,
        "dataset": {
            "name": opt.dataset,
            "root": dataset_root,
            "type": opt.input_type,
            "normalize": dataset_norm,
            "train_batch_size": opt.train_batch_size,
            "test_batch_size": opt.test_batch_size,
            "ratio": 0.2,
            "num_classes": num_classes,
            "signal_length": tsla_conf["seq_len"] if "TSLA" in opt.encoder else None,
            "query_split": dataset_path_dict[opt.dataset].get("query_split", "test"),
            "shot": opt.shot if isinstance(opt.shot, list) else [opt.shot],
            "snr": (opt.snr if isinstance(opt.snr, list) else [opt.snr]) if opt.snr_enable else [None]
        },
        "encoder": {
            "name": opt.encoder,
            "root": model_path_dict[opt.encoder],
            "feature_dim": opt.feature_dim,
            "pretrain_path": os.path.join(
                PROJECT_ROOT,
                "runs",
                "Pretext_random_rot",
                pretrain_exp,
                *([opt.pretrain_date] if opt.pretrain_date else []),
                "best_encoder.pth"
            ) if opt.pretrain_epoch else "",
            "TSLA_config": tsla_conf
        },
        "classifier": {
            "name": opt.classifier.lower(),
            "root": model_path_dict[f"{opt.classifier}Classifier"] if opt.classifier.lower() not in ["lr", "knn", "svm"] else "",
            "in_dim": opt.feature_dim,
            "ratio": 0.2,
            "num_classes": num_classes
        },
        "optimizer": {
            "lr": opt.lr,
            "weight_decay": opt.weight_decay
        },
        "extra_info": opt.extra_info
    }


if __name__ == "__main__":
    import json

    pretrain_conf = pretrain_config()
    finetune_conf = finetune_config()
    print("==> Pretrain Config:", json.dumps(pretrain_conf))
    print("==> Finetune Config:", json.dumps(finetune_conf))
