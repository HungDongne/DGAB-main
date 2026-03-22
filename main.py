import gc
import logging
import os
import random
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import dgl
import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def fix_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def parse_args():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler="resolve"
    )
    parser.add_argument("--method", default="dgab")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dataset",
        type=str,
        default="S-FFSD",
        choices=["S-FFSD", "yelp", "amazon"],
        help="Dataset to use (overrides config file)",
    )

    parsed = parser.parse_args()
    method = vars(parsed)["method"]
    seed = vars(parsed)["seed"]
    dataset = vars(parsed)["dataset"]

    if method == "dgab":
        yaml_file = "config/dgab_cfg.yaml"
    else:
        raise NotImplementedError("Unsupported method.")

    # config = Config().get_config()
    with open(yaml_file) as file:
        args = yaml.safe_load(file)
    args["method"] = method
    args["seed"] = seed
    # Override dataset if specified from command line
    if dataset is not None:
        args["dataset"] = dataset
    fix_seed(seed)
    return args


def main(args):
    if args["method"] == "dgab":
        from methods.dgab.dgab_data import load_dgab_data
        from methods.dgab.dgab_main import dgab_main

        fix_seed(args["seed"])
        feat_data, labels, train_idx, test_idx, g, cat_features = load_dgab_data(
            args["dataset"], args["test_size"], args["seed"]
        )
        auc, ap, f1 = dgab_main(
            feat_data, g, train_idx, test_idx, labels, args, cat_features
        )
        print(f"AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")
        clear_memory()
    else:
        raise NotImplementedError("Unsupported method.")


if __name__ == "__main__":
    main(parse_args())
