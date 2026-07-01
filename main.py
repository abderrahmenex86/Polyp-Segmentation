import argparse
import os

import torch

from src.dataset import build_dataloaders
from src.factory import build_model
from src.helpers import load_json
from src.infer import infer_run
from src.trainer import eval_epoch, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Ex86 Endoscopy Architecture")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--infer", action="store_true")

    known_args, unknown_args = parser.parse_known_args()
    config = vars(known_args)

    for i in range(0, len(unknown_args), 2):
        key = unknown_args[i].lstrip("--")
        val = unknown_args[i + 1]
        if val.isdigit():
            config[key] = int(val)
        elif val.replace(".", "", 1).isdigit():
            config[key] = float(val)
        elif val.lower() in ["true", "false"]:
            config[key] = val.lower() == "true"
        else:
            config[key] = val

    return config


def main():
    config = parse_args()

    if config.get("train", False):
        train_model(config)

    if config.get("test", False):
        run_dir = config.get("run_dir")
        if run_dir and os.path.exists(os.path.join(run_dir, "hyperparameters.json")):
            config.update(load_json(os.path.join(run_dir, "hyperparameters.json")))
            _, _, test_loader = build_dataloaders(config)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = build_model(config).to(device)
            model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device))
            eval_epoch(model, test_loader, device)

    if config.get("infer", False):
        infer_run(config.get("inference_dir", "dataset/inference"), config.get("run_dir"))


if __name__ == "__main__":
    main()
