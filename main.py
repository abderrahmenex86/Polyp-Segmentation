import argparse
import os

import torch

from src.dataset import build_dataloaders
from src.factory import build_model
from src.helpers import load_json
from src.infer import infer_run
from src.profile import profile_hardware_limits
from src.trainer import eval_epoch, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Ex86 Endoscopy Architecture")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--profile", action="store_true")

    parser.add_argument("--architecture", type=str, default="UNet")
    parser.add_argument("--dataset_directory", type=str, default="dataset")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--inference_dir", type=str, default="dataset/inference")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_threshold", type=float, default=1.0)

    parser.add_argument("--image_height", type=int, default=352)
    parser.add_argument("--image_width", type=int, default=352)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--out_classes", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--medsam_checkpoint", type=str, default=None)
    parser.add_argument("--pranet_weights", type=str, default=None)
    parser.add_argument("--backbone_weights", type=str, default=None)

    return vars(parser.parse_args())


def main():
    config = parse_args()

    if config.get("profile"):
        profile_hardware_limits(config)

    if config.get("train"):
        train_model(config)

    if config.get("test"):
        run_dir = config.get("run_dir")
        if run_dir and os.path.exists(os.path.join(run_dir, "hyperparameters.json")):
            config.update(load_json(os.path.join(run_dir, "hyperparameters.json")))
            _, _, test_loader = build_dataloaders(config)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = build_model(config).to(device)
            model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device))
            eval_epoch(model, test_loader, device)

    if config.get("infer"):
        infer_run(config.get("inference_dir"), config.get("run_dir"))


if __name__ == "__main__":
    main()
