import glob
import os

import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Resized,
    ScaleIntensityd,
)

from src.factory import build_model
from src.helpers import load_json, log_message


def infer_run(inference_dir, run_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_json(os.path.join(run_dir, "hyperparameters.json"))

    model = build_model(config)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), map_location=device))
    model.to(device)
    model.eval()

    image_paths = sorted(glob.glob(os.path.join(inference_dir, "*.*")))
    data_dicts = [{"image": path} for path in image_paths]

    h = config.get("image_height", 352)
    w = config.get("image_width", 352)

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(h, w)),
        ]
    )

    loader = DataLoader(Dataset(data=data_dicts, transform=transforms), batch_size=1, shuffle=False)
    out_dir = os.path.join(run_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    log_message("infer", f"Starting inference on {len(data_dicts)} frames.")

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            out_name = os.path.basename(image_paths[idx])
            torch.save(preds.cpu(), os.path.join(out_dir, out_name.replace(".png", ".pt").replace(".jpg", ".pt")))

    log_message("infer", f"Inference saved to {out_dir}")
