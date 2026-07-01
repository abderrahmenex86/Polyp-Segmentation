import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def log_message(category, message):
    timestamp = datetime.now().strftime("%m/%d - %H:%M")
    tqdm.write(f"[{category.upper()}] [{timestamp}] {message}")


def create_directory(path):
    os.makedirs(path, exist_ok=True)


def generate_run_dir(architecture):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("artifacts", f"{timestamp}_{architecture}")
    create_directory(run_dir)
    return run_dir


def save_json(data, path):
    with open(path, "w") as file_handle:
        json.dump(data, file_handle, indent=4)


def load_json(path):
    with open(path, "r") as file_handle:
        return json.load(file_handle)


def plot_pre_training_batch(dataloader, save_path):
    batch = next(iter(dataloader))
    images = batch["image"]
    masks = batch["mask"]

    batch_size = images.shape[0]
    display_count = min(batch_size, 4)

    fig, axes = plt.subplots(display_count, 3, figsize=(12, 4 * display_count))
    if display_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(display_count):
        image_np = images[i].cpu().numpy().transpose(1, 2, 0)
        mask_np = masks[i].cpu().numpy()[0]

        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)

        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_np, cmap="gray")
        axes[i, 1].set_title("Ground Truth Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(image_np)
        axes[i, 2].imshow(mask_np, cmap="jet", alpha=0.4)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    create_directory(os.path.dirname(save_path))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
