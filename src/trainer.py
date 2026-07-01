import os

import torch
from monai.metrics import DiceMetric, MeanIoU
from tqdm import tqdm

from src.dataset import build_dataloaders
from src.factory import build_model, build_optimizer, build_scheduler
from src.helpers import (
    generate_run_dir,
    log_message,
    plot_pre_training_batch,
    save_json,
)
from src.models import build_loss


def eval_epoch(model, dataloader, device):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images, masks) if hasattr(model, "sam") else model(images)
            preds = (torch.sigmoid(logits) > 0.5).float()

            dice_metric(y_pred=preds, y=masks)
            iou_metric(y_pred=preds, y=masks)

    dice = dice_metric.aggregate().item()
    iou = iou_metric.aggregate().item()
    dice_metric.reset()
    iou_metric.reset()

    return dice, iou


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = generate_run_dir(config.get("architecture", "UNet"))
    save_json(config, os.path.join(run_dir, "hyperparameters.json"))

    train_loader, val_loader, _ = build_dataloaders(config)
    plot_pre_training_batch(train_loader, os.path.join("docs", "figs", f"pre_train_{os.path.basename(run_dir)}.png"))

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    loss_fn = build_loss()

    with open(os.path.join(run_dir, "architecture.txt"), "w") as file_handle:
        file_handle.write(str(model))

    epochs = config.get("epochs", 100)
    patience = config.get("patience", 15)
    clip_thresh = config.get("clip_threshold", 1.0)

    best_dice = 0.0
    stagnant_epochs = 0
    history = {"train_loss": [], "val_dice": [], "val_iou": []}

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images, masks) if hasattr(model, "sam") else model(images)
                loss = loss_fn(logits, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh)
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)

        val_dice, val_iou = eval_epoch(model, val_loader, device)

        history["train_loss"].append(avg_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        save_json(history, os.path.join(run_dir, "model_history.json"))

        log_message(
            "train",
            f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val DSC: {val_dice:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}",
        )

        if val_dice > best_dice:
            best_dice = val_dice
            stagnant_epochs = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
            log_message("save", f"New best DSC: {best_dice:.4f} serialized.")
        else:
            stagnant_epochs += 1

        if stagnant_epochs >= patience:
            log_message("stop", f"Early stopping triggered at epoch {epoch}.")
            break
