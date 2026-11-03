
# ==================================================
# ORIGINAL FILE: src/helpers.py
# ==================================================

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



# ==================================================
# ORIGINAL FILE: src/dataset.py
# ==================================================

import glob
import os

import torch
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandFlipd,
    Resized,
    ScaleIntensityd,
    Spacingd,
)


def build_transforms(is_train, target_height, target_width):
    base_transforms = [
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityd(keys=["image", "mask"]),
        Resized(keys=["image", "mask"], spatial_size=(target_height, target_width)),
    ]

    augmentation_transforms = (
        [
            RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
            RandAffined(keys=["image", "mask"], rotate_range=0.3, scale_range=0.1, prob=0.5),
        ]
        if is_train
        else []
    )

    return Compose(base_transforms + augmentation_transforms)


def get_data_dicts(data_dir):
    image_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.*")))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "masks", "*.*")))
    return [{"image": img, "mask": msk} for img, msk in zip(image_paths, mask_paths)]


def build_dataloaders(config):
    data_dicts = get_data_dicts(config.get("dataset_directory"))

    total_samples = len(data_dicts)
    train_size = int(total_samples * 0.70)
    val_size = int(total_samples * 0.15)
    test_size = total_samples - train_size - val_size

    generator = torch.Generator().manual_seed(config.get("random_seed"))
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        data_dicts, [train_size, val_size, test_size], generator=generator
    )

    target_h = config.get("image_height")
    target_w = config.get("image_width")

    train_ds = CacheDataset(
        data=list(train_subset), transform=build_transforms(True, target_h, target_w), cache_rate=1.0, num_workers=8
    )
    val_ds = CacheDataset(
        data=list(val_subset), transform=build_transforms(False, target_h, target_w), cache_rate=1.0, num_workers=4
    )
    test_ds = CacheDataset(
        data=list(test_subset), transform=build_transforms(False, target_h, target_w), cache_rate=1.0, num_workers=4
    )

    batch_size = config.get("batch_size")
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4,
    )

    return train_loader, val_loader, test_loader



# ==================================================
# ORIGINAL FILE: src/models.py
# ==================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR, UNet, ViT


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype="normal"):
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [
            block(self.inplanes, planes, stride, downsample, stype="stage", baseWidth=self.baseWidth, scale=self.scale)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))
        return nn.Sequential(*layers)


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ReceptiveFieldBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch0 = nn.Sequential(BasicConv(in_channels, out_channels, 1))
        self.branch1 = nn.Sequential(
            BasicConv(in_channels, out_channels, 1), BasicConv(out_channels, out_channels, 3, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_channels, out_channels, 1), BasicConv(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_channels, out_channels, 1), BasicConv(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.conv_cat = BasicConv(4 * out_channels, out_channels, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        return F.relu(x_cat + self.conv_res(x), inplace=True)


class PartialDecoder(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = BasicConv(channel, channel, 3, padding=1)
        self.conv_up2 = BasicConv(channel, channel, 3, padding=1)
        self.conv_concat = BasicConv(3 * channel, channel, 3, padding=1)

    def forward(self, f2, f3, f4):
        f3_up = self.upsample(f4)
        f3_out = self.conv_up1(f3_up * f3)
        f2_up = self.upsample(f3_out)
        f2_out = self.conv_up2(f2_up * f2)
        f4_up_2x = self.upsample(self.upsample(f4))
        f3_up_1x = self.upsample(f3_out)
        return self.conv_concat(torch.cat((f2_out, f3_up_1x, f4_up_2x), 1))


class ReverseAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Sequential(BasicConv(channel, channel, 3, padding=1), nn.Conv2d(channel, 1, 1))

    def forward(self, features, prior_mask):
        reverse_mask = -1 * torch.sigmoid(prior_mask) + 1
        return self.conv(features * reverse_mask)


class PraNet(nn.Module):
    def __init__(self, backbone_weights=None, pranet_weights=None):
        super().__init__()
        self.resnet = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)

        if backbone_weights and os.path.exists(backbone_weights):
            self.resnet.load_state_dict(torch.load(backbone_weights))

        self.rfb2 = ReceptiveFieldBlock(512, 32)
        self.rfb3 = ReceptiveFieldBlock(1024, 32)
        self.rfb4 = ReceptiveFieldBlock(2048, 32)

        self.pd = PartialDecoder(32)
        self.ra3 = ReverseAttention(32)
        self.ra2 = ReverseAttention(32)

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.map_out = nn.Conv2d(32, 1, 1)

        if pranet_weights and os.path.exists(pranet_weights):
            self.load_state_dict(torch.load(pranet_weights))

    def forward(self, x):
        l0 = self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x)))
        l0_pool = self.resnet.maxpool(l0)
        l1 = self.resnet.layer1(l0_pool)
        l2 = self.resnet.layer2(l1)
        l3 = self.resnet.layer3(l2)
        l4 = self.resnet.layer4(l3)

        f2 = self.rfb2(l2)
        f3 = self.rfb3(l3)
        f4 = self.rfb4(l4)

        global_map = self.pd(f2, f3, f4)
        global_pred = self.map_out(global_map)

        ra3_out = self.ra3(f3, global_pred)
        ra3_pred = global_pred + self.upsample2(ra3_out)

        ra2_out = self.ra2(f2, ra3_pred)
        ra2_pred = ra3_pred + self.upsample2(ra2_out)

        return self.upsample4(ra2_pred)


def build_loss():
    return DiceCELoss(
        include_background=False, sigmoid=True, jaccard=False, reduction="mean", weight=torch.tensor([1.0])
    )



# ==================================================
# ORIGINAL FILE: src/factory.py
# ==================================================

import torch
from monai.networks.nets import UNETR, UNet, ViT





def build_model(config):
    architecture = config.get("architecture").lower()
    in_channels = config.get("in_channels")
    out_classes = config.get("out_classes")
    resolution = (config.get("image_height"), config.get("image_width"))

    if architecture == "unet":
        return UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
    elif architecture == "unetr":
        return UNETR(in_channels=in_channels, out_channels=out_classes, img_size=resolution, spatial_dims=2)
    elif architecture == "vit":
        return ViT(
            in_channels=in_channels,
            img_size=resolution,
            patch_size=(16, 16),
            classification=False,
            post_activation="Sigmoid",
        )
    elif architecture == "pranet":
        return PraNet(backbone_weights=config.get("backbone_weights"), pranet_weights=config.get("pranet_weights"))
    else:
        log_message("error", f"Architecture {architecture} not recognized.")
        raise ValueError("Invalid architecture.")


def build_optimizer(model, config):
    lr = float(config.get("learning_rate"))
    weight_decay = float(config.get("weight_decay"))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, config):
    total_epochs = int(config.get("epochs"))
    warmup_epochs = int(config.get("warmup_epochs"))

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-7)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])



# ==================================================
# ORIGINAL FILE: src/profile.py
# ==================================================

import gc

import torch
import torchinfo






def profile_hardware_limits(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log_message("profile", "Profiling is designed for CUDA memory limits. Skipping.")
        return

    log_message("profile", "Initiating batch size limits profiling...")

    image_h = config.get("image_height")
    image_w = config.get("image_width")
    in_channels = config.get("in_channels")

    model_initial = build_model(config).to(device)
    dummy_input_initial = torch.randn(1, in_channels, image_h, image_w, device=device)

    model_stats = torchinfo.summary(model_initial, input_data=dummy_input_initial, verbose=0)

    log_message("profile", f"Model Architecture Summary:\n{str(model_stats)}")

    del model_initial, dummy_input_initial
    torch.cuda.empty_cache()
    gc.collect()

    current_batch_size = 2
    max_safe_batch_size = 2

    while True:
        try:
            model = build_model(config).to(device)
            optimizer = build_optimizer(model, config)
            loss_fn = build_loss()

            dummy_images = torch.randn(current_batch_size, in_channels, image_h, image_w, device=device)
            dummy_masks = torch.randint(
                0, 2, (current_batch_size, 1, image_h, image_w), device=device, dtype=torch.float32
            )

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(dummy_images)
                loss = loss_fn(logits, dummy_masks)

            loss.backward()
            optimizer.step()

            max_safe_batch_size = current_batch_size
            current_batch_size += 2

            del model, optimizer, loss_fn, dummy_images, dummy_masks, logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                break
            else:
                raise e

    recommended_batch_size = max(2, max_safe_batch_size - 2)
    log_message("profile", f"Absolute Max Batch Size: {max_safe_batch_size}")
    log_message("profile", f"Recommended Safe Batch Size: {recommended_batch_size}")



# ==================================================
# ORIGINAL FILE: src/trainer.py
# ==================================================

import os

import torch
import torchinfo
from monai.metrics import DiceMetric, MeanIoU
from tqdm import tqdm




    generate_run_dir,
    log_message,
    plot_pre_training_batch,
    save_json,
)



def eval_epoch(model, dataloader, device):
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    iou_metric = MeanIoU(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
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
    run_dir = generate_run_dir(config.get("architecture"))
    save_json(config, os.path.join(run_dir, "hyperparameters.json"))

    train_loader, val_loader, _ = build_dataloaders(config)
    plot_pre_training_batch(train_loader, os.path.join("docs", "figs", f"pre_train_{os.path.basename(run_dir)}.png"))

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    loss_fn = build_loss()

    dummy_input = torch.randn(1, config.get("in_channels"), config.get("image_height"), config.get("image_width")).to(
        device
    )
    model_stats = torchinfo.summary(model, input_data=dummy_input, verbose=0)

    with open(os.path.join(run_dir, "architecture.txt"), "w") as file_handle:
        file_handle.write(str(model_stats))

    epochs = config.get("epochs")
    patience = config.get("patience")
    clip_thresh = config.get("clip_threshold")

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
                logits = model(images)
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



# ==================================================
# ORIGINAL FILE: src/infer.py
# ==================================================

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



# ==================================================
# ORIGINAL FILE: tools.py
# ==================================================

import argparse
import glob
import os
import shutil
import subprocess

import matplotlib.pyplot as plt




def execute_plot_generation(run_dir):
    history_path = os.path.join(run_dir, "model_history.json")
    if not os.path.exists(history_path):
        raise FileNotFoundError("Model history artifact missing.")

    history = load_json(history_path)
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Training Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["val_dice"], label="Validation DSC")
    plt.plot(epochs_range, history["val_iou"], label="Validation IoU")
    plt.legend()

    create_directory("docs/figs")
    plt.savefig(f"docs/figs/performance_{os.path.basename(run_dir)}.png")
    log_message("plot", "Performance figure generated in docs/figs/")


def execute_dataset_verification(data_dir):
    images_count = len(glob.glob(os.path.join(data_dir, "images", "*.*")))
    masks_count = len(glob.glob(os.path.join(data_dir, "masks", "*.*")))
    log_message("verify", f"Images: {images_count} | Masks: {masks_count}")


def execute_unified_download_and_mapping(target_dir):
    raw_dir = "dataset_raw"
    create_directory(raw_dir)
    create_directory(os.path.join(target_dir, "images"))
    create_directory(os.path.join(target_dir, "masks"))

    datasets = {
        "kvasir": {"type": "direct", "url": "https://datasets.simula.no/downloads/kvasir-seg.zip"},
        "cvc_clinic": {"type": "kaggle", "url": "balraj98/cvcclinicdb"},
        "cvc_colon": {"type": "kaggle", "url": "longvil/cvc-colondb"},
        "etis_larib": {"type": "kaggle", "url": "nguyenvoquocduong/etis-laribpolypdb"},
        "cvc_300": {"type": "kaggle", "url": "nourabentaher/cvc-300"},
    }

    for name, info in datasets.items():
        extract_dir = os.path.join(raw_dir, name)

        # Check if already downloaded and extracted
        if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) > 0:
            log_message("download", f"[{name}] already exists. Skipping download.")
            continue

        log_message("download", f"Processing {name}...")
        create_directory(extract_dir)

        try:
            if info["type"] == "direct":
                archive_path = os.path.join(raw_dir, f"{name}.zip")
                subprocess.run(
                    [
                        "aria2c",
                        "-x",
                        "16",
                        "-s",
                        "16",
                        "--check-certificate=false",
                        "-d",
                        raw_dir,
                        "-o",
                        f"{name}.zip",
                        info["url"],
                    ],
                    check=True,
                )
                shutil.unpack_archive(archive_path, extract_dir)

            elif info["type"] == "kaggle":
                subprocess.run(
                    ["kaggle", "datasets", "download", "-d", info["url"], "-p", raw_dir, "--unzip"], check=True
                )
                kaggle_extracted_files = glob.glob(os.path.join(raw_dir, "*"))
                for file_path in kaggle_extracted_files:
                    # Move extracted kaggle files into their specific dataset directory to avoid collisions
                    if (
                        os.path.isdir(file_path)
                        and file_path != extract_dir
                        and os.path.basename(file_path) not in datasets.keys()
                    ):
                        shutil.move(file_path, extract_dir)
        except Exception as e:
            log_message("error", f"Failed to download/extract {name}: {str(e)}")
            continue

    log_message("map", "Aggregating and mapping extracted files...")

    image_folder_variants = [
        "images",
        "Original",
        "CVC-300/images",
        "CVC-ColonDB/images",
        "ETIS-LaribPolypDB/images",
        "PNG/Original",
    ]
    mask_folder_variants = [
        "masks",
        "Ground Truth",
        "CVC-300/masks",
        "CVC-ColonDB/masks",
        "ETIS-LaribPolypDB/masks",
        "PNG/Ground Truth",
    ]

    for name in datasets.keys():
        dataset_base_path = os.path.join(raw_dir, name)

        found_images_dir = None
        found_masks_dir = None

        for root, dirs, files in os.walk(dataset_base_path):
            folder_name = os.path.basename(root)
            if folder_name in [os.path.basename(v) for v in image_folder_variants] and not found_images_dir:
                found_images_dir = root
            if folder_name in [os.path.basename(v) for v in mask_folder_variants] and not found_masks_dir:
                found_masks_dir = root

        if found_images_dir and found_masks_dir:
            image_files = sorted(glob.glob(os.path.join(found_images_dir, "*.*")))
            mask_files = sorted(glob.glob(os.path.join(found_masks_dir, "*.*")))

            for img_path, msk_path in zip(image_files, mask_files):
                img_name = os.path.basename(img_path)
                msk_name = os.path.basename(msk_path)

                new_img_path = os.path.join(target_dir, "images", f"{name}_{img_name}")
                new_msk_path = os.path.join(target_dir, "masks", f"{name}_{msk_name}")

                shutil.copy(img_path, new_img_path)
                shutil.copy(msk_path, new_msk_path)
        else:
            log_message(
                "warning", f"Could not automatically resolve image/mask folders for {name}. Manual inspection required."
            )

    log_message("map", "Consolidation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utilities")
    parser.add_argument("--mode", choices=["download", "verify", "plot"], required=True)
    parser.add_argument("--path", type=str, default="")
    args = parser.parse_args()

    if args.mode == "verify":
        execute_dataset_verification(args.path or "dataset")
    elif args.mode == "plot":
        execute_plot_generation(args.path)
    elif args.mode == "download":
        execute_unified_download_and_mapping(args.path or "dataset")

