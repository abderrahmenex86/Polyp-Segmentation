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
