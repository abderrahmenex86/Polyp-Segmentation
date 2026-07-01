import os
import glob
from typing import Tuple, List, Dict
import torch
from torch.utils.data import random_split
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Resized,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
)
from monai.data import Dataset, DataLoader
from src.helpers import log_system_message


def build_data_transformations(
    is_training_phase_flag: bool, target_spatial_resolution: Tuple[int, int] = (256, 256)
) -> Compose:
    base_transformations_list = [
        LoadImaged(keys=["image_data", "segmentation_mask_data"]),
        EnsureChannelFirstd(keys=["image_data", "segmentation_mask_data"]),
        ScaleIntensityd(keys=["image_data", "segmentation_mask_data"]),
        Resized(keys=["image_data", "segmentation_mask_data"], spatial_size=target_spatial_resolution),
    ]

    augmentation_transformations_list = (
        [
            RandFlipd(keys=["image_data", "segmentation_mask_data"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image_data", "segmentation_mask_data"], spatial_axis=1, prob=0.5),
            RandAffined(keys=["image_data", "segmentation_mask_data"], rotate_range=0.3, scale_range=0.1, prob=0.5),
            RandGaussianNoised(keys=["image_data"], prob=0.2),
        ]
        if is_training_phase_flag
        else []
    )

    return Compose(base_transformations_list + augmentation_transformations_list)


def discover_dataset_filepaths(dataset_root_directory: str) -> List[Dict[str, str]]:
    image_filepaths_list = sorted(glob.glob(os.path.join(dataset_root_directory, "images", "*.*")))
    mask_filepaths_list = sorted(glob.glob(os.path.join(dataset_root_directory, "masks", "*.*")))

    if len(image_filepaths_list) != len(mask_filepaths_list) or len(image_filepaths_list) == 0:
        log_system_message("error", "Dataset image and mask counts are mismatched or empty.")
        raise RuntimeError("Dataset inconsistency detected.")

    return [
        {"image_data": image_path, "segmentation_mask_data": mask_path}
        for image_path, mask_path in zip(image_filepaths_list, mask_filepaths_list)
    ]


def construct_dataloaders(hyperparameter_dictionary: Dict[str, any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_dictionary_list = discover_dataset_filepaths(hyperparameter_dictionary.get("dataset_directory", "dataset"))

    total_samples_count = len(dataset_dictionary_list)
    training_samples_count = int(total_samples_count * 0.70)
    validation_samples_count = int(total_samples_count * 0.15)
    testing_samples_count = total_samples_count - training_samples_count - validation_samples_count

    training_subset, validation_subset, testing_subset = random_split(
        dataset_dictionary_list,
        [training_samples_count, validation_samples_count, testing_samples_count],
        generator=torch.Generator().manual_seed(hyperparameter_dictionary.get("random_seed_value", 42)),
    )

    target_resolution_tuple = (
        hyperparameter_dictionary.get("image_height", 256),
        hyperparameter_dictionary.get("image_width", 256),
    )

    training_dataset_instance = Dataset(
        data=list(training_subset), transform=build_data_transformations(True, target_resolution_tuple)
    )
    validation_dataset_instance = Dataset(
        data=list(validation_subset), transform=build_data_transformations(False, target_resolution_tuple)
    )
    testing_dataset_instance = Dataset(
        data=list(testing_subset), transform=build_data_transformations(False, target_resolution_tuple)
    )

    batch_size_integer = hyperparameter_dictionary.get("batch_size", 8)
    number_of_workers_integer = hyperparameter_dictionary.get("number_of_workers", 4)
    pin_memory_boolean = torch.cuda.is_available()

    training_dataloader_instance = DataLoader(
        training_dataset_instance,
        batch_size=batch_size_integer,
        shuffle=True,
        num_workers=number_of_workers_integer,
        pin_memory=pin_memory_boolean,
        persistent_workers=True,
        prefetch_factor=4,
    )
    validation_dataloader_instance = DataLoader(
        validation_dataset_instance,
        batch_size=batch_size_integer,
        shuffle=False,
        num_workers=number_of_workers_integer,
        pin_memory=pin_memory_boolean,
        persistent_workers=True,
        prefetch_factor=4,
    )
    testing_dataloader_instance = DataLoader(
        testing_dataset_instance,
        batch_size=batch_size_integer,
        shuffle=False,
        num_workers=number_of_workers_integer,
        pin_memory=pin_memory_boolean,
        persistent_workers=True,
        prefetch_factor=4,
    )

    return training_dataloader_instance, validation_dataloader_instance, testing_dataloader_instance
