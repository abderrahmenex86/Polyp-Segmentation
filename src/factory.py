import torch
import torch.nn as nn
from typing import Dict, Any
from monai.networks.nets import UNet, UNETR, ViT
from src.models import CustomPolypSegmentationArchitecture, PraNetStubArchitecture, MedSAMStubArchitecture
from src.helpers import log_system_message


def instantiate_model_architecture(hyperparameter_dictionary: Dict[str, Any]) -> nn.Module:
    architecture_name_string = hyperparameter_dictionary.get("architecture_name", "UNet").lower()
    input_channels_integer = hyperparameter_dictionary.get("input_channels", 3)
    output_classes_integer = hyperparameter_dictionary.get("output_classes", 1)
    image_resolution_tuple = (
        hyperparameter_dictionary.get("image_height", 256),
        hyperparameter_dictionary.get("image_width", 256),
    )

    if architecture_name_string == "unet":
        model_instance = UNet(
            spatial_dims=2,
            in_channels=input_channels_integer,
            out_channels=output_classes_integer,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
    elif architecture_name_string == "unetr":
        model_instance = UNETR(
            in_channels=input_channels_integer,
            out_channels=output_classes_integer,
            img_size=image_resolution_tuple,
            spatial_dims=2,
        )
    elif architecture_name_string == "vit":
        model_instance = ViT(
            in_channels=input_channels_integer,
            img_size=image_resolution_tuple,
            patch_size=(16, 16),
            classification=False,
            post_activation="Sigmoid",
        )
    elif architecture_name_string == "pranet":
        model_instance = PraNetStubArchitecture(input_channels_integer, output_classes_integer)
    elif architecture_name_string == "medsam":
        model_instance = MedSAMStubArchitecture(input_channels_integer, output_classes_integer, image_resolution_tuple)
    elif architecture_name_string == "custom":
        model_instance = CustomPolypSegmentationArchitecture(input_channels_integer, output_classes_integer)
    else:
        log_system_message("error", f"Unknown architecture requested: {architecture_name_string}")
        raise ValueError("Invalid architecture.")

    return model_instance


def instantiate_optimizer_algorithm(
    model_instance: nn.Module, hyperparameter_dictionary: Dict[str, Any]
) -> torch.optim.Optimizer:
    learning_rate_float = float(hyperparameter_dictionary.get("learning_rate", 1e-4))
    weight_decay_float = float(hyperparameter_dictionary.get("weight_decay", 1e-5))
    return torch.optim.AdamW(model_instance.parameters(), lr=learning_rate_float, weight_decay=weight_decay_float)


def instantiate_learning_rate_scheduler(
    optimizer_instance: torch.optim.Optimizer, hyperparameter_dictionary: Dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    maximum_epochs_integer = int(hyperparameter_dictionary.get("maximum_epochs", 100))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_instance, T_max=maximum_epochs_integer, eta_min=1e-7)
