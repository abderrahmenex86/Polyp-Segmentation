import torch
import torch.nn as nn
from monai.networks.nets import UNet, UNETR, ViT
from monai.losses import DiceCELoss


class CustomPolypSegmentationArchitecture(nn.Module):
    def __init__(self, input_channels_count: int, output_classes_count: int):
        super().__init__()
        self.initial_convolutional_layer = nn.Conv2d(input_channels_count, 64, kernel_size=3, padding=1)
        self.activation_function_layer = nn.ReLU(inplace=True)
        self.final_convolutional_layer = nn.Conv2d(64, output_classes_count, kernel_size=1)

    def forward(self, input_image_tensor: torch.Tensor) -> torch.Tensor:
        intermediate_feature_tensor = self.activation_function_layer(
            self.initial_convolutional_layer(input_image_tensor)
        )
        return self.final_convolutional_layer(intermediate_feature_tensor)


class PraNetStubArchitecture(nn.Module):
    def __init__(self, input_channels_count: int, output_classes_count: int):
        super().__init__()
        self.fallback_network = UNet(
            spatial_dims=2,
            in_channels=input_channels_count,
            out_channels=output_classes_count,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )

    def forward(self, input_image_tensor: torch.Tensor) -> torch.Tensor:
        return self.fallback_network(input_image_tensor)


class MedSAMStubArchitecture(nn.Module):
    def __init__(self, input_channels_count: int, output_classes_count: int, image_size_tuple: tuple):
        super().__init__()
        self.fallback_network = ViT(
            in_channels=input_channels_count,
            img_size=image_size_tuple,
            patch_size=(16, 16),
            classification=True,
            num_classes=output_classes_count,
        )

    def forward(self, input_image_tensor: torch.Tensor) -> torch.Tensor:
        return self.fallback_network(input_image_tensor)[0]


def build_combined_loss_function() -> nn.Module:
    return DiceCELoss(
        include_background=False, sigmoid=True, jaccard=False, reduction="mean", weight=torch.tensor([1.0])
    )
