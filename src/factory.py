import torch
from monai.networks.nets import UNETR, UNet, ViT

from src.helpers import log_message
from src.models import MedSAMWrapper, PraNet


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
    elif architecture == "medsam":
        return MedSAMWrapper(checkpoint_path=config.get("medsam_checkpoint"))
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
