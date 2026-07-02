import gc

import torch
import torchinfo

from src.factory import build_model, build_optimizer
from src.helpers import log_message
from src.models import build_loss


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
            loss_fn = build_loss().to(device)

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
