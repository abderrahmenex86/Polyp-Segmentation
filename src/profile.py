import torch
import gc
from src.factory import build_model, build_optimizer
from src.models import build_loss
from src.helpers import log_message


def profile_hardware_limits(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        log_message("profile", "Profiling is designed for CUDA memory limits. Skipping.")
        return

    log_message("profile", "Initiating batch size limits profiling...")

    current_batch_size = 2
    max_safe_batch_size = 2
    image_h = config.get("image_height")
    image_w = config.get("image_width")
    in_channels = config.get("in_channels")

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
                logits = model(dummy_images, dummy_masks) if hasattr(model, "sam") else model(dummy_images)
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
