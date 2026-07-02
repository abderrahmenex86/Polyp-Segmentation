import argparse
import glob
import os
import shutil
import subprocess

import matplotlib.pyplot as plt

from src.helpers import create_directory, load_json, log_message


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
