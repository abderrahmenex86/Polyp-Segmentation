import argparse
import os
import glob
import matplotlib.pyplot as plt
from src.helpers import load_dictionary_from_json, log_system_message


def execute_plot_generation(artifact_directory_path: str) -> None:
    history_filepath_string = os.path.join(artifact_directory_path, "model_history.json")
    if not os.path.exists(history_filepath_string):
        raise FileNotFoundError("Model history artifact missing.")

    history_dictionary = load_dictionary_from_json(history_filepath_string)
    epochs_range_list = range(1, len(history_dictionary["training_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range_list, history_dictionary["training_loss"], label="Training Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range_list, history_dictionary["validation_dice"], label="Validation DSC")
    plt.plot(epochs_range_list, history_dictionary["validation_iou"], label="Validation IoU")
    plt.legend()

    os.makedirs("docs/figs", exist_ok=True)
    plt.savefig(f"docs/figs/performance_{os.path.basename(artifact_directory_path)}.png")
    log_system_message("plot", "Performance figure generated in docs/figs/")


def execute_dataset_verification(dataset_directory_path: str) -> None:
    images_count_integer = len(glob.glob(os.path.join(dataset_directory_path, "images", "*.*")))
    masks_count_integer = len(glob.glob(os.path.join(dataset_directory_path, "masks", "*.*")))
    log_system_message("verify", f"Images: {images_count_integer} | Masks: {masks_count_integer}")


if __name__ == "__main__":
    argument_parser_instance = argparse.ArgumentParser(description="Utilities")
    argument_parser_instance.add_argument("--mode", choices=["download", "verify", "plot"], required=True)
    argument_parser_instance.add_argument("--path", type=str, default="")
    arguments_namespace = argument_parser_instance.parse_args()

    if arguments_namespace.mode == "verify":
        execute_dataset_verification(arguments_namespace.path or "dataset")
    elif arguments_namespace.mode == "plot":
        execute_plot_generation(arguments_namespace.path)
