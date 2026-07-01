import os
import json
from datetime import datetime
from typing import Dict, Any
from tqdm import tqdm
import torch


def log_system_message(message_type: str, message_content: str) -> None:
    current_timestamp_string = datetime.now().strftime("%m/%d - %H:%M")
    tqdm.write(f"[{message_type.upper()}] [{current_timestamp_string}] {message_content}")


def create_directory_if_missing(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)


def generate_artifact_directory_path(model_architecture_name: str) -> str:
    current_timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_directory_path = os.path.join("artifacts", f"{current_timestamp_string}_{model_architecture_name}")
    create_directory_if_missing(artifact_directory_path)
    return artifact_directory_path


def save_dictionary_to_json(data_dictionary: Dict[str, Any], file_path: str) -> None:
    with open(file_path, "w") as json_file_handle:
        json.dump(data_dictionary, json_file_handle, indent=4)


def load_dictionary_from_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as json_file_handle:
        return json.load(json_file_handle)


def determine_optimal_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
