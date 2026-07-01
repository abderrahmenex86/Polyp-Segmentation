import os
import glob
import torch
from monai.transforms import LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, Compose
from monai.data import Dataset, DataLoader
from typing import Dict, Any
from src.helpers import determine_optimal_device, load_dictionary_from_json, log_system_message
from src.factory import instantiate_model_architecture


def execute_smart_inference(inference_directory_path: str, target_artifact_directory_path: str) -> None:
    execution_device = determine_optimal_device()
    hyperparameter_dictionary = load_dictionary_from_json(
        os.path.join(target_artifact_directory_path, "hyperparameters.json")
    )

    model_instance = instantiate_model_architecture(hyperparameter_dictionary)
    model_instance.load_state_dict(
        torch.load(os.path.join(target_artifact_directory_path, "best_model.pth"), map_location=execution_device)
    )
    model_instance.to(execution_device)
    model_instance.eval()

    inference_image_filepaths_list = sorted(glob.glob(os.path.join(inference_directory_path, "*.*")))
    inference_dataset_dictionary_list = [{"image_data": image_path} for image_path in inference_image_filepaths_list]

    target_resolution_tuple = (
        hyperparameter_dictionary.get("image_height", 256),
        hyperparameter_dictionary.get("image_width", 256),
    )

    inference_transformations = Compose(
        [
            LoadImaged(keys=["image_data"]),
            EnsureChannelFirstd(keys=["image_data"]),
            ScaleIntensityd(keys=["image_data"]),
            Resized(keys=["image_data"], spatial_size=target_resolution_tuple),
        ]
    )

    inference_dataset_instance = Dataset(data=inference_dataset_dictionary_list, transform=inference_transformations)
    inference_dataloader_instance = DataLoader(inference_dataset_instance, batch_size=1, shuffle=False)

    output_predictions_directory = os.path.join(target_artifact_directory_path, "predictions")
    os.makedirs(output_predictions_directory, exist_ok=True)

    log_system_message("infer", f"Starting inference on {len(inference_dataset_dictionary_list)} samples.")

    with torch.no_grad():
        for batch_index_integer, batch_dictionary in enumerate(inference_dataloader_instance):
            input_image_tensor = batch_dictionary["image_data"].to(execution_device)
            prediction_logits_tensor = model_instance(input_image_tensor)
            prediction_probabilities_tensor = torch.sigmoid(prediction_logits_tensor)
            prediction_binary_mask_tensor = (prediction_probabilities_tensor > 0.5).float()

            original_filename_string = os.path.basename(inference_image_filepaths_list[batch_index_integer])
            output_filepath_string = os.path.join(output_predictions_directory, original_filename_string)
            torch.save(
                prediction_binary_mask_tensor.cpu(),
                output_filepath_string.replace(".png", ".pt").replace(".jpg", ".pt"),
            )

    log_system_message("infer", f"Inference complete. Artifacts saved to {output_predictions_directory}")
