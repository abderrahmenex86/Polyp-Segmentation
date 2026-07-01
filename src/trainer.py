import os
import torch
from typing import Dict, Any
from tqdm import tqdm
from src.helpers import (
    determine_optimal_device,
    generate_artifact_directory_path,
    save_dictionary_to_json,
    log_system_message,
)
from src.dataset import construct_dataloaders
from src.factory import (
    instantiate_model_architecture,
    instantiate_optimizer_algorithm,
    instantiate_learning_rate_scheduler,
)
from src.models import build_combined_loss_function
from src.tester import execute_evaluation_lifecycle


def execute_training_lifecycle(hyperparameter_dictionary: Dict[str, Any]) -> None:
    execution_device = determine_optimal_device()
    artifact_directory_path = generate_artifact_directory_path(
        hyperparameter_dictionary.get("architecture_name", "UNet")
    )

    save_dictionary_to_json(hyperparameter_dictionary, os.path.join(artifact_directory_path, "hyperparameters.json"))

    training_dataloader, validation_dataloader, _ = construct_dataloaders(hyperparameter_dictionary)

    model_instance = instantiate_model_architecture(hyperparameter_dictionary).to(execution_device)
    optimizer_instance = instantiate_optimizer_algorithm(model_instance, hyperparameter_dictionary)
    scheduler_instance = instantiate_learning_rate_scheduler(optimizer_instance, hyperparameter_dictionary)
    loss_criterion_function = build_combined_loss_function()

    with open(os.path.join(artifact_directory_path, "architecture.txt"), "w") as architecture_file_handle:
        architecture_file_handle.write(str(model_instance))

    maximum_epochs_integer = hyperparameter_dictionary.get("maximum_epochs", 100)
    early_stopping_patience_integer = hyperparameter_dictionary.get("early_stopping_patience", 15)
    gradient_clipping_threshold_float = hyperparameter_dictionary.get("gradient_clipping_threshold", 1.0)

    best_validation_dice_score_float = 0.0
    epochs_without_improvement_integer = 0
    training_history_dictionary = {"training_loss": [], "validation_dice": [], "validation_iou": []}

    for current_epoch_integer in range(1, maximum_epochs_integer + 1):
        model_instance.train()
        accumulated_training_loss_float = 0.0

        progress_bar_instance = tqdm(
            training_dataloader, desc=f"Epoch {current_epoch_integer}/{maximum_epochs_integer}", leave=False
        )
        for batch_dictionary in progress_bar_instance:
            input_image_tensor = batch_dictionary["image_data"].to(execution_device)
            ground_truth_mask_tensor = batch_dictionary["segmentation_mask_data"].to(execution_device)

            optimizer_instance.zero_grad()
            prediction_logits_tensor = model_instance(input_image_tensor)

            batch_loss_tensor = loss_criterion_function(prediction_logits_tensor, ground_truth_mask_tensor)
            batch_loss_tensor.backward()

            torch.nn.utils.clip_grad_norm_(model_instance.parameters(), gradient_clipping_threshold_float)
            optimizer_instance.step()

            accumulated_training_loss_float += batch_loss_tensor.item()
            progress_bar_instance.set_postfix({"loss": f"{batch_loss_tensor.item():.4f}"})

        scheduler_instance.step()
        average_epoch_loss_float = accumulated_training_loss_float / len(training_dataloader)

        validation_dice_score_float, validation_iou_score_float = execute_evaluation_lifecycle(
            model_instance, validation_dataloader
        )

        training_history_dictionary["training_loss"].append(average_epoch_loss_float)
        training_history_dictionary["validation_dice"].append(validation_dice_score_float)
        training_history_dictionary["validation_iou"].append(validation_iou_score_float)
        save_dictionary_to_json(
            training_history_dictionary, os.path.join(artifact_directory_path, "model_history.json")
        )

        log_system_message(
            "train",
            f"Epoch {current_epoch_integer} | Loss: {average_epoch_loss_float:.4f} | Val DSC: {validation_dice_score_float:.4f}",
        )

        if validation_dice_score_float > best_validation_dice_score_float:
            best_validation_dice_score_float = validation_dice_score_float
            epochs_without_improvement_integer = 0
            torch.save(model_instance.state_dict(), os.path.join(artifact_directory_path, "best_model.pth"))
            log_system_message("save", "New best model serialized to disk.")
        else:
            epochs_without_improvement_integer += 1

        if epochs_without_improvement_integer >= early_stopping_patience_integer:
            log_system_message("stop", f"Early stopping triggered at epoch {current_epoch_integer}.")
            break
