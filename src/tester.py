import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from tqdm import tqdm
from monai.metrics import DiceMetric, MeanIoU
from src.helpers import determine_optimal_device, log_system_message


def execute_evaluation_lifecycle(model_instance: nn.Module, evaluation_dataloader: DataLoader) -> Tuple[float, float]:
    execution_device = determine_optimal_device()
    model_instance.eval()
    model_instance.to(execution_device)

    dice_metric_calculator = DiceMetric(include_background=False, reduction="mean")
    iou_metric_calculator = MeanIoU(include_background=False, reduction="mean")

    with torch.no_grad():
        for batch_dictionary in tqdm(evaluation_dataloader, desc="Evaluating", leave=False):
            input_image_tensor = batch_dictionary["image_data"].to(execution_device)
            ground_truth_mask_tensor = batch_dictionary["segmentation_mask_data"].to(execution_device)

            prediction_logits_tensor = model_instance(input_image_tensor)
            prediction_probabilities_tensor = torch.sigmoid(prediction_logits_tensor)
            prediction_binary_mask_tensor = (prediction_probabilities_tensor > 0.5).float()

            dice_metric_calculator(y_pred=prediction_binary_mask_tensor, y=ground_truth_mask_tensor)
            iou_metric_calculator(y_pred=prediction_binary_mask_tensor, y=ground_truth_mask_tensor)

    aggregated_dice_score_float = dice_metric_calculator.aggregate().item()
    aggregated_iou_score_float = iou_metric_calculator.aggregate().item()

    dice_metric_calculator.reset()
    iou_metric_calculator.reset()

    log_system_message(
        "eval", f"Evaluation Completed - DSC: {aggregated_dice_score_float:.4f} | IoU: {aggregated_iou_score_float:.4f}"
    )
    return aggregated_dice_score_float, aggregated_iou_score_float
