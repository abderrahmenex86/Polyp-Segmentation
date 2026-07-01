import argparse
from typing import Dict, Any
from src.trainer import execute_training_lifecycle
from src.tester import execute_evaluation_lifecycle
from src.infer import execute_smart_inference
from src.dataset import construct_dataloaders
from src.factory import instantiate_model_architecture
from src.helpers import load_dictionary_from_json
import torch
import os


def parse_dynamic_arguments() -> Dict[str, Any]:
    argument_parser_instance = argparse.ArgumentParser(description="Ex86 Pure-PyTorch Endoscopy Polyp Segmentation")
    argument_parser_instance.add_argument("--train", action="store_true")
    argument_parser_instance.add_argument("--test", action="store_true")
    argument_parser_instance.add_argument("--infer", action="store_true")

    known_arguments_namespace, unknown_arguments_list = argument_parser_instance.parse_known_args()

    hyperparameter_dictionary = vars(known_arguments_namespace)

    for index_integer in range(0, len(unknown_arguments_list), 2):
        key_string = unknown_arguments_list[index_integer].lstrip("--")
        value_string = unknown_arguments_list[index_integer + 1]
        if value_string.isdigit():
            hyperparameter_dictionary[key_string] = int(value_string)
        elif value_string.replace(".", "", 1).isdigit():
            hyperparameter_dictionary[key_string] = float(value_string)
        elif value_string.lower() in ["true", "false"]:
            hyperparameter_dictionary[key_string] = value_string.lower() == "true"
        else:
            hyperparameter_dictionary[key_string] = value_string

    return hyperparameter_dictionary


def execute_main_application_entrypoint() -> None:
    hyperparameter_dictionary = parse_dynamic_arguments()

    if hyperparameter_dictionary.get("train", False):
        execute_training_lifecycle(hyperparameter_dictionary)

    if hyperparameter_dictionary.get("test", False):
        artifact_target_path = hyperparameter_dictionary.get("artifact_directory", None)
        if artifact_target_path and os.path.exists(os.path.join(artifact_target_path, "hyperparameters.json")):
            hyperparameter_dictionary.update(
                load_dictionary_from_json(os.path.join(artifact_target_path, "hyperparameters.json"))
            )
            _, _, testing_dataloader_instance = construct_dataloaders(hyperparameter_dictionary)
            model_instance = instantiate_model_architecture(hyperparameter_dictionary)
            model_instance.load_state_dict(torch.load(os.path.join(artifact_target_path, "best_model.pth")))
            execute_evaluation_lifecycle(model_instance, testing_dataloader_instance)

    if hyperparameter_dictionary.get("infer", False):
        inference_directory_path = hyperparameter_dictionary.get("inference_directory", "dataset/inference")
        target_artifact_directory_path = hyperparameter_dictionary.get("artifact_directory", None)
        execute_smart_inference(inference_directory_path, target_artifact_directory_path)


if __name__ == "__main__":
    execute_main_application_entrypoint()
