"""
File: analyse_efficiency.py

Description:
    Analyses the computational efficiency of the eight trained
    CNN experiment configurations.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

import csv
import time

import torch

from src.config.paths import (
    MODEL_PATH,
    RESULTS_PATH,
)
from src.models.mobilenet import build_mobilenet
from src.models.resnet import build_resnet
from src.training.dataloader import create_dataloaders

EXPERIMENTS = [
    {
        "name": "resnet_feature_extraction",
        "model": "resnet",
    },
    {
        "name": "resnet_feature_extraction_augmentation",
        "model": "resnet",
    },
    {
        "name": "resnet_fine_tuning",
        "model": "resnet",
    },
    {
        "name": "resnet_fine_tuning_augmentation",
        "model": "resnet",
    },
    {
        "name": "mobilenet_feature_extraction",
        "model": "mobilenet",
    },
    {
        "name": "mobilenet_feature_extraction_augmentation",
        "model": "mobilenet",
    },
    {
        "name": "mobilenet_fine_tuning",
        "model": "mobilenet",
    },
    {
        "name": "mobilenet_fine_tuning_augmentation",
        "model": "mobilenet",
    },
]


def build_model(model_name):
    """Build the requested CNN architecture."""

    if model_name == "resnet":
        return build_resnet()

    if model_name == "mobilenet":
        return build_mobilenet()

    raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    """Return total and trainable parameter counts."""

    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    return total_parameters, trainable_parameters


def calculate_model_size(model):
    """Calculate model size in megabytes."""

    parameter_size = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )

    buffer_size = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )

    total_size = parameter_size + buffer_size

    return total_size / (1024**2)


def measure_inference_time(
    model,
    test_loader,
    device,
    num_batches=10,
):
    """Measure average inference time per image."""

    model.eval()

    total_time = 0.0
    total_images = 0

    with torch.no_grad():

        for batch_index, (images, _) in enumerate(test_loader):

            if batch_index >= num_batches:
                break

            images = images.to(device)

            if device.type == "mps":
                torch.mps.synchronize()

            start_time = time.perf_counter()

            model(images)

            if device.type == "mps":
                torch.mps.synchronize()

            end_time = time.perf_counter()

            total_time += end_time - start_time
            total_images += images.size(0)

    return total_time / total_images


def main():
    """Analyse all eight trained experiments."""

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    _, _, test_loader = create_dataloaders()

    results = []

    for experiment in EXPERIMENTS:

        experiment_name = experiment["name"]

        print("-" * 70)
        print(f"Analysing: {experiment_name}")

        model = build_model(experiment["model"]).to(device)

        model_path = MODEL_PATH / f"{experiment_name}.pth"

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
            )
        )

        total_parameters, trainable_parameters = count_parameters(model)

        model_size_mb = calculate_model_size(model)

        inference_time = measure_inference_time(
            model,
            test_loader,
            device,
        )

        print(f"Total Parameters    : " f"{total_parameters:,}")

        print(f"Trainable Parameters: " f"{trainable_parameters:,}")

        print(f"Model Size          : " f"{model_size_mb:.2f} MB")

        print(f"Inference Time/Image: " f"{inference_time * 1000:.3f} ms")

        results.append(
            [
                experiment_name,
                total_parameters,
                trainable_parameters,
                model_size_mb,
                inference_time * 1000,
            ]
        )

    results_path = RESULTS_PATH / "computational_efficiency.csv"

    with open(
        results_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Experiment",
                "Total Parameters",
                "Trainable Parameters",
                "Model Size (MB)",
                "Inference Time per Image (ms)",
            ]
        )

        writer.writerows(results)

    print("-" * 70)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
