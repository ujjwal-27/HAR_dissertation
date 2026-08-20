"""
File: run_experiment.py

Description:
    Executes the dissertation experiments by
    configuring the selected CNN architecture
    and training strategy.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

import csv
import time

import torch
import torch.nn as nn
import torch.optim as optim

from src.config.constants import (
    EXPERIMENT_NAME,
    LEARNING_RATE,
    MODEL_FILENAME,
    NUM_EPOCHS,
    RESULTS_FILENAME,
)
from src.config.paths import (
    MODEL_PATH,
    RESULTS_PATH,
)
from src.models import build_model
from src.training.dataloader import create_dataloaders
from src.training.evaluate import evaluate
from src.training.train import train_one_epoch


def synchronise_device(device):
    """
    Synchronise the computation device before measuring elapsed time.

    This is particularly important for hardware accelerators such as
    Apple MPS and CUDA because operations may execute asynchronously.
    """

    if device.type == "mps":
        torch.mps.synchronize()

    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    """
    Execute one configured dissertation experiment.
    """

    # Select computation device
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    elif torch.cuda.is_available():
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("-" * 70)

    # Create DataLoaders
    train_loader, validation_loader, _ = create_dataloaders()

    # Build model
    model = build_model().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        filter(
            lambda parameter: parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
    )

    # Track the best validation accuracy
    best_validation_accuracy = 0.0

    # Create CSV file
    csv_path = RESULTS_PATH / RESULTS_FILENAME

    # Write training-log header
    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Epoch",
                "Train Loss",
                "Train Accuracy",
                "Validation Loss",
                "Validation Accuracy",
            ]
        )

    # Synchronise before starting the training timer
    synchronise_device(device)

    training_start_time = time.perf_counter()

    # Training loop
    for epoch in range(NUM_EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_loss, validation_accuracy = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        with open(
            csv_path,
            "a",
            newline="",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_accuracy,
                    validation_loss,
                    validation_accuracy,
                ]
            )

        if validation_accuracy > best_validation_accuracy:

            best_validation_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH / MODEL_FILENAME,
            )

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.2f}%"
        )

    # Synchronise before stopping the training timer
    synchronise_device(device)

    training_end_time = time.perf_counter()

    training_time_seconds = training_end_time - training_start_time

    training_time_minutes = training_time_seconds / 60

    print("\nTraining completed.")

    print(
        f"Total Training Time : "
        f"{training_time_seconds:.2f} seconds "
        f"({training_time_minutes:.2f} minutes)"
    )

    print(f"Best Validation Accuracy : " f"{best_validation_accuracy:.2f}%")

    print(f"Best model saved to: " f"{MODEL_PATH / MODEL_FILENAME}")

    print(f"Training log saved to: " f"{RESULTS_PATH / RESULTS_FILENAME}")


if __name__ == "__main__":
    main()
