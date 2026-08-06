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

import torch
import torch.nn as nn
import torch.optim as optim

from src.config.constants import (
    LEARNING_RATE,
    NUM_EPOCHS,
)
from src.models.resnet import build_resnet
from src.training.dataloader import create_dataloaders
from src.training.evaluate import evaluate
from src.training.train import train_one_epoch


def main():
    """
    Execute Experiment 1.
    """

    # Select computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Create DataLoaders
    train_loader, validation_loader, _ = create_dataloaders()

    # Build model
    model = build_resnet().to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE,
    )

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

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {validation_loss:.4f} | "
            f"Val Acc: {validation_accuracy:.2f}%"
        )


if __name__ == "__main__":
    main()
