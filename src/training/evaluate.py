"""
File: evaluate.py

Description:
    Evaluates a trained model on a validation or testing dataset.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

import torch
import torch.nn as nn


def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device,
):
    """
    Evaluate the model.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.

    dataloader
        Validation or testing DataLoader.

    criterion
        Loss function.

    device
        CPU or GPU.

    Returns
    -------
    tuple[float, float]
        Validation loss and accuracy.
    """

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predictions = torch.max(outputs, 1)

            total_predictions += labels.size(0)

            correct_predictions += (predictions == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = (correct_predictions / total_predictions) * 100

    return epoch_loss, epoch_accuracy
