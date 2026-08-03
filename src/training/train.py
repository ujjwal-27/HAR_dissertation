"""
File: train.py

Description:
    Implements the training loop for transfer learning models.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

import torch
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device,
):
    """
    Train the model for one epoch.

    Returns
    -------
    tuple[float, float]
        Training loss and accuracy.
    """

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predictions = torch.max(outputs, 1)

        total_predictions += labels.size(0)

        correct_predictions += (predictions == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = (correct_predictions / total_predictions) * 100

    return epoch_loss, epoch_accuracy
