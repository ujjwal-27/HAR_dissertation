"""
File: resnet.py

Description:
    Implements the pretrained ResNet50 model for
    feature extraction.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)

from src.config.constants import NUM_CLASSES


def build_resnet() -> nn.Module:
    """
    Build a pretrained ResNet50 model for feature extraction.

    Returns
    -------
    nn.Module
        ResNet50 model.
    """

    # Load pretrained ResNet50
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT,
    )

    # Freeze all pretrained layers
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace the classifier
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    return model
