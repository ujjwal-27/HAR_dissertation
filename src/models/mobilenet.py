"""
File: mobilenet.py

Description:
    Implements the pretrained MobileNetV3-Large model for
    feature extraction.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)

from src.config.constants import (
    NUM_CLASSES,
    TRAINING_MODE,
)


def build_mobilenet() -> nn.Module:
    """
    Build a pretrained MobileNetV3-Large model for feature extraction.

    Returns
    -------
    nn.Module
        MobileNetV3-Large model.
    """

    # Load pretrained MobileNetV3-Large
    model = mobilenet_v3_large(
        weights=MobileNet_V3_Large_Weights.DEFAULT,
    )

    # Configure trainable layers
    if TRAINING_MODE == "feature_extraction":

        for parameter in model.parameters():
            parameter.requires_grad = False

    elif TRAINING_MODE == "fine_tuning":

        for parameter in model.parameters():
            parameter.requires_grad = True

    else:
        raise ValueError(f"Unsupported training mode: {TRAINING_MODE}")

    # Replace classifier
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        NUM_CLASSES,
    )

    return model
