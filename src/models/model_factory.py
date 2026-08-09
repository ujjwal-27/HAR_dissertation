"""
File: model_factory.py

Description:
    Creates the requested CNN architecture for the selected experiment.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

from src.config.constants import MODEL_NAME
from src.models.mobilenet import build_mobilenet
from src.models.resnet import build_resnet


def build_model():
    """
    Build the selected CNN model.

    Returns
    -------
    torch.nn.Module
        Configured CNN model.

    Raises
    ------
    ValueError
        If an unsupported model is requested.
    """

    if MODEL_NAME == "resnet":
        return build_resnet()

    if MODEL_NAME == "mobilenet":
        return build_mobilenet()

    raise ValueError(f"Unsupported model: {MODEL_NAME}")
