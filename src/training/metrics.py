"""
File: metrics.py

Description:
    Utility functions for calculating evaluation metrics.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(
    true_labels,
    predicted_labels,
):
    """
    Calculate classification metrics.

    Returns
    -------
    dict
        Classification metrics.
    """

    return {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "precision": precision_score(
            true_labels,
            predicted_labels,
            average="weighted",
        ),
        "recall": recall_score(
            true_labels,
            predicted_labels,
            average="weighted",
        ),
        "f1_score": f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
        ),
    }
