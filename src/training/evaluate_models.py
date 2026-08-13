"""
File: evaluate_models.py

Description:
    Evaluates trained CNN models on the official Stanford40
    test set using classification performance metrics.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

import csv

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

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


def get_model(model_name: str):
    """
    Build the requested model architecture.
    """

    if model_name == "resnet":
        return build_resnet()

    if model_name == "mobilenet":
        return build_mobilenet()

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(
    model,
    test_loader,
    device,
):
    """
    Evaluate a trained model on the test set.
    """

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            all_predictions.extend(predictions.cpu().numpy())

            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions,
    )


def main():
    """
    Evaluate all eight trained experiments.
    """

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
        print(f"Evaluating: {experiment_name}")

        model = get_model(experiment["model"]).to(device)

        model_path = MODEL_PATH / f"{experiment_name}.pth"

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=device,
            )
        )

        (
            accuracy,
            precision,
            recall,
            f1,
            all_labels,
            all_predictions,
        ) = evaluate_model(
            model,
            test_loader,
            device,
        )

        print(f"Accuracy  : {accuracy * 100:.2f}%")
        print(f"Precision : {precision * 100:.2f}%")
        print(f"Recall    : {recall * 100:.2f}%")
        print(f"F1 Score  : {f1 * 100:.2f}%")

        # Generate confusion matrix
        matrix = confusion_matrix(
            all_labels,
            all_predictions,
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
        )

        display.plot(
            xticks_rotation="vertical",
        )

        plt.title(
            f"Confusion Matrix - {experiment_name}"
        )

        confusion_matrix_path = (
            RESULTS_PATH
            / "confusion_matrices"
            / f"{experiment_name}.png"
        )

        plt.savefig(
            confusion_matrix_path,
            bbox_inches="tight",
        )

        plt.close()

        results.append(
            [
                experiment_name,
                accuracy,
                precision,
                recall,
                f1,
            ]
        )

    results_path = RESULTS_PATH / "test_set_evaluation.csv"

    with open(
        results_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Experiment",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
            ]
        )

        writer.writerows(results)

    print("-" * 70)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
