"""
File: compare_experiments.py

Description:
    Combines test-set performance and computational-efficiency
    results for all CNN experiments.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

import csv

from src.config.paths import RESULTS_PATH


def format_decimal(value):
    """
    Format a numeric value to two decimal places.
    """

    return f"{float(value):.2f}"


def main():
    """
    Create the final comparison table for all experiments.
    """

    evaluation_path = RESULTS_PATH / "test_set_evaluation.csv"

    efficiency_path = RESULTS_PATH / "computational_efficiency.csv"

    output_path = RESULTS_PATH / "final_experiment_comparison.csv"

    # --------------------------------------------------
    # Read test-set evaluation results
    # --------------------------------------------------

    evaluation_results = {}

    with open(
        evaluation_path,
        "r",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            evaluation_results[row["Experiment"]] = row

    # --------------------------------------------------
    # Read computational-efficiency results
    # --------------------------------------------------

    efficiency_results = {}

    with open(
        efficiency_path,
        "r",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            efficiency_results[row["Experiment"]] = row

    # --------------------------------------------------
    # Combine results
    # --------------------------------------------------

    combined_results = []

    for experiment_name in evaluation_results:

        evaluation = evaluation_results[experiment_name]

        efficiency = efficiency_results[experiment_name]

        if experiment_name.startswith("resnet"):
            architecture = "ResNet50"
        else:
            architecture = "MobileNetV3-Large"

        if "fine_tuning" in experiment_name:
            training_strategy = "Fine-Tuning"
        else:
            training_strategy = "Feature Extraction"

        if "augmentation" in experiment_name:
            augmentation = "Yes"
        else:
            augmentation = "No"

        combined_results.append(
            [
                experiment_name,
                architecture,
                training_strategy,
                augmentation,
                # Performance metrics
                format_decimal(evaluation["Accuracy"]),
                format_decimal(evaluation["Precision"]),
                format_decimal(evaluation["Recall"]),
                format_decimal(evaluation["F1 Score"]),
                # Computational efficiency
                f"{int(float(efficiency['Total Parameters'])):,}",
                f"{int(float(efficiency['Trainable Parameters'])):,}",
                format_decimal(efficiency["Model Size (MB)"]),
                format_decimal(efficiency["Inference Time per Image (ms)"]),
                format_decimal(efficiency["Training Time (seconds)"]),
                format_decimal(efficiency["Training Time (minutes)"]),
            ]
        )

    # --------------------------------------------------
    # Write final comparison
    # --------------------------------------------------

    with open(
        output_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Experiment",
                "Architecture",
                "Training Strategy",
                "Data Augmentation",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "Total Parameters",
                "Trainable Parameters",
                "Model Size (MB)",
                "Inference Time per Image (ms)",
                "Training Time (seconds)",
                "Training Time (minutes)",
            ]
        )

        writer.writerows(combined_results)

    print(f"Final comparison saved to: " f"{output_path}")


if __name__ == "__main__":
    main()
