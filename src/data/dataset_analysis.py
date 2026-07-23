"""
File: dataset_analysis.py

Description:
    Performs an initial analysis of the Stanford40 dataset by
    validating the dataset structure and identifying the available
    human activity classes.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

from pathlib import Path

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Stanford40 dataset directory
DATASET_PATH = PROJECT_ROOT / "dataset"


def validate_dataset_structure(dataset_path: Path) -> tuple[Path, Path]:
    """
    Validate the Stanford40 dataset structure.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset directory.

    Returns
    -------
    tuple[Path, Path]
        Paths to the images and ImageSplits directories.

    Raises
    ------
    FileNotFoundError
        If a required dataset directory does not exist.
    """

    # Required dataset directories
    images_path = dataset_path / "images"
    splits_path = dataset_path / "ImageSplits"

    # Check dataset structure
    if not images_path.is_dir():
        raise FileNotFoundError(f"Images directory not found:\n{images_path}")

    if not splits_path.is_dir():
        raise FileNotFoundError(f"ImageSplits directory not found:\n{splits_path}")

    return images_path, splits_path


def get_activity_classes(images_path: Path) -> list[str]:
    """
    Retrieve all activity classes from the dataset.

    Parameters
    ----------
    images_path : Path
        Path to the images directory.

    Returns
    -------
    list[str]
        Alphabetically sorted list of activity class names.
    """

    # Retrieve activity class names
    class_names = sorted(
        folder.name for folder in images_path.iterdir() if folder.is_dir()
    )

    return class_names


def count_images_per_class(images_path: Path) -> dict[str, int]:
    """
    Count the number of images in each activity class.

    Parameters
    ----------
    images_path : Path
        Path to the images directory.

    Returns
    -------
    dict[str, int]
        Dictionary containing activity classes and
        their corresponding image counts.
    """

    image_counts = {}

    # Count images for each activity class
    for class_directory in sorted(images_path.iterdir()):

        if not class_directory.is_dir():
            continue

        image_count = sum(1 for file in class_directory.iterdir() if file.is_file())

        image_counts[class_directory.name] = image_count

    return image_counts


def validate_train_test_splits(
    images_path: Path, splits_path: Path
) -> dict[str, dict[str, int]]:
    """
    Validate the official Stanford40 train/test split files.

    Parameters
    ----------
    images_path : Path
        Path to the images directory.

    splits_path : Path
        Path to the ImageSplits directory.

    Returns
    -------
    dict[str, dict[str, int]]
        Dictionary containing the number of training images,
        testing images and total images for each activity.

    Raises
    ------
    FileNotFoundError
        If an image listed in a split file does not exist.
    """

    split_summary = {}

    # Process each activity
    for class_directory in sorted(images_path.iterdir()):

        if not class_directory.is_dir():
            continue

        activity = class_directory.name

        train_file = splits_path / f"{activity}_train.txt"
        test_file = splits_path / f"{activity}_test.txt"

        # Read training image names
        with train_file.open("r") as file:
            train_images = [line.strip() for line in file if line.strip()]

        # Read testing image names
        with test_file.open("r") as file:
            test_images = [line.strip() for line in file if line.strip()]

        # Check that every image exists
        for image_name in train_images + test_images:

            image_path = class_directory / image_name

            if not image_path.is_file():
                raise FileNotFoundError(f"Missing image:\n{image_path}")

        split_summary[activity] = {
            "train": len(train_images),
            "test": len(test_images),
            "total": len(train_images) + len(test_images),
        }

    return split_summary


def analyse_dataset(dataset_path: Path) -> None:
    """
    Perform an initial analysis of the Stanford40 dataset.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset directory.

    Returns
    -------
    None
    """

    # --------------------------------------------------
    # Validate dataset structure
    # --------------------------------------------------

    images_path, splits_path = validate_dataset_structure(dataset_path)

    # --------------------------------------------------
    # Retrieve activity classes
    # --------------------------------------------------

    class_names = get_activity_classes(images_path)

    # --------------------------------------------------
    # Count images per activity
    # --------------------------------------------------

    image_counts = count_images_per_class(images_path)
    total_images = sum(image_counts.values())

    # --------------------------------------------------
    # Validate train/test splits
    # --------------------------------------------------

    split_summary = validate_train_test_splits(
        images_path,
        splits_path,
    )

    # --------------------------------------------------
    # Display dataset summary
    # --------------------------------------------------

    print("=" * 70)
    print("Stanford40 Dataset Analysis")
    print("=" * 70)

    print(f"Dataset Path      : {dataset_path}")
    print(f"Images Directory  : {images_path}")
    print(f"ImageSplits       : {splits_path}")

    print(f"\nTotal Classes     : {len(class_names)}")
    print(f"Total Images      : {total_images:,}")

    print("\nActivity Distribution")
    print("-" * 70)
    print(f"{'Activity':<40}{'Images':>10}")
    print("-" * 70)

    for activity, count in image_counts.items():
        print(f"{activity:<40}{count:>10}")

    print("-" * 70)

    print("\nTrain/Test Split Summary")

    print("-" * 70)
    print(f"{'Activity':<30}" f"{'Train':>8}" f"{'Test':>8}" f"{'Total':>10}")
    print("-" * 70)

    for activity, summary in split_summary.items():
        print(
            f"{activity:<30}"
            f"{summary['train']:>8}"
            f"{summary['test']:>8}"
            f"{summary['total']:>10}"
        )

    print("-" * 70)

    # --------------------------------------------------
    # Verify image counts
    # --------------------------------------------------

    split_total = sum(summary["total"] for summary in split_summary.values())

    print(f"\nImages from split files : {split_total:,}")
    print(f"Images in dataset       : {total_images:,}")

    # --------------------------------------------------
    # Display validation results
    # --------------------------------------------------

    print("\nDataset Validation")
    print("-" * 70)

    if split_total == total_images:
        print("✓ All images referenced in the train/test split were found.")
        print("✓ Split file count matches the dataset.")
        print("✓ Dataset integrity check passed.")
    else:
        print("✗ Split file count does not match the dataset.")
        print("✗ Dataset integrity check failed.")


def main() -> None:
    """
    Execute the dataset analysis workflow.
    """

    analyse_dataset(DATASET_PATH)


if __name__ == "__main__":
    main()
