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
        raise FileNotFoundError(
            f"Images directory not found:\n{images_path}"
        )

    if not splits_path.is_dir():
        raise FileNotFoundError(
            f"ImageSplits directory not found:\n{splits_path}"
        )

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
        folder.name
        for folder in images_path.iterdir()
        if folder.is_dir()
    )

    return class_names


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
    # Display dataset summary
    # --------------------------------------------------

    print("=" * 70)
    print("Stanford40 Dataset Analysis")
    print("=" * 70)

    print(f"Dataset Path      : {dataset_path}")
    print(f"Images Directory  : {images_path}")
    print(f"ImageSplits       : {splits_path}")

    print(f"\nTotal Classes     : {len(class_names)}")

    print("\nActivity Classes")
    print("-" * 70)

    for index, class_name in enumerate(class_names, start=1):
        print(f"{index:>2}. {class_name}")


def main() -> None:
    """
    Execute the dataset analysis workflow.
    """

    analyse_dataset(DATASET_PATH)


if __name__ == "__main__":
    main()
