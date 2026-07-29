"""
File: preprocessing.py

Description:
    Creates the directory structure required for the
    preprocessed Stanford40 dataset.

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

# Original dataset directory
DATASET_PATH = PROJECT_ROOT / "dataset"

# Original images directory
IMAGES_PATH = DATASET_PATH / "images"

# Preprocessed dataset directory
PREPROCESSED_PATH = DATASET_PATH / "preprocessed"

# Preprocessed images directory
PREPROCESSED_IMAGES_PATH = PREPROCESSED_PATH / "images"


def create_directory_structure() -> int:
    """
    Create the directory structure for the preprocessed dataset.

    Returns
    -------
    int
        Number of activity class directories created.
    """

    # Create the preprocessed images directory
    PREPROCESSED_IMAGES_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    directory_count = 0

    # Create one directory for each activity class
    for class_directory in sorted(IMAGES_PATH.iterdir()):

        if not class_directory.is_dir():
            continue

        output_directory = PREPROCESSED_IMAGES_PATH / class_directory.name

        output_directory.mkdir(exist_ok=True)

        directory_count += 1

    return directory_count


def main() -> None:
    """
    Execute the preprocessing setup.
    """

    directory_count = create_directory_structure()

    print("=" * 70)
    print("Stanford40 Preprocessing")
    print("=" * 70)

    print(f"Output Directory     : {PREPROCESSED_PATH}")
    print(f"Activity Directories : {directory_count}")

    print("\nPreprocessing Setup")
    print("-" * 70)
    print("✓ Preprocessed dataset directory created.")
    print("✓ Activity class directories created.")


if __name__ == "__main__":
    main()
