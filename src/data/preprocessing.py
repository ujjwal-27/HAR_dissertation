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

from PIL import Image

from src.config.constants import (
    DATASET_NAME,
)
from src.config.paths import (
    IMAGES_PATH,
    PREPROCESSED_IMAGES_PATH,
    PREPROCESSED_PATH,
)


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


def preprocess_dataset() -> int:
    """
    Read every image in the dataset.

    Returns
    -------
    int
        Total number of images successfully processed.
    """

    processed_images = 0

    # Traverse every activity class
    for class_directory in sorted(IMAGES_PATH.iterdir()):

        if not class_directory.is_dir():
            continue

        # Traverse every image
        for image_path in sorted(class_directory.iterdir()):

            if not image_path.is_file():
                continue

            # Open the image
            with Image.open(image_path):
                processed_images += 1

    return processed_images


def main() -> None:
    """
    Execute the preprocessing setup.
    """

    directory_count = create_directory_structure()

    processed_images = preprocess_dataset()

    print("=" * 70)
    print(f"{DATASET_NAME} Preprocessing")
    print("=" * 70)

    print(f"Output Directory     : {PREPROCESSED_PATH}")
    print(f"Activity Directories : {directory_count}")
    print(f"Images Processed     : {processed_images:,}")

    print("\nPreprocessing Setup")
    print("-" * 70)
    print("✓ Preprocessed dataset directory created.")
    print("✓ Activity class directories created.")


if __name__ == "__main__":
    main()
