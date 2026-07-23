"""
File: image_analysis.py

Description:
    Validates all images in the Stanford40 dataset by checking
    whether each image can be opened successfully.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

from pathlib import Path

from PIL import Image

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Stanford40 dataset directory
DATASET_PATH = PROJECT_ROOT / "dataset"


def validate_images(images_path: Path) -> tuple[int, int, list[Path]]:
    """
    Validate all images in the dataset.

    Parameters
    ----------
    images_path : Path
        Path to the images directory.

    Returns
    -------
    tuple[int, int, list[Path]]
        Number of valid images, number of corrupted images,
        and a list containing corrupted image paths.
    """

    valid_images = 0
    corrupted_images = []

    # Check every image in each activity class
    for class_directory in sorted(images_path.iterdir()):

        if not class_directory.is_dir():
            continue

        for image_path in sorted(class_directory.iterdir()):

            if not image_path.is_file():
                continue

            try:
                with Image.open(image_path) as image:
                    image.verify()

                valid_images += 1

            except Exception:
                corrupted_images.append(image_path)

    return valid_images, len(corrupted_images), corrupted_images


def analyse_images(dataset_path: Path) -> None:
    """
    Perform image validation for the Stanford40 dataset.

    Parameters
    ----------
    dataset_path : Path
        Path to the dataset directory.

    Returns
    -------
    None
    """

    # --------------------------------------------------
    # Locate images directory
    # --------------------------------------------------

    images_path = dataset_path / "images"

    # --------------------------------------------------
    # Validate images
    # --------------------------------------------------

    valid_images, corrupted_count, corrupted_images = validate_images(images_path)

    # --------------------------------------------------
    # Display validation results
    # --------------------------------------------------

    print("=" * 70)
    print("Stanford40 Image Validation")
    print("=" * 70)

    print(f"Images Directory   : {images_path}")
    print(f"Valid Images       : {valid_images:,}")
    print(f"Corrupted Images   : {corrupted_count}")

    if corrupted_images:

        print("\nCorrupted Image Files")
        print("-" * 70)

        for image in corrupted_images:
            print(image)

    else:

        print("\nImage Validation")
        print("-" * 70)
        print("✓ All images were opened successfully.")
        print("✓ No corrupted images were found.")


def main() -> None:
    """
    Execute the image validation workflow.
    """

    analyse_images(DATASET_PATH)


if __name__ == "__main__":
    main()
