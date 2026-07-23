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

from collections import Counter
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


def analyse_image_dimensions(images_path: Path) -> dict[str, float]:
    """
    Analyse image dimensions in the dataset.

    Parameters
    ----------
    images_path : Path
        Path to the images directory.

    Returns
    -------
    dict[str, float]
        Summary statistics for image width and height.
    """

    widths = []
    heights = []

    # Read the dimensions of every image
    for class_directory in sorted(images_path.iterdir()):

        if not class_directory.is_dir():
            continue

        for image_path in sorted(class_directory.iterdir()):

            if not image_path.is_file():
                continue

            with Image.open(image_path) as image:
                width, height = image.size

            widths.append(width)
            heights.append(height)

    return {
        "min_width": min(widths),
        "max_width": max(widths),
        "avg_width": sum(widths) / len(widths),
        "min_height": min(heights),
        "max_height": max(heights),
        "avg_height": sum(heights) / len(heights),
    }


def analyse_image_properties(images_path: Path) -> dict[str, Counter]:
    """
    Analyse image formats and colour modes.

    Parameters
    ----------
    images_path : Path
        Path to the images directory.

    Returns
    -------
    dict[str, Counter]
        Frequency of image formats and colour modes.
    """

    formats = Counter()
    colour_modes = Counter()

    # Analyse image properties
    for class_directory in sorted(images_path.iterdir()):

        if not class_directory.is_dir():
            continue

        for image_path in sorted(class_directory.iterdir()):

            if not image_path.is_file():
                continue

            with Image.open(image_path) as image:
                formats[image.format] += 1
                colour_modes[image.mode] += 1

    return {
        "formats": formats,
        "colour_modes": colour_modes,
    }


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
    # Analyse image dimensions
    # --------------------------------------------------

    dimension_summary = analyse_image_dimensions(images_path)

    # --------------------------------------------------
    # Analyse image properties
    # --------------------------------------------------

    property_summary = analyse_image_properties(images_path)

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

    # --------------------------------------------------
    # Display image dimensions
    # --------------------------------------------------

    print("\nImage Dimensions")
    print("-" * 70)

    print(f"Minimum Width     : {dimension_summary['min_width']} px")
    print(f"Maximum Width     : {dimension_summary['max_width']} px")
    print(f"Average Width     : {dimension_summary['avg_width']:.2f} px")

    print()

    print(f"Minimum Height    : {dimension_summary['min_height']} px")
    print(f"Maximum Height    : {dimension_summary['max_height']} px")
    print(f"Average Height    : {dimension_summary['avg_height']:.2f} px")

    # --------------------------------------------------
    # Display image formats
    # --------------------------------------------------

    print("\nImage Formats")
    print("-" * 70)

    for image_format, count in sorted(property_summary["formats"].items()):
        print(f"{image_format:<15} {count:,}")

    # --------------------------------------------------
    # Display colour modes
    # --------------------------------------------------

    print("\nColour Modes")
    print("-" * 70)

    for mode, count in sorted(property_summary["colour_modes"].items()):
        print(f"{mode:<15} {count:,}")


def main() -> None:
    """
    Execute the image validation workflow.
    """

    analyse_images(DATASET_PATH)


if __name__ == "__main__":
    main()
