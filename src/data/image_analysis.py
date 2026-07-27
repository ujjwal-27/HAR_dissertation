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
    non_rgb_images = []

    # Analyse image properties
    for class_directory in sorted(images_path.iterdir()):

        if not class_directory.is_dir():
            continue

        for image_path in sorted(class_directory.iterdir()):

            if not image_path.is_file():
                continue

            with Image.open(image_path) as image:
                if image.mode != "RGB":
                    non_rgb_images.append((image_path, image.mode))

                formats[image.format] += 1
                colour_modes[image.mode] += 1

    return {
        "formats": formats,
        "colour_modes": colour_modes,
        "non_rgb_images": non_rgb_images,
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

    # --------------------------------------------------
    # Display non-RGB images
    # --------------------------------------------------

    print("\nNon-RGB Images")
    print("-" * 70)

    non_rgb_images = property_summary["non_rgb_images"]

    if non_rgb_images:

        for image_path, mode in non_rgb_images:
            relative_path = image_path.relative_to(images_path)
            print(f"{relative_path} ({mode})")

    else:
        print("✓ All images are RGB.")

    # --------------------------------------------------
    # Display image analysis summary
    # --------------------------------------------------

    print("\n" + "-" * 70)
    print("Image Analysis Summary")
    print("-" * 70)

    format_count = len(property_summary["formats"])

    if format_count == 1:
        image_format = next(iter(property_summary["formats"]))
        print(f"✓ All images are stored in {image_format} format.")
    else:
        print(f"✓ Dataset contains {format_count} image formats.")

    rgb_count = property_summary["colour_modes"].get("RGB", 0)
    non_rgb_count = len(non_rgb_images)

    print(f"✓ {rgb_count:,} images are RGB.")

    if non_rgb_count == 0:
        print("✓ No colour conversion is required.")
    elif non_rgb_count == 1:
        print("✓ 1 grayscale image will be converted to RGB during preprocessing.")
    else:
        print(
            f"✓ {non_rgb_count} non-RGB images will be converted to RGB during preprocessing."
        )

    print(
        f"✓ Image dimensions range from "
        f"{dimension_summary['min_width']}×{dimension_summary['min_height']} px "
        f"to "
        f"{dimension_summary['max_width']}×{dimension_summary['max_height']} px."
    )


def main() -> None:
    """
    Execute the image validation workflow.
    """

    analyse_images(DATASET_PATH)


if __name__ == "__main__":
    main()
