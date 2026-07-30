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

import time
from pathlib import Path

from PIL import Image

from src.config.constants import (
    DATASET_NAME,
    RGB_MODE,
    TARGET_IMAGE_SIZE,
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


def process_image(
    image_path: Path,
    output_path: Path,
) -> bool:
    """
    Open an image, convert it to RGB if necessary,
    and save it to the preprocessed dataset.

    Parameters
    ----------
    image_path : Path
        Path to the input image.

    output_path : Path
        Path where the processed image will be saved.

    Returns
    -------
    bool
        True if the image was converted to RGB,
        otherwise False.
    """

    with Image.open(image_path) as image:

        # Flag to indicate if the image was converted to RGB
        converted = False

        # Convert the image to RGB if it's not already in that mode
        if image.mode != RGB_MODE:
            image = image.convert(RGB_MODE)
            converted = True

        # Resize the image to the target size
        image = image.resize(
            TARGET_IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        )

        # Save the processed image to the output path
        image.save(output_path)

    return converted


def preprocess_dataset() -> tuple[int, int]:
    """
    Read every image in the dataset.

    Returns
    -------
    int
        Total number of images successfully processed.
    """

    processed_images = 0
    converted_images = 0

    # Traverse every activity class
    for class_directory in sorted(IMAGES_PATH.iterdir()):

        if not class_directory.is_dir():
            continue

        # Traverse every image
        for image_path in sorted(class_directory.iterdir()):

            if not image_path.is_file():
                continue

            output_path = (
                PREPROCESSED_IMAGES_PATH / class_directory.name / image_path.name
            )

            if process_image(image_path, output_path):
                converted_images += 1

            processed_images += 1

    return processed_images, converted_images


def verify_preprocessed_images() -> tuple[int, int, int]:
    """
    Verify that all preprocessed images have the expected
    colour mode and image size.

    Returns
    -------
    tuple[int, int, int]
        Number of RGB images,
        number of correctly resized images,
        number of incorrect images.
    """

    rgb_images = 0
    correct_size = 0
    incorrect_images = 0

    for class_directory in sorted(PREPROCESSED_IMAGES_PATH.iterdir()):

        if not class_directory.is_dir():
            continue

        for image_path in sorted(class_directory.iterdir()):

            if not image_path.is_file():
                continue

            with Image.open(image_path) as image:

                if image.mode == RGB_MODE:
                    rgb_images += 1

                if image.size == TARGET_IMAGE_SIZE:
                    correct_size += 1
                else:
                    incorrect_images += 1
                    print(
                        f"Verification failed: "
                        f"{image_path.name} "
                        f"(Mode={image.mode}, Size={image.size})"
                    )

    return rgb_images, correct_size, incorrect_images


def main() -> None:
    """
    Execute the preprocessing setup.
    """

    start_time = time.perf_counter()

    directory_count = create_directory_structure()

    processed_images, converted_images = preprocess_dataset()

    rgb_images, correct_size, incorrect_images = verify_preprocessed_images()

    elapsed_time = time.perf_counter() - start_time

    print("=" * 70)
    print(f"{DATASET_NAME} Preprocessing")
    print("=" * 70)

    print(f"Output Directory     : {PREPROCESSED_PATH}")
    print(f"Activity Directories : {directory_count}")
    print(f"Images Processed     : {processed_images:,}")
    print(f"Images Converted     : {converted_images:,}")

    print("\nPreprocessing Summary")
    print("-" * 70)
    print("✓ Preprocessed dataset directory created.")
    print("✓ Activity class directories created.")
    print("✓ Images successfully processed and saved.")

    print("\nVerification")
    print("-" * 70)
    print(f"RGB Images          : {rgb_images:,}")
    print(f"Correct Size        : {correct_size:,}")
    print(f"Incorrect Images    : {incorrect_images:,}")

    if incorrect_images == 0:
        print("✓ Verification passed.")
    else:
        print("✗ Verification failed.")

    print(f"\nProcessing Time      : {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
