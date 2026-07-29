"""
File: paths.py

Description:
    Defines the project directory structure and commonly used
    filesystem paths.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

from pathlib import Path

# --------------------------------------------------
# Project Root
# --------------------------------------------------

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------
# Dataset Directories
# --------------------------------------------------

# Dataset root directory
DATASET_PATH = PROJECT_ROOT / "dataset"

# Original dataset directories
IMAGES_PATH = DATASET_PATH / "images"
IMAGE_SPLITS_PATH = DATASET_PATH / "ImageSplits"

# Preprocessed dataset directories
PREPROCESSED_PATH = DATASET_PATH / "preprocessed"
PREPROCESSED_IMAGES_PATH = PREPROCESSED_PATH / "images"
