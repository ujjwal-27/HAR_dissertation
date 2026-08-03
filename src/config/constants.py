"""
File: constants.py

Description:
    Defines project-wide constant values used throughout the
    dissertation implementation.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

# --------------------------------------------------
# Dataset
# --------------------------------------------------

# Dataset name
DATASET_NAME = "Stanford40"

# Total number of activity classes
NUM_CLASSES = 40

# Supported image file extensions
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
)

# Validation split ratio
VALIDATION_SPLIT = 0.2

# --------------------------------------------------
# Image
# --------------------------------------------------

# Target image size for pretrained CNN models
TARGET_IMAGE_SIZE = (224, 224)

# Image colour mode
RGB_MODE = "RGB"

# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

# Random seed
RANDOM_SEED = 42

# --------------------------------------------------
# Training
# --------------------------------------------------

BATCH_SIZE = 32
NUM_WORKERS = 0
