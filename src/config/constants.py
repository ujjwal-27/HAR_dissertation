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

# Batch size
BATCH_SIZE = 32

# Number of worker processes
# Use 0 for macOS during development
NUM_WORKERS = 0

# Learning rate
LEARNING_RATE = 0.001

# Number of training epochs
NUM_EPOCHS = 20

# --------------------------------------------------
# Experiment Configuration
# --------------------------------------------------
#
# Available MODEL_NAME values:
#   - "resnet"
#   - "mobilenet"
#
# Available TRAINING_MODE values:
#   - "feature_extraction"
#   - "fine_tuning"
#
# Available USE_DATA_AUGMENTATION values:
#   - False (No data augmentation)
#   - True  (Apply data augmentation)
#
# Experiment Mapping:
#   Experiment 1: ResNet     + Feature Extraction + No Augmentation
#   Experiment 2: ResNet     + Feature Extraction + Augmentation
#   Experiment 3: ResNet     + Fine-Tuning        + No Augmentation
#   Experiment 4: ResNet     + Fine-Tuning        + Augmentation
#   Experiment 5: MobileNet  + Feature Extraction + No Augmentation
#   Experiment 6: MobileNet  + Feature Extraction + Augmentation
#   Experiment 7: MobileNet  + Fine-Tuning        + No Augmentation
#   Experiment 8: MobileNet  + Fine-Tuning        + Augmentation
# --------------------------------------------------

MODEL_NAME = "resnet"

TRAINING_MODE = "feature_extraction"

USE_DATA_AUGMENTATION = False

EXPERIMENT_NAME = f"{MODEL_NAME}_{TRAINING_MODE}"

# --------------------------------------------------
# Output Files
# --------------------------------------------------

MODEL_FILENAME = f"{EXPERIMENT_NAME}.pth"

RESULTS_FILENAME = f"{EXPERIMENT_NAME}.csv"
