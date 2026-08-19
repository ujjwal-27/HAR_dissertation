"""
File: transforms.py

Description:
    Defines image transformations for training, validation and testing.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

from torchvision import transforms

from src.config.constants import USE_DATA_AUGMENTATION

# ImageNet normalization values used by the pretrained CNN models.
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def get_train_transform():
    """
    Return transformations for the training dataset.

    Data augmentation is applied only when enabled through
    USE_DATA_AUGMENTATION.
    """

    if USE_DATA_AUGMENTATION:

        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
                transforms.ToTensor(),
                NORMALIZE,
            ]
        )

    return transforms.Compose(
        [
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )


def get_validation_transform():
    """
    Return transformations for the validation dataset.

    No data augmentation is applied to validation data.
    """

    return transforms.Compose(
        [
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )


def get_test_transform():
    """
    Return transformations for the test dataset.

    No data augmentation is applied to test data.
    """

    return transforms.Compose(
        [
            transforms.ToTensor(),
            NORMALIZE,
        ]
    )
