"""
File: dataloader.py

Description:
    Creates PyTorch DataLoaders for the Stanford40 dataset.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

from torch.utils.data import (
    DataLoader,
    random_split,
)

from src.config.constants import (
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    VALIDATION_SPLIT,
)
from src.training.dataset import Stanford40Dataset
from src.training.transforms import (
    get_test_transform,
    get_train_transform,
)


def create_dataloaders():
    """
    Create training, validation and testing DataLoaders.

    Returns
    -------
    tuple
        Training, validation and testing DataLoaders.
    """

    # Create the training dataset
    train_dataset = Stanford40Dataset(
        split="train",
        transform=get_train_transform(),
    )

    # Create the testing dataset
    test_dataset = Stanford40Dataset(
        split="test",
        transform=get_test_transform(),
    )

    # Determine validation size
    validation_size = int(len(train_dataset) * VALIDATION_SPLIT)
    training_size = len(train_dataset) - validation_size

    # Split the official training set
    generator = __import__("torch").Generator().manual_seed(RANDOM_SEED)

    train_subset, validation_subset = random_split(
        train_dataset,
        [training_size, validation_size],
        generator=generator,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        validation_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )
