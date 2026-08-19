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
    get_validation_transform,
)


def create_dataloaders():
    """
    Create training, validation and testing DataLoaders.

    The official Stanford40 training split is divided into
    training and validation subsets using a fixed random seed.

    Data augmentation is applied only to the training subset.
    Validation and test data use deterministic transformations.
    """

    # Create the full training dataset without augmentation.
    base_train_dataset = Stanford40Dataset(
        split="train",
        transform=get_validation_transform(),
    )

    # Create a separate training dataset with the training transform.
    augmented_train_dataset = Stanford40Dataset(
        split="train",
        transform=get_train_transform(),
    )

    # Create the testing dataset.
    test_dataset = Stanford40Dataset(
        split="test",
        transform=get_test_transform(),
    )

    # Determine validation size.
    validation_size = int(len(base_train_dataset) * VALIDATION_SPLIT)

    training_size = len(base_train_dataset) - validation_size

    # Create reproducible indices.
    generator = __import__("torch").Generator().manual_seed(RANDOM_SEED)

    indices = __import__("torch").randperm(
        len(base_train_dataset),
        generator=generator,
    )

    train_indices = indices[:training_size]
    validation_indices = indices[training_size:]

    # Create separate subsets using the same indices.
    train_subset = __import__("torch").utils.data.Subset(
        augmented_train_dataset,
        train_indices,
    )

    validation_subset = __import__("torch").utils.data.Subset(
        base_train_dataset,
        validation_indices,
    )

    # Create DataLoaders.
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
