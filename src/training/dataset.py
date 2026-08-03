"""
File: dataset.py

Description:
    Implements a custom PyTorch Dataset for loading the
    Stanford40 dataset using the official train/test split.

Author:
    Ujjwal Shrestha

Dissertation:
    A Comparative Study of Transfer Learning-Based CNN Architectures
    for Image-Based Human Activity Recognition
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.config.paths import (
    IMAGE_SPLITS_PATH,
    PREPROCESSED_IMAGES_PATH,
)


class Stanford40Dataset(Dataset):
    """
    Custom Dataset for the Stanford40 dataset.
    """

    def __init__(
        self,
        split: str,
        transform=None,
    ) -> None:
        """
        Initialise the dataset.

        Parameters
        ----------
        split : str
            Dataset split ("train" or "test").
        """

        self.split = split
        self.transform = transform

        self.class_names = self._load_class_names()

        self.class_to_index = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }

        self.samples = self._load_samples()

    def _load_class_names(self) -> list[str]:
        """
        Retrieve all activity class names.

        Returns
        -------
        list[str]
            Sorted activity class names.
        """

        return sorted(
            folder.name
            for folder in PREPROCESSED_IMAGES_PATH.iterdir()
            if folder.is_dir()
        )

    def _load_samples(self) -> list[tuple[Path, int]]:
        """
        Load image paths and labels from the official split files.

        Returns
        -------
        list[tuple[Path, int]]
            Image path and corresponding class index.
        """

        samples = []

        for class_name in self.class_names:

            split_file = IMAGE_SPLITS_PATH / f"{class_name}_{self.split}.txt"

            with split_file.open("r") as file:

                for line in file:

                    image_name = line.strip()

                    if not image_name:
                        continue

                    image_path = PREPROCESSED_IMAGES_PATH / class_name / image_name

                    samples.append(
                        (
                            image_path,
                            self.class_to_index[class_name],
                        )
                    )

        return samples

    def __len__(self) -> int:
        """
        Return dataset size.
        """

        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Retrieve an image and its corresponding label.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        tuple
            Image and corresponding class label.
        """

        image_path, label = self.samples[index]

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        return image, label
