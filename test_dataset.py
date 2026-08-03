from src.training.dataset import Stanford40Dataset
from src.training.transforms import (
    get_test_transform,
)

dataset = Stanford40Dataset(
    split="train",
    transform=get_test_transform(),
)

image, label = dataset[0]

print(type(image))
print(image.shape)
print(label)
