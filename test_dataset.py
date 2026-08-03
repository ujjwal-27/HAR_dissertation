from src.training.dataset import Stanford40Dataset

train_dataset = Stanford40Dataset("train")

image, label = train_dataset[0]

print(type(image))
print(label)
print(image.size)
