from src.training.dataset import Stanford40Dataset

train_dataset = Stanford40Dataset("train")
test_dataset = Stanford40Dataset("test")

print(f"Training Images : {len(train_dataset):,}")
print(f"Testing Images  : {len(test_dataset):,}")
