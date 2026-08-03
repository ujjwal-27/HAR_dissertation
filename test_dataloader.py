from src.training.dataloader import create_dataloaders

train_loader, validation_loader, test_loader = create_dataloaders()

print(f"Training batches   : {len(train_loader)}")
print(f"Validation batches : {len(validation_loader)}")
print(f"Testing batches    : {len(test_loader)}")

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
