from src.data.dataloader import get_dataloaders


train_loader, val_loader, test_loader = get_dataloaders()

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("First label:", labels[0])