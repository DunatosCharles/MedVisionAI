from src.data.resnet_dataloader import get_resnet_dataloaders


train_loader, val_loader, test_loader = get_resnet_dataloaders(
    dataset="busi"
)


print("Train batches:", len(train_loader))
print("Val batches:", len(val_loader))
print("Test batches:", len(test_loader))


images, labels = next(iter(train_loader))

print("Image shape:", images.shape)
print("Labels:", labels)