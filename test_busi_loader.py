from src.data.busi_dataset import BUSIDataset


dataset = BUSIDataset(
    "datasets/busi/Dataset_BUSI_with_GT"
)


print("Total images:", len(dataset))

print(
    "First image label:",
    dataset[0][1]
)