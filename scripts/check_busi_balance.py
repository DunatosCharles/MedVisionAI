from src.data.busi_dataset import BUSIDataset
from collections import Counter


dataset = BUSIDataset(
    root_dir="datasets/busi/Dataset_BUSI_with_GT"
)


labels = []

for _, label in dataset:
    labels.append(label)


print(Counter(labels))