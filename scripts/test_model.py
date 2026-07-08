import torch
from src.models.cnn import BreastCancerCNN


model = BreastCancerCNN()

print(model)


dummy_image = torch.randn(1,1,28,28)

output = model(dummy_image)

print("Output shape:", output.shape)