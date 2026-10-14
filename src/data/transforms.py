from torchvision import transforms


IMAGE_SIZE = 224

# Grayscale normalisation used by both training and inference
MEAN = [0.4829]
STD = [0.229]


def get_train_transform():

    return transforms.Compose([

        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=MEAN,
            std=STD
        )
    ])


def get_eval_transform():

    return transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=MEAN,
            std=STD
        )
    ])
