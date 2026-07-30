import os
from PIL import Image

from torch.utils.data import Dataset


class BUSIDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.labels = []


        classes = {
            "benign": 0,
            "malignant": 1
        }


        for class_name, label in classes.items():

            class_dir = os.path.join(
                root_dir,
                class_name
            )


            for filename in os.listdir(class_dir):

                # Ignore segmentation masks
                if "_mask" in filename:
                    continue


                if filename.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):

                    self.images.append(
                        os.path.join(
                            class_dir,
                            filename
                        )
                    )

                    self.labels.append(label)



    def __len__(self):

        return len(self.images)



    def __getitem__(self, index):

        image_path = self.images[index]

        label = self.labels[index]


        image = Image.open(
            image_path
        ).convert("L")


        if self.transform:
            image = self.transform(image)


        return image, label