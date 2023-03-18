import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import cv2

class A2D2SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, csv_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.annotations = pd.read_csv(csv_file)  # csv file in the form of image name, task labels in cosecutive colomns
        self.transform = transform


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        mask_path =os.path.join(self.mask_dir, self.annotations.iloc[index, 1])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)


        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def test():
    pass

if __name__ == "__main__":
    test()