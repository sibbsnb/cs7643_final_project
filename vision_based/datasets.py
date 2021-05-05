import torch
from torch.utils.data import Dataset

import numpy as np
import cv2

# Dataset loader class
class SHOPEEDataset(Dataset):
    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]  # get the row for this sample which contains all info
        img = cv2.imread(row.file_path)  # read in image as numpy ndarray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert opencv BGR color channel order to GB

        # Optional: If doing data augmentation
        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        img = img.astype(np.float32)  # convert to float32
        img = img.transpose(2, 0, 1)  # put image channel into first dimension like 3x128x128

        # Finally convert to torch tensors

        # If running test set then just return the image, no labels
        if self.mode == 'test':
            return torch.tensor(img).float()
        else:
            # return both image and label, our label being the label_group number
            return torch.tensor(img).float(), torch.tensor(row.label_group).float()

class ShopeeDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index()
        self.augmentations = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # load in image
        image = cv2.imread(row.file_path)

        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        return image, torch.tensor(row.label_group)



