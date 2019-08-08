import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path


class ImageDataset(Dataset):
    def __init__(self, online, offline, transforms_=None, unaligned=False, mode='train', online_test=None, offline_test=None):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        if mode!="train":
            if online_test is None:
                online = online.replace("train", mode)
            if offline_test is None:
                offline = offline.replace("train", mode)
                
        self.files_A = list(Path(online).rglob("*.tif"))
        self.files_B = list(Path(offline).rglob("*.png"))

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))

        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

if __name__ == "__main__":
    a = r"/media/data/GitHub/handwriting_data/train_online_cropped"
    b = r"/media/data/GitHub/handwriting_data/train_offline_preprocessed"
    
    dataloader = ImageDataset(a, b, transforms_=None, unaligned=True)
