import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class OpenImagesDataset(Dataset):
    """ Google open images dataset """

    def __init__(self, args, split="train"):
        self.root = args.database_root
        self.split = split
        self.args = args
        self.csv_file = os.path.join(self.root, self.split + ".csv")

        self.csv_data = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        img_path = self.csv_data[index][0]
        bbox = self.csv_data[index][1, ...]

        # reading image:
        img = Image.load(img_path).convert('RGB')

        sample = {"image": img, "bbox": bbox}

        # applying image transforms
        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == "val":
            return self.transform_val(sample)
        elif self.split == "test":
            return self.transform_test(sample)




