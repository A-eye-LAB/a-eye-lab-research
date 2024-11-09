import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        for label in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, label)):
                for img_file in os.listdir(os.path.join(self.root_dir, label)):
                    self.image_paths.append(os.path.join(self.root_dir, label, img_file))
                    self.labels.append(int(label))

        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open image and apply transforms if provided
        image = Image.open(img_path).convert("RGB")

        return self.transforms(image), torch.tensor(label, dtype=torch.long)
