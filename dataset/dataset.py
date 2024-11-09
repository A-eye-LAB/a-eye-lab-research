import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SingleImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transforms = transforms.Compose([transforms.ToTensor()])

        self.image_paths, self.labels = self.get_label_image_paths(self.root_dir)

    def get_label_image_paths(self, root_dir):
        image_paths = []
        labels = []
        for label in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, label)):
                for img_file in os.listdir(os.path.join(root_dir, label)):
                    image_paths.append(os.path.join(root_dir, label, img_file))
                    labels.append(int(label))

        return image_paths, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.get_image(idx)

        return self.transforms(image), torch.tensor(label, dtype=torch.long)

    def get_image(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return image


class MultipleImageDataset(SingleImageDataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.image_paths = []
        self.labels = []

        for dataset_dir in os.listdir(self.root_dir):
            image_path, label = self.get_label_image_paths(os.path.join(self.root_dir, dataset_dir))
            self.image_paths += image_path
            self.labels += label
