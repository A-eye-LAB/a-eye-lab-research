# Torch Imports
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset

# ETC Imports
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from PIL import Image
from typing import Tuple, List

class TransformableSubset(Subset):
    def __init__(self, dataset: Dataset, indices: List[int], transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if self.transform:
            image, label = item
            image = self.transform(image)
            return image, label
        return item
    
class CombinedDataset:
    def __init__(self, root_dirs: List[str], n_folds: int = 5, fold_idx: int = 0, random_seed: int = 321):
        self.root_dirs = root_dirs
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.random_seed = random_seed
        self.norm = {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225)
        }
        
        # ✅ RandAugment 추가
        self.transform_train = transforms.Compose([
            transforms.Resize(256),  # 더 작은 크기로 리사이즈
            transforms.RandomCrop(224),  # 224로 크롭
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(**self.norm),
        ])

        
        self.transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**self.norm),
        ])

        self.combined_train_dataset, self.combined_val_dataset = self.process_datasets()

    def create_dataset(self, root_dir: str) -> Dataset:
        return datasets.ImageFolder(root=root_dir, allow_empty=True)

    def process_datasets(self) -> Tuple[Dataset, Dataset]:
        all_train_datasets = []
        all_val_datasets = []

        for root_dir in self.root_dirs:
            print(f"Dataset from {root_dir} created")
            dataset = self.create_dataset(root_dir)
            root_dir_name = root_dir.split('/')[-2]

            print(f"Processing dataset from {root_dir_name}")
            train_dataset, val_dataset = self.split_dataset(dataset, 
                                                            fold_idx=self.fold_idx, 
                                                            n_folds=self.n_folds,
                                                            random_seed=self.random_seed)
            
            # Apply transform to subsets
            train_dataset = TransformableSubset(train_dataset.dataset, train_dataset.indices, transform=self.transform_train)
            val_dataset = TransformableSubset(val_dataset.dataset, val_dataset.indices, transform=self.transform_val)

            self.print_class_statistics(train_dataset, val_dataset)
            
            all_train_datasets.append(train_dataset)
            all_val_datasets.append(val_dataset)

        combined_train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
        combined_val_dataset = torch.utils.data.ConcatDataset(all_val_datasets)
        
        return combined_train_dataset, combined_val_dataset

    @staticmethod
    def split_dataset(dataset: Dataset, 
                       fold_idx: int, 
                       n_folds: int, 
                       random_seed: int) -> Tuple[Dataset, Dataset]:
        targets = dataset.targets
        indices = list(range(len(targets)))
        
        if n_folds == 1:
            train_idx, val_idx = train_test_split(
                                    np.arange(len(targets)),
                                    test_size=0.2,
                                    random_state=random_seed,
                                    shuffle=True,
                                    stratify=targets
                                )
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, val_idx)
            return train_dataset, val_dataset
        
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        
            for i, (train_idx, val_idx) in enumerate(skf.split(indices, targets)):
                if i == fold_idx:
                    train_dataset = Subset(dataset, train_idx)
                    val_dataset = Subset(dataset, val_idx)
                    return train_dataset, val_dataset
        
        raise ValueError(f"Fold index {fold_idx} is out of range")

    @staticmethod
    def print_class_statistics(train_dataset: Subset, val_dataset: Subset):
        from tabulate import tabulate

        def count_and_ratio(subset: Subset, class_idx: int) -> Tuple[int, float]:
            count = sum(1 for idx in subset.indices if subset.dataset.targets[idx] == class_idx)
            ratio = count / len(subset) if len(subset) > 0 else 0
            return count, ratio

        class_indices = sorted(set(train_dataset.dataset.targets))
        table_data = []
        headers = ["Class", "Train Count", "Train Ratio", "Val Count", "Val Ratio"]
        
        for class_idx in class_indices:
            train_class_count, train_class_ratio = count_and_ratio(train_dataset, class_idx)
            val_class_count, val_class_ratio = count_and_ratio(val_dataset, class_idx)
            
            table_data.append([
                f"Class {class_idx}",
                train_class_count,
                f"{train_class_ratio:.2%}",
                val_class_count,
                f"{val_class_ratio:.2%}"
            ])
        
        print("\nDataset Statistics:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()

    def __iter__(self):
        yield self.combined_train_dataset
        yield self.combined_val_dataset

import torchvision
class CustomImageDataset(Dataset):
    def __init__(self, dataset_path):
        """
        커스텀 데이터셋 클래스 (이미지와 레이블을 로드)
        """

        self.norm = {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225)
        }

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),  # 더 큰 크기로 리사이즈
            transforms.CenterCrop(224),  # 최종 크기 384로 크롭
            transforms.ToTensor(),
            transforms.Normalize(**self.norm),
        ])
        self.dataset = torchvision.datasets.ImageFolder(
            root=dataset_path,
            transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

if __name__ == '__main__':
    import utils
    cfg = utils.load_yaml('/workspace/a-eye-lab-research/src/configs/train.yaml')
    root_dirs = cfg['DATASET']['TRAIN_DATA_DIR']

    combined_train_dataset, combined_val_dataset = CombinedDataset(root_dirs)
    
    print(f"Total number of samples in combined train dataset: {len(combined_train_dataset)}")
    print(f"Total number of samples in combined validation dataset: {len(combined_val_dataset)}")
