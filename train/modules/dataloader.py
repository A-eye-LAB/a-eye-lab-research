import os
from typing import List, Tuple
from sklearn.model_selection import StratifiedKFold
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch
from tabulate import tabulate


class CombinedDataset:
    def __init__(self, root_dirs: List[str], n_folds: int = 5, fold_idx: int = 0, random_seed: int = 42):
        """
        Initialize the dataset loader to combine and process multiple directories for 5-fold CV.
        """
        self.root_dirs = root_dirs
        self.n_folds = n_folds
        self.fold_idx = fold_idx
        self.random_seed = random_seed
        self.norm = {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225)
        }
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(**self.norm),
        ])
        self.combined_train_dataset, self.combined_val_dataset = self._process_datasets()

    def _create_dataset(self, root_dir: str) -> Dataset:
        """
        Create a dataset using ImageFolder for the root directory containing class subdirectories.
        """
        print(f"Creating dataset from {root_dir}")
        return datasets.ImageFolder(root=root_dir, transform=self.transform)

    def _process_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Process all datasets from multiple root directories, split into train and validation.
        """
        all_train_datasets = []
        all_val_datasets = []

        for root_dir in self.root_dirs:
            print(f"Processing root directory: {root_dir}")

            dataset = self._create_dataset(root_dir)  # Pass the parent directory
            train_dataset, val_dataset = self._split_dataset(dataset)

            self._print_class_statistics(train_dataset, val_dataset, os.path.basename(root_dir))

            all_train_datasets.append(train_dataset)
            all_val_datasets.append(val_dataset)

        combined_train_dataset = ConcatDataset(all_train_datasets)
        combined_val_dataset = ConcatDataset(all_val_datasets)

        print(f"\n[INFO] Combined train samples: {len(combined_train_dataset)}")
        print(f"[INFO] Combined validation samples: {len(combined_val_dataset)}\n")
        return combined_train_dataset, combined_val_dataset

    def _split_dataset(self, dataset: Dataset) -> Tuple[Subset, Subset]:
        """
        Split a dataset into training and validation using StratifiedKFold.
        """
        targets = dataset.targets
        indices = list(range(len(targets)))

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        for i, (train_idx, val_idx) in enumerate(skf.split(indices, targets)):
            if i == self.fold_idx:
                train_dataset = Subset(dataset, train_idx)
                val_dataset = Subset(dataset, val_idx)
                return train_dataset, val_dataset

        raise ValueError(f"Invalid fold index: {self.fold_idx}")

    @staticmethod
    def _print_class_statistics(train_dataset: Subset, val_dataset: Subset, dataset_name: str):
        """
        Print statistics for training and validation subsets.
        """
        def count_by_class(subset: Subset):
            targets = [subset.dataset.targets[idx] for idx in subset.indices]
            class_counts = {cls: targets.count(cls) for cls in set(targets)}
            return class_counts

        train_counts = count_by_class(train_dataset)
        val_counts = count_by_class(val_dataset)

        print(f"\n[Dataset: {dataset_name}]")
        print(tabulate([
            ["Train"] + [f"Class {cls}: {train_counts.get(cls, 0)}" for cls in sorted(train_counts)],
            ["Val"] + [f"Class {cls}: {val_counts.get(cls, 0)}" for cls in sorted(val_counts)],
        ], headers=["Subset", *sorted(train_counts)], tablefmt="grid"))

    def get_dataloaders(self, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for train and validation datasets.
        """
        train_loader = DataLoader(self.combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(self.combined_val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader


if __name__ == "__main__":
    # main example
    root_dirs = [
        "/home/mia/a-eye-lab-research/dataset/dataset/data/Cataract-Detection-and-Classification",
        "/home/mia/a-eye-lab-research/dataset/dataset/data/Cataract_Detection-using-CNN"
    ]

    # Process all folds and print stats for each
    n_folds = 5  # Total number of folds
    for fold_idx in range(n_folds):
        print(f"\n--- Processing Fold {fold_idx} ---\n")
        dataset = CombinedDataset(root_dirs, fold_idx=fold_idx, n_folds=n_folds)
        train_loader, val_loader = dataset.get_dataloaders(batch_size=32)

        print(f"Fold {fold_idx}: Train samples = {len(train_loader.dataset)}, Val samples = {len(val_loader.dataset)}\n")

        # Print a few batches for debugging
        for batch_idx, (images, labels) in enumerate(train_loader):
            print(f"Train Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}")
            if batch_idx > 1:  # Limit printing to avoid excessive logs
                break

        for batch_idx, (images, labels) in enumerate(val_loader):
            print(f"Val Batch {batch_idx}: Images shape {images.shape}, Labels shape {labels.shape}")
            if batch_idx > 1:
                break

