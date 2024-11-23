import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, train=True):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # MNIST 데이터 로드
        mnist_dataset = datasets.MNIST(
            root='./dataset/data',
            train=train,
            download=True,
            transform=transform
        )
        
        # 데이터와 레이블을 분리하여 저장
        self.x_data = mnist_dataset.data.float()  # [N, 28, 28]
        self.x_data = self.x_data.unsqueeze(1)    # [N, 1, 28, 28]
        self.y_data = mnist_dataset.targets       # [N]
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return len(self.x_data)

def get_mnist_dataloaders(batch_size=64, num_workers=4):

    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader