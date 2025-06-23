from torch.utils.data import Dataset
from torchvision import transforms
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
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm['mean'], std=self.norm['std']),
        ])
        self.dataset = torchvision.datasets.ImageFolder(
            root=dataset_path,
            transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
