import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

class MultiFolderDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.norm = {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225)
        }

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**self.norm),
        ])

        self.samples = []  # (img_path, label) 튜플 목록
        self.class_to_idx = {'0': 0, '1': 1}

        for root in root_dirs:
            for class_name in ['0', '1']:
                class_dir = os.path.join(root, class_name)
                if not os.path.isdir(class_dir):
                    continue
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        img_path = os.path.join(class_dir, fname)
                        label = self.class_to_idx[class_name]
                        self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    from torch.utils.data import random_split

    root_dirs = [
        "/workspace/a-eye-lab-research/data2/C002",
        "/workspace/a-eye-lab-research/data2/C003"
    ]
    
    dataset = MultiFolderDataset(root_dirs)



    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
