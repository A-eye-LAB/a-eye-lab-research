import os
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

class CustomImageDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        커스텀 데이터셋 클래스 (이미지와 레이블을 로드)
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        """
        디렉토리에서 라벨이 된 이미지 로드
        디렉토리 구조
        dataset
          ㄴ 0      : Nomal
          ㄴ 1      : Cataract
        """
        print(f"Loading images from: {self.dataset_path}")
        classes = {"0": 0, "1": 1}

        for label_dir, label in classes.items():
            class_path = os.path.join(self.dataset_path, label_dir)
            if not os.path.isdir(class_path):
                raise ValueError(f"Directory '{class_path}' does not exist.")

            for file_name in os.listdir(class_path):
                file_path = os.path.join(class_path, file_name)
                try:
                    image = Image.open(file_path).convert("RGB")
                    if self.transform:
                        image = self.transform(image)
                    self.data.append(image)
                    self.labels.append(label)
                except Exception as e:
                    print(f"Failed to process image: {file_path}. Error: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ImageTestEvaluator:
    def __init__(self, model, dataset_path, batch_size=16, image_size=(224, 224), device="cpu"):
        """
        모델과 데이터셋 경로로 Class 초기화
        """
        self.model = model.to(device)
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.device = device
        self.test_data = []
        self.test_labels = []
        self.batch_size = batch_size

        # Image transformation
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def load_data(self):
        """
        DataLoader를 사용하여 데이터 로드
        """
        dataset = CustomImageDataset(self.dataset_path, transform=self.transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        print(f"Loaded {len(dataset)} images.")

    def evaluate(self):
        """
        Test dataset 에서 Evaluation 진행
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, label in tqdm(self.dataloader, desc='Evaluationg', unit='batch'):
                data = data.to(self.device)

                output = self.model(data)
                preds = torch.argmax(output, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(label.cpu().numpy())

        y_true = all_labels
        y_pred = all_preds

        # Calculate metrics
        metrics = {
            "Confusion Matrix": confusion_matrix(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
        }

        return metrics


if __name__ == "__main__":
    from torchvision.models import resnet18

    # Test Model
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)


    ###########
    dataset_path = "../test/Test"
    evaluator = ImageTestEvaluator(model, dataset_path, image_size=(224, 224), device="cpu")

    evaluator.load_data()
    results = evaluator.evaluate()

    # Display results
    for metric, value in results.items():
        print(f"{metric}:")
        print(value)
        print('\n')

    ###########