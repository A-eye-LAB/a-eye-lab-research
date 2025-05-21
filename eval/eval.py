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
            transforms.Resize(256),
            transforms.CenterCrop(224),
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


def specificity_score(y_true, y_pred, zero_division=0):
    """
    특이도(Specificity) 계산 함수
    TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)
    if len(cm) <= 1:
        return zero_division
    tn, fp = cm[0][0], cm[0][1]
    if tn + fp == 0:
        return zero_division
    return tn / (tn + fp)

class ImageTestEvaluator:
    def __init__(self, model, dataset_path, batch_size=16, image_size=(224, 224), device="cpu"):
        """
        모델과 데이터셋 경로로 Class 초기화
        """
        self.model = model.to(device)
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.device = device
        self.batch_size = batch_size

    def load_data(self):
        """
        DataLoader를 사용하여 데이터 로드
        """
        dataset = CustomImageDataset(self.dataset_path)
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
            "Specificity": specificity_score(y_true, y_pred, zero_division=0),
        }

        return metrics


if __name__ == "__main__":
    #from models.vit import ViT_Large
    import sys
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),".."))
    from train.models import *

    # Test Model
    model = FastViT(num_classes=2, pretrained=False)
    # model = MobileNet_V3_Large(num_classes=2, pretrained=False)
    model_path = "/workspace/outputs/FastViT_20250407_092255/weights/checkpoint_epoch_67.pt"

    model.load_state_dict(torch.load(model_path, map_location="cuda", weights_only=True))
    model.to("cuda")
    model.eval()


    # ###########
    # dataset_path = "/workspace/a-eye-lab-research/dataset/data/kaggle_cataract_nand"
    # evaluator = ImageTestEvaluator(model, dataset_path, image_size=(224, 224), device="cuda")

    # evaluator.load_data()
    # results = evaluator.evaluate()

    # # Display results
    # for metric, value in results.items():
    #     if metric=="Confusion Matrix":
    #         print(f"\n{metric}")
    #         print(value,"\n")
    #     else:
    #         print(f"{metric:<15} : {value:.4f}")


    dataset_path = "/workspace/a-eye-lab-research/dataset/real_data"
    evaluator = ImageTestEvaluator(model, dataset_path, image_size=(224, 224), device="cuda")

    evaluator.load_data()
    results = evaluator.evaluate()

    # Display results
    for metric, value in results.items():
        if metric=="Confusion Matrix":
            print(f"\n{metric}")
            print(value,"\n")
        else:
            print(f"{metric:<15} : {value:.4f}")