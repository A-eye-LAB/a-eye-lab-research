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
    accuracy_score,
    roc_curve,
    auc
)
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt

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

class ONNXImageTestEvaluator:
    def __init__(self, onnx_path, dataset_path, batch_size=16, image_size=(224, 224)):
        self.onnx_path = onnx_path
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        
        # CUDA를 사용할 수 없는 경우를 위해 providers 목록 수정
        providers = ['CPUExecutionProvider']
        try:
            # CUDA 사용 가능성 확인
            ort_session = ort.InferenceSession(
                onnx_path, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        except Exception as e:
            print("CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
            print(f"에러 메시지: {str(e)}")
        
        # 수정된 providers로 세션 초기화
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 입력 이름 가져오기
        self.input_name = self.session.get_inputs()[0].name

        self.norm = {
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225)
        }

        # Image transformation
        self.transform = transforms.Compose([
            #transforms.Resize(image_size),
            transforms.Resize(image_size[0]),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(**self.norm),
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
        Test dataset에서 ONNX 모델 Evaluation 진행하고 ROC 커브 생성
        """
        all_preds = []
        all_labels = []
        all_probs = []  # 확률값 저장을 위한 리스트

        for data, label in tqdm(self.dataloader, desc='Evaluating', unit='batch'):
            input_data = data.numpy()
            outputs = self.session.run(None, {self.input_name: input_data})
            
            # 클래스별 확률값
            probabilities = outputs[0]
            positive_probs = probabilities[:, 1]  # 양성 클래스(1)의 확률
            
            preds = np.argmax(outputs[0], axis=1)
            
            all_probs.extend(positive_probs)
            all_preds.extend(preds)
            all_labels.extend(label.numpy())

        y_true = all_labels
        y_pred = all_preds
        y_prob = all_probs

        # ROC 커브 계산
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        # ROC 커브 그리기
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # ROC 커브 저장
        plt.savefig('roc_curve.png')
        plt.close()

        # Calculate metrics
        metrics = {
            "Confusion Matrix": confusion_matrix(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Specificity": specificity_score(y_true, y_pred, zero_division=0),
            "AUC-ROC": roc_auc
        }

        return metrics

if __name__ == "__main__":
    # ONNX 모델 경로
    onnx_path = "/workspace/model_quantized.onnx"
    dataset_path = "/workspace/a-eye-lab-research/dataset/data/kaggle_cataract_nand"
    
    # ONNX 평가기 초기화
    evaluator = ONNXImageTestEvaluator(
        onnx_path=onnx_path,
        dataset_path=dataset_path,
        image_size=(224, 224)
    )

    # 데이터 로드 및 평가
    evaluator.load_data()
    results = evaluator.evaluate()

    # 결과 출력
    for metric, value in results.items():
        if metric=="Confusion Matrix":
            print(f"\n{metric}")
            print(value,"\n")
        else:
            print(f"{metric:<15} : {value:.4f}")