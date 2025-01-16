"""유틸리티 코드
- load model, preprocess, evaluate metrics
"""

import torch
import pickle
import torchmetrics

from PIL import Image
from typing import Dict, List

from torchvision import models
from torchvision import transforms
from models.mv3 import MobileNet_V3_Large

from scipy.spatial.distance import cosine


def load_model(tuning_status, model_path=""):
    """MobileNetV3 모델 로드"""

    if tuning_status:
        model = MobileNet_V3_Large(num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(model_path, weights_only=True))

    else:
        model = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )

    model.classifier = torch.nn.Identity()
    model.eval()

    return model


def preprocess_image(image_path):
    """이미지 전처리 함수"""

    image = Image.open(image_path).convert("RGB")
    min_size = min(image.size)

    transform = transforms.Compose(
        [
            transforms.CenterCrop(min_size),
            # transforms.CenterCrop(224),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    out_tensor = transform(image)

    return out_tensor.unsqueeze(0)


def load_mean_embedding(file_path):
    """평균 임베딩 로드 함수

    Args :
        - file_path : pickle file path
    """

    with open(file_path, "rb") as f:
        data = pickle.load(f)

        return data


def extract_embedding(model, image_path):
    """임베딩 추출 함수"""

    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        embedding = model(input_tensor).squeeze().numpy()

    return embedding


def is_eye_image(model, image_path, mean_embedding, threshold=0.65):
    """이미지 유사성 판단 함수 (cosine similarity)

    Args :
        - model : model class
        - image_path : target image file path.
        - mean_embedding : mean embedding value.
        - threshold : similarity threshold (Defaults to 0.65.)

    Returns :
        - result : True / False
        - similarity : similarity value.
    """

    embedding = extract_embedding(model, image_path)
    similarity = 1 - cosine(embedding, mean_embedding)

    result = similarity > threshold

    return result, similarity


def evaluate_model(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    # Tensor로 변환
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    accuracy = torchmetrics.Accuracy(task="binary")(y_pred, y_true)
    f1_score = torchmetrics.F1Score(task="binary")(y_pred, y_true)
    precision = torchmetrics.Precision(task="binary")(y_pred, y_true)
    recall = torchmetrics.Recall(task="binary")(y_pred, y_true)
    specificity = torchmetrics.Specificity(task="binary")(y_pred, y_true)

    metrics = {
        "valid/accuracy": accuracy.item(),
        "valid/f1_score": f1_score.item(),
        "valid/precision": precision.item(),
        "valid/recall": recall.item(),
        "valid/specificity": specificity.item(),
    }

    return metrics


def print_evaluation(
    phase: str, epoch: int, loss: float, metrics: Dict[str, float]
) -> None:
    formatted_metrics = " | ".join(
        [f"{metric_name}: {value:.4f}" for metric_name, value in metrics.items()]
    )
    print(f"{phase} | Epoch {epoch} | Loss: {loss:.4f} | {formatted_metrics}")
