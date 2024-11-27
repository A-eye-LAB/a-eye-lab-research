# Torch Imports
import torch
import torchmetrics

# ETC Imports
from typing import Dict, List, Any

def evaluate_model(y_true: List[int], y_pred: List[int], cfg: Dict[str, Any]) -> Dict[str, float]:
    # Tensor로 변환
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=cfg["MODEL"]["NUM_CLASSES"])(y_pred, y_true)
    f1_score = torchmetrics.F1Score(task="multiclass", num_classes=cfg["MODEL"]["NUM_CLASSES"])(y_pred, y_true)
    precision = torchmetrics.Precision(task="multiclass", num_classes=cfg["MODEL"]["NUM_CLASSES"])(y_pred, y_true)
    recall = torchmetrics.Recall(task="multiclass", num_classes=cfg["MODEL"]["NUM_CLASSES"])(y_pred, y_true)
    specificity = torchmetrics.Specificity(task="multiclass", num_classes=cfg["MODEL"]["NUM_CLASSES"])(y_pred, y_true)

    metrics = {
        'valid/accuracy': accuracy.item(),
        'valid/f1_score': f1_score.item(),
        'valid/precision': precision.item(),
        'valid/recall': recall.item(),
        'valid/specificity': specificity.item()
    }

    return metrics

def print_evaluation(phase: str, epoch: int, loss: float, metrics: Dict[str, float]) -> None:
    formatted_metrics = " | ".join([f'{metric_name}: {value:.4f}' for metric_name, value in metrics.items()])
    print(f"{phase} | Epoch {epoch} | Loss: {loss:.4f} | {formatted_metrics}")

