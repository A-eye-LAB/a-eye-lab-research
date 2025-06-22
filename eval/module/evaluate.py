from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt

def specificity_score(y_true, y_pred, zero_division="warn"):
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

def evaluate(y_true, y_pred):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    metrics = {
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division="warn"),
        "Recall": recall_score(y_true, y_pred, zero_division="warn"),
        "F1 Score": f1_score(y_true, y_pred),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Specificity": specificity_score(y_true, y_pred, zero_division="warn"),
        "AUC-ROC": roc_auc
        }


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
    plt.savefig('roc_curve.png')
    plt.close()

    return metrics