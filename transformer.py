
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from torch import optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import numpy as np

# ====== 설정 ======
data_dir_train = "/home/mia/a-eye-lab-research/cropped_eyes_haar"
data_dir_val = "/home/mia/a-eye-lab-research/dataset/data/real_data"
save_path = "/home/mia/a-eye-lab-research/eye_transformer_attention2.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 전처리 ======
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


train_dataset = datasets.ImageFolder(data_dir_train, transform=train_transform)
val_dataset = datasets.ImageFolder(data_dir_val, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


class AttentionEnhancedSwin(nn.Module):
    def __init__(self, base_model="swin_tiny_patch4_window7_224", num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(base_model, pretrained=True, features_only=True)
        self.feature_dim = self.backbone.feature_info.channels()[-1]

        self.attention_conv = nn.Sequential(
            nn.Conv2d(self.feature_dim, 4, kernel_size=1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)[-1]
        if features.shape[1] < 10:
            features = features.permute(0, 3, 1, 2)
        att_map = self.attention_conv(features)
        att_map = att_map.unsqueeze(2)
        features = features.unsqueeze(1)
        attended = (features * att_map).mean(dim=1)
        return self.classifier(attended)

model = AttentionEnhancedSwin().to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)


best_auc = 0.0
patience = 7
counter = 0

for epoch in range(100):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step(epoch + 1)


    model.eval()
    y_val_true, y_val_pred, y_val_score = [], [], []
    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)

            outputs1 = model(val_images)
            outputs2 = model(torch.flip(val_images, dims=[3]))  # TTA: Horizontal flip
            outputs = (outputs1 + outputs2) / 2

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            y_val_true.extend(val_labels.cpu().numpy())
            y_val_pred.extend(preds)
            y_val_score.extend(probs)

    val_auc = roc_auc_score(y_val_true, y_val_score)
    val_acc = accuracy_score(y_val_true, y_val_pred)
    val_f1 = f1_score(y_val_true, y_val_pred)
    tn, fp, fn, tp = confusion_matrix(y_val_true, y_val_pred).ravel()
    val_spec = tn / (tn + fp)

    print(f"\nEpoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, Spec: {val_spec:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), save_path)
        print(" Best model saved.")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(" Early stopping triggered.")
            break
