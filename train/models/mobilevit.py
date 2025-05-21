import torch.nn as nn
import timm

class LeViT(nn.Module):
    def __init__(self, num_classes, pretrained, img_size=384):
        super(LeViT, self).__init__()
        self.drop_rate = 0.3
        
        # LeViT 모델을 불러옵니다
        self.model = timm.create_model(
            'levit_384.fb_dist_in1k',
            pretrained=pretrained,
            drop_path_rate=self.drop_rate
        )
        
        # stage 0, 1 및 기타 레이어들을 프리즈
        for name, param in self.model.named_parameters():
            if not name.startswith('stages.2'):
                param.requires_grad = False
                
        # 분류 헤드 수정
        in_features = self.model.head.linear.in_features
        self.model.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(256, num_classes)
        )
        
        # head_dist 제거
        self.model.head_dist = None

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = LeViT(num_classes=2, pretrained=True)
    
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
