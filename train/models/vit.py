import torch.nn as nn
import timm

class ViT_Large(nn.Module):
    def __init__(self, num_classes, pretrained, img_size=384):
        super(ViT_Large, self).__init__()
        self.drop_rate = 0.4
        self.drop_rate_head = 0.3
        # timm을 사용해 vit_large 모델을 불러옵니다.
        self.model = timm.create_model(
            'vit_large_patch16_224.augreg_in21k', 
            pretrained=pretrained,
            drop_path_rate=self.drop_rate
        )
        
        # 모든 파라미터를 freeze
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 23번 블록부터 마지막까지의 모든 파라미터를 학습 가능하도록 설정
        for name, param in self.model.blocks[23:].named_parameters():
            param.requires_grad = True

        # model.norm의 파라미터를 학습 가능하도록 설정
        self.model.norm.weight.requires_grad = True
        self.model.norm.bias.requires_grad = True

        # 분류 헤드를 num_classes에 맞게 수정하고 배치 정규화 추가
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(), 
            nn.Dropout(p=self.drop_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = ViT_Large(num_classes=2, pretrained=True)

    for name, param in model.named_parameters():  
        print(f"{name}: requires_grad={param.requires_grad}")