import torch
import torch.nn as nn
import timm
from copy import deepcopy

class MobileNet_V3_Large(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(MobileNet_V3_Large, self).__init__()
        self.drop_rate = 0.4

        # From timm
        self.model = timm.create_model(
            'mobilenetv3_large_100.miil_in21k_ft_in1k',
            pretrained = pretrained,
            drop_path_rate=self.drop_rate
        )
        
        # 모든 파라미터를 freeze
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 마지막 두 개의 블록만 학습하도록 설정
        for name, param in self.model.blocks[-2:].named_parameters():
            param.requires_grad = True
            
        
        # 분류 헤드 수정
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=self.drop_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        #features = self.model.forward_features(x)
        features, x = self.model(x)
        features = self.org_block5(features)
        features = self.org_block6(features)
        features = self.global_pool(features)

        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x, features

if __name__ == '__main__':
    model = MobileNet_V3_Large(num_classes=2, pretrained=False)

    for name, param in model.named_parameters():  
        print(f"{name}: requires_grad={param.requires_grad}")