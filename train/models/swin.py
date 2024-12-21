import torch.nn as nn
import timm

class Swin_Large(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(Swin_Large, self).__init__()

        # timm을 사용해 Swin-Large 모델을 불러옵니다.
        self.model = timm.create_model(
            'swin_large_patch4_window7_224',  # Swin-Large 모델 이름
            pretrained=pretrained
        )

        # 모든 파라미터를 freeze
        for param in self.model.parameters():
            param.requires_grad = False

        # 마지막 stage의 블록만 학습 가능하도록 설정
        for param in self.model.layers[-1].parameters():
            param.requires_grad = True

        # norm 파라미터를 학습 가능하도록 설정
        self.model.norm.weight.requires_grad = True
        self.model.norm.bias.requires_grad = True

        # 분류 헤드를 num_classes에 맞게 수정
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.mean(dim=(1,2))
        return x

if __name__ == '__main__':
    model = Swin_Large(num_classes=2, pretrained=True)

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")