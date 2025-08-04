import torch.nn as nn
import timm

class FastViT(nn.Module):
    def __init__(self, num_classes, pretrained, img_size=224):
        super(FastViT, self).__init__()
        self.drop_rate = 0.4
        self.drop_rate_head = 0.3
        
        # timm을 사용해 FastViT 모델을 불러옵니다
        self.model = timm.create_model(
            'fastvit_ma36.apple_dist_in1k', # 기본 모델인 T8 사용 
            pretrained=pretrained,
            drop_path_rate=self.drop_rate,
            num_classes=num_classes
        )
        
        # 모든 파라미터를 freeze
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 마지막 stage의 블록만 학습 가능하도록 설정
        for param in self.model.stages[-1].parameters():
            param.requires_grad = True
            
        # final_conv 파라미터를 학습 가능하도록 설정
        for name, param in self.model.named_parameters():
            if 'final_conv' in name:
                param.requires_grad = True
            
        # norm 파라미터를 학습 가능하도록 설정
        if hasattr(self.model, 'norm'):
            self.model.norm.weight.requires_grad = True
            self.model.norm.bias.requires_grad = True


    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = FastViT(num_classes=2, pretrained=True)
    
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
