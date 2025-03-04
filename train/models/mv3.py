import torch
import torch.nn as nn
import timm
from copy import deepcopy

class MobileNet_V3_Large(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(MobileNet_V3_Large, self).__init__()

        # From timm
        self.model = timm.create_model(
            'mobilenetv3_large_100',
            pretrained = pretrained,
            num_classes = num_classes
        )


    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = MobileNet_V3_Large(num_classes=2, pretrained=False)

    for name, param in model.named_parameters():  
        print(f"{name}: requires_grad={param.requires_grad}")