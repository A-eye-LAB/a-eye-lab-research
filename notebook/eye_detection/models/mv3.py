"""mobilenetv3 model
- from Hyungkeun-park
"""

import timm
import torch.nn as nn


class MobileNet_V3_Large(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(MobileNet_V3_Large, self).__init__()

        self.model = timm.create_model(
            "mobilenetv3_large_100.miil_in21k_ft_in1k",
            pretrained=pretrained,
        )

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(
            in_features=in_features, out_features=num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = MobileNet_V3_Large(num_classes=2, pretrained=False)

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
