import torch.nn as nn
from torchvision.models import convnext_base


class ConvNextBase(nn.Module):
    def __init__(self, num_classes, pretrained):
        super(ConvNextBase, self).__init__()

        self.model = convnext_base(pretrained=pretrained)

        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ConvNextBase(num_classes=2, pretrained=True)

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
