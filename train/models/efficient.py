import torch.nn as nn
import timm


class EfficientNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNet, self).__init__()

        self.model = timm.create_model(
            "efficientnet_b4",
            num_classes=num_classes,
            pretrained=pretrained,
        )

        for param in self.model.parameters():
            param.requires_grad = False
  
        for name, param in self.model.blocks[-3:].named_parameters():   
            param.requires_grad = True
 
        if hasattr(self.model, "classifier"):   
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    model = EfficientNet(num_classes=2, pretrained=True)

    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
