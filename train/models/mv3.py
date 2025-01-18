import torch
import torch.nn as nn
import timm

class MobileNet_V3_Large(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(MobileNet_V3_Large, self).__init__()

        # From timm
        self.model = timm.create_model(
            'mobilenetv3_large_100.miil_in21k_ft_in1k',
            pretrained = pretrained,
            features_only = True,
            out_indices = [3, 4],
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_head = nn.Conv2d(960, 1280, kernel_size=1, stride=1)
        self.act2 = nn.Hardswish()
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(1280, num_classes)

        for param in self.model.parameters():
            param.requires_grad = False

        for i in range(5, len(self.model.blocks)):
            for param in self.model.blocks[i].parameters():
                param.requires_grad = True
        #for param in self.model.blocks[0:]:
        #for param in self.model.blocks[5:]:   
            #param.requires_grad = True
        #self.model.conv_head.weight.requires_grad = True
        #self.model.conv_head.bias.requires_grad = True

        # For timm version
        #in_features = self.model.classifier.in_features
        #self.model.classifier = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x):
        #features = self.model.forward_features(x)
        features, x = self.model(x)
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