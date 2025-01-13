import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class DenseNetTest(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        """
        DenseNet121 model with customizable output classes and pre-trained weights.

        Args:
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to use ImageNet pre-trained weights.
        """
        super(DenseNetTest, self).__init__()

        # Load DenseNet121 model with or without pre-trained weights
        if pretrained:
            print("Loading DenseNet121 with ImageNet pre-trained weights...")
            self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            print("Loading DenseNet121 without pre-trained weights...")
            self.densenet = densenet121(weights=None)

        # Modify the classifier for the desired number of output classes
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Add dropout for regularization
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.densenet(x)


if __name__ == '__main__':
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model with 2 output classes and pre-trained weights
    num_classes = 2
    pretrained = True

    model = DenseNetTest(num_classes=num_classes, pretrained=pretrained).to(device)

    # Generate a dummy input tensor for testing
    dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Batch size 1, 3-channel 224x224 image

    # Forward pass
    output = model(dummy_input)

    # Print model and output shape
    print(model)
    print(f"Output shape: {output.shape}")
