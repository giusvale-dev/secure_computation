import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetBinaryClassifier(nn.Module):
    def __init__(self, freeze_features=True):
        super(MobileNetBinaryClassifier, self).__init__()

        # Load pretrained MobileNetV2
        weights = MobileNet_V2_Weights.DEFAULT
        self.base_model = models.mobilenet_v2(weights=weights)

        # freeze feature extractor
        if freeze_features:
            for param in self.base_model.features.parameters():
                param.requires_grad = False

        # Replace classifier with binary output (2 classes)
        self.base_model.classifier[1] = nn.Linear(
            in_features=self.base_model.classifier[1].in_features,
            out_features=2
        )

    def forward(self, x):
        return self.base_model(x)
