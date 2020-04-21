from torch import nn
from torch.nn import functional as F
from torchvision import models


def resnet(variant, pretrained=True):
    backbone = getattr(models, variant)(pretrained=pretrained)
    in_features = backbone.fc.in_features
    
    backbone = ResNet(backbone)
    
    return backbone, in_features


class ResNet(nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, 3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
