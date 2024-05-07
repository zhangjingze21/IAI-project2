import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobileNet(nn.Module):
    def __init__(self, num_classes=6):
        super(MobileNet, self).__init__()
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),       # Approximates first layer in AlexNet
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),     # Approximates second layer in AlexNet
            conv_dw(128, 256, 2),    # Approximates third layer in AlexNet
            conv_dw(256, 512, 2),    # Approximates fourth layer in AlexNet
            conv_dw(512, 1024, 2),   # Approximates fifth layer in AlexNet
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x