import torch
import torch.nn as nn
import torchvision

class ResNet50_pretrained(nn.Module):
    def __init__(self):
        super(ResNet50_pretrained, self).__init__()
        self.resnet = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.resnet.fc = nn.Linear(2048, 6)
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)