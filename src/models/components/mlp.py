import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.3, height: int = 256, width: int = 256, channels: int = 3):
        super(MLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(channels * height * width, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x