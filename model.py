import torch.nn as nn


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28 → 28x28
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28x28 → 14x14
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14x14 → 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
