"""
model.py — Simple CNN for federated learning
=============================================
A lightweight convolutional neural network suitable for medical image
classification tasks. Each hospital trains this model locally on its
private data, and only the model weights (or their hash) are shared.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FederatedCNN(nn.Module):
    """A compact CNN designed for federated learning across hospital nodes."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model_hash(model: nn.Module) -> str:
    """Compute a SHA-256 hash of the model's state dict for on-chain submission."""
    import hashlib
    import io

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()
