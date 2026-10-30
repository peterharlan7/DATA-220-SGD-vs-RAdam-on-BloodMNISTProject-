"""ResNet-8 for 28x28 BloodMNIST (Trello: [S1] Implement ResNet-8).

Everyone else only calls build_model() and load_init_weights(), so keep those
names the same.

Architecture (8 weighted layers = 1 stem conv + 3 blocks x 2 convs + 1 linear):
  input 3x28x28
  -> 3x3 conv, 16 channels                 28x28
  -> residual block 16 -> 16, stride 1     28x28
  -> residual block 16 -> 32, stride 2     14x14
  -> residual block 32 -> 64, stride 2      7x7
  -> global average pooling                64
  -> linear                                 8 class scores
Total trainable parameters: 77,912.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import set_seed


class BasicBlock(nn.Module):
    """Two 3x3 convs with a skip connection: out = relu(F(x) + shortcut(x))."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # When the shape changes, a 1x1 conv resizes x so it can be added to F(x).
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet8(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.layer1 = BasicBlock(16, 16, stride=1)
        self.layer2 = BasicBlock(16, 32, stride=2)
        self.layer3 = BasicBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)


def build_model() -> nn.Module:
    return ResNet8(num_classes=8)


def load_init_weights(model: nn.Module, path: str = "init.pt") -> nn.Module:
    """Load the shared starting weights so SGD and RAdam start identically."""
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def make_init_weights(path: str = "init.pt", seed: int = 42, overwrite: bool = False) -> str:
    """Create init.pt ONCE for the whole team, then commit it.

    Refuses to replace an existing file unless overwrite=True, because changing
    the starting weights would make old and new runs incomparable.
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists. Everyone must use the same one.")
    set_seed(seed)
    model = build_model()
    torch.save(model.state_dict(), path)
    return path
