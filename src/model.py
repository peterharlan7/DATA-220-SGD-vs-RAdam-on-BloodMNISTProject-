"""ResNet-8. Owner: Jason (Trello: [S1] Implement ResNet-8).

Everyone else only calls build_model() and load_init_weights(), so keep those
names the same.
"""
import torch
import torch.nn as nn


class ResNet8(nn.Module):
    """3x3 conv stem -> 3 residual blocks (16 -> 32 -> 64) -> global avg pool -> 8 classes.

    TODO (Jason):
      - BasicBlock with skip connection (1x1 conv on the skip when channels/stride change)
      - forward(x) with x of shape (batch, 3, 28, 28) returns (batch, 8)
    """

    def __init__(self, num_classes: int = 8):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def build_model() -> nn.Module:
    return ResNet8(num_classes=8)


def load_init_weights(model: nn.Module, path: str = "init.pt") -> nn.Module:
    """Load the shared starting weights so SGD and RAdam start identically."""
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
