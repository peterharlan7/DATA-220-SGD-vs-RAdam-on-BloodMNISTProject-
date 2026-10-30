"""Shared helpers. Owner: Peter."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Make Python, NumPy and PyTorch random numbers repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int = 42) -> torch.Generator:
    """Generator for DataLoader shuffling, so SGD and RAdam see the same batch order."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device() -> torch.device:
    """The project runs on Colab CPU by design, so this always returns CPU."""
    return torch.device("cpu")


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Total bytes of every tensor the optimizer keeps between steps.

    Plain SGD (no momentum) keeps nothing, so this returns 0.
    RAdam keeps exp_avg and exp_avg_sq for every parameter.
    Call it after at least one optimizer.step(), since state is created lazily.
    """
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
