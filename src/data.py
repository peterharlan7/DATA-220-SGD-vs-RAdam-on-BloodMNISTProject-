"""BloodMNIST loading. Owner: Peter (Trello: [S1] BloodMNIST data pipeline).

Everyone else only calls get_loaders(), so keep this function name and its
return value the same even if the inside changes.
"""
from torch.utils.data import DataLoader

from src.utils import make_generator

NUM_CLASSES = 8
CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "immature granulocytes",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]


def get_loaders(batch_size: int = 64, seed: int = 42, root: str = "data"):
    """Return (train_loader, val_loader, test_loader) for BloodMNIST 28x28.

    TODO (Peter):
      - medmnist.BloodMNIST(split=..., download=True, size=28, root=root)
      - transform: ToTensor() + Normalize(mean=[.5]*3, std=[.5]*3)
      - labels come as shape (1,); squeeze them to a plain int
      - train loader: shuffle=True, generator=make_generator(seed)
      - val/test loaders: shuffle=False
      - check sizes are 11,959 / 1,712 / 3,421
    """
    raise NotImplementedError
