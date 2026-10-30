"""BloodMNIST loading. Owner: Peter (Trello: [S1] BloodMNIST data pipeline).

Downloads BloodMNIST into data/, which your .gitignore already keeps out of git.
Checks that the splits are exactly 11,959 / 1,712 / 3,421 images.
Normalizes the images and turns the labels into plain numbers so the training loop can use them.
Shuffles the training batches the same way on every run with the same seed, so SGD and RAdam see identical batch order.
Has two functions that make your card’s slide figures: a sample image grid and a class-count bar chart.

Everyone else only calls get_loaders(), so keep this function name and its
return value the same even if the inside changes.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from medmnist import BloodMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils import make_generator

NUM_CLASSES = 8
CLASS_NAMES = [
    "basophil", "eosinophil", "erythroblast", "immature granulocytes",
    "lymphocyte", "monocyte", "neutrophil", "platelet",
]
EXPECTED_SIZES = {"train": 11959, "val": 1712, "test": 3421}

# Scales pixel values from [0, 1] to [-1, 1] on each RGB channel.
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def _to_int(label):
    """medmnist gives each label as an array like [3]; the loss function wants 3."""
    return int(label[0])


def get_datasets(root: str = "data", transform=TRANSFORM):
    """Return (train, val, test) BloodMNIST datasets at 28x28, downloading on first use."""
    os.makedirs(root, exist_ok=True)  # medmnist refuses a folder that doesn't exist
    splits = []
    for split in ("train", "val", "test"):
        ds = BloodMNIST(split=split, transform=transform, target_transform=_to_int,
                        download=True, size=28, root=root)
        assert len(ds) == EXPECTED_SIZES[split], f"{split}: got {len(ds)} images"
        splits.append(ds)
    return tuple(splits)


def get_loaders(batch_size: int = 64, seed: int = 42, root: str = "data"):
    """Return (train_loader, val_loader, test_loader).

    The training loader shuffles with a seeded generator, so every run with the
    same seed sees the same batches in the same order, for SGD and RAdam alike.
    """
    train_ds, val_ds, test_ds = get_datasets(root)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=make_generator(seed), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# ---------- Figures for the slides ----------

def save_sample_grid(path: str = "results/sample_grid.png", per_class: int = 4,
                     root: str = "data", seed: int = 0):
    """One row per class, `per_class` random training images in each row."""
    train_ds, _, _ = get_datasets(root, transform=None)
    labels = train_ds.labels.ravel()
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(NUM_CLASSES, per_class,
                             figsize=(per_class * 1.4, NUM_CLASSES * 1.4))
    for c in range(NUM_CLASSES):
        picks = rng.choice(np.flatnonzero(labels == c), per_class, replace=False)
        for j, idx in enumerate(picks):
            ax = axes[c, j]
            ax.imshow(train_ds.imgs[idx])
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(CLASS_NAMES[c], rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("BloodMNIST training images (28 × 28)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_class_counts(path: str = "results/class_counts.png", root: str = "data"):
    """Bar chart of images per class in the train / val / test splits."""
    datasets = get_datasets(root, transform=None)
    counts = {name: np.bincount(ds.labels.ravel(), minlength=NUM_CLASSES)
              for name, ds in zip(("train", "val", "test"), datasets)}
    x = np.arange(NUM_CLASSES)
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (name, c) in enumerate(counts.items()):
        ax.bar(x + (i - 1) * width, c, width, label=f"{name} ({c.sum():,})")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Images")
    ax.set_title("BloodMNIST images per class")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return counts


if __name__ == "__main__":
    tr, va, te = get_loaders()
    images, labels = next(iter(tr))
    print("batch:", tuple(images.shape), "labels:", labels[:8].tolist())
    print("sizes:", len(tr.dataset), len(va.dataset), len(te.dataset))
