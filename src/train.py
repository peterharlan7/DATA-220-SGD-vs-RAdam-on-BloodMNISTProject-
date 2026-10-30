"""Training and evaluation loop (Trello: [S1] Training + evaluation loop).

Notebooks call run_experiment(); keep its name and arguments the same.

Every run:
  - starts from init.pt (same weights for every optimizer)
  - sees the same batch order for the same seed
  - logs one CSV row per epoch to results/
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.data import CLASS_NAMES, get_loaders
from src.model import build_model, load_init_weights
from src.utils import optimizer_state_bytes, set_seed

RESULTS_DIR = "results"


def make_optimizer(name: str, params, lr: float):
    """"sgd" = plain SGD (no momentum, no weight decay); "radam" = PyTorch RAdam defaults."""
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "radam":
        return torch.optim.RAdam(params, lr=lr)
    raise ValueError(f"Unknown optimizer {name!r}; use 'sgd' or 'radam'.")


def train_one_epoch(model, loader, optimizer, loss_fn):
    """One pass over the training set. Returns (average loss, accuracy)."""
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, return_predictions: bool = False):
    """Returns {"loss", "acc", "auc"}; AUC is one-vs-rest, averaged over the 8 classes
    (the same metric MedMNIST reports)."""
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * images.size(0)
        n += images.size(0)
        all_probs.append(torch.softmax(logits, 1))
        all_labels.append(labels)
    probs = torch.cat(all_probs).numpy()
    y = torch.cat(all_labels).numpy()
    out = {
        "loss": total_loss / n,
        "acc": float((probs.argmax(1) == y).mean()),
        "auc": float(roc_auc_score(y, probs, multi_class="ovr")),
    }
    if return_predictions:
        out["y_true"], out["y_pred"], out["probs"] = y, probs.argmax(1), probs
    return out


def _results_path(name: str) -> str:
    path = name if os.path.dirname(name) else os.path.join(RESULTS_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def run_experiment(optimizer_name: str, lr: float, epochs: int,
                   batch_size: int = 64, seed: int = 42,
                   out_csv: str = None, save_model: str = None,
                   init_path: str = "init.pt", verbose: bool = True):
    """Train from init.pt and evaluate on the validation set after every epoch.

    Returns (DataFrame of per-epoch results, trained model).
    Row 0 is the untrained model, so learning curves start at the same point.

    CSV columns: epoch, train_loss, train_acc, val_loss, val_acc, val_auc,
                 epoch_seconds (wall clock), cpu_seconds (process CPU time),
                 optimizer_state_bytes, optimizer, lr, seed, batch_size

    save_model: optional path such as "checkpoints/final_sgd.pth" (git-ignored).
    """
    set_seed(seed)
    train_loader, val_loader, _ = get_loaders(batch_size=batch_size, seed=seed)
    model = load_init_weights(build_model(), init_path)
    optimizer = make_optimizer(optimizer_name, model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()

    v = evaluate(model, val_loader, loss_fn)
    rows = [dict(epoch=0, train_loss=np.nan, train_acc=np.nan,
                 val_loss=v["loss"], val_acc=v["acc"], val_auc=v["auc"],
                 epoch_seconds=0.0, cpu_seconds=0.0,
                 optimizer_state_bytes=optimizer_state_bytes(optimizer))]

    for epoch in range(1, epochs + 1):
        wall0, cpu0 = time.perf_counter(), time.process_time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
        v = evaluate(model, val_loader, loss_fn)
        rows.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                         val_loss=v["loss"], val_acc=v["acc"], val_auc=v["auc"],
                         epoch_seconds=wall, cpu_seconds=cpu,
                         optimizer_state_bytes=optimizer_state_bytes(optimizer)))
        if verbose:
            print(f"[{optimizer_name} lr={lr}] epoch {epoch}/{epochs}  "
                  f"train_loss {train_loss:.4f}  val_loss {v['loss']:.4f}  "
                  f"val_acc {v['acc']:.4f}  val_auc {v['auc']:.4f}  {wall:.1f}s")

    df = pd.DataFrame(rows)
    df["optimizer"], df["lr"], df["seed"], df["batch_size"] = optimizer_name, lr, seed, batch_size

    if out_csv:
        path = _results_path(out_csv)
        df.to_csv(path, index=False)
        if verbose:
            print("saved", path)
    if save_model:
        os.makedirs(os.path.dirname(save_model) or ".", exist_ok=True)
        torch.save(model.state_dict(), save_model)
    return df, model


def test_model(model, batch_size: int = 64, seed: int = 42):
    """Final test-set evaluation. Use ONCE per final model (Sprint 4), never for choosing LRs.

    Returns {"loss", "acc", "auc", "confusion_matrix", "per_class_acc"}.
    """
    _, _, test_loader = get_loaders(batch_size=batch_size, seed=seed)
    r = evaluate(model, test_loader, nn.CrossEntropyLoss(), return_predictions=True)
    cm = confusion_matrix(r["y_true"], r["y_pred"], labels=list(range(len(CLASS_NAMES))))
    per_class = cm.diagonal() / cm.sum(axis=1)
    return {"loss": r["loss"], "acc": r["acc"], "auc": r["auc"],
            "confusion_matrix": cm,
            "per_class_acc": dict(zip(CLASS_NAMES, per_class.round(4)))}
