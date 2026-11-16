"""[S3] Verify optimizer-state memory. Owner: Sai.

Checks the measured optimizer-state bytes for plain SGD and RAdam against the
hand calculation 2 x params x 4 bytes (the 0.59 MiB on the proposal slide).

Run from the repo root:  python notebooks/sai_memory_check.py
Writes results/memory_check.csv
"""
import sys
import os

sys.path.insert(0, os.getcwd())

import pandas as pd
import torch
import torch.nn as nn

from src.data import get_loaders
from src.model import build_model, load_init_weights
from src.train import make_optimizer
from src.utils import count_parameters, optimizer_state_bytes, set_seed

MIB = 1024 ** 2


def measure(name, lr, momentum=None):
    set_seed(42)
    model = load_init_weights(build_model(), "init.pt")
    if momentum is None:
        opt = make_optimizer(name, model.parameters(), lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    before = optimizer_state_bytes(opt)

    images, labels = next(iter(train_loader))
    opt.zero_grad()
    nn.CrossEntropyLoss()(model(images), labels).backward()
    opt.step()

    # break the total down by what the optimizer stores
    by_key = {}
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                by_key[key] = by_key.get(key, 0) + value.numel() * value.element_size()
    return before, optimizer_state_bytes(opt), by_key


set_seed(42)
train_loader, _, _ = get_loaders(batch_size=64, seed=42)
model = build_model()
n_params = count_parameters(model)
n_tensors = sum(1 for p in model.parameters() if p.requires_grad)

hand_radam = 2 * n_params * 4          # exp_avg + exp_avg_sq, float32
hand_step = n_tensors * 4              # one float32 "step" scalar per parameter tensor

rows = []
for label, name, lr, mom, hand in [
    ("SGD (plain, as used)", "sgd", 0.01, None, 0),
    ("RAdam (defaults, as used)", "radam", 0.001, None, hand_radam + hand_step),
    ("SGD + momentum 0.9 (reference only)", "sgd", 0.01, 0.9, n_params * 4),
]:
    before, after, by_key = measure(name, lr, mom)
    rows.append(dict(optimizer=label, bytes_before_first_step=before,
                     measured_bytes=after, hand_calc_bytes=hand,
                     match=after == hand, measured_MiB=round(after / MIB, 4),
                     breakdown=" + ".join(f"{k}={v:,}" for k, v in by_key.items()) or "-"))

df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
df.to_csv("results/memory_check.csv", index=False)

early = pd.read_csv("results/early_radam_lr0.001_seed42.csv")
print(f"trainable parameters : {n_params:,}")
print(f"parameter tensors    : {n_tensors}")
print(f"2 x params x 4 bytes : {hand_radam:,} B = {hand_radam / MIB:.4f} MiB")
print(f"+ step counters      : {n_tensors} x 4 = {hand_step} B")
print(f"RAdam bytes in early run CSV (epochs 1-3): {sorted(set(early.optimizer_state_bytes[1:]))}")
print()
print(df.to_string(index=False))
