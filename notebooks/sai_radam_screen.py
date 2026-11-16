"""RAdam LR screen (Trello card [S3] LR screen: RAdam, owned by Jason).

Run locally by Sai to unblock [S4] Final Run: RAdam. Same settings as the card:
3 epochs, seed 42, batch 64, from init.pt. Also re-runs lr=0.001 to check that
a local run reproduces the team's Colab early run.

Run from the repo root:  python notebooks/sai_radam_screen.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

from src.train import run_experiment

lrs = [0.0003, 0.001, 0.003]
for lr in lrs:
    run_experiment("radam", lr=lr, epochs=3, batch_size=64, seed=42,
                   out_csv=f"screen_radam_lr{lr}_seed42.csv")

def final(lr):
    return pd.read_csv(f"results/screen_radam_lr{lr}_seed42.csv").iloc[-1]

# card rule: if the smallest rate wins, also try one step lower
best = max(lrs, key=lambda lr: (final(lr).val_acc, -final(lr).val_loss))
if best == 0.0003:
    run_experiment("radam", lr=0.0001, epochs=3, batch_size=64, seed=42,
                   out_csv="screen_radam_lr0.0001_seed42.csv")
    lrs.insert(0, 0.0001)

# selection rule (Peter's card): highest val acc after the screen, ties -> lower val loss
summary = pd.DataFrame([final(lr) for lr in lrs])[["lr", "val_acc", "val_loss", "val_auc", "epoch_seconds"]]
summary.to_csv("results/screen_radam_summary.csv", index=False)
best = max(lrs, key=lambda lr: (final(lr).val_acc, -final(lr).val_loss))
print(summary.to_string(index=False))
print("BEST RAdam LR:", best)

colab = pd.read_csv("results/early_radam_lr0.001_seed42.csv")
local = pd.read_csv("results/screen_radam_lr0.001_seed42.csv")
print("\nReproducibility check, lr=0.001 val_acc per epoch")
print(pd.DataFrame({"colab_early": colab.val_acc, "local": local.val_acc}).to_string())
