"""[S4] Final Run: RAdam, 10 epochs. Owner: Sai.

Uses the best LR from results/screen_radam_summary.csv (selection rule: highest
val acc after the 3-epoch screen, ties -> lower val loss). Seeds 42, 43, 44.
Peter's SGD final run must use the same epochs and seeds.
The test set is NOT touched here ([S4] Evaluate on the test set once, Ashritha).

Run from the repo root:  python notebooks/sai_radam_final.py
"""
import os
import sys

sys.path.insert(0, os.getcwd())

import pandas as pd

from src.train import run_experiment

EPOCHS = 10
SEEDS = [42, 43, 44]

s = pd.read_csv("results/screen_radam_summary.csv")
best_lr = s.sort_values(["val_acc", "val_loss"], ascending=[False, True]).iloc[0].lr
best_lr = float(f"{best_lr:g}")
print("best RAdam LR from screen:", best_lr)

finals = []
for seed in SEEDS:
    df, _ = run_experiment("radam", lr=best_lr, epochs=EPOCHS, batch_size=64, seed=seed,
                           out_csv=f"final_radam_lr{best_lr:g}_seed{seed}.csv",
                           save_model=f"checkpoints/final_radam_seed{seed}.pth")
    finals.append(df)

all_runs = pd.concat(finals)
last = all_runs[all_runs.epoch == EPOCHS]
print("\nFinal epoch, per seed")
print(last[["seed", "val_acc", "val_auc", "val_loss", "epoch_seconds"]].to_string(index=False))
print("\nmean +- std over seeds")
for col in ["val_acc", "val_auc", "val_loss"]:
    print(f"  {col}: {last[col].mean():.4f} +- {last[col].std():.4f}")
print(f"  sec/epoch (epochs 1-{EPOCHS}): {all_runs[all_runs.epoch > 0].epoch_seconds.mean():.1f}")
for target in (0.85, 0.90):
    hits = [d.loc[d.val_acc >= target, "epoch"].min() for d in finals]
    print(f"  first epoch with val_acc >= {target}: {hits}")
