"""Plots from the CSV logs in results/. Same colors for SGD and RAdam on every chart."""
import os

import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"sgd": "#1f77b4", "radam": "#d62728"}
LINESTYLES = ["-", "--", ":", "-."]
LABELS = {"train_loss": "Training loss", "val_loss": "Validation loss",
          "train_acc": "Training accuracy", "val_acc": "Validation accuracy",
          "val_auc": "Validation AUC"}


def _read(name):
    return pd.read_csv(name if os.path.dirname(name) else os.path.join("results", name))


def plot_curves(csv_files, metrics=("train_loss", "val_loss", "val_acc"),
                out_png=None, title=None):
    """One panel per metric; one line per CSV file, colored by optimizer.

    Example:
        plot_curves(["early_sgd_lr0.01_seed42.csv", "early_radam_lr0.001_seed42.csv"],
                    out_png="results/early_curves.png")
    """
    runs = [_read(f) for f in csv_files]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 3.6))
    axes = [axes] if len(metrics) == 1 else axes
    style_count = {}
    for df in runs:
        opt = str(df["optimizer"].iloc[0]).lower()
        k = style_count.get(opt, 0)
        style_count[opt] = k + 1
        label = f"{'SGD' if opt == 'sgd' else 'RAdam'} (lr={df['lr'].iloc[0]:g})"
        for ax, m in zip(axes, metrics):
            d = df.dropna(subset=[m])
            ax.plot(d["epoch"], d[m], marker="o", ms=4, label=label,
                    color=COLORS.get(opt), linestyle=LINESTYLES[k % len(LINESTYLES)])
    for ax, m in zip(axes, metrics):
        ax.set_xlabel("Epoch")
        ax.set_title(LABELS.get(m, m))
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
    return fig


def summary_table(csv_files):
    """Last-epoch numbers plus average seconds per epoch and optimizer memory, one row per run."""
    rows = []
    for f in csv_files:
        df = _read(f)
        last = df.iloc[-1]
        trained = df[df["epoch"] > 0]
        rows.append({"optimizer": last["optimizer"], "lr": last["lr"], "epochs": int(last["epoch"]),
                     "val_acc": round(last["val_acc"], 4), "val_auc": round(last["val_auc"], 4),
                     "sec_per_epoch": round(trained["epoch_seconds"].mean(), 1),
                     "optimizer_state_bytes": int(last["optimizer_state_bytes"])})
    return pd.DataFrame(rows)
