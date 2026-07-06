#!/usr/bin/env python3
"""Convert a training metrics.json (written by train_lora.py) into a PNG chart.

Usage:
    python roentgenv2/train_code/plot_metrics.py --metrics /path/to/metrics.json
    python roentgenv2/train_code/plot_metrics.py --metrics metrics.json --out curve.png --smooth 0.9

The JSON schema is:
    {
      "meta":  {...hyperparameters...},
      "train": {"step": [...], "loss": [...], "lr": [...]},
      "val":   {"step": [...], "loss": [...]}
    }
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless / no display (Colab, servers)
import matplotlib.pyplot as plt


def ema(values, alpha):
    """Exponential moving average to tame the very noisy per-step diffusion loss."""
    if not values or alpha <= 0:
        return list(values)
    smoothed, acc = [], values[0]
    for v in values:
        acc = alpha * acc + (1 - alpha) * v
        smoothed.append(acc)
    return smoothed


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training/validation curves from metrics.json.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.json.")
    parser.add_argument("--out", default=None, help="Output PNG path (default: <metrics_dir>/training_curve.png).")
    parser.add_argument("--smooth", type=float, default=0.9,
                        help="EMA factor 0..1 for the train-loss curve (0 = raw). Default 0.9.")
    parser.add_argument("--show-lr", action="store_true", help="Overlay the learning-rate schedule on a second axis.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.metrics, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    train = data.get("train", {})
    val = data.get("val", {})
    tsteps, tloss, tlr = train.get("step", []), train.get("loss", []), train.get("lr", [])
    vsteps, vloss = val.get("step", []), val.get("loss", [])

    if not tsteps and not vsteps:
        raise SystemExit(f"No data points found in {args.metrics}.")

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.metrics)), "training_curve.png")

    fig, ax = plt.subplots(figsize=(10, 6))

    if tsteps:
        ax.plot(tsteps, tloss, color="tab:blue", alpha=0.25, linewidth=1, label="train loss (raw)")
        if args.smooth > 0:
            ax.plot(tsteps, ema(tloss, args.smooth), color="tab:blue", linewidth=2,
                    label=f"train loss (EMA {args.smooth})")
    if vsteps:
        ax.plot(vsteps, vloss, color="tab:red", marker="o", markersize=4, linewidth=2, label="validation loss")

    ax.set_xlabel("training step")
    ax.set_ylabel("MSE loss")
    ax.set_title("RoentGen-v2 LoRA training")
    ax.grid(True, alpha=0.3)

    if args.show_lr and tlr:
        ax_lr = ax.twinx()
        ax_lr.plot(tsteps, tlr, color="tab:green", alpha=0.6, linestyle="--", label="learning rate")
        ax_lr.set_ylabel("learning rate")
        ax_lr.legend(loc="upper right")

    ax.legend(loc="upper left" if not (args.show_lr and tlr) else "upper center")

    meta = data.get("meta", {})
    if meta:
        subtitle = "  ".join(f"{k}={v}" for k, v in meta.items())
        fig.text(0.5, 0.005, subtitle, ha="center", va="bottom", fontsize=7, color="gray", wrap=True)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved chart -> {out_path}")
    if vloss:
        best_i = min(range(len(vloss)), key=lambda i: vloss[i])
        print(f"Best validation loss: {vloss[best_i]:.5f} at step {vsteps[best_i]}")


if __name__ == "__main__":
    main()
