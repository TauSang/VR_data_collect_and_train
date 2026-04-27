import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="./artifacts/checkpoints_smoke20/metrics.json")
    parser.add_argument("--out", default="./artifacts/checkpoints_smoke20/training_curve.png")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    if not rows:
        raise RuntimeError("metrics 为空，先训练再画图")

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Behavior Cloning Training Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved: {out_path}")


if __name__ == "__main__":
    main()
