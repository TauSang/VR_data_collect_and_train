import argparse
import csv
import json
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act.scripts.train_act import ACTPolicy
from src.vrtrain.utils.config import load_yaml


class H5SequenceDataset(Dataset):
    def __init__(self, h5_path: str):
        self._h5 = h5py.File(h5_path, "r")
        self.obs_seq = self._h5["obs_seq"]
        self.act = self._h5["act"]

    def __len__(self):
        return len(self.obs_seq)

    def __getitem__(self, idx):
        return torch.from_numpy(self.obs_seq[idx]), torch.from_numpy(self.act[idx])

    def close(self):
        if self._h5:
            self._h5.close()
            self._h5 = None


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def compute_baselines(act: np.ndarray):
    zero = np.zeros_like(act)
    mean_action = np.mean(act, axis=0, keepdims=True)
    mean_pred = np.repeat(mean_action, repeats=act.shape[0], axis=0)

    return {
        "zero_action_mse": mse(zero, act),
        "mean_action_mse": mse(mean_pred, act),
    }


def load_model(cfg: dict, ckpt_path: Path, h5_path: Path, device: str):
    with h5py.File(h5_path, "r") as h5:
        obs_dim = int(h5.attrs["obs_dim"])
        act_dim = int(h5.attrs["act_dim"])

    mcfg = cfg["model"]
    model = ACTPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        d_model=int(mcfg.get("d_model", 128)),
        nhead=int(mcfg.get("nhead", 4)),
        num_layers=int(mcfg.get("num_layers", 3)),
        dim_feedforward=int(mcfg.get("dim_feedforward", 256)),
        dropout=float(mcfg.get("dropout", 0.1)),
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def collect_predictions(model, loader, device: str):
    preds = []
    gts = []
    with torch.no_grad():
        for obs_seq, act in loader:
            obs_seq = obs_seq.to(device)
            pred = model(obs_seq).cpu().numpy()
            preds.append(pred)
            gts.append(act.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(gts, axis=0)


def save_training_curve(metrics_path: Path, out_png: Path):
    with metrics_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("ACT Training Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_test_bar(act_mse: float, baselines: dict, out_png: Path):
    names = ["ACT", "Zero", "Mean"]
    values = [act_mse, baselines["zero_action_mse"], baselines["mean_action_mse"]]

    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(names, values)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("Test MSE")
    plt.title("ACT vs Baselines on collector2")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_joint_rmse(pred: np.ndarray, gt: np.ndarray, joint_names: list[str], out_png: Path):
    # each joint has xyz deltas (3 dims)
    rmse = np.sqrt(np.mean((pred - gt) ** 2, axis=0))
    joint_rmse = []
    for i, jn in enumerate(joint_names):
        seg = rmse[i * 3 : (i + 1) * 3]
        joint_rmse.append(float(np.mean(seg)))

    plt.figure(figsize=(10, 4.8))
    plt.bar(joint_names, joint_rmse)
    plt.ylabel("RMSE")
    plt.title("ACT Test RMSE by Joint (avg xyz)")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_bc_vs_act_compare(act_mse: float, bc_summary_path: Path, out_png: Path):
    with bc_summary_path.open("r", encoding="utf-8") as f:
        bc = json.load(f)
    bc_mse = float(bc.get("cross_test_mse", np.nan))

    names = ["BC", "ACT"]
    vals = [bc_mse, act_mse]

    plt.figure(figsize=(6, 4.5))
    bars = plt.bar(names, vals)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Test MSE (collector2)")
    plt.title("BC vs ACT Cross-dataset")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_pred_vs_gt_scatter(pred: np.ndarray, gt: np.ndarray, out_png: Path, max_points: int = 6000):
    # flatten all action dimensions
    p = pred.reshape(-1)
    g = gt.reshape(-1)

    n = p.shape[0]
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(np.int64)
        p = p[idx]
        g = g[idx]

    lo = float(min(np.min(p), np.min(g)))
    hi = float(max(np.max(p), np.max(g)))

    plt.figure(figsize=(5.6, 5.6))
    plt.scatter(g, p, s=7, alpha=0.35)
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y=x")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title("ACT Prediction vs Ground Truth")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def save_error_histogram(pred: np.ndarray, gt: np.ndarray, out_png: Path):
    err = (pred - gt).reshape(-1)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    plt.figure(figsize=(7.2, 4.8))
    plt.hist(err, bins=80, alpha=0.9)
    plt.axvline(0.0, color="r", linestyle="--", linewidth=1.2)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title(f"ACT Error Distribution (MAE={mae:.4f}, RMSE={rmse:.4f})")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="act/configs/base_act.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--bc-summary", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_dir = Path(args.run_dir)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"未找到 ACT 训练指标: {metrics_path}")

    save_training_curve(metrics_path, figure_dir / "act_training_curve.png")

    test_h5 = Path(cfg["data"]["test_h5_path"])
    ckpt_path = Path(args.ckpt) if args.ckpt else (run_dir / "checkpoints" / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(cfg, ckpt_path, test_h5, device)

    ds = H5SequenceDataset(str(test_h5))
    try:
        loader = DataLoader(ds, batch_size=int(cfg["train"].get("batch_size", 128)), shuffle=False, num_workers=0)
        pred, gt = collect_predictions(model, loader, device)
    finally:
        ds.close()

    act_mse = mse(pred, gt)
    baselines = compute_baselines(gt)

    joint_names = list(cfg["data"]["joint_names"])

    save_test_bar(act_mse, baselines, figure_dir / "act_vs_baseline_bar.png")
    save_joint_rmse(pred, gt, joint_names, figure_dir / "act_joint_rmse.png")
    save_pred_vs_gt_scatter(pred, gt, figure_dir / "act_pred_vs_gt_scatter.png")
    save_error_histogram(pred, gt, figure_dir / "act_error_hist.png")

    if args.bc_summary:
        bc_summary_path = Path(args.bc_summary)
        if bc_summary_path.exists():
            save_bc_vs_act_compare(act_mse, bc_summary_path, figure_dir / "bc_vs_act_bar.png")

    summary = {
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_val_loss": float(ckpt.get("val_loss", -1.0)),
        "test_mse": float(act_mse),
        "baselines": baselines,
    }

    with (run_dir / "result_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (run_dir / "result_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["best_epoch", "best_val_loss", "test_mse", "zero_action_mse", "mean_action_mse"])
        w.writerow(
            [
                summary["best_epoch"],
                f"{summary['best_val_loss']:.8f}",
                f"{summary['test_mse']:.8f}",
                f"{baselines['zero_action_mse']:.8f}",
                f"{baselines['mean_action_mse']:.8f}",
            ]
        )

    print(f"[fig] saved: {figure_dir / 'act_training_curve.png'}")
    print(f"[fig] saved: {figure_dir / 'act_vs_baseline_bar.png'}")
    print(f"[fig] saved: {figure_dir / 'act_joint_rmse.png'}")
    print(f"[fig] saved: {figure_dir / 'act_pred_vs_gt_scatter.png'}")
    print(f"[fig] saved: {figure_dir / 'act_error_hist.png'}")
    if args.bc_summary and Path(args.bc_summary).exists():
        print(f"[fig] saved: {figure_dir / 'bc_vs_act_bar.png'}")
    print(f"[summary] saved: {run_dir / 'result_summary.json'}")


if __name__ == "__main__":
    main()
