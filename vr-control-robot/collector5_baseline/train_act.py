import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import ensure_dir, load_config, load_episodes, save_json, set_seed, split_by_episode


class ACTEncoder(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, act_dim))

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(obs_seq)
        z = self.encoder(x)
        return self.head(z[:, -1, :])


def make_seq_dataset(episodes, seq_len: int):
    xs = []
    ys = []
    for ep in episodes:
        T = ep.obs.shape[0]
        if T < seq_len:
            continue
        for end in range(seq_len - 1, T):
            start = end - seq_len + 1
            xs.append(ep.obs[start : end + 1])
            ys.append(ep.act[end])
    if not xs:
        raise RuntimeError("No sequence samples. Check seq_len/data.")
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def evaluate(model, loader, device):
    model.eval()
    losses = []
    maes = []
    with torch.no_grad():
        for obs_seq, act in loader:
            obs_seq = obs_seq.to(device)
            act = act.to(device)
            pred = model(obs_seq)
            loss = F.mse_loss(pred, act)
            mae = torch.mean(torch.abs(pred - act))
            losses.append(float(loss.item()))
            maes.append(float(mae.item()))
    return float(np.mean(losses)), float(np.mean(maes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)

    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    data_cfg = cfg["data"]
    seq_len = int(data_cfg["seq_len"])

    episodes = load_episodes(
        root / data_cfg["episodes_jsonl"],
        joint_names=list(data_cfg["joint_names"]),
        use_joint_velocities=bool(data_cfg["use_joint_velocities"]),
    )
    train_eps, val_eps = split_by_episode(episodes, float(data_cfg["train_split"]))

    x_train, y_train = make_seq_dataset(train_eps, seq_len)
    x_val, y_val = make_seq_dataset(val_eps, seq_len)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    obs_dim = x_train.shape[-1]
    act_dim = y_train.shape[-1]
    m = cfg["act"]
    model = ACTEncoder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        d_model=int(m["d_model"]),
        nhead=int(m["nhead"]),
        num_layers=int(m["num_layers"]),
        dim_feedforward=int(m["dim_feedforward"]),
        dropout=float(m["dropout"]),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    out_root = root / args.out
    run_dir = out_root / "act" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    epochs = int(cfg["train"]["epochs"])
    history = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for obs_seq, act in train_loader:
            obs_seq = obs_seq.to(device)
            act = act.to(device)
            pred = model(obs_seq)
            loss = F.mse_loss(pred, act)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss, val_mae = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "val_mse": val_loss,
                "val_mae": val_mae,
            }
        )
        print(f"[ACT] epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f} val_mae={val_mae:.6f}")

        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "best.pt")

    save_json(
        run_dir / "metrics.json",
        {
            "model": "ACTEncoder",
            "seq_len": seq_len,
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "num_train_episodes": len(train_eps),
            "num_val_episodes": len(val_eps),
            "num_train_sequences": int(x_train.shape[0]),
            "num_val_sequences": int(x_val.shape[0]),
            "history": history,
        },
    )

    epochs_x = [h["epoch"] for h in history]
    train_mse = [h["train_mse"] for h in history]
    val_mse = [h["val_mse"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_x, train_mse, label="train_mse")
    plt.plot(epochs_x, val_mse, label="val_mse")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("ACT Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=140)
    plt.close()

    model.eval()
    with torch.no_grad():
        n = min(2000, x_val.shape[0])
        pred = model(torch.from_numpy(x_val[:n]).to(device)).cpu().numpy()
    gt = y_val[:n]

    plt.figure(figsize=(6, 6))
    plt.scatter(gt.reshape(-1), pred.reshape(-1), s=2, alpha=0.2)
    lo = float(min(gt.min(), pred.min()))
    hi = float(max(gt.max(), pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("ground truth")
    plt.ylabel("prediction")
    plt.title("ACT Prediction Scatter (Val)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "pred_scatter.png", dpi=140)
    plt.close()

    print(f"[ACT] done -> {run_dir}")


if __name__ == "__main__":
    main()
