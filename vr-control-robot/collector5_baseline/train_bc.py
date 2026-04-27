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


class BCMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [obs_dim] + hidden_dims + [act_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def make_frame_dataset(episodes):
    obs = np.concatenate([ep.obs for ep in episodes], axis=0)
    act = np.concatenate([ep.act for ep in episodes], axis=0)
    return obs, act


def evaluate(model, loader, device):
    model.eval()
    losses = []
    maes = []
    with torch.no_grad():
        for obs, act in loader:
            obs = obs.to(device)
            act = act.to(device)
            pred = model(obs)
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
    episodes = load_episodes(
        root / data_cfg["episodes_jsonl"],
        joint_names=list(data_cfg["joint_names"]),
        use_joint_velocities=bool(data_cfg["use_joint_velocities"]),
    )
    train_eps, val_eps = split_by_episode(episodes, float(data_cfg["train_split"]))

    x_train, y_train = make_frame_dataset(train_eps)
    x_val, y_val = make_frame_dataset(val_eps)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    obs_dim = x_train.shape[1]
    act_dim = y_train.shape[1]

    model = BCMLP(obs_dim, act_dim, hidden_dims=list(cfg["bc"]["hidden_dims"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    out_root = root / args.out
    run_dir = out_root / "bc" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    epochs = int(cfg["train"]["epochs"])
    history = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for obs, act in train_loader:
            obs = obs.to(device)
            act = act.to(device)
            pred = model(obs)
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
        print(f"[BC] epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f} val_mae={val_mae:.6f}")

        torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch}, ckpt_dir / "best.pt")

    save_json(
        run_dir / "metrics.json",
        {
            "model": "BCMLP",
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "num_train_episodes": len(train_eps),
            "num_val_episodes": len(val_eps),
            "num_train_frames": int(x_train.shape[0]),
            "num_val_frames": int(x_val.shape[0]),
            "history": history,
        },
    )

    # plot
    epochs_x = [h["epoch"] for h in history]
    train_mse = [h["train_mse"] for h in history]
    val_mse = [h["val_mse"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_x, train_mse, label="train_mse")
    plt.plot(epochs_x, val_mse, label="val_mse")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("BC Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=140)
    plt.close()

    # prediction scatter (best-effort with last model)
    model.eval()
    with torch.no_grad():
        n = min(4000, x_val.shape[0])
        obs_t = torch.from_numpy(x_val[:n]).to(device)
        pred = model(obs_t).cpu().numpy()
    gt = y_val[:n]

    plt.figure(figsize=(6, 6))
    plt.scatter(gt.reshape(-1), pred.reshape(-1), s=2, alpha=0.2)
    lo = float(min(gt.min(), pred.min()))
    hi = float(max(gt.max(), pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("ground truth")
    plt.ylabel("prediction")
    plt.title("BC Prediction Scatter (Val)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "pred_scatter.png", dpi=140)
    plt.close()

    print(f"[BC] done -> {run_dir}")


if __name__ == "__main__":
    main()
