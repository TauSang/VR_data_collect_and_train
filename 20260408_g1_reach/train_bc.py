import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from common import (
    build_action_names,
    build_feature_names,
    denormalize_act,
    ensure_dir,
    fit_normalizer,
    load_config,
    load_segments,
    make_frame_dataset,
    save_json,
    set_seed,
    split_segments_by_episode,
    summarize_segments,
)


class TaskBCMLP(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, act_dim)
        self.success_head = nn.Linear(in_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(obs)
        return self.action_head(feat), self.success_head(feat).squeeze(-1)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-6)


@torch.no_grad()
def evaluate(model, loader, device, success_loss_weight: float):
    model.eval()
    total_losses, action_mses, action_maes, success_bces, success_accs = [], [], [], [], []
    for obs, act, success, weights in loader:
        obs, act, success, weights = obs.to(device), act.to(device), success.to(device), weights.to(device)
        pred_act, pred_success = model(obs)
        mse_each = torch.mean((pred_act - act) ** 2, dim=1)
        mae_each = torch.mean(torch.abs(pred_act - act), dim=1)
        bce_each = F.binary_cross_entropy_with_logits(pred_success, success, reduction="none")
        total_loss = _weighted_mean(mse_each, weights) + success_loss_weight * _weighted_mean(bce_each, weights)
        success_acc = torch.mean(((pred_success > 0).float() == success).float())
        total_losses.append(float(total_loss.item()))
        action_mses.append(float(torch.mean(mse_each).item()))
        action_maes.append(float(torch.mean(mae_each).item()))
        success_bces.append(float(torch.mean(bce_each).item()))
        success_accs.append(float(success_acc.item()))
    return {
        "total_loss": float(np.mean(total_losses)),
        "action_mse": float(np.mean(action_mses)),
        "action_mae": float(np.mean(action_maes)),
        "success_bce": float(np.mean(success_bces)),
        "success_acc": float(np.mean(success_accs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config)
    g1_joints = list(cfg["data"]["g1_joint_names"])

    set_seed(int(cfg["train"]["seed"]))
    segments, data_summary = load_segments(cfg)
    train_segments, val_segments, split_summary = split_segments_by_episode(
        segments, train_split=float(cfg["data"]["train_split"]), seed=int(cfg["train"]["seed"]),
    )
    normalizer = fit_normalizer(train_segments)

    x_train, y_train, s_train, w_train, _ = make_frame_dataset(
        train_segments, normalizer=normalizer,
        obs_clip_z=float(cfg["data"]["obs_clip_z"]),
        act_clip_z=float(cfg["data"]["act_clip_z"]),
    )
    x_val, y_val, s_val, w_val, _ = make_frame_dataset(
        val_segments, normalizer=normalizer,
        obs_clip_z=float(cfg["data"]["obs_clip_z"]),
        act_clip_z=float(cfg["data"]["act_clip_z"]),
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(s_train), torch.from_numpy(w_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val), torch.from_numpy(s_val), torch.from_numpy(w_val))

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    obs_dim = int(x_train.shape[1])
    act_dim = int(y_train.shape[1])
    feature_names = build_feature_names(g1_joints)
    action_names = build_action_names(g1_joints)

    assert obs_dim == len(g1_joints) * 2 + 15, f"Expected obs_dim={len(g1_joints)*2+15}, got {obs_dim}"
    assert act_dim == len(g1_joints), f"Expected act_dim={len(g1_joints)}, got {act_dim}"

    model = TaskBCMLP(
        obs_dim=obs_dim, act_dim=act_dim,
        hidden_dims=list(cfg["bc"]["hidden_dims"]),
        dropout=float(cfg["bc"].get("dropout", 0.1)),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]))
    success_loss_weight = float(cfg["train"].get("success_loss_weight", 0.2))
    grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 1.0))

    out_root = root / args.out
    run_dir = out_root / "bc" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("early_stop_patience", 12))
    best_val_mse = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_total_losses, train_action_mses, train_success_bces, train_success_accs = [], [], [], []

        for obs, act, success, weights in train_loader:
            obs, act, success, weights = obs.to(device), act.to(device), success.to(device), weights.to(device)
            pred_act, pred_success = model(obs)
            mse_each = torch.mean((pred_act - act) ** 2, dim=1)
            bce_each = F.binary_cross_entropy_with_logits(pred_success, success, reduction="none")
            total_loss = _weighted_mean(mse_each, weights) + success_loss_weight * _weighted_mean(bce_each, weights)

            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            opt.step()

            success_acc = torch.mean(((pred_success > 0).float() == success).float())
            train_total_losses.append(float(total_loss.item()))
            train_action_mses.append(float(torch.mean(mse_each).item()))
            train_success_bces.append(float(torch.mean(bce_each).item()))
            train_success_accs.append(float(success_acc.item()))

        val_metrics = evaluate(model, val_loader, device, success_loss_weight)
        row = {
            "epoch": epoch,
            "train_total_loss": float(np.mean(train_total_losses)),
            "train_action_mse": float(np.mean(train_action_mses)),
            "train_success_bce": float(np.mean(train_success_bces)),
            "train_success_acc": float(np.mean(train_success_accs)),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"[BC] epoch={epoch:03d}"
            f" train_mse={row['train_action_mse']:.6f}"
            f" val_mse={row['val_action_mse']:.6f}"
            f" val_mae={row['val_action_mae']:.6f}"
            f" val_succ_acc={row['val_success_acc']:.4f}"
        )

        checkpoint = {"model": model.state_dict(), "epoch": epoch, "normalizer": normalizer.to_dict(), "config": cfg}
        torch.save(checkpoint, ckpt_dir / "last.pt")

        if row["val_action_mse"] < best_val_mse:
            best_val_mse = row["val_action_mse"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(checkpoint, ckpt_dir / "best.pt")
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(f"[BC] early stop at epoch={epoch} (patience={patience})")
            break

    # ── Evaluation on best model ──
    best_state = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_state["model"])
    model.eval()
    with torch.no_grad():
        n = min(4000, x_val.shape[0])
        pred_z, pred_success = model(torch.from_numpy(x_val[:n]).to(device))
        pred = denormalize_act(pred_z.cpu().numpy(), normalizer)
        gt = denormalize_act(y_val[:n], normalizer)
        success_prob = torch.sigmoid(pred_success).cpu().numpy()

    save_json(run_dir / "metrics.json", {
        "model": "TaskBCMLP",
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "feature_names": feature_names,
        "action_names": action_names,
        "best_epoch": best_epoch,
        "best_val_action_mse": best_val_mse,
        "num_train_frames": int(x_train.shape[0]),
        "num_val_frames": int(x_val.shape[0]),
        "data_summary": data_summary,
        "train_summary": summarize_segments(train_segments),
        "val_summary": summarize_segments(val_segments),
        "split_summary": split_summary,
        "normalizer": normalizer.to_dict(),
        "history": history,
    })

    # ── Plots ──
    epochs_x = [h["epoch"] for h in history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_x, [h["train_action_mse"] for h in history], label="train")
    plt.plot(epochs_x, [h["val_action_mse"] for h in history], label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title("BC Action Loss"); plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_x, [h["train_success_acc"] for h in history], label="train")
    plt.plot(epochs_x, [h["val_success_acc"] for h in history], label="val")
    plt.xlabel("epoch"); plt.ylabel("Accuracy"); plt.title("BC Success Aux"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(run_dir / "loss_curve.png", dpi=140); plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(gt.reshape(-1), pred.reshape(-1), s=2, alpha=0.2)
    lo, hi = float(min(gt.min(), pred.min())), float(max(gt.max(), pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("ground truth"); plt.ylabel("predicted"); plt.title("BC Pred vs GT (Val, Best)")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(run_dir / "pred_scatter.png", dpi=140); plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(success_prob[s_val[:n] == 0], bins=30, alpha=0.6, label="timeout", density=True)
    plt.hist(success_prob[s_val[:n] == 1], bins=30, alpha=0.6, label="success", density=True)
    plt.xlabel("P(success)"); plt.ylabel("density"); plt.title("BC Success Head (Val, Best)")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.savefig(run_dir / "success_head_hist.png", dpi=140); plt.close()

    print(f"[BC] done -> {run_dir}")


if __name__ == "__main__":
    main()
