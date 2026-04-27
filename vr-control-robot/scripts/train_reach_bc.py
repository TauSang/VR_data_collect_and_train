import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.models.gru_policy import SequenceGRUPolicy
from src.vrtrain.utils.config import load_yaml, resolve_path
from src.vrtrain.utils.seed import set_seed


class H5ReachDataset(Dataset):
    def __init__(self, h5_path: str, indices=None, mean=None, std=None):
        self._h5 = h5py.File(h5_path, "r")
        self.obs_seq = self._h5["obs_seq"]
        self.act = self._h5["act"]
        n = len(self.obs_seq)
        self.indices = list(range(n)) if indices is None else list(indices)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        obs = np.asarray(self.obs_seq[i], dtype=np.float32)
        if self.mean is not None and self.std is not None:
            obs = (obs - self.mean[None, :]) / self.std[None, :]
        act = np.asarray(self.act[i], dtype=np.float32)
        return torch.from_numpy(obs), torch.from_numpy(act)

    def close(self):
        if self._h5:
            self._h5.close()
            self._h5 = None


def split_by_episode(episode_ids: np.ndarray, val_ratio: float):
    uniq = sorted(np.unique(episode_ids).tolist())
    val_ep_n = max(1, int(round(len(uniq) * val_ratio)))
    if val_ep_n >= len(uniq):
        val_ep_n = max(1, len(uniq) - 1)
    val_eps = set(uniq[-val_ep_n:])
    train_idx = [i for i, ep in enumerate(episode_ids) if int(ep) not in val_eps]
    val_idx = [i for i, ep in enumerate(episode_ids) if int(ep) in val_eps]
    return train_idx, val_idx, sorted(val_eps)


def compute_norm_stats(h5_path: Path, indices: list[int]):
    with h5py.File(h5_path, "r") as h5:
        obs = np.asarray(h5["obs_seq"][indices], dtype=np.float32)
    flat = obs.reshape(-1, obs.shape[-1])
    mean = np.mean(flat, axis=0)
    std = np.std(flat, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    for obs_seq, act in loader:
        obs_seq = obs_seq.to(device)
        act = act.to(device)
        pred = model(obs_seq)
        loss = F.mse_loss(pred, act)
        total += float(loss.item())
        n += 1
    return total / max(1, n)


@torch.no_grad()
def inference(model, loader, device):
    model.eval()
    preds = []
    gts = []
    for obs_seq, act in loader:
        obs_seq = obs_seq.to(device)
        preds.append(model(obs_seq).cpu().numpy())
        gts.append(act.numpy())
    if not preds:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(preds, axis=0), np.concatenate(gts, axis=0)


def mse(a: np.ndarray, b: np.ndarray):
    return float(np.mean((a - b) ** 2))


def compute_baselines(h5_path: Path):
    with h5py.File(h5_path, "r") as h5:
        act = np.asarray(h5["act"], dtype=np.float32)

    zero = np.zeros_like(act)
    mean_action = np.mean(act, axis=0, keepdims=True)
    mean_pred = np.repeat(mean_action, repeats=act.shape[0], axis=0)
    return {
        "zero_action_mse": mse(zero, act),
        "mean_action_mse": mse(mean_pred, act),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reach_bc_task.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    train_h5 = resolve_path(args.config, cfg["data"]["train_h5_path"])
    test_h5_raw = cfg["data"].get("test_h5_path", None)
    test_h5 = resolve_path(args.config, test_h5_raw) if test_h5_raw else None

    if not train_h5.exists():
        raise FileNotFoundError(f"未找到 reach 训练集: {train_h5}，请先运行 build_reach_seq_hdf5.py")

    with h5py.File(train_h5, "r") as h5:
        obs_dim = int(h5.attrs["obs_dim"])
        act_dim = int(h5.attrs["act_dim"])
        episode_ids = np.asarray(h5["episode_id"], dtype=np.int32)

    train_idx, val_idx, val_eps = split_by_episode(episode_ids, float(cfg["train"].get("val_ratio", 0.2)))
    print(f"[reach-split] val_eps={val_eps}")

    mean, std = compute_norm_stats(train_h5, train_idx)

    exp_name = cfg.get("experiment", {}).get("name", "reach_bc")
    out_root = resolve_path(args.config, cfg.get("experiment", {}).get("out_dir", "./artifacts/experiments_reach"))
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / exp_name / f"run_{run_id}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "normalization_stats.json").open("w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "obs_dim": obs_dim}, f, ensure_ascii=False, indent=2)

    train_ds = H5ReachDataset(str(train_h5), indices=train_idx, mean=mean, std=std)
    val_ds = H5ReachDataset(str(train_h5), indices=val_idx, mean=mean, std=std)
    test_ds = None

    try:
        batch_size = int(cfg["train"].get("batch_size", 128))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = SequenceGRUPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
            num_layers=int(cfg["model"].get("num_layers", 2)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["train"].get("lr", 3e-4)),
            weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
        )

        best_val = float("inf")
        no_improve = 0
        history = []
        epochs = int(cfg["train"].get("epochs", 30))
        log_interval = int(cfg["train"].get("log_interval", 20))
        grad_clip_norm = float(cfg["train"].get("grad_clip_norm", 1.0))
        early_stop_patience = int(cfg["train"].get("early_stop_patience", 0))

        for epoch in range(1, epochs + 1):
            model.train()
            total = 0.0
            for step, (obs_seq, act) in enumerate(train_loader, start=1):
                obs_seq = obs_seq.to(device)
                act = act.to(device)
                pred = model(obs_seq)
                loss = F.mse_loss(pred, act)

                opt.zero_grad()
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                opt.step()

                total += float(loss.item())
                if step % log_interval == 0:
                    print(f"[reach-train] epoch={epoch} step={step} loss={loss.item():.6f}")

            train_loss = total / max(1, len(train_loader))
            val_loss = evaluate(model, val_loader, device)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            print(f"[reach-epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f}")

            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_dir / "best.pt")
            else:
                no_improve += 1

            if early_stop_patience > 0 and no_improve >= early_stop_patience:
                print(f"[reach-early-stop] epoch={epoch}")
                break

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(history)

        best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt["model"])

        test_result = None
        if test_h5 is not None and test_h5.exists():
            test_ds = H5ReachDataset(str(test_h5), indices=None, mean=mean, std=std)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            pred, gt = inference(model, test_loader, device)
            test_result = {
                "test_mse": mse(pred, gt),
                "baselines": compute_baselines(test_h5),
                "test_samples": int(len(test_ds)),
            }

        summary = {
            "run_id": run_id,
            "config": str(Path(args.config).resolve()),
            "train_h5": str(train_h5),
            "test_h5": str(test_h5) if test_h5 is not None else None,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "train_samples": int(len(train_ds)),
            "val_samples": int(len(val_ds)),
            "best_epoch": int(best_ckpt.get("epoch", -1)),
            "best_val_loss": float(best_ckpt.get("val_loss", -1.0)),
            "test_result": test_result,
            "created_at": dt.datetime.now().isoformat(),
        }
        with (run_dir / "result_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[reach-done] run_dir={run_dir}")

    finally:
        train_ds.close()
        val_ds.close()
        if test_ds is not None:
            test_ds.close()


if __name__ == "__main__":
    main()
