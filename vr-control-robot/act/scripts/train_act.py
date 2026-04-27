import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.utils.config import load_yaml
from src.vrtrain.utils.seed import set_seed


class H5SequenceDataset(Dataset):
    def __init__(self, h5_path: str, indices=None):
        self._h5 = h5py.File(h5_path, "r")
        self.obs_seq = self._h5["obs_seq"]
        self.act = self._h5["act"]
        n = len(self.obs_seq)
        self.indices = list(range(n)) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return torch.from_numpy(self.obs_seq[i]), torch.from_numpy(self.act[i])

    def close(self):
        if self._h5:
            self._h5.close()
            self._h5 = None


class ACTPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, d_model=128, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(obs_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, act_dim))

    def forward(self, obs_seq):
        # obs_seq: [B, T, obs_dim]
        x = self.input_proj(obs_seq)
        x = self.encoder(x)
        last = x[:, -1, :]
        return self.head(last)


def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for obs_seq, act in loader:
            obs_seq = obs_seq.to(device)
            act = act.to(device)
            pred = model(obs_seq)
            loss = F.mse_loss(pred, act)
            total += float(loss.item())
            n += 1
    return total / max(1, n)


def split_train_val_by_episode(episode_ids: np.ndarray, val_ratio: float):
    uniq = sorted(np.unique(episode_ids).tolist())
    val_ep_n = max(1, int(round(len(uniq) * val_ratio)))
    if val_ep_n >= len(uniq):
        val_ep_n = max(1, len(uniq) - 1)
    val_eps = set(uniq[-val_ep_n:])
    train_idx = [i for i, ep in enumerate(episode_ids) if int(ep) not in val_eps]
    val_idx = [i for i, ep in enumerate(episode_ids) if int(ep) in val_eps]
    return train_idx, val_idx, sorted(val_eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="act/configs/base_act.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    data_cfg = cfg["data"]
    train_h5 = Path(data_cfg["train_h5_path"])
    if not train_h5.exists():
        raise FileNotFoundError(f"未找到序列训练集: {train_h5}，请先运行 convert_jsonl_to_seq_hdf5.py")

    with h5py.File(train_h5, "r") as h5:
        obs_dim = int(h5.attrs["obs_dim"])
        act_dim = int(h5.attrs["act_dim"])
        episode_ids = np.asarray(h5["episode_id"]).astype(np.int32)

    train_idx, val_idx, val_eps = split_train_val_by_episode(episode_ids, val_ratio=0.2)
    print(f"[act-split] val_eps={val_eps}")

    train_ds = H5SequenceDataset(str(train_h5), train_idx)
    val_ds = H5SequenceDataset(str(train_h5), val_idx)

    try:
        batch_size = int(cfg["train"].get("batch_size", 128))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        tcfg = cfg["train"]
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(tcfg.get("lr", 3e-4)),
            weight_decay=float(tcfg.get("weight_decay", 1e-5)),
        )

        exp_name = cfg.get("experiment", {}).get("name", "act")
        out_root = Path(cfg.get("experiment", {}).get("out_dir", "./artifacts/experiments_act"))
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = out_root / exp_name / f"run_{run_id}"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        history = []
        best_val = float("inf")
        epochs = int(tcfg.get("epochs", 20))
        log_interval = int(tcfg.get("log_interval", 20))

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
                opt.step()

                total += float(loss.item())
                if step % log_interval == 0:
                    print(f"[act-train] epoch={epoch} step={step} loss={loss.item():.6f}")

            train_loss = total / max(1, len(train_loader))
            val_loss = evaluate(model, val_loader, device)
            print(f"[act-epoch {epoch}] train={train_loss:.6f} val={val_loss:.6f}")

            history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": float(val_loss)})

            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_dir / "best.pt")

        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        print(f"[act-done] run_dir={run_dir}")

    finally:
        train_ds.close()
        val_ds.close()


if __name__ == "__main__":
    main()
