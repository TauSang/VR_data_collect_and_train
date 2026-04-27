import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.models.gru_policy import SequenceGRUPolicy
from src.vrtrain.utils.config import load_yaml, resolve_path


class NormalizedReachPolicy(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, obs_seq):
        x = (obs_seq - self.mean[None, None, :]) / self.std[None, None, :]
        return self.model(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reach_bc_task.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_h5 = resolve_path(args.config, cfg["data"]["train_h5_path"])
    ckpt_path = Path(args.ckpt).resolve()
    stats_path = Path(args.stats).resolve() if args.stats else ckpt_path.parents[1] / "normalization_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"未找到 normalization stats: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)

    with h5py.File(train_h5, "r") as h5:
        obs_dim = int(h5.attrs["obs_dim"])
        act_dim = int(h5.attrs["act_dim"])
        seq_len = int(h5.attrs["seq_len"])

    model = SequenceGRUPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        num_layers=int(cfg["model"].get("num_layers", 2)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    wrapped = NormalizedReachPolicy(model, mean=mean, std=std).eval()
    example = torch.zeros(1, seq_len, obs_dim, dtype=torch.float32)
    traced = torch.jit.trace(wrapped, example, check_trace=False)

    out_path = Path(args.out).resolve() if args.out else resolve_path(args.config, cfg["export"]["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"[reach-export] saved: {out_path}")


if __name__ == "__main__":
    main()
