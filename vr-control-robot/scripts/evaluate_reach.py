import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.models.gru_policy import SequenceGRUPolicy
from src.vrtrain.utils.config import load_yaml, resolve_path
from train_reach_bc import H5ReachDataset, compute_baselines, compute_norm_stats, inference, mse, split_by_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reach_bc_task.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", choices=["train", "test"], default="test")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_h5 = resolve_path(args.config, cfg["data"]["train_h5_path"])
    data_h5 = train_h5 if args.dataset == "train" else resolve_path(args.config, cfg["data"]["test_h5_path"])

    if not data_h5.exists():
        raise FileNotFoundError(f"未找到数据集: {data_h5}")

    with h5py.File(train_h5, "r") as h5:
        train_episode_ids = np.asarray(h5["episode_id"], dtype=np.int32)
        obs_dim = int(h5.attrs["obs_dim"])
        act_dim = int(h5.attrs["act_dim"])

    train_idx, _, _ = split_by_episode(train_episode_ids, float(cfg["train"].get("val_ratio", 0.2)))
    mean, std = compute_norm_stats(train_h5, train_idx)

    model = SequenceGRUPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=int(cfg["model"].get("hidden_dim", 256)),
        num_layers=int(cfg["model"].get("num_layers", 2)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    ds = H5ReachDataset(str(data_h5), indices=None, mean=mean, std=std)
    try:
        loader = DataLoader(ds, batch_size=int(cfg["train"].get("batch_size", 128)), shuffle=False, num_workers=0)
        pred, gt = inference(model, loader, device)
        result = {
            "dataset": args.dataset,
            "mse": mse(pred, gt),
            "samples": int(len(ds)),
            "baselines": compute_baselines(data_h5),
        }
        print(result)
    finally:
        ds.close()


if __name__ == "__main__":
    main()
