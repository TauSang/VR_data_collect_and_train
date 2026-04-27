import argparse
from pathlib import Path
import sys

import h5py
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.utils.config import load_yaml
from src.vrtrain.data.dataset import H5BehaviorCloningDataset
from src.vrtrain.models.mlp_policy import MLPPolicy
from src.vrtrain.trainers.bc_trainer import evaluate_bc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--ckpt", default="./artifacts/checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    h5_path = Path(cfg["data"]["h5_path"])

    with h5py.File(h5_path, "r") as h5:
        obs_dim = int(h5.attrs["obs_dim"])
        act_dim = int(h5.attrs["act_dim"])
        n = int(h5.attrs["num_samples"])

    ds = H5BehaviorCloningDataset(str(h5_path), list(range(n)))
    loader = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)

    model = MLPPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=tuple(cfg["model"].get("hidden_dims", [256, 256])),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    mse = evaluate_bc(model, loader, device)
    print(f"[eval] mse={mse:.6f}")
    ds.close()


if __name__ == "__main__":
    main()
