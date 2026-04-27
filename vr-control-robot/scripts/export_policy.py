import argparse
from pathlib import Path
import sys

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.utils.config import load_yaml
from src.vrtrain.models.mlp_policy import MLPPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--ckpt", default="./artifacts/checkpoints/best.pt")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    h5_path = Path(cfg["data"]["h5_path"])
    out_path = Path(cfg["export"]["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        obs_dim = int(h5.attrs["obs_dim"])
        act_dim = int(h5.attrs["act_dim"])

    model = MLPPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=tuple(cfg["model"].get("hidden_dims", [256, 256])),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    example = torch.randn(1, obs_dim)
    traced = torch.jit.trace(model, example)
    traced.save(str(out_path))
    print(f"[export] saved: {out_path}")


if __name__ == "__main__":
    main()
