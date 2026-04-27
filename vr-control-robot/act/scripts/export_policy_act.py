import argparse
from pathlib import Path
import sys

import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act.scripts.train_act import ACTPolicy
from src.vrtrain.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="act/configs/base_act.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_h5 = Path(cfg["data"]["train_h5_path"])

    with h5py.File(train_h5, "r") as h5:
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

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_path = Path(args.out or cfg.get("export", {}).get("out_path", "./artifacts/exported/policy_act.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    example = torch.randn(1, int(cfg["data"].get("seq_len", 16)), obs_dim)
    traced = torch.jit.trace(model, example, check_trace=False)
    traced.save(str(out_path))
    print(f"[act-export] saved: {out_path}")


if __name__ == "__main__":
    main()
