import argparse
from pathlib import Path
import sys

import h5py
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from act.scripts.train_act import ACTPolicy  # 复用模型定义
from src.vrtrain.utils.config import load_yaml


class H5SequenceDataset(Dataset):
    def __init__(self, h5_path: str):
        self._h5 = h5py.File(h5_path, "r")
        self.obs_seq = self._h5["obs_seq"]
        self.act = self._h5["act"]

    def __len__(self):
        return len(self.obs_seq)

    def __getitem__(self, idx):
        return torch.from_numpy(self.obs_seq[idx]), torch.from_numpy(self.act[idx])

    def close(self):
        if self._h5:
            self._h5.close()
            self._h5 = None


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse = 0.0
    n = 0
    for obs_seq, act in loader:
        obs_seq = obs_seq.to(device)
        act = act.to(device)
        pred = model(obs_seq)
        loss = torch.mean((pred - act) ** 2)
        mse += float(loss.item())
        n += 1
    return mse / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="act/configs/base_act.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset", choices=["train", "test"], default="test")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    h5_path = Path(cfg["data"]["test_h5_path"] if args.dataset == "test" else cfg["data"]["train_h5_path"])
    if not h5_path.exists():
        raise FileNotFoundError(f"未找到数据集: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    ds = H5SequenceDataset(str(h5_path))
    try:
        loader = DataLoader(ds, batch_size=int(cfg["train"].get("batch_size", 128)), shuffle=False, num_workers=0)
        mse = evaluate(model, loader, device)
        print(f"[act-eval] dataset={args.dataset} mse={mse:.6f}")
    finally:
        ds.close()


if __name__ == "__main__":
    main()
