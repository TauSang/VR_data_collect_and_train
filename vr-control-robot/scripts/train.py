import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sys
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.utils.config import load_yaml
from src.vrtrain.utils.seed import set_seed
from src.vrtrain.data.dataset import H5BehaviorCloningDataset
from src.vrtrain.models.mlp_policy import MLPPolicy
from src.vrtrain.trainers.bc_trainer import train_bc


def prepare_run_cfg(cfg: dict, config_path: str, out_root: str, run_name: Optional[str], legacy_static_output: bool):
    """
    统一训练输出目录策略：
    - 默认：每次训练写入新的 run_时间戳 目录，避免覆盖
    - 可选：legacy 模式沿用配置中的固定 ckpt_dir
    """
    runtime_cfg = copy.deepcopy(cfg)

    if legacy_static_output:
        ckpt_dir = Path(runtime_cfg["train"]["ckpt_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        metrics_json = Path(runtime_cfg["train"].get("metrics_json", ckpt_dir / "metrics.json"))
        metrics_csv = Path(runtime_cfg["train"].get("metrics_csv", ckpt_dir / "metrics.csv"))
        return runtime_cfg, ckpt_dir, metrics_json, metrics_csv, None

    cfg_stem = Path(config_path).stem
    exp_name = run_name or cfg_stem
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = Path(out_root) / exp_name
    run_dir = exp_root / f"run_{run_id}"
    ckpt_dir = run_dir / "checkpoints"
    metrics_json = run_dir / "metrics.json"
    metrics_csv = run_dir / "metrics.csv"

    runtime_cfg["train"]["ckpt_dir"] = str(ckpt_dir)
    runtime_cfg["train"]["metrics_json"] = str(metrics_json)
    runtime_cfg["train"]["metrics_csv"] = str(metrics_csv)

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": dt.datetime.now().isoformat(),
                "config_path": str(config_path),
                "exp_name": exp_name,
                "run_id": run_id,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(runtime_cfg, f, ensure_ascii=False, indent=2)

    latest_file = exp_root / "latest_run.txt"
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(str(run_dir), encoding="utf-8")

    return runtime_cfg, ckpt_dir, metrics_json, metrics_csv, run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out-root", default="./artifacts/experiments")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--legacy-static-output", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg, ckpt_dir, metrics_json, metrics_csv, run_dir = prepare_run_cfg(
        cfg,
        config_path=args.config,
        out_root=args.out_root,
        run_name=args.run_name,
        legacy_static_output=bool(args.legacy_static_output),
    )

    print(f"[output] ckpt_dir={ckpt_dir}")
    print(f"[output] metrics_json={metrics_json}")
    print(f"[output] metrics_csv={metrics_csv}")
    if run_dir is not None:
        print(f"[output] run_dir={run_dir}")

    set_seed(int(cfg.get("seed", 42)))

    h5_path = Path(cfg["data"]["h5_path"])
    if not h5_path.exists():
        raise FileNotFoundError(f"未找到训练集: {h5_path}，请先运行 convert_jsonl_to_hdf5.py")

    with h5py.File(h5_path, "r") as h5:
        obs_dim = int(h5.attrs["obs_dim"])
        act_dim = int(h5.attrs["act_dim"])
        n = int(h5.attrs["num_samples"])
        episode_ids = np.asarray(h5["episode_id"]).astype(np.int32)

    split_by_episode = bool(cfg["train"].get("split_by_episode", True))
    val_ratio = float(cfg["train"].get("val_ratio", 0.1))

    if split_by_episode:
        uniq_eps = sorted(np.unique(episode_ids).tolist())
        val_ep_n = max(1, int(round(len(uniq_eps) * val_ratio)))
        if val_ep_n >= len(uniq_eps):
            val_ep_n = max(1, len(uniq_eps) - 1)
        val_eps = set(uniq_eps[-val_ep_n:])
        train_idx = [i for i, ep in enumerate(episode_ids) if int(ep) not in val_eps]
        val_idx = [i for i, ep in enumerate(episode_ids) if int(ep) in val_eps]
        print(f"[split] mode=episode, total_eps={len(uniq_eps)}, val_eps={sorted(val_eps)}")
    else:
        val_n = max(1, int(n * val_ratio))
        train_n = n - val_n
        indices = list(range(n))
        train_idx = indices[:train_n]
        val_idx = indices[train_n:]
        print(f"[split] mode=frame, val_ratio={val_ratio}")

    train_ds = H5BehaviorCloningDataset(str(h5_path), train_idx)
    val_ds = H5BehaviorCloningDataset(str(h5_path), val_idx)

    batch_size = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MLPPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dims=tuple(cfg["model"].get("hidden_dims", [256, 256])),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}, samples={n}, train={len(train_ds)}, val={len(val_ds)}")
    train_bc(model, train_loader, val_loader, cfg, device)

    train_ds.close()
    val_ds.close()


if __name__ == "__main__":
    main()
