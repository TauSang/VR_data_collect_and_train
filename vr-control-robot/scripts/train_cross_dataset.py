import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import sys

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
from src.vrtrain.data.vectorize import frame_to_obs_action
from src.vrtrain.models.mlp_policy import MLPPolicy
from src.vrtrain.trainers.bc_trainer import evaluate_bc, train_bc


def convert_jsonl_to_h5(raw_jsonl: Path, out_h5: Path, joint_names: list[str], use_joint_velocities: bool, include_gripper: bool):
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    obs_all = []
    act_all = []
    episode_ids = []

    with raw_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            ep = int(frame.get("episodeId", 0))
            if ep <= 0:
                continue

            obs_vec, act_vec = frame_to_obs_action(frame, joint_names, use_joint_velocities, include_gripper)
            obs_all.append(obs_vec)
            act_all.append(act_vec)
            episode_ids.append(ep)

    if not obs_all:
        raise RuntimeError(f"没有可用样本: {raw_jsonl}")

    obs_np = np.stack(obs_all, axis=0)
    act_np = np.stack(act_all, axis=0)
    ep_np = np.asarray(episode_ids, dtype=np.int32)

    with h5py.File(out_h5, "w") as h5:
        h5.create_dataset("obs", data=obs_np)
        h5.create_dataset("act", data=act_np)
        h5.create_dataset("episode_id", data=ep_np)
        h5.attrs["obs_dim"] = obs_np.shape[1]
        h5.attrs["act_dim"] = act_np.shape[1]
        h5.attrs["num_samples"] = obs_np.shape[0]
        h5.attrs["joint_names_json"] = json.dumps(joint_names, ensure_ascii=False)

    print(f"[convert] {raw_jsonl} -> {out_h5}")
    print(f"[convert] samples={obs_np.shape[0]} obs_dim={obs_np.shape[1]} act_dim={act_np.shape[1]}")


def maybe_prepare_h5(raw_jsonl: Path, out_h5: Path, joint_names: list[str], use_joint_velocities: bool, include_gripper: bool, rebuild: bool):
    if out_h5.exists() and not rebuild:
        print(f"[convert] skip existing: {out_h5}")
        return
    convert_jsonl_to_h5(raw_jsonl, out_h5, joint_names, use_joint_velocities, include_gripper)


def split_train_val_indices(episode_ids: np.ndarray, val_ratio: float, split_by_episode: bool):
    n = int(len(episode_ids))
    if n <= 1:
        return [0], [0]

    if split_by_episode:
        uniq_eps = sorted(np.unique(episode_ids).tolist())
        val_ep_n = max(1, int(round(len(uniq_eps) * val_ratio)))
        if val_ep_n >= len(uniq_eps):
            val_ep_n = max(1, len(uniq_eps) - 1)
        val_eps = set(uniq_eps[-val_ep_n:])
        train_idx = [i for i, ep in enumerate(episode_ids) if int(ep) not in val_eps]
        val_idx = [i for i, ep in enumerate(episode_ids) if int(ep) in val_eps]
        print(f"[split] mode=episode total_eps={len(uniq_eps)} val_eps={sorted(val_eps)}")
        return train_idx, val_idx

    val_n = max(1, int(n * val_ratio))
    train_n = n - val_n
    idx = list(range(n))
    print(f"[split] mode=frame val_ratio={val_ratio}")
    return idx[:train_n], idx[train_n:]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def compute_baselines_for_h5(h5_path: Path):
    with h5py.File(h5_path, "r") as h5:
        act = np.asarray(h5["act"], dtype=np.float32)
        ep = np.asarray(h5["episode_id"], dtype=np.int32)

    pred_zero = np.zeros_like(act)
    zero_mse = mse(pred_zero, act)

    mean_action = np.mean(act, axis=0, keepdims=True)
    pred_mean = np.repeat(mean_action, repeats=act.shape[0], axis=0)
    mean_mse = mse(pred_mean, act)

    pred_prev = np.zeros_like(act)
    pred_prev[1:] = act[:-1]
    boundary = np.where(ep[1:] != ep[:-1])[0] + 1
    pred_prev[boundary] = 0.0
    prev_mse = mse(pred_prev, act)

    return {
        "zero_action_mse": float(zero_mse),
        "mean_action_mse": float(mean_mse),
        "prev_action_mse": float(prev_mse),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cross_c3_train_c2_test.yaml")
    parser.add_argument("--rebuild-h5", action="store_true", help="强制重新从 JSONL 生成 H5")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    data_cfg = cfg["data"]
    train_raw = Path(data_cfg["train_raw_jsonl"])
    train_h5 = Path(data_cfg["train_h5_path"])
    test_raw = Path(data_cfg["test_raw_jsonl"])
    test_h5 = Path(data_cfg["test_h5_path"])

    joint_names = list(data_cfg["joint_names"])
    use_joint_vel = bool(data_cfg.get("use_joint_velocities", True))
    include_gripper = bool(data_cfg.get("include_gripper", False))

    maybe_prepare_h5(train_raw, train_h5, joint_names, use_joint_vel, include_gripper, rebuild=args.rebuild_h5)
    maybe_prepare_h5(test_raw, test_h5, joint_names, use_joint_vel, include_gripper, rebuild=args.rebuild_h5)

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg.get("experiment", {}).get("name", "cross_dataset")
    exp_root = Path(cfg.get("experiment", {}).get("out_dir", "./artifacts/experiments")) / exp_name
    run_dir = exp_root / f"run_{run_id}"
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = copy.deepcopy(cfg)
    run_cfg["train"]["ckpt_dir"] = str(ckpt_dir)
    run_cfg["train"]["metrics_json"] = str(run_dir / "metrics.json")
    run_cfg["train"]["metrics_csv"] = str(run_dir / "metrics.csv")

    with h5py.File(train_h5, "r") as h5:
        obs_dim = int(h5.attrs["obs_dim"])
        act_dim = int(h5.attrs["act_dim"])
        train_episode_ids = np.asarray(h5["episode_id"]).astype(np.int32)

    train_idx, val_idx = split_train_val_indices(
        train_episode_ids,
        val_ratio=float(run_cfg["train"].get("val_ratio", 0.2)),
        split_by_episode=bool(run_cfg["train"].get("split_by_episode", True)),
    )

    train_ds = H5BehaviorCloningDataset(str(train_h5), train_idx)
    val_ds = H5BehaviorCloningDataset(str(train_h5), val_idx)
    test_ds = None

    try:
        train_loader = DataLoader(train_ds, batch_size=int(run_cfg["train"]["batch_size"]), shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=int(run_cfg["train"]["batch_size"]), shuffle=False, num_workers=0)

        model = MLPPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=tuple(run_cfg["model"].get("hidden_dims", [256, 256])),
            dropout=float(run_cfg["model"].get("dropout", 0.1)),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[train] device={device} train_samples={len(train_ds)} val_samples={len(val_ds)}")
        train_bc(model, train_loader, val_loader, run_cfg, device)

        best_ckpt = ckpt_dir / "best.pt"
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.to(device)

        with h5py.File(test_h5, "r") as h5:
            n_test = int(h5.attrs["num_samples"])

        test_ds = H5BehaviorCloningDataset(str(test_h5), list(range(n_test)))
        test_loader = DataLoader(test_ds, batch_size=int(run_cfg["train"]["batch_size"]), shuffle=False, num_workers=0)
        test_mse = float(evaluate_bc(model, test_loader, device))

        baselines = compute_baselines_for_h5(test_h5)

        summary = {
            "run_id": run_id,
            "config": str(args.config),
            "train_raw_jsonl": str(train_raw),
            "train_h5_path": str(train_h5),
            "test_raw_jsonl": str(test_raw),
            "test_h5_path": str(test_h5),
            "joint_names": joint_names,
            "device": device,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": n_test,
            "best_epoch": int(ckpt.get("epoch", -1)),
            "best_val_loss": float(ckpt.get("val_loss", -1.0)),
            "cross_test_mse": test_mse,
            "test_baselines": baselines,
            "created_at": dt.datetime.now().isoformat(),
        }

        summary_path = run_dir / "result_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        summary_csv_path = run_dir / "result_summary.csv"
        with summary_csv_path.open("w", encoding="utf-8") as f:
            f.write(
                "run_id,best_epoch,best_val_loss,cross_test_mse,zero_action_mse,mean_action_mse,prev_action_mse,train_samples,val_samples,test_samples\n"
            )
            f.write(
                f"{run_id},{summary['best_epoch']},{summary['best_val_loss']:.8f},{summary['cross_test_mse']:.8f},"
                f"{baselines['zero_action_mse']:.8f},{baselines['mean_action_mse']:.8f},{baselines['prev_action_mse']:.8f},"
                f"{summary['train_samples']},{summary['val_samples']},{summary['test_samples']}\n"
            )

        print(f"[done] run_dir={run_dir}")
        print(f"[done] cross_test_mse={test_mse:.6f}")
        print(f"[done] summary={summary_path}")

    finally:
        train_ds.close()
        val_ds.close()
        if test_ds is not None:
            test_ds.close()


if __name__ == "__main__":
    main()
