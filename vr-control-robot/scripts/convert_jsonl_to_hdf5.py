import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.utils.config import load_yaml
from src.vrtrain.data.vectorize import frame_to_obs_action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    raw_path = Path(cfg["data"]["raw_jsonl"])
    h5_path = Path(cfg["data"]["h5_path"])
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    joint_names = cfg["data"]["joint_names"]
    use_joint_vel = bool(cfg["data"].get("use_joint_velocities", True))
    include_gripper = bool(cfg["data"].get("include_gripper", False))

    obs_all = []
    act_all = []
    episode_ids = []

    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            ep = int(frame.get("episodeId", 0))
            if ep <= 0:
                continue
            obs_vec, act_vec = frame_to_obs_action(frame, joint_names, use_joint_vel, include_gripper)
            obs_all.append(obs_vec)
            act_all.append(act_vec)
            episode_ids.append(ep)

    if not obs_all:
        raise RuntimeError("没有可用样本，请检查输入 JSONL 或 episode 过滤条件")

    obs_np = np.stack(obs_all, axis=0)
    act_np = np.stack(act_all, axis=0)
    ep_np = np.asarray(episode_ids, dtype=np.int32)

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("obs", data=obs_np)
        h5.create_dataset("act", data=act_np)
        h5.create_dataset("episode_id", data=ep_np)
        h5.attrs["obs_dim"] = obs_np.shape[1]
        h5.attrs["act_dim"] = act_np.shape[1]
        h5.attrs["num_samples"] = obs_np.shape[0]
        h5.attrs["joint_names_json"] = json.dumps(joint_names, ensure_ascii=False)

    print(f"[convert] done: samples={obs_np.shape[0]}, obs_dim={obs_np.shape[1]}, act_dim={act_np.shape[1]}")
    print(f"[convert] output: {h5_path}")


if __name__ == "__main__":
    main()
