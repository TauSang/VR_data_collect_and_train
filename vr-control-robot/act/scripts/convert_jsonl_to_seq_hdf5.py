import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.data.vectorize import frame_to_obs_action
from src.vrtrain.utils.config import load_yaml


def load_frames_grouped_by_episode(raw_jsonl: Path):
    by_ep = defaultdict(list)
    with raw_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            ep = int(frame.get("episodeId", 0))
            if ep <= 0:
                continue
            by_ep[ep].append(frame)
    return by_ep


def build_seq_dataset(by_ep: dict, joint_names: list[str], use_joint_velocities: bool, include_gripper: bool, seq_len: int, stride: int):
    obs_seq_all = []
    act_all = []
    ep_all = []

    for ep, frames in by_ep.items():
        if len(frames) < seq_len:
            continue
        vec_pairs = [frame_to_obs_action(fr, joint_names, use_joint_velocities, include_gripper) for fr in frames]

        obs_list = [v[0] for v in vec_pairs]
        act_list = [v[1] for v in vec_pairs]

        for end_idx in range(seq_len - 1, len(frames), stride):
            start = end_idx - seq_len + 1
            obs_window = np.stack(obs_list[start : end_idx + 1], axis=0)
            act_target = act_list[end_idx]
            obs_seq_all.append(obs_window)
            act_all.append(act_target)
            ep_all.append(ep)

    if not obs_seq_all:
        raise RuntimeError("序列样本为空，请检查数据量或 seq_len")

    return (
        np.stack(obs_seq_all, axis=0).astype(np.float32),
        np.stack(act_all, axis=0).astype(np.float32),
        np.asarray(ep_all, dtype=np.int32),
    )


def write_h5(path: Path, obs_seq: np.ndarray, act: np.ndarray, episode_id: np.ndarray, joint_names: list[str], seq_len: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("obs_seq", data=obs_seq)
        h5.create_dataset("act", data=act)
        h5.create_dataset("episode_id", data=episode_id)

        h5.attrs["num_samples"] = int(obs_seq.shape[0])
        h5.attrs["seq_len"] = int(seq_len)
        h5.attrs["obs_dim"] = int(obs_seq.shape[-1])
        h5.attrs["act_dim"] = int(act.shape[-1])
        h5.attrs["joint_names_json"] = json.dumps(joint_names, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="act/configs/base_act.yaml")
    parser.add_argument("--dataset", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    seq_len = int(data_cfg.get("seq_len", 16))
    stride = int(data_cfg.get("stride", 1))
    joint_names = list(data_cfg["joint_names"])
    use_joint_velocities = bool(data_cfg.get("use_joint_velocities", True))
    include_gripper = bool(data_cfg.get("include_gripper", False))

    targets = []
    if args.dataset in ("train", "both"):
        targets.append((Path(data_cfg["train_raw_jsonl"]), Path(data_cfg["train_h5_path"]), "train"))
    if args.dataset in ("test", "both"):
        targets.append((Path(data_cfg["test_raw_jsonl"]), Path(data_cfg["test_h5_path"]), "test"))

    for raw_jsonl, out_h5, tag in targets:
        by_ep = load_frames_grouped_by_episode(raw_jsonl)
        obs_seq, act, episode_id = build_seq_dataset(
            by_ep,
            joint_names,
            use_joint_velocities,
            include_gripper,
            seq_len,
            stride,
        )
        write_h5(out_h5, obs_seq, act, episode_id, joint_names, seq_len)
        print(
            f"[act-convert] {tag}: samples={obs_seq.shape[0]} seq_len={obs_seq.shape[1]} "
            f"obs_dim={obs_seq.shape[2]} act_dim={act.shape[1]} -> {out_h5}"
        )


if __name__ == "__main__":
    main()
