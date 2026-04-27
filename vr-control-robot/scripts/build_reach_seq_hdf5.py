import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.data.vectorize_reach import frame_to_reach_obs_action, infer_assigned_hand, task_distance
from src.vrtrain.utils.config import load_yaml, resolve_path


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _iter_frames(raw_paths: list[Path]):
    for raw_path in raw_paths:
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                frame["_source_path"] = str(raw_path)
                yield frame


def _group_by_target_segment(raw_paths: list[Path]):
    grouped = defaultdict(list)
    for frame in _iter_frames(raw_paths):
        ep = int(frame.get("episodeId", 0))
        obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
        task = obs.get("task", {}) if isinstance(obs.get("task", {}), dict) else {}
        target_index = int(task.get("targetIndex", 0) or 0)
        target_id = int(task.get("targetId", 0) or 0)
        if ep <= 0 or target_index <= 0 or target_id <= 0:
            continue
        key = (frame["_source_path"], ep, target_index, target_id)
        grouped[key].append(frame)
    return grouped


def _segment_success(frames: list[dict]):
    for frame in frames:
        obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
        task = obs.get("task", {}) if isinstance(obs.get("task", {}), dict) else {}
        if task.get("phaseLabel") == "success":
            return True
        if task.get("successHand") in ("left", "right"):
            return True
    return False


def _segment_assigned_hand(frames: list[dict], default_hand: str | None = None):
    for frame in frames:
        obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
        task = obs.get("task", {}) if isinstance(obs.get("task", {}), dict) else {}
        hand = infer_assigned_hand(task, fallback=None)
        if hand in ("left", "right"):
            return hand
    return default_hand


def _assigned_hand_index(hand: str | None):
    if hand == "left":
        return 0
    if hand == "right":
        return 1
    return -1


def _build_target_sequences(
    grouped: dict,
    joint_names: list[str],
    use_joint_velocities: bool,
    include_phase: bool,
    seq_len: int,
    stride: int,
    label_shift: int,
    hand_filter: str,
    include_failures: bool,
):
    obs_seq_all = []
    act_all = []
    episode_all = []
    segment_all = []
    target_all = []
    assigned_hand_all = []
    start_dist_all = []
    end_dist_all = []
    success_all = []
    step_all = []

    segment_id = 0
    for key in sorted(grouped.keys()):
        frames = grouped[key]
        source_path, episode_id, _, target_id = key
        assigned_hand = _segment_assigned_hand(frames, default_hand=None)
        seg_success = _segment_success(frames)

        if hand_filter in ("left", "right") and assigned_hand != hand_filter:
            continue
        if (not include_failures) and (not seg_success):
            continue

        feat_list = []
        act_list = []
        dist_list = []
        for frame in frames:
            feat, act = frame_to_reach_obs_action(
                frame,
                joint_names=joint_names,
                use_joint_velocities=use_joint_velocities,
                include_phase=include_phase,
                default_assigned_hand=assigned_hand,
            )
            obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
            task = obs.get("task", {}) if isinstance(obs.get("task", {}), dict) else {}
            feat_list.append(feat)
            act_list.append(act)
            dist_list.append(task_distance(task, assigned_hand=assigned_hand))

        if len(feat_list) < seq_len + max(0, label_shift):
            continue

        feat_np = np.stack(feat_list, axis=0)
        act_np = np.stack(act_list, axis=0)
        dist_np = np.asarray(dist_list, dtype=np.float32)

        max_t = len(feat_np) - 1 - max(0, label_shift)
        segment_id += 1
        for t in range(seq_len - 1, max_t + 1, stride):
            y_idx = t + max(0, label_shift)
            obs_seq_all.append(feat_np[t - seq_len + 1 : t + 1])
            act_all.append(act_np[y_idx])
            episode_all.append(int(episode_id))
            segment_all.append(int(segment_id))
            target_all.append(int(target_id))
            assigned_hand_all.append(int(_assigned_hand_index(assigned_hand)))
            start_dist_all.append(float(dist_np[t - seq_len + 1]))
            end_dist_all.append(float(dist_np[y_idx]))
            success_all.append(int(seg_success))
            step_all.append(int(y_idx))

    if not obs_seq_all:
        raise RuntimeError("没有可用的 target-conditioned sequence 样本，请检查数据或 hand_filter")

    return {
        "obs_seq": np.stack(obs_seq_all, axis=0).astype(np.float32),
        "act": np.stack(act_all, axis=0).astype(np.float32),
        "episode_id": np.asarray(episode_all, dtype=np.int32),
        "segment_id": np.asarray(segment_all, dtype=np.int32),
        "target_id": np.asarray(target_all, dtype=np.int32),
        "assigned_hand": np.asarray(assigned_hand_all, dtype=np.int8),
        "start_distance": np.asarray(start_dist_all, dtype=np.float32),
        "end_distance": np.asarray(end_dist_all, dtype=np.float32),
        "segment_success": np.asarray(success_all, dtype=np.int8),
        "step_index": np.asarray(step_all, dtype=np.int32),
    }


def _write_h5(path: Path, payload: dict, joint_names: list[str], seq_len: int, raw_paths: list[Path]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for key, value in payload.items():
            h5.create_dataset(key, data=value)

        h5.attrs["num_samples"] = int(payload["obs_seq"].shape[0])
        h5.attrs["seq_len"] = int(seq_len)
        h5.attrs["obs_dim"] = int(payload["obs_seq"].shape[-1])
        h5.attrs["act_dim"] = int(payload["act"].shape[-1])
        h5.attrs["joint_names_json"] = json.dumps(joint_names, ensure_ascii=False)
        h5.attrs["raw_paths_json"] = json.dumps([str(p) for p in raw_paths], ensure_ascii=False)
        h5.attrs["schema"] = "robot_state + target_condition -> jointDelta"


def _run_one(config_path: str, raw_values, out_value, tag: str, data_cfg: dict):
    raw_paths = [resolve_path(config_path, p) for p in _ensure_list(raw_values)]
    if not raw_paths:
        print(f"[reach-build] skip {tag}: no raw paths configured")
        return

    grouped = _group_by_target_segment(raw_paths)
    payload = _build_target_sequences(
        grouped=grouped,
        joint_names=list(data_cfg["joint_names"]),
        use_joint_velocities=bool(data_cfg.get("use_joint_velocities", True)),
        include_phase=bool(data_cfg.get("include_phase", False)),
        seq_len=int(data_cfg.get("seq_len", 16)),
        stride=int(data_cfg.get("stride", 1)),
        label_shift=int(data_cfg.get("label_shift", 0)),
        hand_filter=str(data_cfg.get("hand_filter", "any")),
        include_failures=bool(data_cfg.get("include_failures", True)),
    )
    out_path = resolve_path(config_path, out_value)
    _write_h5(out_path, payload, list(data_cfg["joint_names"]), int(data_cfg.get("seq_len", 16)), raw_paths)
    print(
        f"[reach-build] {tag}: samples={payload['obs_seq'].shape[0]} "
        f"seq_len={payload['obs_seq'].shape[1]} obs_dim={payload['obs_seq'].shape[2]} "
        f"act_dim={payload['act'].shape[1]} -> {out_path}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reach_bc_task.yaml")
    parser.add_argument("--dataset", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    data_cfg = cfg["data"]

    if args.dataset in ("train", "both"):
        _run_one(args.config, data_cfg.get("train_raw_jsonl"), data_cfg["train_h5_path"], "train", data_cfg)
    if args.dataset in ("test", "both") and data_cfg.get("test_raw_jsonl"):
        _run_one(args.config, data_cfg.get("test_raw_jsonl"), data_cfg["test_h5_path"], "test", data_cfg)


if __name__ == "__main__":
    main()
