import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


PHASES = ["idle", "reach", "align", "grasp", "hold", "timeout", "success", "unknown"]
PHASE_TO_INDEX = {name: idx for idx, name in enumerate(PHASES)}
NEAREST_HANDS = ["none", "left", "right"]
NEAREST_HAND_TO_INDEX = {name: idx for idx, name in enumerate(NEAREST_HANDS)}


@dataclass
class SegmentData:
    episode_id: int
    target_id: int
    target_index: int
    outcome: str
    obs: np.ndarray
    act: np.ndarray
    success: int
    weights: np.ndarray


@dataclass
class Normalizer:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    def to_dict(self) -> Dict:
        return {
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "act_mean": self.act_mean.tolist(),
            "act_std": self.act_std.tolist(),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _get3(d: Dict, key: str) -> List[float]:
    v = d.get(key, [0.0, 0.0, 0.0])
    if not isinstance(v, list) or len(v) != 3:
        return [0.0, 0.0, 0.0]
    out = []
    for x in v:
        try:
            out.append(float(x))
        except Exception:
            out.append(0.0)
    return out


def _get_pose_p(node) -> List[float]:
    if isinstance(node, dict):
        p = node.get("p", None)
        if isinstance(p, list) and len(p) == 3:
            out = []
            for x in p:
                try:
                    out.append(float(x))
                except Exception:
                    out.append(0.0)
            return out
    return [0.0, 0.0, 0.0]


def _get_pose_q(node) -> List[float]:
    if isinstance(node, dict):
        q = node.get("q", None)
        if isinstance(q, list) and len(q) == 4:
            out = []
            for x in q:
                try:
                    out.append(float(x))
                except Exception:
                    out.append(0.0)
            return out
    return [0.0, 0.0, 0.0, 1.0]


def _get_scalar(d: Dict, key: str, default: float = 0.0) -> float:
    try:
        v = d.get(key, default)
        if v is None:
            return float(default)
        out = float(v)
        if not math.isfinite(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _get_bool(d: Dict, key: str, default: bool = False) -> float:
    v = d.get(key, default)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return 1.0 if float(v) != 0.0 else 0.0
    return 1.0 if default else 0.0


def _one_hot(size: int, index: int) -> List[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def _task_distance(task: Dict) -> float:
    vals = [
        _get_scalar(task, "distToTarget", float("nan")),
        _get_scalar(task, "distToTargetLeft", float("nan")),
        _get_scalar(task, "distToTargetRight", float("nan")),
    ]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return 0.0
    return float(min(vals))


def build_feature_names(
    joint_names: List[str],
    use_joint_velocities: bool,
    include_end_effector_quat: bool,
    include_phase: bool,
    include_ee_velocity: bool = False,
    action_repr: str = "euler_delta",
) -> List[str]:
    names: List[str] = []
    axes3 = ["x", "y", "z"]
    axes4 = ["x", "y", "z", "w"]
    for joint in joint_names:
        names.extend([f"joint_pos.{joint}.{axis}" for axis in axes3])
        if use_joint_velocities:
            names.extend([f"joint_vel.{joint}.{axis}" for axis in axes3])
    for hand in ("left", "right"):
        names.extend([f"ee.{hand}.p.{axis}" for axis in axes3])
        if include_end_effector_quat:
            names.extend([f"ee.{hand}.q.{axis}" for axis in axes4])
        if include_ee_velocity:
            names.extend([f"ee_vel.{hand}.linear.{axis}" for axis in axes3])
            names.extend([f"ee_vel.{hand}.angular.{axis}" for axis in axes3])
    names.extend([f"target_rel_base.p.{axis}" for axis in axes3])
    names.extend([f"target_rel_left.p.{axis}" for axis in axes3])
    names.extend([f"target_rel_right.p.{axis}" for axis in axes3])
    names.extend(["dist.target", "dist.target_left", "dist.target_right"])
    names.extend(["contact.any", "contact.left", "contact.right"])
    names.extend(["hold.any_sec", "hold.left_sec", "hold.right_sec"])
    names.extend(["progress.completed_targets", "progress.targets_per_episode", "progress.ratio"])
    names.extend([f"nearest_hand.{name}" for name in NEAREST_HANDS])
    if include_phase:
        names.extend([f"phase.{name}" for name in PHASES])
    return names


def build_action_names(joint_names: List[str], action_repr: str = "euler_delta") -> List[str]:
    names: List[str] = []
    if action_repr == "rot_vec":
        for joint in joint_names:
            names.extend([f"joint_rotvec.{joint}.{axis}" for axis in ["x", "y", "z"]])
    elif action_repr == "target_quat":
        for joint in joint_names:
            names.extend([f"joint_target_q.{joint}.{axis}" for axis in ["x", "y", "z", "w"]])
    else:  # euler_delta (default / legacy)
        for joint in joint_names:
            names.extend([f"joint_delta.{joint}.{axis}" for axis in ["x", "y", "z"]])
    return names


def load_segment_outcomes(events_jsonl: Path) -> Tuple[Dict[Tuple[int, int, int], str], Dict[int, str]]:
    segment_outcomes: Dict[Tuple[int, int, int], str] = {}
    episode_outcomes: Dict[int, str] = {}
    with events_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ev = json.loads(s)
            episode_id = int(ev.get("episodeId", 0) or 0)
            ev_type = str(ev.get("type", ""))
            payload = ev.get("payload", {}) if isinstance(ev.get("payload", {}), dict) else {}
            if ev_type == "target_spawned":
                key = (
                    episode_id,
                    int(payload.get("targetIndex", 0) or 0),
                    int(payload.get("targetId", 0) or 0),
                )
                if key[1] > 0 and key[2] > 0:
                    segment_outcomes.setdefault(key, "unknown")
            elif ev_type == "grasp_success":
                key = (
                    episode_id,
                    int(payload.get("targetIndex", 0) or 0),
                    int(payload.get("targetId", 0) or 0),
                )
                if key[1] > 0 and key[2] > 0:
                    segment_outcomes[key] = "success"
            elif ev_type == "episode_timeout":
                key = (
                    episode_id,
                    int(payload.get("targetIndex", 0) or 0),
                    int(payload.get("targetId", 0) or 0),
                )
                if key[1] > 0 and key[2] > 0 and segment_outcomes.get(key) != "success":
                    segment_outcomes[key] = "timeout"
            elif ev_type == "episode_end":
                episode_outcomes[episode_id] = str(payload.get("outcome", "unknown"))

    finalized: Dict[Tuple[int, int, int], str] = {}
    for key, outcome in segment_outcomes.items():
        if outcome == "unknown":
            ep_outcome = episode_outcomes.get(key[0], "unknown")
            if ep_outcome == "auto_end_by_new_start":
                outcome = "truncated"
            elif ep_outcome == "timeout":
                outcome = "timeout"
        finalized[key] = outcome
    return finalized, episode_outcomes


def _frame_is_valid(frame: Dict, joint_names: List[str], max_abs_joint_velocity: float, max_abs_joint_delta: float) -> bool:
    obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
    act = frame.get("action", {}) if isinstance(frame.get("action", {}), dict) else {}
    joint_vel = obs.get("jointVelocities", {}) if isinstance(obs.get("jointVelocities", {}), dict) else {}
    joint_delta = act.get("jointDelta", {}) if isinstance(act.get("jointDelta", {}), dict) else {}

    for joint in joint_names:
        vel = joint_vel.get(joint, [])
        if isinstance(vel, list):
            for x in vel:
                try:
                    v = float(x)
                except Exception:
                    return False
                if not math.isfinite(v) or abs(v) > max_abs_joint_velocity:
                    return False
        delta = joint_delta.get(joint, [])
        if isinstance(delta, list):
            for x in delta:
                try:
                    v = float(x)
                except Exception:
                    return False
                if not math.isfinite(v) or abs(v) > max_abs_joint_delta:
                    return False
    return True


def frame_to_robot_task_obs_act(
    frame: Dict,
    joint_names: List[str],
    use_joint_velocities: bool,
    include_end_effector_quat: bool,
    include_phase: bool,
    weighting_cfg: Dict,
    segment_outcome: str,
    include_ee_velocity: bool = False,
    action_repr: str = "euler_delta",
) -> Tuple[np.ndarray, np.ndarray, float]:
    obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
    act = frame.get("action", {}) if isinstance(frame.get("action", {}), dict) else {}
    task = obs.get("task", {}) if isinstance(obs.get("task", {}), dict) else {}

    joint_pos = obs.get("jointPositions", {}) if isinstance(obs.get("jointPositions", {}), dict) else {}
    joint_vel = obs.get("jointVelocities", {}) if isinstance(obs.get("jointVelocities", {}), dict) else {}
    end_eff = obs.get("endEffector", {}) if isinstance(obs.get("endEffector", {}), dict) else {}
    ee_vel = obs.get("endEffectorVelocity", {}) if isinstance(obs.get("endEffectorVelocity", {}), dict) else {}

    # Select action source based on representation
    if action_repr == "rot_vec":
        act_source = act.get("jointDeltaRotVec", {}) if isinstance(act.get("jointDeltaRotVec", {}), dict) else {}
    elif action_repr == "target_quat":
        act_source = act.get("jointTargetQuat", {}) if isinstance(act.get("jointTargetQuat", {}), dict) else {}
    else:  # euler_delta (default / legacy)
        act_source = act.get("jointDelta", {}) if isinstance(act.get("jointDelta", {}), dict) else {}

    obs_vec: List[float] = []
    act_vec: List[float] = []

    for joint in joint_names:
        obs_vec.extend(_get3(joint_pos, joint))
        if use_joint_velocities:
            obs_vec.extend(_get3(joint_vel, joint))
        if action_repr == "target_quat":
            q = act_source.get(joint, [0.0, 0.0, 0.0, 1.0])
            if not isinstance(q, list) or len(q) != 4:
                q = [0.0, 0.0, 0.0, 1.0]
            act_vec.extend([float(x) for x in q])
        else:
            act_vec.extend(_get3(act_source, joint))

    for hand in ("left", "right"):
        node = end_eff.get(hand, None)
        obs_vec.extend(_get_pose_p(node))
        if include_end_effector_quat:
            obs_vec.extend(_get_pose_q(node))
        if include_ee_velocity:
            vel_node = ee_vel.get(hand, None)
            if isinstance(vel_node, dict):
                lv = vel_node.get("linearVelocity", [0.0, 0.0, 0.0])
                av = vel_node.get("angularVelocity", [0.0, 0.0, 0.0])
            else:
                lv, av = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            obs_vec.extend([float(x) if isinstance(x, (int, float)) else 0.0 for x in lv[:3]])
            obs_vec.extend([float(x) if isinstance(x, (int, float)) else 0.0 for x in av[:3]])

    obs_vec.extend(_get_pose_p(task.get("targetRelToRobotBase", None) or task.get("targetPose", None)))
    obs_vec.extend(_get_pose_p(task.get("targetRelToLeftHand", None)))
    obs_vec.extend(_get_pose_p(task.get("targetRelToRightHand", None)))

    completed = _get_scalar(task, "completedTargets", 0.0)
    total_targets = max(_get_scalar(task, "targetsPerEpisode", 0.0), 1.0)
    progress_ratio = completed / total_targets

    nearest_hand = str(task.get("nearestHand", "none") or "none")
    nearest_hand_idx = NEAREST_HAND_TO_INDEX.get(nearest_hand, 0)

    obs_vec.extend(
        [
            _get_scalar(task, "distToTarget", 0.0),
            _get_scalar(task, "distToTargetLeft", 0.0),
            _get_scalar(task, "distToTargetRight", 0.0),
            _get_bool(task, "contactFlag", False),
            _get_bool(task, "contactFlagLeft", False),
            _get_bool(task, "contactFlagRight", False),
            _get_scalar(task, "contactHoldMs", 0.0) / 1000.0,
            _get_scalar(task, "contactHoldMsLeft", 0.0) / 1000.0,
            _get_scalar(task, "contactHoldMsRight", 0.0) / 1000.0,
            completed,
            total_targets,
            progress_ratio,
        ]
    )
    obs_vec.extend(_one_hot(len(NEAREST_HANDS), nearest_hand_idx))

    if include_phase:
        phase = str(task.get("phaseLabel", "unknown") or "unknown")
        phase_idx = PHASE_TO_INDEX.get(phase, PHASE_TO_INDEX["unknown"])
        obs_vec.extend(_one_hot(len(PHASES), phase_idx))

    weight = 1.0
    near_target_threshold = float(weighting_cfg.get("near_target_threshold", 0.25))
    dist = _task_distance(task)
    if segment_outcome == "success":
        weight += float(weighting_cfg.get("success_bonus", 0.0))
    if dist <= near_target_threshold:
        weight += float(weighting_cfg.get("near_target_bonus", 0.0))
    if _get_bool(task, "contactFlag", False) > 0.5:
        weight += float(weighting_cfg.get("contact_bonus", 0.0))
    if str(task.get("phaseLabel", "")) in ("align", "grasp", "hold"):
        weight += float(weighting_cfg.get("phase_bonus", 0.0))

    # Frame activity label importance weighting (from v3 collection schema)
    frame_label = str(frame.get("frameLabel", ""))
    idle_discount = float(weighting_cfg.get("idle_discount", 0.3))
    moving_bonus = float(weighting_cfg.get("moving_bonus", 0.0))
    if frame_label == "idle":
        weight *= idle_discount
    elif frame_label == "approaching":
        weight += float(weighting_cfg.get("approaching_bonus", 0.15))
    elif frame_label in ("contacting", "holding"):
        weight += float(weighting_cfg.get("contact_bonus", 0.0))
    elif frame_label == "moving":
        weight += moving_bonus

    weight = min(weight, float(weighting_cfg.get("max_weight", 3.0)))

    return (
        np.asarray(obs_vec, dtype=np.float32),
        np.asarray(act_vec, dtype=np.float32),
        float(weight),
    )


def load_segments(config: Dict) -> Tuple[List[SegmentData], Dict]:
    data_cfg = config["data"]
    root = Path(__file__).resolve().parent
    episodes_jsonl = root / data_cfg["episodes_jsonl"]
    events_jsonl = root / data_cfg["events_jsonl"]
    joint_names = list(data_cfg["joint_names"])
    allowed_outcomes = set(data_cfg.get("allowed_outcomes", ["success", "timeout"]))
    segment_outcomes, episode_outcomes = load_segment_outcomes(events_jsonl)

    grouped: Dict[Tuple[int, int, int], List[Dict]] = defaultdict(list)
    invalid_frame_count = 0
    with episodes_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            frame = json.loads(s)
            episode_id = int(frame.get("episodeId", 0) or 0)
            obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
            task = obs.get("task", {}) if isinstance(obs.get("task", {}), dict) else {}
            target_index = int(task.get("targetIndex", 0) or 0)
            target_id = int(task.get("targetId", 0) or 0)
            if episode_id <= 0 or target_index <= 0 or target_id <= 0:
                continue
            if not _frame_is_valid(
                frame,
                joint_names=joint_names,
                max_abs_joint_velocity=float(data_cfg.get("max_abs_joint_velocity", 10.0)),
                max_abs_joint_delta=float(data_cfg.get("max_abs_joint_delta", 1.0)),
            ):
                invalid_frame_count += 1
                continue
            grouped[(episode_id, target_index, target_id)].append(frame)

    segments: List[SegmentData] = []
    raw_outcome_counter: Counter = Counter()
    kept_outcome_counter: Counter = Counter()
    episode_segment_counter: Counter = Counter()

    for key in sorted(grouped.keys()):
        frames = grouped[key]
        if not frames:
            continue
        outcome = segment_outcomes.get(key, "unknown")
        raw_outcome_counter[outcome] += 1
        if outcome not in allowed_outcomes:
            continue

        feat_list = []
        act_list = []
        weight_list = []
        for frame in frames:
            obs_vec, act_vec, sample_weight = frame_to_robot_task_obs_act(
                frame,
                joint_names=joint_names,
                use_joint_velocities=bool(data_cfg.get("use_joint_velocities", True)),
                include_end_effector_quat=bool(data_cfg.get("include_end_effector_quat", True)),
                include_phase=bool(data_cfg.get("include_phase", True)),
                weighting_cfg=dict(data_cfg.get("sample_weighting", {})),
                segment_outcome=outcome,
                include_ee_velocity=bool(data_cfg.get("include_ee_velocity", False)),
                action_repr=str(data_cfg.get("action_repr", "euler_delta")),
            )
            feat_list.append(obs_vec)
            act_list.append(act_vec)
            weight_list.append(sample_weight)

        if not feat_list:
            continue

        episode_id, target_index, target_id = key
        segments.append(
            SegmentData(
                episode_id=episode_id,
                target_id=target_id,
                target_index=target_index,
                outcome=outcome,
                obs=np.stack(feat_list, axis=0),
                act=np.stack(act_list, axis=0),
                success=1 if outcome == "success" else 0,
                weights=np.asarray(weight_list, dtype=np.float32),
            )
        )
        kept_outcome_counter[outcome] += 1
        episode_segment_counter[episode_id] += 1

    summary = {
        "episodes_jsonl": str(episodes_jsonl),
        "events_jsonl": str(events_jsonl),
        "num_total_grouped_segments": int(sum(raw_outcome_counter.values())),
        "num_kept_segments": int(len(segments)),
        "num_invalid_frames_dropped": int(invalid_frame_count),
        "raw_segment_outcomes": dict(raw_outcome_counter),
        "kept_segment_outcomes": dict(kept_outcome_counter),
        "episode_segment_counts": {str(k): int(v) for k, v in sorted(episode_segment_counter.items())},
    }
    return segments, summary


def _episode_outcome_counts(segments: List[SegmentData]) -> Dict[int, Counter]:
    out: Dict[int, Counter] = defaultdict(Counter)
    for seg in segments:
        out[seg.episode_id][seg.outcome] += 1
    return out


def split_segments_by_episode(segments: List[SegmentData], train_split: float, seed: int) -> Tuple[List[SegmentData], List[SegmentData], Dict]:
    episodes = sorted({seg.episode_id for seg in segments})
    if len(episodes) < 2:
        raise RuntimeError("Need at least 2 episodes with usable segments.")

    val_n = max(1, int(round(len(episodes) * (1.0 - train_split))))
    val_n = min(val_n, len(episodes) - 1)
    rng = random.Random(seed)
    outcome_counts = _episode_outcome_counts(segments)
    desired_outcomes = {name for name in ("success", "timeout") if any(outcome_counts[ep][name] > 0 for ep in episodes)}

    def split_score(train_eps: List[int], val_eps: List[int]) -> Tuple[float, float, float]:
        train_counter = Counter()
        val_counter = Counter()
        for ep in train_eps:
            train_counter.update(outcome_counts[ep])
        for ep in val_eps:
            val_counter.update(outcome_counts[ep])

        missing = float(len(desired_outcomes - set(k for k, v in train_counter.items() if v > 0)))
        missing += float(len(desired_outcomes - set(k for k, v in val_counter.items() if v > 0)))

        train_success = float(train_counter.get("success", 0))
        train_total = float(train_success + train_counter.get("timeout", 0))
        val_success = float(val_counter.get("success", 0))
        val_total = float(val_success + val_counter.get("timeout", 0))
        train_ratio = train_success / train_total if train_total > 0 else 0.0
        val_ratio = val_success / val_total if val_total > 0 else 0.0
        ratio_gap = abs(train_ratio - val_ratio)

        size_gap = abs(len(train_eps) - len(val_eps) / max(1.0 - train_split, 1e-6))
        return (missing, ratio_gap, size_gap)

    best_train_eps = episodes[:-val_n]
    best_val_eps = episodes[-val_n:]
    best_score = split_score(best_train_eps, best_val_eps)

    for _ in range(2048):
        shuffled = episodes[:]
        rng.shuffle(shuffled)
        val_eps = sorted(shuffled[-val_n:])
        train_eps = sorted(shuffled[:-val_n])
        score = split_score(train_eps, val_eps)
        if score < best_score:
            best_score = score
            best_train_eps = train_eps
            best_val_eps = val_eps
            if score[0] == 0.0 and score[1] < 0.05:
                break

    train_set = set(best_train_eps)
    val_set = set(best_val_eps)
    train_segments = [seg for seg in segments if seg.episode_id in train_set]
    val_segments = [seg for seg in segments if seg.episode_id in val_set]

    split_summary = {
        "train_episodes": best_train_eps,
        "val_episodes": best_val_eps,
        "train_segment_outcomes": dict(Counter(seg.outcome for seg in train_segments)),
        "val_segment_outcomes": dict(Counter(seg.outcome for seg in val_segments)),
        "train_num_segments": len(train_segments),
        "val_num_segments": len(val_segments),
    }
    return train_segments, val_segments, split_summary


def fit_normalizer(segments: List[SegmentData]) -> Normalizer:
    obs = np.concatenate([seg.obs for seg in segments], axis=0)
    act = np.concatenate([seg.act for seg in segments], axis=0)
    obs_mean = obs.mean(axis=0)
    act_mean = act.mean(axis=0)
    obs_std = obs.std(axis=0)
    act_std = act.std(axis=0)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std).astype(np.float32)
    act_std = np.where(act_std < 1e-6, 1.0, act_std).astype(np.float32)
    return Normalizer(
        obs_mean=obs_mean.astype(np.float32),
        obs_std=obs_std,
        act_mean=act_mean.astype(np.float32),
        act_std=act_std,
    )


def normalize_obs(obs: np.ndarray, normalizer: Normalizer, clip_z: float) -> np.ndarray:
    z = (obs - normalizer.obs_mean) / normalizer.obs_std
    if clip_z is not None and clip_z > 0:
        z = np.clip(z, -clip_z, clip_z)
    return z.astype(np.float32)


def normalize_act(act: np.ndarray, normalizer: Normalizer, clip_z: float) -> np.ndarray:
    z = (act - normalizer.act_mean) / normalizer.act_std
    if clip_z is not None and clip_z > 0:
        z = np.clip(z, -clip_z, clip_z)
    return z.astype(np.float32)


def denormalize_act(act_z: np.ndarray, normalizer: Normalizer) -> np.ndarray:
    return (act_z * normalizer.act_std + normalizer.act_mean).astype(np.float32)


def make_frame_dataset(segments: List[SegmentData], normalizer: Normalizer, obs_clip_z: float, act_clip_z: float):
    xs = []
    ys = []
    success = []
    weights = []
    meta = []
    for seg_id, seg in enumerate(segments, start=1):
        obs_z = normalize_obs(seg.obs, normalizer, obs_clip_z)
        act_z = normalize_act(seg.act, normalizer, act_clip_z)
        xs.append(obs_z)
        ys.append(act_z)
        success.append(np.full((obs_z.shape[0],), seg.success, dtype=np.float32))
        weights.append(seg.weights.astype(np.float32))
        meta.extend(
            [
                {
                    "segment_id": seg_id,
                    "episode_id": seg.episode_id,
                    "target_id": seg.target_id,
                    "target_index": seg.target_index,
                    "outcome": seg.outcome,
                }
            ]
            * obs_z.shape[0]
        )
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(success, axis=0),
        np.concatenate(weights, axis=0),
        meta,
    )


def make_seq_dataset(segments: List[SegmentData], seq_len: int, normalizer: Normalizer, obs_clip_z: float, act_clip_z: float):
    xs = []
    ys = []
    success = []
    weights = []
    meta = []
    for seg_id, seg in enumerate(segments, start=1):
        if seg.obs.shape[0] < seq_len:
            continue
        obs_z = normalize_obs(seg.obs, normalizer, obs_clip_z)
        act_z = normalize_act(seg.act, normalizer, act_clip_z)
        for end in range(seq_len - 1, seg.obs.shape[0]):
            start = end - seq_len + 1
            xs.append(obs_z[start : end + 1])
            ys.append(act_z[end])
            success.append(float(seg.success))
            weights.append(float(seg.weights[end]))
            meta.append(
                {
                    "segment_id": seg_id,
                    "episode_id": seg.episode_id,
                    "target_id": seg.target_id,
                    "target_index": seg.target_index,
                    "outcome": seg.outcome,
                    "step_index": end,
                }
            )
    if not xs:
        raise RuntimeError("No sequence samples after filtering; please check seq_len or dataset settings.")
    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.stack(ys, axis=0).astype(np.float32),
        np.asarray(success, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        meta,
    )


def summarize_segments(segments: List[SegmentData]) -> Dict:
    outcome_counter = Counter(seg.outcome for seg in segments)
    episode_counter = Counter(seg.episode_id for seg in segments)
    lengths = [int(seg.obs.shape[0]) for seg in segments]
    frame_counter = Counter()
    for seg in segments:
        frame_counter[seg.outcome] += int(seg.obs.shape[0])
    return {
        "num_segments": len(segments),
        "num_episodes": len(episode_counter),
        "segment_outcomes": dict(outcome_counter),
        "frames_by_outcome": dict(frame_counter),
        "min_segment_len": int(min(lengths)) if lengths else 0,
        "max_segment_len": int(max(lengths)) if lengths else 0,
        "mean_segment_len": float(np.mean(lengths)) if lengths else 0.0,
    }
