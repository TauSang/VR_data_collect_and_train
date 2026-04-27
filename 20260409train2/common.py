"""
G1 Reach Task — Data Loading & Feature Construction

Observation space (nJ*2 + 15 D, currently 31D with 8 active joints):
    nJ   G1 joint positions  (scalar per revolute joint)
    nJ   G1 joint velocities (scalar per revolute joint)
     6D  end-effector positions (left xyz + right xyz, relative to robot base)
     3D  target relative to robot base (xyz)
     3D  target relative to left hand  (xyz)
     3D  target relative to right hand (xyz)

Action space (nJ D, currently 8D with 8 active joints):
    nJ   G1 joint position delta (scalar per joint, from VR euler delta)

VR bone → G1 joint mapping:
    leftUpperArm  euler(x,y,z) → shoulder(roll, pitch, yaw)
    leftLowerArm  euler(y)     → elbow
    leftHand      euler(x,y,z) → wrist(roll, pitch, yaw)
    (right side symmetric)
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ─── VR bone → G1 joint mapping ─────────────────────────────────────
# Each entry: (g1_joint_name, euler_axis_index, sign)
VR_TO_G1_MAPPING: Dict[str, List[Tuple[str, int, float]]] = {
    "leftUpperArm": [
        ("left_shoulder_pitch_joint", 1, 1.0),   # euler.y → pitch
        ("left_shoulder_roll_joint",  0, 1.0),   # euler.x → roll
        ("left_shoulder_yaw_joint",   2, 1.0),   # euler.z → yaw
    ],
    "leftLowerArm": [
        ("left_elbow_joint", 1, 1.0),            # euler.y → elbow
    ],
    "leftHand": [
        ("left_wrist_roll_joint",  0, 1.0),      # euler.x → wrist roll
        ("left_wrist_pitch_joint", 1, 1.0),      # euler.y → wrist pitch
        ("left_wrist_yaw_joint",   2, 1.0),      # euler.z → wrist yaw
    ],
    "rightUpperArm": [
        ("right_shoulder_pitch_joint", 1, 1.0),
        ("right_shoulder_roll_joint",  0, 1.0),
        ("right_shoulder_yaw_joint",   2, 1.0),
    ],
    "rightLowerArm": [
        ("right_elbow_joint", 1, 1.0),
    ],
    "rightHand": [
        ("right_wrist_roll_joint",  0, 1.0),
        ("right_wrist_pitch_joint", 1, 1.0),
        ("right_wrist_yaw_joint",   2, 1.0),
    ],
}

# ─── Data structures ────────────────────────────────────────────────

@dataclass
class SegmentData:
    episode_id: int
    target_id: int
    target_index: int
    outcome: str
    obs: np.ndarray          # (T, obs_dim)
    act: np.ndarray          # (T, act_dim)
    success: int             # 1 if outcome == "success"
    weights: np.ndarray      # (T,)


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

    @classmethod
    def from_dict(cls, d: Dict) -> "Normalizer":
        return cls(
            obs_mean=np.asarray(d["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(d["obs_std"], dtype=np.float32),
            act_mean=np.asarray(d["act_mean"], dtype=np.float32),
            act_std=np.asarray(d["act_std"], dtype=np.float32),
        )


# ─── Utility helpers ────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_float(v, default: float = 0.0) -> float:
    try:
        out = float(v)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _get_pose_p(node) -> List[float]:
    """Extract position [x,y,z] from a pose dict with key 'p'."""
    if isinstance(node, dict):
        p = node.get("p", [0.0, 0.0, 0.0])
        if isinstance(p, list) and len(p) >= 3:
            return [_safe_float(p[0]), _safe_float(p[1]), _safe_float(p[2])]
    return [0.0, 0.0, 0.0]


def _get_scalar(d: Dict, key: str, default: float = 0.0) -> float:
    return _safe_float(d.get(key, default), default)


def _get_bool(d: Dict, key: str, default: bool = False) -> float:
    v = d.get(key, default)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return 1.0 if float(v) != 0.0 else 0.0
    return 1.0 if default else 0.0


# ─── VR → G1 mapping ───────────────────────────────────────────────

def _vr_to_g1_scalars(
    vr_data: Dict[str, list],
    g1_joint_names: List[str],
) -> List[float]:
    """
    Extract G1 joint scalars from VR bone euler/velocity/delta data.

    vr_data: e.g. obs["jointPositions"] = {"leftUpperArm": [x,y,z], ...}
    Returns: list of 14 floats in g1_joint_names order.
    """
    values: Dict[str, float] = {j: 0.0 for j in g1_joint_names}
    for vr_bone, mappings in VR_TO_G1_MAPPING.items():
        euler = vr_data.get(vr_bone)
        if euler is None or not isinstance(euler, list) or len(euler) < 3:
            continue
        for g1_joint, axis_idx, sign in mappings:
            if g1_joint in values:
                try:
                    v = sign * float(euler[axis_idx])
                    if math.isfinite(v):
                        values[g1_joint] = v
                except (IndexError, ValueError, TypeError):
                    pass
    return [values[j] for j in g1_joint_names]


# ─── Feature / action name builders ────────────────────────────────

def build_feature_names(g1_joint_names: List[str]) -> List[str]:
    """Build observation feature names (nJ*2 + 15)."""
    names: List[str] = []
    for joint in g1_joint_names:
        names.append(f"g1_pos.{joint}")
    for joint in g1_joint_names:
        names.append(f"g1_vel.{joint}")
    for hand in ("left", "right"):
        for axis in ("x", "y", "z"):
            names.append(f"ee.{hand}.p.{axis}")
    for prefix in ("target_rel_base", "target_rel_left", "target_rel_right"):
        for axis in ("x", "y", "z"):
            names.append(f"{prefix}.p.{axis}")
    return names


def build_action_names(g1_joint_names: List[str]) -> List[str]:
    """Build action feature names (nJ)."""
    return [f"g1_delta.{joint}" for joint in g1_joint_names]


# ─── Segment outcome parsing ───────────────────────────────────────

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
            elif ev_type in ("target_success", "target_reached"):
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


# ─── Frame validation ───────────────────────────────────────────────

def _extract_g1_values(
    container: Dict,
    precomputed_key: str,
    fallback_key: str,
    g1_joint_names: List[str],
) -> List[float]:
    """Extract G1 joint values: prefer pre-computed dict, fallback to VR→G1 mapping."""
    precomputed = container.get(precomputed_key)
    if isinstance(precomputed, dict) and len(precomputed) >= len(g1_joint_names):
        return [_safe_float(precomputed.get(j, 0.0)) for j in g1_joint_names]
    fallback = container.get(fallback_key, {})
    if isinstance(fallback, dict):
        return _vr_to_g1_scalars(fallback, g1_joint_names)
    return [0.0] * len(g1_joint_names)


def _frame_is_valid(
    frame: Dict,
    g1_joint_names: List[str],
    max_abs_velocity: float,
    max_abs_delta: float,
) -> bool:
    obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
    act = frame.get("action", {}) if isinstance(frame.get("action", {}), dict) else {}

    # Validate G1 joint velocities
    g1_vel = _extract_g1_values(obs, "g1JointVelocities", "jointVelocities", g1_joint_names)
    for v in g1_vel:
        if not math.isfinite(v) or abs(v) > max_abs_velocity:
            return False

    # Validate G1 joint deltas
    g1_delta = _extract_g1_values(act, "g1JointDelta", "jointDelta", g1_joint_names)
    for v in g1_delta:
        if not math.isfinite(v) or abs(v) > max_abs_delta:
            return False

    return True


# ─── Frame → obs/act extraction ────────────────────────────────────

def _task_distance(task: Dict) -> float:
    vals = [
        _get_scalar(task, "distToTarget", float("nan")),
        _get_scalar(task, "distToTargetLeft", float("nan")),
        _get_scalar(task, "distToTargetRight", float("nan")),
    ]
    vals = [v for v in vals if math.isfinite(v)]
    return float(min(vals)) if vals else 0.0


def frame_to_obs_act(
    frame: Dict,
    g1_joint_names: List[str],
    weighting_cfg: Dict,
    segment_outcome: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build 43D observation + 14D action from one VR JSONL frame.

    Returns: (obs_vec[43], act_vec[14], sample_weight)
    """
    obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
    act = frame.get("action", {}) if isinstance(frame.get("action", {}), dict) else {}
    task = obs.get("task", {}) if isinstance(obs.get("task", {}), dict) else {}
    end_eff = obs.get("endEffector", {}) if isinstance(obs.get("endEffector", {}), dict) else {}

    obs_vec: List[float] = []

    # ── G1 joint positions (14D) ── prefer pre-computed g1JointPositions
    obs_vec.extend(_extract_g1_values(obs, "g1JointPositions", "jointPositions", g1_joint_names))

    # ── G1 joint velocities (14D) ── prefer pre-computed g1JointVelocities
    obs_vec.extend(_extract_g1_values(obs, "g1JointVelocities", "jointVelocities", g1_joint_names))

    # ── End-effector positions (6D: left xyz + right xyz) ──
    obs_vec.extend(_get_pose_p(end_eff.get("left", None)))
    obs_vec.extend(_get_pose_p(end_eff.get("right", None)))

    # ── Target-relative positions (9D) ──
    obs_vec.extend(_get_pose_p(task.get("targetRelToRobotBase", None) or task.get("targetPose", None)))
    obs_vec.extend(_get_pose_p(task.get("targetRelToLeftHand", None)))
    obs_vec.extend(_get_pose_p(task.get("targetRelToRightHand", None)))

    # ── G1 joint deltas (14D action) ── prefer pre-computed g1JointDelta
    act_vec = _extract_g1_values(act, "g1JointDelta", "jointDelta", g1_joint_names)

    # ── Sample weight ──
    weight = 1.0
    near_target_threshold = float(weighting_cfg.get("near_target_threshold", 0.25))
    dist = _task_distance(task)

    if segment_outcome == "success":
        weight += float(weighting_cfg.get("success_bonus", 0.0))
    if dist <= near_target_threshold:
        weight += float(weighting_cfg.get("near_target_bonus", 0.0))

    # Frame activity label weighting
    frame_label = str(frame.get("frameLabel", ""))
    if frame_label == "idle":
        weight *= float(weighting_cfg.get("idle_discount", 0.3))
    elif frame_label == "approaching":
        weight += float(weighting_cfg.get("approaching_bonus", 0.15))
    elif frame_label == "moving":
        weight += float(weighting_cfg.get("moving_bonus", 0.0))

    weight = min(weight, float(weighting_cfg.get("max_weight", 3.0)))

    return (
        np.asarray(obs_vec, dtype=np.float32),
        np.asarray(act_vec, dtype=np.float32),
        float(weight),
    )


# ─── Segment loading ───────────────────────────────────────────────

def _load_single_source(
    episodes_jsonl: Path,
    events_jsonl: Path,
    g1_joint_names: List[str],
    allowed_outcomes: set,
    max_abs_velocity: float,
    max_abs_delta: float,
    weighting_cfg: Dict,
    episode_id_offset: int = 0,
) -> Tuple[List[SegmentData], Dict]:
    """Load segments from a single data source, offseting episode IDs to avoid collisions."""
    segment_outcomes, episode_outcomes = load_segment_outcomes(events_jsonl)

    grouped: Dict[Tuple[int, int, int], List[Dict]] = defaultdict(list)
    invalid_frame_count = 0

    with episodes_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            frame = json.loads(s)
            raw_ep_id = int(frame.get("episodeId", 0) or 0)
            obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
            task = obs.get("task", {}) if isinstance(obs.get("task", {}), dict) else {}
            target_index = int(task.get("targetIndex", 0) or 0)
            target_id = int(task.get("targetId", 0) or 0)
            if raw_ep_id <= 0 or target_index <= 0 or target_id <= 0:
                continue
            if not _frame_is_valid(frame, g1_joint_names, max_abs_velocity, max_abs_delta):
                invalid_frame_count += 1
                continue
            # offset episode id so different sources don't collide
            episode_id = raw_ep_id + episode_id_offset
            grouped[(episode_id, target_index, target_id)].append(frame)
            # also offset the key in segment_outcomes
            if (raw_ep_id, target_index, target_id) in segment_outcomes:
                segment_outcomes[(episode_id, target_index, target_id)] = segment_outcomes.pop(
                    (raw_ep_id, target_index, target_id)
                )

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

        feat_list, act_list, weight_list = [], [], []
        for frame in frames:
            obs_vec, act_vec, sample_weight = frame_to_obs_act(
                frame,
                g1_joint_names=g1_joint_names,
                weighting_cfg=weighting_cfg,
                segment_outcome=outcome,
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
        "episode_id_offset": episode_id_offset,
        "num_total_grouped_segments": int(sum(raw_outcome_counter.values())),
        "num_kept_segments": int(len(segments)),
        "num_invalid_frames_dropped": int(invalid_frame_count),
        "raw_segment_outcomes": dict(raw_outcome_counter),
        "kept_segment_outcomes": dict(kept_outcome_counter),
        "episode_segment_counts": {str(k): int(v) for k, v in sorted(episode_segment_counter.items())},
    }
    return segments, summary


def load_segments(config: Dict) -> Tuple[List[SegmentData], Dict]:
    data_cfg = config["data"]
    root = Path(__file__).resolve().parent
    g1_joint_names = list(data_cfg["g1_joint_names"])
    allowed_outcomes = set(data_cfg.get("allowed_outcomes", ["success", "timeout"]))
    max_abs_velocity = float(data_cfg.get("max_abs_g1_velocity", 10.0))
    max_abs_delta = float(data_cfg.get("max_abs_g1_delta", 1.0))
    weighting_cfg = dict(data_cfg.get("sample_weighting", {}))

    # Support multi-source (data_sources list) or single-source (episodes_jsonl)
    sources = data_cfg.get("data_sources", None)
    if sources is None:
        sources = [{
            "name": "default",
            "episodes_jsonl": data_cfg["episodes_jsonl"],
            "events_jsonl": data_cfg["events_jsonl"],
        }]

    all_segments: List[SegmentData] = []
    source_summaries: List[Dict] = []
    ep_id_offset = 0

    for src in sources:
        ep_path = root / src["episodes_jsonl"]
        ev_path = root / src["events_jsonl"]
        print(f"[data] Loading {src['name']}: {ep_path.name}")
        segs, summ = _load_single_source(
            episodes_jsonl=ep_path,
            events_jsonl=ev_path,
            g1_joint_names=g1_joint_names,
            allowed_outcomes=allowed_outcomes,
            max_abs_velocity=max_abs_velocity,
            max_abs_delta=max_abs_delta,
            weighting_cfg=weighting_cfg,
            episode_id_offset=ep_id_offset,
        )
        summ["source_name"] = src["name"]
        source_summaries.append(summ)
        all_segments.extend(segs)
        # Advance offset past the highest episode id in this source
        if segs:
            max_ep = max(s.episode_id for s in segs)
            ep_id_offset = max_ep + 1000  # large gap to avoid collision
        else:
            ep_id_offset += 1000

    combined_summary = {
        "num_sources": len(sources),
        "total_segments": len(all_segments),
        "total_frames": sum(s.obs.shape[0] for s in all_segments),
        "total_episodes": len({s.episode_id for s in all_segments}),
        "outcome_totals": dict(Counter(s.outcome for s in all_segments)),
        "per_source": source_summaries,
    }
    print(f"[data] Combined: {combined_summary['total_segments']} segments, "
          f"{combined_summary['total_frames']} frames, "
          f"{combined_summary['total_episodes']} episodes")
    return all_segments, combined_summary


# ─── Train / val split ──────────────────────────────────────────────

def _episode_outcome_counts(segments: List[SegmentData]) -> Dict[int, Counter]:
    out: Dict[int, Counter] = defaultdict(Counter)
    for seg in segments:
        out[seg.episode_id][seg.outcome] += 1
    return out


def split_segments_by_episode(
    segments: List[SegmentData],
    train_split: float,
    seed: int,
) -> Tuple[List[SegmentData], List[SegmentData], Dict]:
    episodes = sorted({seg.episode_id for seg in segments})
    if len(episodes) < 2:
        raise RuntimeError("Need at least 2 episodes with usable segments.")

    val_n = max(1, int(round(len(episodes) * (1.0 - train_split))))
    val_n = min(val_n, len(episodes) - 1)
    rng = random.Random(seed)
    outcome_counts = _episode_outcome_counts(segments)
    desired_outcomes = {
        name for name in ("success", "timeout")
        if any(outcome_counts[ep][name] > 0 for ep in episodes)
    }

    def split_score(train_eps, val_eps):
        train_counter, val_counter = Counter(), Counter()
        for ep in train_eps:
            train_counter.update(outcome_counts[ep])
        for ep in val_eps:
            val_counter.update(outcome_counts[ep])
        missing = float(len(desired_outcomes - set(k for k, v in train_counter.items() if v > 0)))
        missing += float(len(desired_outcomes - set(k for k, v in val_counter.items() if v > 0)))
        t_s = float(train_counter.get("success", 0))
        t_t = float(t_s + train_counter.get("timeout", 0))
        v_s = float(val_counter.get("success", 0))
        v_t = float(v_s + val_counter.get("timeout", 0))
        ratio_gap = abs((t_s / t_t if t_t > 0 else 0.0) - (v_s / v_t if v_t > 0 else 0.0))
        size_gap = abs(len(train_eps) - len(val_eps) / max(1.0 - train_split, 1e-6))
        return (missing, ratio_gap, size_gap)

    best_train = episodes[:-val_n]
    best_val = episodes[-val_n:]
    best_score = split_score(best_train, best_val)

    for _ in range(2048):
        shuffled = episodes[:]
        rng.shuffle(shuffled)
        val_eps = sorted(shuffled[-val_n:])
        train_eps = sorted(shuffled[:-val_n])
        score = split_score(train_eps, val_eps)
        if score < best_score:
            best_score = score
            best_train, best_val = train_eps, val_eps
            if score[0] == 0.0 and score[1] < 0.05:
                break

    train_set, val_set = set(best_train), set(best_val)
    train_segs = [s for s in segments if s.episode_id in train_set]
    val_segs = [s for s in segments if s.episode_id in val_set]

    split_summary = {
        "train_episodes": best_train,
        "val_episodes": best_val,
        "train_segment_outcomes": dict(Counter(s.outcome for s in train_segs)),
        "val_segment_outcomes": dict(Counter(s.outcome for s in val_segs)),
        "train_num_segments": len(train_segs),
        "val_num_segments": len(val_segs),
    }
    return train_segs, val_segs, split_summary


# ─── Normalization ──────────────────────────────────────────────────

def fit_normalizer(segments: List[SegmentData]) -> Normalizer:
    obs = np.concatenate([seg.obs for seg in segments], axis=0)
    act = np.concatenate([seg.act for seg in segments], axis=0)
    obs_std = obs.std(axis=0)
    act_std = act.std(axis=0)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std).astype(np.float32)
    act_std = np.where(act_std < 1e-6, 1.0, act_std).astype(np.float32)
    return Normalizer(
        obs_mean=obs.mean(axis=0).astype(np.float32),
        obs_std=obs_std,
        act_mean=act.mean(axis=0).astype(np.float32),
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


# ─── Dataset builders ───────────────────────────────────────────────

def make_frame_dataset(
    segments: List[SegmentData],
    normalizer: Normalizer,
    obs_clip_z: float,
    act_clip_z: float,
):
    xs, ys, success, weights, meta = [], [], [], [], []
    for seg_id, seg in enumerate(segments, start=1):
        obs_z = normalize_obs(seg.obs, normalizer, obs_clip_z)
        act_z = normalize_act(seg.act, normalizer, act_clip_z)
        xs.append(obs_z)
        ys.append(act_z)
        success.append(np.full((obs_z.shape[0],), seg.success, dtype=np.float32))
        weights.append(seg.weights.astype(np.float32))
        meta.extend([{
            "segment_id": seg_id,
            "episode_id": seg.episode_id,
            "target_id": seg.target_id,
            "target_index": seg.target_index,
            "outcome": seg.outcome,
        }] * obs_z.shape[0])
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(success, axis=0),
        np.concatenate(weights, axis=0),
        meta,
    )


def make_seq_dataset(
    segments: List[SegmentData],
    seq_len: int,
    normalizer: Normalizer,
    obs_clip_z: float,
    act_clip_z: float,
):
    xs, ys, success, weights, meta = [], [], [], [], []
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
            meta.append({
                "segment_id": seg_id,
                "episode_id": seg.episode_id,
                "target_id": seg.target_id,
                "target_index": seg.target_index,
                "outcome": seg.outcome,
                "step_index": end,
            })
    if not xs:
        raise RuntimeError("No sequence samples after filtering; check seq_len or dataset settings.")
    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.stack(ys, axis=0).astype(np.float32),
        np.asarray(success, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        meta,
    )


# ─── Summary reporting ──────────────────────────────────────────────

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
