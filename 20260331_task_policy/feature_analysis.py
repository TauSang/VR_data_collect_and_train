"""Feature importance analysis for observation space reduction.

Analyzes which features in the 106D observation space are actually
informative for predicting actions, using:
1. Variance analysis (low-variance features are uninformative)
2. Feature-action correlation (Pearson)
3. Mutual information (nonlinear relevance)
4. PCA explained variance (redundancy detection)
5. Feature group importance summary
"""

import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Avoid importing common.py directly since it depends on torch.
# Instead, we replicate the minimal logic needed for analysis.

PHASES = ["idle", "reach", "align", "grasp", "hold", "timeout", "success", "unknown"]
PHASE_TO_INDEX = {name: idx for idx, name in enumerate(PHASES)}
NEAREST_HANDS = ["none", "left", "right"]
NEAREST_HAND_TO_INDEX = {name: idx for idx, name in enumerate(NEAREST_HANDS)}


def _get3(d, key):
    v = d.get(key, [0.0, 0.0, 0.0])
    if not isinstance(v, list) or len(v) != 3:
        return [0.0, 0.0, 0.0]
    return [float(x) if isinstance(x, (int, float)) else 0.0 for x in v]


def _get_pose_p(node):
    if isinstance(node, dict):
        p = node.get("p", None)
        if isinstance(p, list) and len(p) == 3:
            return [float(x) if isinstance(x, (int, float)) else 0.0 for x in p]
    return [0.0, 0.0, 0.0]


def _get_pose_q(node):
    if isinstance(node, dict):
        q = node.get("q", None)
        if isinstance(q, list) and len(q) == 4:
            return [float(x) if isinstance(x, (int, float)) else 0.0 for x in q]
    return [0.0, 0.0, 0.0, 1.0]


def _get_scalar(d, key, default=0.0):
    try:
        v = d.get(key, default)
        if v is None:
            return float(default)
        out = float(v)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _get_bool(d, key, default=False):
    v = d.get(key, default)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return 1.0 if float(v) != 0.0 else 0.0
    return 1.0 if default else 0.0


def _one_hot(size, index):
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def _task_distance(task):
    vals = [_get_scalar(task, k, float("nan")) for k in ("distToTarget", "distToTargetLeft", "distToTargetRight")]
    vals = [v for v in vals if math.isfinite(v)]
    return float(min(vals)) if vals else 0.0


def build_feature_names(joint_names, use_joint_velocities, include_end_effector_quat,
                        include_phase, include_ee_velocity=False, action_repr="euler_delta"):
    names = []
    for joint in joint_names:
        names.extend([f"joint_pos.{joint}.{a}" for a in "xyz"])
        if use_joint_velocities:
            names.extend([f"joint_vel.{joint}.{a}" for a in "xyz"])
    for hand in ("left", "right"):
        names.extend([f"ee.{hand}.p.{a}" for a in "xyz"])
        if include_end_effector_quat:
            names.extend([f"ee.{hand}.q.{a}" for a in ["x", "y", "z", "w"]])
        if include_ee_velocity:
            names.extend([f"ee_vel.{hand}.linear.{a}" for a in "xyz"])
            names.extend([f"ee_vel.{hand}.angular.{a}" for a in "xyz"])
    names.extend([f"target_rel_base.p.{a}" for a in "xyz"])
    names.extend([f"target_rel_left.p.{a}" for a in "xyz"])
    names.extend([f"target_rel_right.p.{a}" for a in "xyz"])
    names.extend(["dist.target", "dist.target_left", "dist.target_right"])
    names.extend(["contact.any", "contact.left", "contact.right"])
    names.extend(["hold.any_sec", "hold.left_sec", "hold.right_sec"])
    names.extend(["progress.completed_targets", "progress.targets_per_episode", "progress.ratio"])
    names.extend([f"nearest_hand.{n}" for n in NEAREST_HANDS])
    if include_phase:
        names.extend([f"phase.{n}" for n in PHASES])
    return names


def frame_to_obs_act(frame, joint_names, use_joint_velocities, include_end_effector_quat,
                     include_phase, include_ee_velocity=False, action_repr="euler_delta",
                     segment_outcome="unknown", weighting_cfg=None):
    if weighting_cfg is None:
        weighting_cfg = {}
    obs_d = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
    act_d = frame.get("action", {}) if isinstance(frame.get("action", {}), dict) else {}
    task = obs_d.get("task", {}) if isinstance(obs_d.get("task", {}), dict) else {}
    joint_pos = obs_d.get("jointPositions", {}) if isinstance(obs_d.get("jointPositions", {}), dict) else {}
    joint_vel = obs_d.get("jointVelocities", {}) if isinstance(obs_d.get("jointVelocities", {}), dict) else {}
    end_eff = obs_d.get("endEffector", {}) if isinstance(obs_d.get("endEffector", {}), dict) else {}
    ee_vel = obs_d.get("endEffectorVelocity", {}) if isinstance(obs_d.get("endEffectorVelocity", {}), dict) else {}

    if action_repr == "rot_vec":
        act_source = act_d.get("jointDeltaRotVec", {}) if isinstance(act_d.get("jointDeltaRotVec", {}), dict) else {}
    else:
        act_source = act_d.get("jointDelta", {}) if isinstance(act_d.get("jointDelta", {}), dict) else {}

    obs_vec = []
    act_vec = []

    for joint in joint_names:
        obs_vec.extend(_get3(joint_pos, joint))
        if use_joint_velocities:
            obs_vec.extend(_get3(joint_vel, joint))
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

    obs_vec.extend([
        _get_scalar(task, "distToTarget", 0.0),
        _get_scalar(task, "distToTargetLeft", 0.0),
        _get_scalar(task, "distToTargetRight", 0.0),
        _get_bool(task, "contactFlag", False),
        _get_bool(task, "contactFlagLeft", False),
        _get_bool(task, "contactFlagRight", False),
        _get_scalar(task, "contactHoldMs", 0.0) / 1000.0,
        _get_scalar(task, "contactHoldMsLeft", 0.0) / 1000.0,
        _get_scalar(task, "contactHoldMsRight", 0.0) / 1000.0,
        completed, total_targets, progress_ratio,
    ])
    obs_vec.extend(_one_hot(len(NEAREST_HANDS), nearest_hand_idx))
    if include_phase:
        phase = str(task.get("phaseLabel", "unknown") or "unknown")
        phase_idx = PHASE_TO_INDEX.get(phase, PHASE_TO_INDEX["unknown"])
        obs_vec.extend(_one_hot(len(PHASES), phase_idx))

    weight = 1.0
    return np.asarray(obs_vec, dtype=np.float32), np.asarray(act_vec, dtype=np.float32), weight


def load_segment_outcomes(events_path):
    segment_outcomes = {}
    episode_outcomes = {}
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            ev = json.loads(s)
            episode_id = int(ev.get("episodeId", 0) or 0)
            ev_type = str(ev.get("type", ""))
            payload = ev.get("payload", {}) if isinstance(ev.get("payload", {}), dict) else {}
            if ev_type == "target_spawned":
                key = (episode_id, int(payload.get("targetIndex", 0) or 0), int(payload.get("targetId", 0) or 0))
                if key[1] > 0 and key[2] > 0:
                    segment_outcomes.setdefault(key, "unknown")
            elif ev_type == "grasp_success":
                key = (episode_id, int(payload.get("targetIndex", 0) or 0), int(payload.get("targetId", 0) or 0))
                if key[1] > 0 and key[2] > 0:
                    segment_outcomes[key] = "success"
            elif ev_type == "episode_timeout":
                key = (episode_id, int(payload.get("targetIndex", 0) or 0), int(payload.get("targetId", 0) or 0))
                if key[1] > 0 and key[2] > 0 and segment_outcomes.get(key) != "success":
                    segment_outcomes[key] = "timeout"
            elif ev_type == "episode_end":
                episode_outcomes[episode_id] = str(payload.get("outcome", "unknown"))
    finalized = {}
    for key, outcome in segment_outcomes.items():
        if outcome == "unknown":
            ep_out = episode_outcomes.get(key[0], "unknown")
            if ep_out == "auto_end_by_new_start":
                outcome = "truncated"
            elif ep_out == "timeout":
                outcome = "timeout"
        finalized[key] = outcome
    return finalized, episode_outcomes


def main():
    config_path = Path(__file__).resolve().parent / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    data_cfg = config["data"]

    print("=" * 70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)

    # Load data directly from JSONL
    root = Path(__file__).resolve().parent
    episodes_jsonl = root / data_cfg["episodes_jsonl"]
    events_jsonl = root / data_cfg["events_jsonl"]
    joint_names = list(data_cfg["joint_names"])
    allowed_outcomes = set(data_cfg.get("allowed_outcomes", ["success", "timeout"]))

    segment_outcomes, episode_outcomes = load_segment_outcomes(events_jsonl)

    # OVERRIDE: use actual data schema (v2 data, not v3)
    # The collector5 data has only jointDelta (euler), no jointDeltaRotVec or endEffectorVelocity
    actual_use_joint_vel = bool(data_cfg.get("use_joint_velocities", True))
    actual_include_ee_quat = bool(data_cfg.get("include_end_effector_quat", True))
    actual_include_phase = bool(data_cfg.get("include_phase", True))
    actual_include_ee_vel = False  # NOT in data
    actual_action_repr = "euler_delta"  # only euler_delta in data

    # Group frames by segment
    grouped = defaultdict(list)
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
            grouped[(episode_id, target_index, target_id)].append(frame)

    # Build observation/action arrays per segment
    obs_list = []
    act_list = []
    for key in sorted(grouped.keys()):
        frames = grouped[key]
        outcome = segment_outcomes.get(key, "unknown")
        if outcome not in allowed_outcomes:
            continue
        for frame in frames:
            obs_vec, act_vec, w = frame_to_obs_act(
                frame, joint_names,
                use_joint_velocities=actual_use_joint_vel,
                include_end_effector_quat=actual_include_ee_quat,
                include_phase=actual_include_phase,
                include_ee_velocity=actual_include_ee_vel,
                action_repr=actual_action_repr,
                segment_outcome=outcome,
            )
            obs_list.append(obs_vec)
            act_list.append(act_vec)

    obs_all = np.stack(obs_list, axis=0)
    act_all = np.stack(act_list, axis=0)

    # Get feature names (using actual data schema)
    feature_names = build_feature_names(
        joint_names,
        use_joint_velocities=actual_use_joint_vel,
        include_end_effector_quat=actual_include_ee_quat,
        include_phase=actual_include_phase,
        include_ee_velocity=actual_include_ee_vel,
        action_repr=actual_action_repr,
    )

    n_frames, obs_dim = obs_all.shape
    act_dim = act_all.shape[1]
    print(f"\nTotal frames used: {n_frames}, obs_dim: {obs_dim}, act_dim: {act_dim}")
    print(f"Feature names count: {len(feature_names)}")

    if len(feature_names) != obs_dim:
        print(f"WARNING: feature_names ({len(feature_names)}) != obs_dim ({obs_dim})")
        # Pad or truncate
        if len(feature_names) < obs_dim:
            feature_names += [f"unknown_{i}" for i in range(len(feature_names), obs_dim)]
        else:
            feature_names = feature_names[:obs_dim]

    # === 1. Variance Analysis ===
    print("\n" + "=" * 70)
    print("1. FEATURE VARIANCE (raw, unnormalized)")
    print("=" * 70)
    variances = np.var(obs_all, axis=0)
    means = np.mean(obs_all, axis=0)

    # Group features
    groups = {}
    for i, name in enumerate(feature_names):
        parts = name.split(".")
        group = parts[0] if len(parts) > 0 else "other"
        if group not in groups:
            groups[group] = []
        groups[group].append(i)

    print(f"\n{'Group':<25} {'Count':>6} {'Mean Var':>12} {'Max Var':>12} {'Zero-Var':>10}")
    print("-" * 70)
    for group_name in sorted(groups.keys()):
        indices = groups[group_name]
        group_var = variances[indices]
        zero_var = np.sum(group_var < 1e-8)
        print(f"{group_name:<25} {len(indices):>6} {np.mean(group_var):>12.6f} "
              f"{np.max(group_var):>12.6f} {int(zero_var):>10}")

    # Low variance features
    low_var_threshold = 1e-6
    low_var_features = [(feature_names[i], variances[i]) for i in range(obs_dim) if variances[i] < low_var_threshold]
    if low_var_features:
        print(f"\n  Low-variance features (var < {low_var_threshold}):")
        for name, var in low_var_features:
            print(f"    {name}: var={var:.2e}, mean={means[feature_names.index(name)]:.4f}")

    # === 2. Feature-Action Correlation ===
    print("\n" + "=" * 70)
    print("2. FEATURE-ACTION CORRELATION (absolute Pearson, max across actions)")
    print("=" * 70)

    # Normalize for correlation
    obs_centered = obs_all - obs_all.mean(axis=0, keepdims=True)
    act_centered = act_all - act_all.mean(axis=0, keepdims=True)
    obs_norms = np.sqrt(np.sum(obs_centered ** 2, axis=0, keepdims=True) + 1e-12)
    act_norms = np.sqrt(np.sum(act_centered ** 2, axis=0, keepdims=True) + 1e-12)
    obs_normed = obs_centered / obs_norms
    act_normed = act_centered / act_norms

    # Correlation matrix: (obs_dim, act_dim)
    corr_matrix = obs_normed.T @ act_normed / n_frames
    max_abs_corr = np.max(np.abs(corr_matrix), axis=1)  # max across action dims
    mean_abs_corr = np.mean(np.abs(corr_matrix), axis=1)

    # Group-level correlation summary
    print(f"\n{'Group':<25} {'Count':>6} {'Mean MaxCorr':>14} {'Best MaxCorr':>14}")
    print("-" * 65)
    for group_name in sorted(groups.keys()):
        indices = groups[group_name]
        group_corr = max_abs_corr[indices]
        print(f"{group_name:<25} {len(indices):>6} {np.mean(group_corr):>14.4f} "
              f"{np.max(group_corr):>14.4f}")

    # Top features by correlation
    sorted_by_corr = np.argsort(-max_abs_corr)
    print(f"\n  Top 20 features by max |correlation| with actions:")
    for rank, idx in enumerate(sorted_by_corr[:20]):
        print(f"    {rank+1:>3}. {feature_names[idx]:<40} corr={max_abs_corr[idx]:.4f}")

    print(f"\n  Bottom 20 features by max |correlation| with actions:")
    for rank, idx in enumerate(sorted_by_corr[-20:]):
        print(f"    {obs_dim-19+rank:>3}. {feature_names[idx]:<40} corr={max_abs_corr[idx]:.4f}")

    # === 3. PCA Analysis ===
    print("\n" + "=" * 70)
    print("3. PCA ANALYSIS (cumulative explained variance)")
    print("=" * 70)

    # Z-score normalize
    obs_mean = obs_all.mean(axis=0)
    obs_std = obs_all.std(axis=0)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)
    obs_normed_z = (obs_all - obs_mean) / obs_std
    obs_normed_z = np.clip(obs_normed_z, -8.0, 8.0)

    # SVD for PCA
    obs_z_centered = obs_normed_z - obs_normed_z.mean(axis=0, keepdims=True)
    try:
        U, S, Vt = np.linalg.svd(obs_z_centered, full_matrices=False)
        explained_var = (S ** 2) / (n_frames - 1)
        total_var = np.sum(explained_var)
        cumulative_ratio = np.cumsum(explained_var) / total_var

        thresholds = [0.90, 0.95, 0.99]
        for t in thresholds:
            n_components = int(np.searchsorted(cumulative_ratio, t) + 1)
            print(f"  {t*100:.0f}% variance explained by {n_components} / {obs_dim} components")

        print(f"\n  Top 20 PCA components explained variance ratio:")
        for i in range(min(20, len(explained_var))):
            print(f"    PC{i+1:>3}: {explained_var[i]/total_var*100:>6.2f}%  "
                  f"(cumulative: {cumulative_ratio[i]*100:>6.2f}%)")
    except Exception as e:
        print(f"  PCA failed: {e}")

    # === 4. Feature Group Importance Ranking ===
    print("\n" + "=" * 70)
    print("4. FEATURE GROUP IMPORTANCE RANKING")
    print("=" * 70)

    group_scores = {}
    for group_name, indices in groups.items():
        avg_corr = float(np.mean(max_abs_corr[indices]))
        avg_var = float(np.mean(variances[indices]))
        n_zero_var = int(np.sum(variances[indices] < 1e-8))
        group_scores[group_name] = {
            "count": len(indices),
            "avg_max_corr": avg_corr,
            "avg_variance": avg_var,
            "n_zero_variance": n_zero_var,
            "importance_score": avg_corr * (1.0 - n_zero_var / max(len(indices), 1)),
        }

    # Sort by importance score
    ranked = sorted(group_scores.items(), key=lambda x: -x[1]["importance_score"])
    print(f"\n{'Rank':>4} {'Group':<25} {'Dims':>5} {'AvgCorr':>9} {'Score':>9} {'Recommendation':<15}")
    print("-" * 75)
    for rank, (name, info) in enumerate(ranked):
        score = info["importance_score"]
        if score > 0.1:
            rec = "KEEP"
        elif score > 0.03:
            rec = "CONSIDER"
        else:
            rec = "DROP"
        print(f"{rank+1:>4} {name:<25} {info['count']:>5} {info['avg_max_corr']:>9.4f} "
              f"{score:>9.4f} {rec:<15}")

    # === 5. Recommended Reduced Feature Set ===
    print("\n" + "=" * 70)
    print("5. RECOMMENDED OBSERVATION SPACE")
    print("=" * 70)

    # Always keep: joint_pos (24D) - essential proprioception
    # Always keep: ee (end effector position) - essential for reaching
    # Keep if useful: target_rel (goal-conditioned)
    # Evaluate: joint_vel, ee_vel, dist, contact, hold, progress, nearest_hand, phase

    keep_groups = []
    drop_groups = []
    consider_groups = []
    for name, info in ranked:
        if info["importance_score"] > 0.1:
            keep_groups.append(name)
        elif info["importance_score"] > 0.03:
            consider_groups.append(name)
        else:
            drop_groups.append(name)

    keep_dims = sum(groups[g].__len__() for g in keep_groups if g in groups)
    consider_dims = sum(groups[g].__len__() for g in consider_groups if g in groups)
    drop_dims = sum(groups[g].__len__() for g in drop_groups if g in groups)

    print(f"\n  KEEP ({keep_dims}D): {keep_groups}")
    print(f"  CONSIDER ({consider_dims}D): {consider_groups}")
    print(f"  DROP ({drop_dims}D): {drop_groups}")
    print(f"\n  Minimal: {keep_dims}D")
    print(f"  With CONSIDER: {keep_dims + consider_dims}D")
    print(f"  Current: {obs_dim}D")

    # Detailed feature-level recommendation
    print("\n  Per-feature recommendation (corr > 0.05 = KEEP):")
    keep_features = []
    drop_features = []
    for i in range(obs_dim):
        if max_abs_corr[i] > 0.05 or feature_names[i].startswith("joint_pos") or feature_names[i].startswith("ee."):
            keep_features.append(feature_names[i])
        else:
            drop_features.append(feature_names[i])

    print(f"\n  Features to KEEP ({len(keep_features)}D):")
    for f in keep_features:
        idx = feature_names.index(f)
        print(f"    {f:<45} corr={max_abs_corr[idx]:.4f}")

    print(f"\n  Features to DROP ({len(drop_features)}D):")
    for f in drop_features:
        idx = feature_names.index(f)
        print(f"    {f:<45} corr={max_abs_corr[idx]:.4f}")

    # Save results
    results = {
        "total_features": obs_dim,
        "keep_features": keep_features,
        "drop_features": drop_features,
        "keep_dim": len(keep_features),
        "drop_dim": len(drop_features),
        "group_importance": {k: v for k, v in group_scores.items()},
    }
    out_path = Path(__file__).resolve().parent / "outputs" / "feature_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
