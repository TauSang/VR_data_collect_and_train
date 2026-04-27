"""
VR 数据回放验证 — 在 MuJoCo G1 模型中回放 VR 采集的动作序列

用途:
  1. 检查 VR euler_delta 转换为 G1 关节角后是否物理可行
  2. 检查是否有关节超限、速度爆炸、穿模等问题
  3. 可视化回放过程, 对比 VR 端末端位置与 MuJoCo 末端位置

用法:
  python validate_replay.py --episodes PATH_TO_EPISODES.jsonl [--visualize]
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from joint_mapping import (
    G1_ARM_JOINTS,
    G1_JOINT_LIMITS,
    VR_TO_G1_MAPPING,
    clamp_to_limits,
    get_g1_stand_pose,
    vr_euler_delta_to_g1_joint_delta,
)


def load_episodes(jsonl_path: Path) -> Dict[int, List[dict]]:
    """Load episode frames grouped by (episodeId, targetIndex)."""
    grouped: Dict[int, List[dict]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            frame = json.loads(s)
            ep_id = int(frame.get("episodeId", 0) or 0)
            if ep_id <= 0:
                continue
            grouped.setdefault(ep_id, []).append(frame)

    # Sort by timestamp within each episode
    for ep_id in grouped:
        grouped[ep_id].sort(key=lambda f: float(f.get("timestamp", 0) or 0))

    return grouped


def extract_vr_joint_deltas(frame: dict) -> Optional[Dict[str, List[float]]]:
    """Extract jointDelta from a VR frame."""
    act = frame.get("action", {})
    if not isinstance(act, dict):
        return None
    # Try jointDelta (euler_delta, Schema v2)
    jd = act.get("jointDelta")
    if isinstance(jd, dict):
        return jd
    # Try jointDeltaRotVec (Schema v3)
    jd = act.get("jointDeltaRotVec")
    if isinstance(jd, dict):
        return jd
    return None


def extract_vr_ee_positions(frame: dict) -> Dict[str, Optional[np.ndarray]]:
    """Extract end-effector positions from VR frame for comparison."""
    obs = frame.get("obs", {})
    if not isinstance(obs, dict):
        return {"left": None, "right": None}
    ee = obs.get("endEffector", {})
    if not isinstance(ee, dict):
        return {"left": None, "right": None}
    result = {}
    for hand in ("left", "right"):
        node = ee.get(hand)
        if isinstance(node, dict) and isinstance(node.get("p"), list) and len(node["p"]) == 3:
            try:
                result[hand] = np.array([float(x) for x in node["p"]])
            except (ValueError, TypeError):
                result[hand] = None
        else:
            result[hand] = None
    return result


def get_g1_ee_body_names() -> Tuple[str, str]:
    """Body names of G1 end-effectors (hands)."""
    return "left_wrist_yaw_link", "right_wrist_yaw_link"


def run_replay(
    episodes: Dict[int, List[dict]],
    model_path: Path,
    visualize: bool = False,
    max_episodes: int = 5,
):
    """Replay VR actions on G1 model and collect statistics."""
    try:
        import mujoco
    except ImportError:
        print("[ERROR] mujoco not installed. Run: pip install mujoco")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Build joint index map
    joint_idx: Dict[str, int] = {}
    joint_qpos_adr: Dict[str, int] = {}
    for name in G1_ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            joint_idx[name] = jid
            joint_qpos_adr[name] = model.jnt_qposadr[jid]
        else:
            print(f"[WARN] Joint {name} not found in model")

    # Build actuator index map
    act_idx: Dict[str, int] = {}
    for name in G1_ARM_JOINTS:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            act_idx[name] = aid

    # EE body indices
    left_ee_name, right_ee_name = get_g1_ee_body_names()
    left_ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, left_ee_name)
    right_ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, right_ee_name)

    # Replay statistics
    stats = {
        "total_episodes": 0,
        "total_frames": 0,
        "joint_limit_violations": 0,
        "large_delta_frames": 0,
        "ee_position_errors": [],
    }

    viewer = None
    if visualize:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as e:
            print(f"[WARN] Could not launch viewer: {e}")
            visualize = False

    ep_ids = sorted(episodes.keys())[:max_episodes]
    print(f"\nReplaying {len(ep_ids)} episodes...")

    for ep_id in ep_ids:
        frames = episodes[ep_id]
        print(f"\n  Episode {ep_id}: {len(frames)} frames")

        # Reset to stand pose
        mujoco.mj_resetData(model, data)
        stand = get_g1_stand_pose()
        current_pos = dict(stand)
        for name, pos in stand.items():
            if name in joint_qpos_adr:
                data.qpos[joint_qpos_adr[name]] = pos
            if name in act_idx:
                data.ctrl[act_idx[name]] = pos
        mujoco.mj_forward(model, data)

        ep_violations = 0
        ep_large_deltas = 0

        for fi, frame in enumerate(frames):
            vr_deltas = extract_vr_joint_deltas(frame)
            if vr_deltas is None:
                continue

            # Convert VR deltas to G1 joint deltas
            g1_deltas = vr_euler_delta_to_g1_joint_delta(vr_deltas)

            # Check for large deltas
            max_delta = max(abs(v) for v in g1_deltas.values())
            if max_delta > 0.5:  # > 0.5 rad in one frame is suspicious
                ep_large_deltas += 1

            # Apply deltas
            for name in G1_ARM_JOINTS:
                current_pos[name] = current_pos.get(name, 0.0) + g1_deltas[name]

            # Clamp and check violations
            clamped = clamp_to_limits(current_pos)
            for name in G1_ARM_JOINTS:
                if abs(clamped[name] - current_pos[name]) > 1e-6:
                    ep_violations += 1
                current_pos[name] = clamped[name]

            # Set actuator targets
            for name, pos in current_pos.items():
                if name in act_idx:
                    data.ctrl[act_idx[name]] = pos

            # Step simulation
            mujoco.mj_step(model, data)

            # Compare EE positions
            vr_ee = extract_vr_ee_positions(frame)
            if left_ee_bid >= 0 and vr_ee["left"] is not None:
                mujoco_left_ee = data.xpos[left_ee_bid].copy()
                err = np.linalg.norm(mujoco_left_ee - vr_ee["left"])
                stats["ee_position_errors"].append(float(err))

            stats["total_frames"] += 1

            if visualize and viewer is not None and viewer.is_running():
                viewer.sync()

        stats["total_episodes"] += 1
        stats["joint_limit_violations"] += ep_violations
        stats["large_delta_frames"] += ep_large_deltas
        print(f"    Joint limit violations: {ep_violations}")
        print(f"    Large delta frames (>0.5 rad): {ep_large_deltas}")

    # Summary
    print("\n" + "=" * 60)
    print("REPLAY SUMMARY")
    print("=" * 60)
    print(f"  Episodes replayed:        {stats['total_episodes']}")
    print(f"  Total frames:             {stats['total_frames']}")
    print(f"  Joint limit violations:   {stats['joint_limit_violations']}")
    print(f"  Large-delta frames:       {stats['large_delta_frames']}")

    if stats["ee_position_errors"]:
        errors = np.array(stats["ee_position_errors"])
        print(f"  EE position error (m):")
        print(f"    mean = {errors.mean():.4f}")
        print(f"    std  = {errors.std():.4f}")
        print(f"    max  = {errors.max():.4f}")
        print(f"    p95  = {np.percentile(errors, 95):.4f}")
    else:
        print("  EE position error: N/A (no VR EE data in frames)")

    if visualize and viewer is not None:
        viewer.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Replay VR data on G1 MuJoCo model")
    parser.add_argument("--episodes", required=True, help="Path to episodes JSONL file")
    parser.add_argument("--visualize", action="store_true", help="Launch MuJoCo viewer")
    parser.add_argument("--max-episodes", type=int, default=5, help="Max episodes to replay")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to MuJoCo XML model (default: model/g1.xml)",
    )
    args = parser.parse_args()

    episodes_path = Path(args.episodes)
    if not episodes_path.exists():
        print(f"[ERROR] Episodes file not found: {episodes_path}")
        sys.exit(1)

    model_path = Path(args.model) if args.model else (Path(__file__).resolve().parent / "model" / "g1.xml")
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}. Run setup_model.py first.")
        sys.exit(1)

    print(f"Loading episodes from {episodes_path}...")
    episodes = load_episodes(episodes_path)
    print(f"Loaded {len(episodes)} episodes")

    run_replay(
        episodes,
        model_path=model_path,
        visualize=args.visualize,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
