"""
MuJoCo 专家数据收集 — Jacobian-transpose IK 控制器

在 MuJoCo 仿真中生成高质量 reach 演示数据，
输出与 VR 数据管线兼容的 JSONL 文件。

用法:
    cd mujoco_sim
    python collect_expert.py --num-episodes 100 --out ../data_collector/mujoco_expert
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from joint_mapping import G1_JOINT_LIMITS

# ── Constants (与 validate_policy.py 一致) ──────────────────────────
SIM_DT = 0.002
CONTROL_DT = 1.0 / 30.0
STEPS_PER_CTRL = max(1, int(CONTROL_DT / SIM_DT))
TARGET_REACH_THRESHOLD = 0.16
HOLD_DURATION_S = 0.25
TARGET_TIMEOUT_S = 15.0
TARGETS_PER_EPISODE = 5

# 8 active joints
JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]
LEFT_JOINT_NAMES = JOINT_NAMES[:4]
RIGHT_JOINT_NAMES = JOINT_NAMES[4:]


def mujoco_to_vr(mj_xyz):
    """MuJoCo Z-up → VR Y-up: (x,y,z) → (x, z, -y)."""
    return np.array([mj_xyz[0], mj_xyz[2], -mj_xyz[1]])


# ── Rotation: VR-world → VR-robot-local frame (same as validate_policy.py) ──
_R_BASE = np.array([[0.0, 0.0, -1.0],
                     [0.0, 1.0,  0.0],
                     [-1.0, 0.0, 0.0]], dtype=np.float64)
_ROBOT_BASE_VR = np.array([0.0, 0.0, 0.0])


def sample_target(rng, mj_model, mj_data, left_sh_bid, right_sh_bid, max_reach=0.28):
    """Sample reachable target (matching validate_policy.py logic)."""
    mujoco.mj_forward(mj_model, mj_data)
    shoulder = mj_data.xpos[left_sh_bid if rng.random() < 0.5 else right_sh_bid].copy()
    for _ in range(100):
        offset = rng.uniform(-max_reach, max_reach, size=3)
        if np.linalg.norm(offset) <= max_reach and offset[2] > -0.15:
            break
    return shoulder + offset


def jacobian_delta(mj_model, mj_data, target_mj, ee_bid, dof_indices, gain, damping=0.01):
    """Damped pseudoinverse IK: delta_q = gain * J^+ @ error, clamped.

    Uses J^+ = J^T @ (J @ J^T + λI)^{-1} for stability near singularities.
    Adaptive gain: larger steps when far from target.
    """
    jacp = np.zeros((3, mj_model.nv))
    mujoco.mj_jacBody(mj_model, mj_data, jacp, None, ee_bid)
    J = jacp[:, dof_indices]  # (3, n_arm_joints)
    error = target_mj - mj_data.xpos[ee_bid]
    err_norm = np.linalg.norm(error)
    # Adaptive damping: lower when far (more aggressive), higher when close (stable)
    adaptive_damping = damping * min(1.0, 0.05 / max(err_norm, 1e-6))
    JJT = J @ J.T + adaptive_damping * np.eye(3)
    delta = gain * J.T @ np.linalg.solve(JJT, error)
    MAX_DELTA = 0.15  # Larger max step for faster convergence
    return np.clip(delta, -MAX_DELTA, MAX_DELTA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="../data_collector/mujoco_expert")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--gain", type=float, default=2.0,
                        help="IK gain, tunes action magnitude (higher = faster convergence)")
    parser.add_argument("--action-noise", type=float, default=0.002,
                        help="Gaussian noise std added to expert actions for diversity")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_xml = Path(__file__).parent / "model" / "task_scene.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
    mj_data = mujoco.MjData(mj_model)
    rng = np.random.default_rng(args.seed)

    # Joint → MuJoCo addresses
    joint_qpos_adr = {}
    joint_dof_adr = {}
    for jname in JOINT_NAMES:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        joint_qpos_adr[jname] = mj_model.jnt_qposadr[jid]
        joint_dof_adr[jname] = mj_model.jnt_dofadr[jid]

    left_dof_idx = [joint_dof_adr[j] for j in LEFT_JOINT_NAMES]
    right_dof_idx = [joint_dof_adr[j] for j in RIGHT_JOINT_NAMES]

    # Actuator indices (for setting ctrl — critical for position actuators)
    act_idx = {}
    for jname in JOINT_NAMES:
        aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        if aid >= 0:
            act_idx[jname] = aid

    left_ee_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_ee_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    left_sh_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link")
    right_sh_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link")

    # Target ball (mocap body) for visualization
    tgt_ball_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
    tgt_mocap_id = mj_model.body_mocapid[tgt_ball_bid] if tgt_ball_bid >= 0 else -1

    # robot base in VR coords — ground level to match VR recordings
    robot_base_vr = _ROBOT_BASE_VR

    episodes_path = out_dir / "episodes.jsonl"
    events_path = out_dir / "events.jsonl"

    total_frames = 0
    total_targets = 0
    total_success = 0

    with open(episodes_path, "w", encoding="utf-8") as ep_f, \
         open(events_path, "w", encoding="utf-8") as ev_f:

        for ep in range(1, args.num_episodes + 1):
            # Reset to random starting pose within inner half of joint range
            mujoco.mj_resetData(mj_model, mj_data)
            for jname in JOINT_NAMES:
                lo, hi = G1_JOINT_LIMITS.get(jname, (-1.0, 1.0))
                center = (lo + hi) / 2
                half_range = (hi - lo) / 4
                mj_data.qpos[joint_qpos_adr[jname]] = np.clip(
                    center + rng.uniform(-half_range, half_range), lo, hi
                )
                if jname in act_idx:
                    mj_data.ctrl[act_idx[jname]] = mj_data.qpos[joint_qpos_adr[jname]]
            mujoco.mj_forward(mj_model, mj_data)

            prev_pos = {j: float(mj_data.qpos[joint_qpos_adr[j]]) for j in JOINT_NAMES}

            for tgt_num in range(1, TARGETS_PER_EPISODE + 1):
                target_mj = sample_target(rng, mj_model, mj_data, left_sh_bid, right_sh_bid)
                target_vr = mujoco_to_vr(target_mj)
                target_id = ep * 100 + tgt_num

                if tgt_mocap_id >= 0:
                    mj_data.mocap_pos[tgt_mocap_id] = target_mj
                    mujoco.mj_forward(mj_model, mj_data)

                # target_spawned event
                ev_f.write(json.dumps({
                    "episodeId": ep,
                    "type": "target_spawned",
                    "payload": {"targetIndex": tgt_num, "targetId": target_id},
                }) + "\n")

                total_targets += 1
                sim_t = 0.0
                hold_t = 0.0
                reached = False

                while sim_t < TARGET_TIMEOUT_S:
                    mujoco.mj_forward(mj_model, mj_data)

                    # Current EE positions (MuJoCo frame)
                    left_ee_mj = mj_data.xpos[left_ee_bid].copy()
                    right_ee_mj = mj_data.xpos[right_ee_bid].copy()
                    d_left = float(np.linalg.norm(left_ee_mj - target_mj))
                    d_right = float(np.linalg.norm(right_ee_mj - target_mj))

                    # Pick closer arm
                    if d_left <= d_right:
                        primary_ee_bid = left_ee_bid
                        primary_dof_idx = left_dof_idx
                        primary_slice = slice(0, 4)
                        secondary_slice = slice(4, 8)
                    else:
                        primary_ee_bid = right_ee_bid
                        primary_dof_idx = right_dof_idx
                        primary_slice = slice(4, 8)
                        secondary_slice = slice(0, 4)

                    d_min = min(d_left, d_right)

                    # Expert action: IK when far, zero when at target
                    delta = np.zeros(8)
                    if d_min > TARGET_REACH_THRESHOLD * 0.6:
                        # Still need to reach — use IK
                        p_delta = jacobian_delta(
                            mj_model, mj_data, target_mj,
                            primary_ee_bid, primary_dof_idx, args.gain,
                        )
                        delta[primary_slice] = p_delta
                        # Secondary arm: near-zero noise
                        delta[secondary_slice] = rng.normal(0, 0.0005, size=4)
                        # Diversity noise on primary arm too
                        delta[primary_slice] += rng.normal(0, args.action_noise, size=4)
                        frame_label = "approaching" if d_min < 0.3 else "moving"
                    else:
                        # At target — output zero delta (crucial for learning to STOP)
                        delta[:] = rng.normal(0, 0.001, size=8)
                        frame_label = "holding"

                    # Build observation in robot-local frame (matching VR recording)
                    left_ee_vr = mujoco_to_vr(left_ee_mj)
                    right_ee_vr = mujoco_to_vr(right_ee_mj)

                    # All spatial features in robot-local frame
                    left_ee_local = _R_BASE @ (left_ee_vr - robot_base_vr)
                    right_ee_local = _R_BASE @ (right_ee_vr - robot_base_vr)
                    target_rel_base = _R_BASE @ (target_vr - robot_base_vr)
                    target_rel_left = _R_BASE @ (target_vr - left_ee_vr)
                    target_rel_right = _R_BASE @ (target_vr - right_ee_vr)

                    positions = {}
                    velocities = {}
                    for jname in JOINT_NAMES:
                        v = float(mj_data.qpos[joint_qpos_adr[jname]])
                        positions[jname] = v
                        velocities[jname] = (v - prev_pos.get(jname, v)) / CONTROL_DT

                    d_min = min(d_left, d_right)
                    frame = {
                        "episodeId": ep,
                        "obs": {
                            "g1JointPositions": positions,
                            "g1JointVelocities": velocities,
                            "endEffector": {
                                "left": {"p": left_ee_local.tolist()},
                                "right": {"p": right_ee_local.tolist()},
                            },
                            "task": {
                                "targetIndex": tgt_num,
                                "targetId": target_id,
                                "targetRelToRobotBase": {"p": target_rel_base.tolist()},
                                "targetRelToLeftHand": {"p": target_rel_left.tolist()},
                                "targetRelToRightHand": {"p": target_rel_right.tolist()},
                                "distToTarget": d_min,
                                "distToTargetLeft": d_left,
                                "distToTargetRight": d_right,
                            },
                        },
                        "action": {
                            "g1JointDelta": {jname: float(delta[i]) for i, jname in enumerate(JOINT_NAMES)},
                        },
                        "frameLabel": frame_label,
                    }
                    ep_f.write(json.dumps(frame) + "\n")
                    total_frames += 1

                    # Apply action (set both qpos AND ctrl for position actuators)
                    curr_pos = {}
                    for i, jname in enumerate(JOINT_NAMES):
                        adr = joint_qpos_adr[jname]
                        v = float(mj_data.qpos[adr]) + delta[i]
                        lo, hi = G1_JOINT_LIMITS.get(jname, (-3.14, 3.14))
                        mj_data.qpos[adr] = np.clip(v, lo, hi)
                        if jname in act_idx:
                            mj_data.ctrl[act_idx[jname]] = mj_data.qpos[adr]
                        curr_pos[jname] = float(mj_data.qpos[adr])

                    # Step physics
                    for _ in range(STEPS_PER_CTRL):
                        mujoco.mj_step(mj_model, mj_data)
                        sim_t += SIM_DT

                    # Check success
                    d = min(
                        float(np.linalg.norm(mj_data.xpos[left_ee_bid] - target_mj)),
                        float(np.linalg.norm(mj_data.xpos[right_ee_bid] - target_mj)),
                    )
                    if d <= TARGET_REACH_THRESHOLD:
                        hold_t += CONTROL_DT
                        if hold_t >= HOLD_DURATION_S:
                            reached = True
                            break
                    else:
                        hold_t = 0.0

                    prev_pos = curr_pos

                # Outcome event
                if reached:
                    total_success += 1
                    ev_f.write(json.dumps({
                        "episodeId": ep,
                        "type": "target_success",
                        "payload": {"targetIndex": tgt_num, "targetId": target_id, "outcome": "success"},
                    }) + "\n")
                else:
                    ev_f.write(json.dumps({
                        "episodeId": ep,
                        "type": "episode_timeout",
                        "payload": {"targetIndex": tgt_num, "targetId": target_id, "outcome": "timeout"},
                    }) + "\n")

            # Episode end
            ev_f.write(json.dumps({
                "episodeId": ep,
                "type": "episode_end",
                "payload": {"outcome": "completed"},
            }) + "\n")

            if ep % 10 == 0:
                rate = total_success / max(total_targets, 1)
                print(f"Episode {ep}/{args.num_episodes}: "
                      f"success={total_success}/{total_targets} ({rate:.1%}), "
                      f"frames={total_frames}")

    rate = total_success / max(total_targets, 1)
    print(f"\nDone! {args.num_episodes} episodes, {total_targets} targets, "
          f"{total_success} reached ({rate:.1%})")
    print(f"Total frames: {total_frames}")
    print(f"Output: {episodes_path}, {events_path}")


if __name__ == "__main__":
    main()
