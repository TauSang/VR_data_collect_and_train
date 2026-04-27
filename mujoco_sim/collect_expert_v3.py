"""
Expert data collector v3 — Iterative IK solver
Instead of single-step Jacobian, run multiple IK iterations to solve
for a joint configuration that reaches the target, then smoothly
interpolate from current to goal config.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from joint_mapping import G1_JOINT_LIMITS

SIM_DT = 0.002
CONTROL_DT = 1.0 / 30.0
STEPS_PER_CTRL = max(1, int(CONTROL_DT / SIM_DT))
TARGET_REACH_THRESHOLD = 0.16
HOLD_DURATION_S = 0.25
TARGET_TIMEOUT_S = 15.0
TARGETS_PER_EPISODE = 5
HOLD_FRAMES_AFTER_REACH = 30  # 1 second of holding at target

JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]
LEFT_JOINT_NAMES = JOINT_NAMES[:4]
RIGHT_JOINT_NAMES = JOINT_NAMES[4:]


def mujoco_to_vr(mj_xyz):
    return np.array([mj_xyz[0], mj_xyz[2], -mj_xyz[1]])


# Rotation and base matching validate_policy.py
_R_BASE = np.array([[0.0, 0.0, -1.0],
                     [0.0, 1.0,  0.0],
                     [-1.0, 0.0, 0.0]], dtype=np.float64)
_ROBOT_BASE_VR = np.array([0.0, 0.0, 0.0])


def sample_target(rng, mj_model, mj_data, left_sh_bid, right_sh_bid, max_reach=0.28):
    mujoco.mj_forward(mj_model, mj_data)
    shoulder = mj_data.xpos[left_sh_bid if rng.random() < 0.5 else right_sh_bid].copy()
    for _ in range(100):
        offset = rng.uniform(-max_reach, max_reach, size=3)
        if np.linalg.norm(offset) <= max_reach and offset[2] > -0.15:
            break
    return shoulder + offset


def ik_solve(mj_model, mj_data, target_mj, ee_bid, dof_indices, joint_qpos_adr,
             joint_names_subset, max_iters=200, tol=0.02):
    """Iterative IK: find joint config that places ee near target.
    
    Uses damped Jacobian pseudoinverse with line search.
    Returns (success, goal_qpos_dict) where goal_qpos_dict maps joint_name -> target angle.
    """
    # Save state
    saved_qpos = mj_data.qpos.copy()
    
    best_dist = float('inf')
    best_qpos = {}
    
    for it in range(max_iters):
        mujoco.mj_forward(mj_model, mj_data)
        error = target_mj - mj_data.xpos[ee_bid]
        dist = np.linalg.norm(error)
        
        if dist < best_dist:
            best_dist = dist
            best_qpos = {jn: float(mj_data.qpos[joint_qpos_adr[jn]]) for jn in joint_names_subset}
        
        if dist < tol:
            # Restore original state
            mj_data.qpos[:] = saved_qpos
            mujoco.mj_forward(mj_model, mj_data)
            return True, best_qpos
        
        # Jacobian
        jacp = np.zeros((3, mj_model.nv))
        mujoco.mj_jacBody(mj_model, mj_data, jacp, None, ee_bid)
        J = jacp[:, dof_indices]
        
        # Damped pseudoinverse
        damping = 0.001 * (1.0 + dist)  # Lower damping when close
        JJT = J @ J.T + damping * np.eye(3)
        delta_q = J.T @ np.linalg.solve(JJT, error)
        
        # Adaptive step size
        step = min(1.0, 0.3 / max(np.max(np.abs(delta_q)), 1e-6))
        delta_q *= step
        
        # Apply with joint limits
        for i, jn in enumerate(joint_names_subset):
            adr = joint_qpos_adr[jn]
            lo, hi = G1_JOINT_LIMITS.get(jn, (-3.14, 3.14))
            mj_data.qpos[adr] = np.clip(float(mj_data.qpos[adr]) + delta_q[i], lo, hi)
    
    # Restore
    mj_data.qpos[:] = saved_qpos
    mujoco.mj_forward(mj_model, mj_data)
    return best_dist < TARGET_REACH_THRESHOLD, best_qpos


def build_frame(positions, velocities, left_ee_vr, right_ee_vr, robot_base_vr,
                target_vr, d_min, d_left, d_right, delta, ep, tgt_num, target_id,
                frame_label):
    left_ee_local = _R_BASE @ (left_ee_vr - robot_base_vr)
    right_ee_local = _R_BASE @ (right_ee_vr - robot_base_vr)
    target_rel_base = _R_BASE @ (target_vr - robot_base_vr)
    target_rel_left = _R_BASE @ (target_vr - left_ee_vr)
    target_rel_right = _R_BASE @ (target_vr - right_ee_vr)
    return {
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
            "g1JointDelta": {jn: float(delta[i]) for i, jn in enumerate(JOINT_NAMES)},
        },
        "frameLabel": frame_label,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="../data_collector/mujoco_expert_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-delta", type=float, default=0.10,
                        help="Max joint delta per control step during interpolation")
    parser.add_argument("--action-noise", type=float, default=0.003)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_xml = Path(__file__).parent / "model" / "task_scene.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
    mj_data = mujoco.MjData(mj_model)
    rng = np.random.default_rng(args.seed)

    joint_qpos_adr = {}
    joint_dof_adr = {}
    for jname in JOINT_NAMES:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        joint_qpos_adr[jname] = mj_model.jnt_qposadr[jid]
        joint_dof_adr[jname] = mj_model.jnt_dofadr[jid]

    left_dof_idx = [joint_dof_adr[j] for j in LEFT_JOINT_NAMES]
    right_dof_idx = [joint_dof_adr[j] for j in RIGHT_JOINT_NAMES]

    # Actuator indices (critical: must set ctrl for position actuators with kp=500)
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

    robot_base_vr = _ROBOT_BASE_VR

    episodes_path = out_dir / "episodes.jsonl"
    events_path = out_dir / "events.jsonl"

    total_frames = 0
    total_targets = 0
    total_ik_solved = 0
    total_reached = 0

    with open(episodes_path, "w", encoding="utf-8") as ep_f, \
         open(events_path, "w", encoding="utf-8") as ev_f:

        for ep in range(1, args.num_episodes + 1):
            mujoco.mj_resetData(mj_model, mj_data)
            # Start from neutral pose (center of joint range)
            for jname in JOINT_NAMES:
                lo, hi = G1_JOINT_LIMITS.get(jname, (-1.0, 1.0))
                center = (lo + hi) / 2
                half_range = (hi - lo) / 6  # Tighter random range
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

                ev_f.write(json.dumps({
                    "episodeId": ep, "type": "target_spawned",
                    "payload": {"targetIndex": tgt_num, "targetId": target_id},
                }) + "\n")

                total_targets += 1

                # Decide which arm to use
                mujoco.mj_forward(mj_model, mj_data)
                d_left = np.linalg.norm(mj_data.xpos[left_ee_bid] - target_mj)
                d_right = np.linalg.norm(mj_data.xpos[right_ee_bid] - target_mj)
                if d_left <= d_right:
                    primary_ee_bid = left_ee_bid
                    primary_dof_idx = left_dof_idx
                    primary_names = LEFT_JOINT_NAMES
                    primary_slice = slice(0, 4)
                    secondary_slice = slice(4, 8)
                else:
                    primary_ee_bid = right_ee_bid
                    primary_dof_idx = right_dof_idx
                    primary_names = RIGHT_JOINT_NAMES
                    primary_slice = slice(4, 8)
                    secondary_slice = slice(0, 4)

                # Phase 1: Solve IK to find goal configuration
                ik_ok, goal_qpos = ik_solve(
                    mj_model, mj_data, target_mj, primary_ee_bid,
                    primary_dof_idx, joint_qpos_adr, primary_names,
                    max_iters=300, tol=0.02,
                )
                if ik_ok:
                    total_ik_solved += 1

                # Phase 2: Interpolate from current to goal config
                sim_t = 0.0
                hold_t = 0.0
                reached = False
                hold_frames_left = HOLD_FRAMES_AFTER_REACH

                while sim_t < TARGET_TIMEOUT_S:
                    mujoco.mj_forward(mj_model, mj_data)

                    left_ee_mj = mj_data.xpos[left_ee_bid].copy()
                    right_ee_mj = mj_data.xpos[right_ee_bid].copy()
                    d_left_curr = float(np.linalg.norm(left_ee_mj - target_mj))
                    d_right_curr = float(np.linalg.norm(right_ee_mj - target_mj))
                    d_min = min(d_left_curr, d_right_curr)

                    # Compute action delta
                    delta = np.zeros(8)
                    if ik_ok and not reached:
                        # Move toward IK goal
                        for i, jn in enumerate(primary_names):
                            curr = float(mj_data.qpos[joint_qpos_adr[jn]])
                            goal = goal_qpos[jn]
                            diff = goal - curr
                            # Proportional control toward goal, clamped
                            delta_j = np.clip(diff * 0.3, -args.max_delta, args.max_delta)
                            idx = JOINT_NAMES.index(jn)
                            delta[idx] = delta_j + rng.normal(0, args.action_noise)
                        # Secondary arm: near zero
                        for idx in range(8):
                            if delta[idx] == 0:
                                delta[idx] = rng.normal(0, 0.001)
                        frame_label = "approaching" if d_min < 0.3 else "moving"
                    elif reached and hold_frames_left > 0:
                        # At target — zero delta (teach the model to STOP)
                        delta[:] = rng.normal(0, 0.001, size=8)
                        frame_label = "holding"
                        hold_frames_left -= 1
                    else:
                        # IK failed or done holding — use online Jacobian fallback
                        if not reached:
                            jacp = np.zeros((3, mj_model.nv))
                            mujoco.mj_jacBody(mj_model, mj_data, jacp, None, primary_ee_bid)
                            J = jacp[:, primary_dof_idx]
                            error = target_mj - mj_data.xpos[primary_ee_bid]
                            JJT = J @ J.T + 0.01 * np.eye(3)
                            p_delta = 5.0 * J.T @ np.linalg.solve(JJT, error)
                            p_delta = np.clip(p_delta, -args.max_delta, args.max_delta)
                            delta[primary_slice] = p_delta + rng.normal(0, args.action_noise, size=4)
                            delta[secondary_slice] = rng.normal(0, 0.001, size=4)
                            frame_label = "approaching" if d_min < 0.3 else "moving"
                        else:
                            break  # Done with this target

                    # Build and record observation
                    left_ee_vr = mujoco_to_vr(left_ee_mj)
                    right_ee_vr = mujoco_to_vr(right_ee_mj)

                    positions = {}
                    velocities = {}
                    for jname in JOINT_NAMES:
                        v = float(mj_data.qpos[joint_qpos_adr[jname]])
                        positions[jname] = v
                        velocities[jname] = (v - prev_pos.get(jname, v)) / CONTROL_DT

                    frame = build_frame(
                        positions, velocities, left_ee_vr, right_ee_vr,
                        robot_base_vr, target_vr, d_min, d_left_curr, d_right_curr,
                        delta, ep, tgt_num, target_id, frame_label,
                    )
                    ep_f.write(json.dumps(frame) + "\n")
                    total_frames += 1

                    # Apply action (set both qpos AND ctrl for position actuators)
                    for i, jname in enumerate(JOINT_NAMES):
                        adr = joint_qpos_adr[jname]
                        v = float(mj_data.qpos[adr]) + delta[i]
                        lo, hi = G1_JOINT_LIMITS.get(jname, (-3.14, 3.14))
                        mj_data.qpos[adr] = np.clip(v, lo, hi)
                        if jname in act_idx:
                            mj_data.ctrl[act_idx[jname]] = mj_data.qpos[adr]
                    prev_pos = {j: float(mj_data.qpos[joint_qpos_adr[j]]) for j in JOINT_NAMES}

                    # Step physics
                    for _ in range(STEPS_PER_CTRL):
                        mujoco.mj_step(mj_model, mj_data)
                        sim_t += SIM_DT

                    # Check success
                    mujoco.mj_forward(mj_model, mj_data)
                    d = min(
                        float(np.linalg.norm(mj_data.xpos[left_ee_bid] - target_mj)),
                        float(np.linalg.norm(mj_data.xpos[right_ee_bid] - target_mj)),
                    )
                    if d <= TARGET_REACH_THRESHOLD:
                        hold_t += CONTROL_DT
                        if hold_t >= HOLD_DURATION_S:
                            reached = True
                    else:
                        hold_t = 0.0

                outcome = "success" if reached else "timeout"
                if reached:
                    total_reached += 1
                ev_f.write(json.dumps({
                    "episodeId": ep,
                    "type": "target_reached" if reached else "episode_timeout",
                    "payload": {"targetIndex": tgt_num, "targetId": target_id, "outcome": outcome},
                }) + "\n")

            ev_f.write(json.dumps({
                "episodeId": ep, "type": "episode_end",
                "payload": {"outcome": "completed"},
            }) + "\n")

            if ep % 10 == 0:
                print(f"Episode {ep}/{args.num_episodes}: ik_solved={total_ik_solved}/{total_targets} "
                      f"({100*total_ik_solved/total_targets:.1f}%), "
                      f"reached={total_reached}/{total_targets} ({100*total_reached/total_targets:.1f}%), "
                      f"frames={total_frames}")

    print(f"\nDone! {args.num_episodes} episodes, {total_targets} targets")
    print(f"  IK solved: {total_ik_solved}/{total_targets} ({100*total_ik_solved/total_targets:.1f}%)")
    print(f"  Reached: {total_reached}/{total_targets} ({100*total_reached/total_targets:.1f}%)")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {episodes_path}, {events_path}")


if __name__ == "__main__":
    main()
