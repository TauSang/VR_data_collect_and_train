"""
诊断ACT模型行为：记录关节角度轨迹，判断是有目的的到达还是在乱转。
同时对比 BC 和 ACT（不同scale）的关节运动方式。
"""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "mujoco_sim"))

import mujoco
from joint_mapping import G1_JOINT_LIMITS
from validate_policy import (
    PolicyRunner, build_obs, sample_reachable_target,
    mujoco_to_vr_pos, TARGET_REACH_THRESHOLD, HOLD_DURATION_S,
    EPISODE_TIMEOUT_S, SIM_DT, CONTROL_DT,
)

XML = str(Path("mujoco_sim/model/task_scene.xml"))

def run_one_target(policy, mj_model, mj_data, joint_qpos_adr, act_idx,
                   joint_names, tgt_mj, tgt_vr, left_ee_bid, right_ee_bid,
                   action_scale=1.0, action_ema=0.0, max_steps=300):
    """Run one target, return trajectory data."""
    policy.reset()
    prev_jp = {j: float(mj_data.qpos[a]) for j, a in joint_qpos_adr.items()}
    steps_per_ctrl = max(1, int(CONTROL_DT / SIM_DT))

    traj = {"joint_pos": [], "ee_dist": [], "actions": [], "joint_vel_from_delta": []}
    sim_t = 0.0

    for step in range(max_steps):
        obs, curr_jp = build_obs(mj_data, joint_names, joint_qpos_adr, prev_jp,
                                 tgt_vr, left_ee_bid, right_ee_bid)
        delta = policy.predict(obs, action_ema=action_ema, action_scale=action_scale)

        # Record
        jp = [float(mj_data.qpos[joint_qpos_adr[j]]) for j in joint_names]
        traj["joint_pos"].append(jp)
        traj["actions"].append(delta.tolist())

        # Apply
        for i, jname in enumerate(joint_names):
            adr = joint_qpos_adr.get(jname)
            if adr is not None:
                v = float(mj_data.qpos[adr]) + float(delta[i])
                lo, hi = G1_JOINT_LIMITS.get(jname, (-3.14, 3.14))
                mj_data.qpos[adr] = max(lo, min(hi, v))
                if jname in act_idx:
                    mj_data.ctrl[act_idx[jname]] = mj_data.qpos[adr]

        for _ in range(steps_per_ctrl):
            mujoco.mj_step(mj_model, mj_data)
            sim_t += SIM_DT

        l_ee = mj_data.xpos[left_ee_bid].copy()
        r_ee = mj_data.xpos[right_ee_bid].copy()
        d = min(np.linalg.norm(l_ee - tgt_mj), np.linalg.norm(r_ee - tgt_mj))
        traj["ee_dist"].append(float(d))
        prev_jp = curr_jp

    return traj


def analyze_trajectory(traj, joint_names, label):
    """Analyze a trajectory for spinning vs purposeful motion."""
    jp = np.array(traj["joint_pos"])    # (T, 8)
    actions = np.array(traj["actions"]) # (T, 8)
    dists = np.array(traj["ee_dist"])   # (T,)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    # 1. Distance to target over time
    print(f"\n距离变化:")
    print(f"  初始距离:   {dists[0]:.4f}m")
    print(f"  最小距离:   {dists.min():.4f}m (step {dists.argmin()})")
    print(f"  最终距离:   {dists[-1]:.4f}m")
    print(f"  距离趋势:   {'递减(好)' if dists[-1] < dists[0] else '递增或不变(差)'}")

    # 2. Joint range of motion (how much each joint moved total)
    total_travel = np.sum(np.abs(np.diff(jp, axis=0)), axis=0)
    net_displacement = jp[-1] - jp[0]
    print(f"\n关节运动分析:")
    print(f"  {'关节':<30s} {'总移动量(rad)':>12s} {'净位移(rad)':>12s} {'效率':>8s}")
    for i, jn in enumerate(joint_names):
        eff = abs(net_displacement[i]) / (total_travel[i] + 1e-10)
        # 效率: 1.0=直线运动, 0.0=来回做无用功
        print(f"  {jn:<30s} {total_travel[i]:>12.3f} {net_displacement[i]:>12.3f} {eff:>8.1%}")

    avg_eff = np.mean(np.abs(net_displacement) / (total_travel + 1e-10))
    print(f"\n  平均运动效率: {avg_eff:.1%}  (>30%=有目的, <10%=在乱转)")

    # 3. Check if joints hit limits frequently (sign of spinning)
    limits = [G1_JOINT_LIMITS.get(j, (-3.14, 3.14)) for j in joint_names]
    limit_hits = 0
    for i, (lo, hi) in enumerate(limits):
        at_lo = np.sum(np.abs(jp[:, i] - lo) < 0.01)
        at_hi = np.sum(np.abs(jp[:, i] - hi) < 0.01)
        if at_lo + at_hi > 10:
            limit_hits += 1
            print(f"  ⚠ {joint_names[i]} 频繁撞限位: 下限{at_lo}次, 上限{at_hi}次")

    # 4. Action direction consistency (do actions flip sign frequently?)
    sign_changes = np.sum(np.diff(np.sign(actions), axis=0) != 0, axis=0)
    print(f"\n动作方向翻转次数 (300步中):")
    for i, jn in enumerate(joint_names):
        status = "正常" if sign_changes[i] < 100 else "频繁翻转⚠"
        print(f"  {jn:<30s} {sign_changes[i]:>5d} 次  {status}")

    # 5. Action magnitude
    act_mag = np.linalg.norm(actions, axis=1)
    print(f"\n动作幅度: mean={act_mag.mean():.6f}, max={act_mag.max():.6f}")

    return avg_eff


# ── Setup ──
mj_model = mujoco.MjModel.from_xml_path(XML)
mj_data = mujoco.MjData(mj_model)

bc_policy = PolicyRunner("20260409train2/outputs/bc_oldparams/bc/run_20260410_012304/checkpoints/best.pt")
act_policy = PolicyRunner("20260409train2/outputs/act_v4/act/run_20260410_035001/checkpoints/best.pt")

joint_names = bc_policy.joint_names
joint_qpos_adr, act_idx = {}, {}
for name in joint_names:
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid >= 0:
        joint_qpos_adr[name] = mj_model.jnt_qposadr[jid]
    aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if aid >= 0:
        act_idx[name] = aid

left_ee_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
right_ee_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
left_sh_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link")
right_sh_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link")

# Sample a fixed target
rng = np.random.default_rng(42)
mujoco.mj_resetData(mj_model, mj_data)
for jname, adr in joint_qpos_adr.items():
    mj_data.qpos[adr] = bc_policy.init_joint_pos.get(jname, 0.0)
mujoco.mj_forward(mj_model, mj_data)
init_qpos = mj_data.qpos.copy()
init_qvel = mj_data.qvel.copy()

tgt_mj, tgt_vr = sample_reachable_target(rng, mj_model, mj_data, left_sh_bid, right_sh_bid)
print(f"目标位置 (MuJoCo): {np.round(tgt_mj, 4)}")

# ── Run BC (no scale) ──
mj_data.qpos[:] = init_qpos; mj_data.qvel[:] = init_qvel
mujoco.mj_forward(mj_model, mj_data)
bc_traj = run_one_target(bc_policy, mj_model, mj_data, joint_qpos_adr, act_idx,
                          joint_names, tgt_mj, tgt_vr, left_ee_bid, right_ee_bid,
                          action_scale=1.0, action_ema=0.5)
bc_eff = analyze_trajectory(bc_traj, joint_names, "BC模型 (scale=1.0, ema=0.5)")

# ── Run ACT scale=2.2 ──
mj_data.qpos[:] = init_qpos; mj_data.qvel[:] = init_qvel
mujoco.mj_forward(mj_model, mj_data)
act22_traj = run_one_target(act_policy, mj_model, mj_data, joint_qpos_adr, act_idx,
                             joint_names, tgt_mj, tgt_vr, left_ee_bid, right_ee_bid,
                             action_scale=2.2, action_ema=0.5)
act22_eff = analyze_trajectory(act22_traj, joint_names, "ACT v4 (scale=2.2, ema=0.5) — 你看到的版本")

# ── Run ACT scale=1.0 (no scale) ──
mj_data.qpos[:] = init_qpos; mj_data.qvel[:] = init_qvel
mujoco.mj_forward(mj_model, mj_data)
act10_traj = run_one_target(act_policy, mj_model, mj_data, joint_qpos_adr, act_idx,
                             joint_names, tgt_mj, tgt_vr, left_ee_bid, right_ee_bid,
                             action_scale=1.0, action_ema=0.5)
act10_eff = analyze_trajectory(act10_traj, joint_names, "ACT v4 (scale=1.0, 无缩放)")

# ── Run ACT scale=1.3 ──
mj_data.qpos[:] = init_qpos; mj_data.qvel[:] = init_qvel
mujoco.mj_forward(mj_model, mj_data)
act13_traj = run_one_target(act_policy, mj_model, mj_data, joint_qpos_adr, act_idx,
                             joint_names, tgt_mj, tgt_vr, left_ee_bid, right_ee_bid,
                             action_scale=1.3, action_ema=0.5)
act13_eff = analyze_trajectory(act13_traj, joint_names, "ACT v4 (scale=1.3, ema=0.5)")

print(f"\n\n{'='*60}")
print(f"  总结")
print(f"{'='*60}")
print(f"  BC  (scale=1.0): 效率={bc_eff:.1%}, 最小距离={min(bc_traj['ee_dist']):.3f}m")
print(f"  ACT (scale=1.0): 效率={act10_eff:.1%}, 最小距离={min(act10_traj['ee_dist']):.3f}m")
print(f"  ACT (scale=1.3): 效率={act13_eff:.1%}, 最小距离={min(act13_traj['ee_dist']):.3f}m")
print(f"  ACT (scale=2.2): 效率={act22_eff:.1%}, 最小距离={min(act22_traj['ee_dist']):.3f}m")
print(f"\n  效率 >30% = 有目的的运动")
print(f"  效率 <10% = 乱转/做无用功")
print(f"  成功阈值: <{TARGET_REACH_THRESHOLD}m")
