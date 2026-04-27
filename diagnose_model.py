"""Quick diagnostic: run the BC model and track joint + action drift over time."""
import sys, argparse
import numpy as np
import mujoco

sys.path.insert(0, "e:/XT/vr-robot-control/mujoco_sim")
from validate_policy import (
    PolicyRunner, build_obs, sample_reachable_target,
    mujoco_to_vr_pos, vr_to_mujoco_pos,
    CONTROL_DT, SIM_DT, G1_JOINT_LIMITS,
)
from pathlib import Path

ckpt = "E:/XT/vr-robot-control/20260409train2/outputs/dagger_v2/bc/run_20260410_123054/checkpoints/best.pt"
xml = Path("e:/XT/vr-robot-control/mujoco_sim/model/task_scene.xml")

policy = PolicyRunner(ckpt)
mj_model = mujoco.MjModel.from_xml_path(str(xml))
mj_data = mujoco.MjData(mj_model)
rng = np.random.default_rng(42)

joint_names = policy.joint_names
joint_qpos_adr = {}
act_idx = {}
for jname in joint_names:
    jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    joint_qpos_adr[jname] = mj_model.jnt_qposadr[jid]
    aid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
    if aid >= 0:
        act_idx[jname] = aid

left_ee_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
right_ee_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
left_sh_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link")
right_sh_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link")

tgt_ball_jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "target_ball_joint")
tgt_ball_adr = mj_model.jnt_qposadr[tgt_ball_jid] if tgt_ball_jid >= 0 else -1

# Initialize
mujoco.mj_resetData(mj_model, mj_data)
for jname, adr in joint_qpos_adr.items():
    mj_data.qpos[adr] = policy.init_joint_pos.get(jname, 0.0)
mujoco.mj_forward(mj_model, mj_data)
policy.reset()

prev_jp = {j: float(mj_data.qpos[a]) for j, a in joint_qpos_adr.items()}

# Sample one target
tgt_mj, tgt_vr = sample_reachable_target(rng, mj_model, mj_data, left_sh_bid, right_sh_bid)
if tgt_ball_adr >= 0:
    mj_data.qpos[tgt_ball_adr:tgt_ball_adr+3] = tgt_mj
    mujoco.mj_forward(mj_model, mj_data)

print(f"Target MJ: {tgt_mj}")
print(f"Target VR: {tgt_vr}")
print(f"Left shoulder MJ: {mj_data.xpos[left_sh_bid]}")
print(f"Right shoulder MJ: {mj_data.xpos[right_sh_bid]}")
print(f"Left EE MJ: {mj_data.xpos[left_ee_bid]}")
print(f"Right EE MJ: {mj_data.xpos[right_ee_bid]}")
print()

# Print normalizer stats for spatial features (last 15 dims)
n = len(policy.obs_mean)
print(f"Normalizer obs_mean (last 15 = spatial):")
for i, label in enumerate(["left_ee_x", "left_ee_y", "left_ee_z",
                            "right_ee_x", "right_ee_y", "right_ee_z",
                            "tgt_base_x", "tgt_base_y", "tgt_base_z",
                            "tgt_left_x", "tgt_left_y", "tgt_left_z",
                            "tgt_right_x", "tgt_right_y", "tgt_right_z"]):
    idx = n - 15 + i
    print(f"  [{idx}] {label}: mean={policy.obs_mean[idx]:.4f}  std={policy.obs_std[idx]:.4f}")

print(f"\nNormalizer act_mean:")
for i, jname in enumerate(joint_names):
    print(f"  [{i}] {jname}: mean={policy.act_mean[i]:.6f}  std={policy.act_std[i]:.6f}")

# Run 600 steps (20s) and track
steps_per_ctrl = max(1, int(CONTROL_DT / SIM_DT))
action_ema = 0.5
all_deltas = []
all_joint_pos = []

for step in range(600):
    obs, curr_jp = build_obs(mj_data, joint_names, joint_qpos_adr, prev_jp,
                              tgt_vr, left_ee_bid, right_ee_bid)
    delta = policy.predict(obs, action_ema=action_ema, action_scale=1.0)
    all_deltas.append(delta.copy())
    
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
    
    jp = [float(mj_data.qpos[joint_qpos_adr[jname]]) for jname in joint_names]
    all_joint_pos.append(jp)
    prev_jp = curr_jp
    
    l_d = np.linalg.norm(mj_data.xpos[left_ee_bid] - tgt_mj)
    r_d = np.linalg.norm(mj_data.xpos[right_ee_bid] - tgt_mj)
    d = min(l_d, r_d)
    
    if step % 60 == 0 or step < 10:
        print(f"Step {step:3d}  t={step/30:.1f}s  "
              f"d_left={l_d:.3f}  d_right={r_d:.3f}  min={d:.3f}  "
              f"delta={[f'{x:.4f}' for x in delta]}")

# Summary
all_deltas = np.array(all_deltas)
all_jp = np.array(all_joint_pos)
print(f"\n=== Action delta statistics (600 steps) ===")
for i, jname in enumerate(joint_names):
    print(f"  {jname}: mean_delta={all_deltas[:,i].mean():.5f}  "
          f"std={all_deltas[:,i].std():.5f}  "
          f"total_drift={all_deltas[:,i].sum():.3f}  "
          f"pos_start={all_jp[0,i]:.3f}  pos_end={all_jp[-1,i]:.3f}")
