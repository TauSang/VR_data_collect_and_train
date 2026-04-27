"""
VR Skeleton → Unitree G1 Dual-Arm Joint Mapping

VR 采集系统使用 Mixamo 骨骼:
  - leftShoulder  (锁骨, 不参与 IK, 零方差)
  - leftUpperArm  (大臂: shoulder pitch/roll/yaw)
  - leftLowerArm  (小臂: elbow)
  - leftHand      (手掌: wrist roll/pitch/yaw)
  右臂同理

G1 dual_arm 关节:
  每臂 7 DOF: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw

映射策略:
  VR 的 euler_delta (3D) 描述的是骨骼旋转增量, 需要分解到 G1 各关节轴上。
  - leftUpperArm euler → left_shoulder_pitch (Y), left_shoulder_roll (X), left_shoulder_yaw (Z)
  - leftLowerArm euler → left_elbow (Y), (roll/yaw 可映射但 VR 通常只有弯曲)
  - leftHand euler → left_wrist_roll (X), left_wrist_pitch (Y), left_wrist_yaw (Z)
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── G1 dual-arm joint names (14 DOF) ──────────────────────────────────
G1_LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

G1_RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_ARM_JOINTS = G1_LEFT_ARM_JOINTS + G1_RIGHT_ARM_JOINTS

# ── G1 joint limits (radians) ──────────────────────────────────────────
G1_JOINT_LIMITS = {
    "left_shoulder_pitch_joint": (-3.0892, 2.6704),
    "left_shoulder_roll_joint": (-1.5882, 2.2515),
    "left_shoulder_yaw_joint": (-2.618, 2.618),
    "left_elbow_joint": (-1.0472, 2.0944),
    "left_wrist_roll_joint": (-1.97222, 1.97222),
    "left_wrist_pitch_joint": (-1.61443, 1.61443),
    "left_wrist_yaw_joint": (-1.61443, 1.61443),
    "right_shoulder_pitch_joint": (-3.0892, 2.6704),
    "right_shoulder_roll_joint": (-2.2515, 1.5882),
    "right_shoulder_yaw_joint": (-2.618, 2.618),
    "right_elbow_joint": (-1.0472, 2.0944),
    "right_wrist_roll_joint": (-1.97222, 1.97222),
    "right_wrist_pitch_joint": (-1.61443, 1.61443),
    "right_wrist_yaw_joint": (-1.61443, 1.61443),
}

# ── VR bone → G1 joint mapping ────────────────────────────────────────
# VR bones 输出 euler_delta[3] = (x, y, z)
# 需要按轴对应到 G1 各关节

# leftUpperArm euler(x,y,z) 映射:
#   euler.y → shoulder_pitch (Y轴)
#   euler.x → shoulder_roll  (X轴)
#   euler.z → shoulder_yaw   (Z轴)
VR_TO_G1_MAPPING = {
    # VR bone name → [(g1_joint, euler_axis_index, sign)]
    "leftUpperArm": [
        ("left_shoulder_pitch_joint", 1, 1.0),   # euler.y → pitch
        ("left_shoulder_roll_joint", 0, 1.0),     # euler.x → roll
        ("left_shoulder_yaw_joint", 2, 1.0),      # euler.z → yaw
    ],
    "leftLowerArm": [
        ("left_elbow_joint", 1, 1.0),             # euler.y → elbow bend
        # wrist roll 部分可能来自 lowerArm 的 x 旋转
        # ("left_wrist_roll_joint", 0, 1.0),
    ],
    "leftHand": [
        ("left_wrist_roll_joint", 0, 1.0),        # euler.x → wrist roll
        ("left_wrist_pitch_joint", 1, 1.0),       # euler.y → wrist pitch
        ("left_wrist_yaw_joint", 2, 1.0),         # euler.z → wrist yaw
    ],
    "rightUpperArm": [
        ("right_shoulder_pitch_joint", 1, 1.0),
        ("right_shoulder_roll_joint", 0, 1.0),
        ("right_shoulder_yaw_joint", 2, 1.0),
    ],
    "rightLowerArm": [
        ("right_elbow_joint", 1, 1.0),
    ],
    "rightHand": [
        ("right_wrist_roll_joint", 0, 1.0),
        ("right_wrist_pitch_joint", 1, 1.0),
        ("right_wrist_yaw_joint", 2, 1.0),
    ],
}

# leftShoulder / rightShoulder 是锁骨, G1 没有对应关节, 跳过


def vr_euler_delta_to_g1_joint_delta(
    vr_joint_deltas: Dict[str, List[float]],
) -> Dict[str, float]:
    """
    将 VR 采集的 euler_delta 转换为 G1 关节角增量。

    Args:
        vr_joint_deltas: VR 的 jointDelta 字典
            e.g. {"leftUpperArm": [dx, dy, dz], "leftLowerArm": [...], ...}

    Returns:
        G1 关节角增量字典, e.g. {"left_shoulder_pitch_joint": 0.02, ...}
    """
    g1_deltas: Dict[str, float] = {j: 0.0 for j in G1_ARM_JOINTS}

    for vr_bone, mappings in VR_TO_G1_MAPPING.items():
        euler = vr_joint_deltas.get(vr_bone)
        if euler is None or not isinstance(euler, list) or len(euler) < 3:
            continue
        for g1_joint, axis_idx, sign in mappings:
            try:
                g1_deltas[g1_joint] += sign * float(euler[axis_idx])
            except (IndexError, ValueError, TypeError):
                pass

    return g1_deltas


def clamp_to_limits(
    joint_positions: Dict[str, float],
) -> Dict[str, float]:
    """Clamp joint positions to G1 hardware limits."""
    clamped = {}
    for joint, pos in joint_positions.items():
        lo, hi = G1_JOINT_LIMITS.get(joint, (-math.pi, math.pi))
        clamped[joint] = max(lo, min(hi, pos))
    return clamped


def get_g1_stand_pose() -> Dict[str, float]:
    """G1 stand keyframe for arms (from g1.xml)."""
    return {
        "left_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 1.28,
        "left_wrist_roll_joint": 0.0,
        "left_wrist_pitch_joint": 0.0,
        "left_wrist_yaw_joint": 0.0,
        "right_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_yaw_joint": 0.0,
        "right_elbow_joint": 1.28,
        "right_wrist_roll_joint": 0.0,
        "right_wrist_pitch_joint": 0.0,
        "right_wrist_yaw_joint": 0.0,
    }


def print_mapping_table():
    """Print a human-readable mapping table for verification."""
    print("=" * 70)
    print("VR Bone → G1 Joint Mapping")
    print("=" * 70)
    for vr_bone, mappings in VR_TO_G1_MAPPING.items():
        print(f"\n  VR: {vr_bone}")
        axis_names = ["x", "y", "z"]
        for g1_joint, axis_idx, sign in mappings:
            sign_str = "+" if sign > 0 else "-"
            print(f"    euler.{axis_names[axis_idx]} ({sign_str}) → {g1_joint}")

    print("\n  VR: leftShoulder → (skipped, clavicle, zero variance)")
    print("  VR: rightShoulder → (skipped, clavicle, zero variance)")
    print("=" * 70)


def visualize_joint_movement():
    """Load G1 in MuJoCo and visualize arm joint movements."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("[ERROR] mujoco not installed. Run: pip install mujoco")
        return

    model_path = Path(__file__).resolve().parent / "model" / "g1.xml"
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}. Run setup_model.py first.")
        return

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Set stand pose
    stand_pose = get_g1_stand_pose()
    for joint_name, pos in stand_pose.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            qpos_adr = model.jnt_qposadr[joint_id]
            data.qpos[qpos_adr] = pos

    # Set ctrl to match
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if act_name in stand_pose:
            data.ctrl[i] = stand_pose[act_name]

    mujoco.mj_forward(model, data)

    print("Launching MuJoCo viewer with G1 stand pose...")
    print("Arm joints will oscillate to verify mapping.")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0
        while viewer.is_running():
            # Oscillate each arm joint with different frequencies
            for i, joint_name in enumerate(G1_ARM_JOINTS):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    continue
                lo, hi = G1_JOINT_LIMITS[joint_name]
                mid = (lo + hi) / 2.0
                amp = (hi - lo) / 4.0
                freq = 0.3 + i * 0.05
                target = mid + amp * math.sin(2 * math.pi * freq * t)
                # Set actuator ctrl
                act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
                if act_id >= 0:
                    data.ctrl[act_id] = target

            mujoco.mj_step(model, data)
            viewer.sync()
            t += model.opt.timestep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VR → G1 Joint Mapping")
    parser.add_argument("--visualize", action="store_true", help="Launch MuJoCo viewer to visualize joint movements")
    args = parser.parse_args()

    print_mapping_table()

    if args.visualize:
        visualize_joint_movement()
