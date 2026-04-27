"""
Post-process VR collector data to align frames with the canonical MuJoCo-expert
observation frame used by the eval pipeline and the strong-pretrain dataset.

Problem
-------
- MuJoCo expert (collect_expert_v5.py) records:
    endEffector.left.p, endEffector.right.p       in ROBOT-LOCAL frame
                                                   (R_BASE @ (ee_vr - robot_origin))
    task.targetRelTo*                              also in ROBOT-LOCAL frame
- VR frontend (RobotVR.vue) records:
    endEffector.left.p, endEffector.right.p       in RAW VR-WORLD frame
                                                   (= three.js RobotExpressive avatar wrist)
    task.targetRelTo*                              in VR-robot-local via inv(q) rotation
                                                   (different axis convention:
                                                    MuJoCo-expert forward = -Z,
                                                    VR frontend      forward = +Z)
- Also, the EE itself is the RobotExpressive avatar's wrist, not G1 FK.

Fix
---
For each VR frame:
  1. Run MuJoCo forward kinematics on `obs.g1JointPositions` to get G1-native
     wrist positions in MuJoCo world.
  2. Convert to VR-world via mujoco_to_vr.
  3. Apply _R_BASE (the canonical rotation used by MuJoCo expert) to yield
     endEffector in the canonical ROBOT-LOCAL frame.
  4. Re-derive targetRelToRobotBase / targetRelToLeftHand / targetRelToRightHand
     from `task.targetPose.p` and `robotBaseWorldPose.p` using the same canonical
     _R_BASE transform.

Output
------
data_collector/<output_name>/
    episodes.jsonl     # aligned frames
    events.jsonl       # copied unchanged
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:
    sys.exit("mujoco is required; pip install mujoco")


ALL_G1_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

_R_BASE = np.array([[0.0, 0.0, -1.0],
                    [0.0, 1.0,  0.0],
                    [-1.0, 0.0, 0.0]], dtype=np.float64)


def mujoco_to_vr(p):
    return np.array([p[0], p[2], -p[1]], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mujoco_sim/g1_dual_arm_scene.xml")
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True,
                    help="output directory (e.g. data_collector/collector10_aligned)")
    ap.add_argument("--action-scale", type=float, default=1.0,
                    help="Multiply g1JointDelta action by this factor (VR actions are "
                         "typically ~3-4x smaller per-frame than MJ expert; use ~3.5 "
                         "to match MJ action magnitude distribution).")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    joint_adr = {}
    for jn in ALL_G1_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            print(f"[WARN] joint not in model: {jn}")
            continue
        joint_adr[jn] = model.jnt_qposadr[jid]

    left_ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    right_ee_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    if left_ee_bid < 0 or right_ee_bid < 0:
        sys.exit("EE bodies not found in model")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_episodes = out_dir / "episodes.jsonl"
    out_events = out_dir / "events.jsonl"

    shutil.copyfile(args.events, out_events)

    n_in, n_out, n_skip = 0, 0, 0
    with open(args.episodes, "r", encoding="utf-8") as fin, \
         open(out_episodes, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                d = json.loads(line)
            except Exception:
                n_skip += 1
                continue

            obs = d.get("obs", {}) or {}
            gp = obs.get("g1JointPositions", {}) or {}
            task = obs.get("task", {}) or {}
            target_pose = task.get("targetPose", {}) or {}
            target_p = target_pose.get("p")
            rbase_pose = obs.get("robotBaseWorldPose", {}) or {}
            rbase_p = rbase_pose.get("p")

            if target_p is None or rbase_p is None or not gp:
                n_skip += 1
                continue

            # FK: set qpos from recorded g1JointPositions
            data.qpos[:] = 0
            for jn, adr in joint_adr.items():
                if jn in gp:
                    data.qpos[adr] = float(gp[jn])
            mujoco.mj_forward(model, data)

            le_mj = data.xpos[left_ee_bid].copy()
            re_mj = data.xpos[right_ee_bid].copy()
            le_vr = mujoco_to_vr(le_mj)
            re_vr = mujoco_to_vr(re_mj)

            # Canonical robot-local frame (matches MuJoCo-expert convention)
            le_loc = _R_BASE @ le_vr
            re_loc = _R_BASE @ re_vr

            target_world = np.array(target_p, dtype=np.float64)
            robot_world = np.array(rbase_p, dtype=np.float64)
            target_rel_base = _R_BASE @ (target_world - robot_world)
            target_rel_left = target_rel_base - le_loc
            target_rel_right = target_rel_base - re_loc

            # Overwrite fields consumed by common.frame_to_obs_act
            end_eff = obs.setdefault("endEffector", {})
            end_eff["left"] = {"p": le_loc.tolist()}
            end_eff["right"] = {"p": re_loc.tolist()}

            task["targetRelToRobotBase"] = {"p": target_rel_base.tolist()}
            task["targetRelToLeftHand"] = {"p": target_rel_left.tolist()}
            task["targetRelToRightHand"] = {"p": target_rel_right.tolist()}

            # Preserve distance metric under canonical frame (EE-to-target)
            d_left = float(np.linalg.norm(target_rel_left))
            d_right = float(np.linalg.norm(target_rel_right))
            task["distToTargetLeft"] = d_left
            task["distToTargetRight"] = d_right
            task["distToTarget"] = min(d_left, d_right)

            obs["task"] = task
            obs["endEffector"] = end_eff
            d["obs"] = obs
            d["alignedToG1FK"] = True

            # Optionally rescale g1JointDelta action to match MJ expert magnitude
            if args.action_scale != 1.0:
                act = d.get("action", {}) or {}
                gd = act.get("g1JointDelta", {}) or {}
                if isinstance(gd, dict):
                    for k, v in list(gd.items()):
                        try:
                            gd[k] = float(v) * args.action_scale
                        except (TypeError, ValueError):
                            pass
                    act["g1JointDelta"] = gd
                    d["action"] = act
                d["actionScaleApplied"] = args.action_scale

            fout.write(json.dumps(d) + "\n")
            n_out += 1

    print(f"[align] in={n_in}  out={n_out}  skip={n_skip}")
    print(f"[align] wrote {out_episodes}")
    print(f"[align] copied {out_events}")


if __name__ == "__main__":
    main()
