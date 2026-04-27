"""
Render MuJoCo policy evaluation to images for visual inspection.
Saves key frames as PNG screenshots.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

# Import from validate_policy
sys.path.insert(0, str(Path(__file__).parent))
from validate_policy import (
    PolicyRunner, build_obs, sample_reachable_target,
    mujoco_to_vr_pos, CONTROL_DT, SIM_DT, EPISODE_TIMEOUT_S,
    TARGETS_PER_EPISODE, TARGET_REACH_THRESHOLD, HOLD_DURATION_S,
)
from joint_mapping import G1_JOINT_LIMITS


def render_frame(mj_model, mj_data, renderer, width=640, height=480):
    """Render current MuJoCo scene to an image array."""
    renderer.update_scene(mj_data)
    return renderer.render()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="render_output")
    parser.add_argument("--frame-interval", type=int, default=15,
                        help="Save every N-th control frame")
    parser.add_argument("--no-ensemble", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_xml = Path(__file__).parent / "model" / "task_scene.xml"
    policy = PolicyRunner(args.checkpoint)
    policy.no_ensemble = args.no_ensemble
    joint_names = policy.joint_names
    rng = np.random.default_rng(args.seed)

    mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
    mj_data = mujoco.MjData(mj_model)

    # Setup renderer
    renderer = mujoco.Renderer(mj_model, height=480, width=640)

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
    left_shoulder_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link")
    right_shoulder_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_shoulder_pitch_link")
    tgt_ball_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
    tgt_mocap_id = mj_model.body_mocapid[tgt_ball_bid] if tgt_ball_bid >= 0 else -1

    total_targets, total_reached = 0, 0

    for trial in range(1, args.num_trials + 1):
        mujoco.mj_resetData(mj_model, mj_data)
        for jname, adr in joint_qpos_adr.items():
            mj_data.qpos[adr] = policy.init_joint_pos.get(jname, 0.0)
        mujoco.mj_forward(mj_model, mj_data)
        policy.reset()
        prev_jp = {j: float(mj_data.qpos[a]) for j, a in joint_qpos_adr.items()}

        for tgt_num in range(1, TARGETS_PER_EPISODE + 1):
            tgt_mj, tgt_vr = sample_reachable_target(
                rng, mj_model, mj_data, left_shoulder_bid, right_shoulder_bid)
            policy.reset()
            if tgt_mocap_id >= 0:
                mj_data.mocap_pos[tgt_mocap_id] = tgt_mj
                mujoco.mj_forward(mj_model, mj_data)

            total_targets += 1
            sim_t, hold_t, min_d, reached = 0.0, 0.0, float("inf"), False
            steps_per_ctrl = max(1, int(CONTROL_DT / SIM_DT))
            ctrl_frame = 0

            # Save initial frame
            img = render_frame(mj_model, mj_data, renderer)
            try:
                from PIL import Image
                fname = out_dir / f"trial{trial}_tgt{tgt_num}_f{ctrl_frame:04d}_start.png"
                Image.fromarray(img).save(str(fname))
            except ImportError:
                import matplotlib.pyplot as plt
                fname = out_dir / f"trial{trial}_tgt{tgt_num}_f{ctrl_frame:04d}_start.png"
                plt.imsave(str(fname), img)

            while sim_t < EPISODE_TIMEOUT_S:
                obs, curr_jp = build_obs(
                    mj_data, joint_names, joint_qpos_adr, prev_jp,
                    tgt_vr, left_ee_bid, right_ee_bid)
                delta = policy.predict(obs)

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
                prev_jp = curr_jp
                ctrl_frame += 1

                # Compute distance
                mujoco.mj_forward(mj_model, mj_data)
                d_left = float(np.linalg.norm(mj_data.xpos[left_ee_bid] - tgt_mj))
                d_right = float(np.linalg.norm(mj_data.xpos[right_ee_bid] - tgt_mj))
                d = min(d_left, d_right)
                if d < min_d:
                    min_d = d

                # Save frames at key moments
                save_frame = False
                label = ""
                if ctrl_frame % args.frame_interval == 0:
                    save_frame = True
                    label = f"d{d:.3f}"
                if d <= TARGET_REACH_THRESHOLD and hold_t < HOLD_DURATION_S:
                    save_frame = True
                    label = f"reaching_d{d:.3f}"

                if save_frame:
                    img = render_frame(mj_model, mj_data, renderer)
                    try:
                        from PIL import Image
                        fname = out_dir / f"trial{trial}_tgt{tgt_num}_f{ctrl_frame:04d}_{label}.png"
                        Image.fromarray(img).save(str(fname))
                    except ImportError:
                        import matplotlib.pyplot as plt
                        fname = out_dir / f"trial{trial}_tgt{tgt_num}_f{ctrl_frame:04d}_{label}.png"
                        plt.imsave(str(fname), img)

                if d <= TARGET_REACH_THRESHOLD:
                    hold_t += CONTROL_DT
                    if hold_t >= HOLD_DURATION_S:
                        reached = True
                        # Save success frame
                        img = render_frame(mj_model, mj_data, renderer)
                        try:
                            from PIL import Image
                            fname = out_dir / f"trial{trial}_tgt{tgt_num}_f{ctrl_frame:04d}_SUCCESS.png"
                            Image.fromarray(img).save(str(fname))
                        except ImportError:
                            import matplotlib.pyplot as plt
                            fname = out_dir / f"trial{trial}_tgt{tgt_num}_f{ctrl_frame:04d}_SUCCESS.png"
                            plt.imsave(str(fname), img)
                        break
                else:
                    hold_t = 0.0

            outcome = "OK" if reached else "FAIL"
            total_reached += int(reached)
            print(f"  Trial {trial} target {tgt_num}/5: {outcome}  t={sim_t:.1f}s  min_d={min_d:.3f}m")

        print(f"[Trial {trial}/{args.num_trials}]")

    print(f"\nOverall: {total_reached}/{total_targets} ({100*total_reached/total_targets:.1f}%)")
    print(f"Frames saved to: {out_dir}")
    renderer.close()


if __name__ == "__main__":
    main()
