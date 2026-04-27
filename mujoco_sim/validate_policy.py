# pyright: reportMissingImports=false
"""
策略推理验证 — 在 MuJoCo G1 模型中运行 BC/ACT 策略并评估 reach 成功率

观测/动作维度从 checkpoint 自动推断 (g1_joint_names)。
目标在 G1 手臂可达工作空间内生成。

用法:
  python validate_policy.py --checkpoint ../20260408_g1_reach/outputs/act/run_xxx/checkpoints/best.pt [--visualize] [--num-trials 10]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from joint_mapping import G1_JOINT_LIMITS

# ── Task parameters ─────────────────────────────────────────────────
# Reach is measured from the palm centre (wrist_yaw_link origin + PALM_OFFSET
# along the link's local +X axis) to the target-ball centre. PALM_OFFSET is
# chosen from the G1 XML: rubber_hand geom is attached at +0.0415 m and its
# inertial CoM is at +0.0708 m along wrist_yaw_link's +X; the palm surface
# sits roughly ~0.09 m out.  With a ball radius of 0.04 m, a palm-to-centre
# distance of 0.08 m corresponds to the palm being within ~4 cm of the
# ball surface (i.e. essentially touching it).
TARGET_REACH_THRESHOLD = 0.12   # meters — palm-centre to ball-centre
HOLD_DURATION_S = 0.3           # seconds — must hold on target, not fly by
EPISODE_TIMEOUT_S = 20.0        # seconds
TARGETS_PER_EPISODE = 5
SIM_DT = 0.002                  # MuJoCo timestep
CONTROL_DT = 1.0 / 30.0        # policy frequency = 30 Hz
PALM_OFFSET_LOCAL = np.array([0.09, 0.0, 0.0], dtype=np.float64)  # wrist_yaw_link → palm centre


def vr_to_mujoco_pos(vr_xyz):
    """VR Y-up → MuJoCo Z-up: (x,y,z) → (x, -z, y)."""
    return np.array([vr_xyz[0], -vr_xyz[2], vr_xyz[1]])


def mujoco_to_vr_pos(mj_xyz):
    """MuJoCo Z-up → VR Y-up: (x,y,z) → (x, z, -y)."""
    return np.array([mj_xyz[0], mj_xyz[2], -mj_xyz[1]])


# ── Rotation: VR-world → VR-robot-local frame ──────────────────────
# MuJoCo robot faces +X (= VR +X).  VR robot faces +Z (after π rotation).
# VR robot-local: +X = left, +Y = up, -Z = forward.
# This matrix converts (pos_vr_world − base_vr) → robot-local coords
# so that observations match what the VR recording produces via
# poseRelativeToBase / buildRelativePose.
#   robot-local X  =  −(VR-world Z)          (left  ← −world Z)
#   robot-local Y  =    VR-world Y            (up    ←  world Y)
#   robot-local Z  =  −(VR-world X)          (−fwd  ← −world X)
_R_BASE = np.array([[0.0, 0.0, -1.0],
                     [0.0, 1.0,  0.0],
                     [-1.0, 0.0, 0.0]], dtype=np.float64)

# Robot base in VR coords — at feet / ground level to match VR's
# Object3D origin (not pelvis height).
_ROBOT_BASE_VR = np.array([0.0, 0.0, 0.0])


def sample_reachable_target(rng, mj_model, mj_data, left_shoulder_bid, right_shoulder_bid, max_reach=0.28):
    """Sample a random target within the robot arm's reachable workspace.

    Randomly picks a shoulder, then samples a point within max_reach of it,
    constrained to be IN FRONT of the robot (MuJoCo +X) and on the correct side
    of the body to avoid physically impossible reaches (e.g. left arm crossing
    the torso to reach far-right targets, or behind the head).

    MuJoCo frame conventions (G1 faces +X):
      +X = forward, +Y = left, +Z = up
    """
    import mujoco
    mujoco.mj_forward(mj_model, mj_data)
    use_left = rng.random() < 0.5
    if use_left:
        shoulder = mj_data.xpos[left_shoulder_bid].copy()
    else:
        shoulder = mj_data.xpos[right_shoulder_bid].copy()

    for _ in range(200):
        offset = rng.uniform(-max_reach, max_reach, size=3)
        # Must be in front of shoulder (+X) — avoids "behind head" reaches.
        if offset[0] < 0.05:
            continue
        # Keep target on the same side of body (avoids arm crossing through torso).
        if use_left and offset[1] < -0.05:
            continue
        if (not use_left) and offset[1] > 0.05:
            continue
        # Not too far below shoulder (keep natural human-reachable range).
        if offset[2] < -0.15:
            continue
        if np.linalg.norm(offset) <= max_reach:
            break
    target_mj = shoulder + offset
    target_vr = mujoco_to_vr_pos(target_mj)
    return target_mj, target_vr


def build_obs(mj_data, joint_names, joint_qpos_adr, prev_joint_pos, target_vr,
              left_ee_bid, right_ee_bid):
    """Build observation vector (nJ*2 + 15) in VR coordinate frame."""
    obs = []
    joint_pos = {}
    for jname in joint_names:
        adr = joint_qpos_adr.get(jname)
        val = float(mj_data.qpos[adr]) if adr is not None else 0.0
        joint_pos[jname] = val
        obs.append(val)
    for jname in joint_names:
        curr = joint_pos[jname]
        prev = prev_joint_pos.get(jname, curr)
        obs.append((curr - prev) / CONTROL_DT if CONTROL_DT > 0 else 0.0)
    left_ee_vr = mujoco_to_vr_pos(mj_data.xpos[left_ee_bid]) if left_ee_bid >= 0 else np.zeros(3)
    right_ee_vr = mujoco_to_vr_pos(mj_data.xpos[right_ee_bid]) if right_ee_bid >= 0 else np.zeros(3)

    # All spatial features must be in robot-local frame (matching VR recording).
    left_ee_local = _R_BASE @ (left_ee_vr - _ROBOT_BASE_VR)
    right_ee_local = _R_BASE @ (right_ee_vr - _ROBOT_BASE_VR)
    obs.extend(left_ee_local.tolist())
    obs.extend(right_ee_local.tolist())

    target_rel_base = _R_BASE @ (target_vr - _ROBOT_BASE_VR)
    obs.extend(target_rel_base.tolist())

    # Hand-relative: use base rotation as approximation
    target_rel_left = _R_BASE @ (target_vr - left_ee_vr)
    target_rel_right = _R_BASE @ (target_vr - right_ee_vr)
    obs.extend(target_rel_left.tolist())
    obs.extend(target_rel_right.tolist())
    return np.array(obs, dtype=np.float32), joint_pos


class PolicyRunner:
    def __init__(self, checkpoint_path):
        import torch
        self.torch = torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.config = ckpt["config"]
        norm = ckpt["normalizer"]
        self.obs_mean = np.array(norm["obs_mean"], dtype=np.float32)
        self.obs_std = np.array(norm["obs_std"], dtype=np.float32)
        self.act_mean = np.array(norm["act_mean"], dtype=np.float32)
        self.act_std = np.array(norm["act_std"], dtype=np.float32)
        self.obs_clip_z = float(self.config["data"].get("obs_clip_z", 8.0))

        # Read joint names from checkpoint config
        self.joint_names = list(self.config["data"]["g1_joint_names"])
        self.obs_dim = len(self.obs_mean)
        self.act_dim = len(self.act_mean)

        # Initial joint positions from training data mean
        self.init_joint_pos = {}
        for i, jname in enumerate(self.joint_names):
            self.init_joint_pos[jname] = float(self.obs_mean[i])

        state_dict = ckpt["model"]

        # Auto-detect pipeline directory from checkpoint path
        ckpt_path = Path(checkpoint_path).resolve()
        # Walk up from checkpoint to find the training root (contains train_bc.py / train_act.py / train_act_chunk.py)
        pipeline_dir = None
        for parent in ckpt_path.parents:
            if (parent / "train_bc.py").exists() or (parent / "train_act.py").exists() or (parent / "train_act_chunk.py").exists():
                pipeline_dir = str(parent)
                break
        if pipeline_dir is None:
            # Fallback to 20260408_g1_reach
            pipeline_dir = str(Path(__file__).resolve().parent.parent / "20260408_g1_reach")
        if pipeline_dir not in sys.path:
            sys.path.insert(0, pipeline_dir)

        # Detect model type from checkpoint
        model_class = ckpt.get("model_class", None)
        self.chunk_size = int(ckpt.get("chunk_size", 0))

        if model_class == "ACTChunkMLP" or (model_class == "ACTChunk" and str(self.config.get("act_chunk", {}).get("backbone", "transformer")) == "mlp"):
            # MLP-based action chunking
            self.model_type = "act_chunk"
            from train_act_chunk import ACTChunkMLP
            act_cfg = self.config.get("act_chunk", self.config.get("act", {}))
            self.model = ACTChunkMLP(
                obs_dim=self.obs_dim, act_dim=self.act_dim,
                chunk_size=self.chunk_size,
                hidden_dims=list(act_cfg.get("hidden_dims", [256, 256, 128])),
                dropout=0.0,
            )
            self.seq_len = 1  # MLP only uses last obs
            self._action_queue = []

        elif model_class == "ACTChunkMLPFiLM" or (model_class == "ACTChunk" and str(self.config.get("act_chunk", {}).get("backbone", "transformer")) == "mlp_film"):
            # Domain-adaptive FiLM variant; inference uses default domain (MuJoCo=0)
            self.model_type = "act_chunk"
            from train_act_chunk import ACTChunkMLPFiLM
            act_cfg = self.config.get("act_chunk", self.config.get("act", {}))
            self.model = ACTChunkMLPFiLM(
                obs_dim=self.obs_dim, act_dim=self.act_dim,
                chunk_size=self.chunk_size,
                hidden_dims=list(act_cfg.get("hidden_dims", [256, 256, 128])),
                dropout=0.0,
                num_domains=int(act_cfg.get("num_domains", 2)),
                cond_dim=int(act_cfg.get("cond_dim", 16)),
                default_domain_id=int(act_cfg.get("default_domain_id", 0)),
            )
            self.seq_len = 1
            self._action_queue = []

        elif model_class == "ACTChunkTransformerLite" or (model_class == "ACTChunk" and str(self.config.get("act_chunk", {}).get("backbone", "transformer")).lower() in {"transformer_lite", "act_chunk_transformer_lite"}):
            # Tokenized Transformer-Lite encoder + FiLM MLP head; inference uses MuJoCo domain=0.
            self.model_type = "act_chunk"
            from train_act_chunk import ACTChunkTransformerLite
            act_cfg = self.config.get("act_chunk", self.config.get("act", {}))
            self.model = ACTChunkTransformerLite(
                obs_dim=self.obs_dim, act_dim=self.act_dim,
                chunk_size=self.chunk_size,
                d_model=int(act_cfg.get("d_model", 128)),
                nhead=int(act_cfg.get("nhead", 4)),
                num_encoder_layers=int(act_cfg.get("num_encoder_layers", 3)),
                dim_feedforward=int(act_cfg.get("dim_feedforward", 256)),
                hidden_dims=list(act_cfg.get("hidden_dims", [512, 512, 256])),
                dropout=0.0,
                num_domains=int(act_cfg.get("num_domains", 2)),
                cond_dim=int(act_cfg.get("cond_dim", 16)),
                default_domain_id=int(act_cfg.get("default_domain_id", 0)),
                proprio_dim=int(act_cfg.get("proprio_dim", 16)),
            )
            self.seq_len = 1
            self._action_queue = []

        elif model_class == "ACTChunkGatedCrossAttnFiLM" or (model_class == "ACTChunk" and str(self.config.get("act_chunk", {}).get("backbone", "transformer")).lower() in {"gated_cross_attn_film", "cross_attn_film", "act_chunk_gated_cross_attn_film"}):
            # v8 conservative attention variant: MLP-FiLM primary path plus zero-gated cross-attn residual.
            self.model_type = "act_chunk"
            from train_act_chunk import ACTChunkGatedCrossAttnFiLM
            act_cfg = self.config.get("act_chunk", self.config.get("act", {}))
            self.model = ACTChunkGatedCrossAttnFiLM(
                obs_dim=self.obs_dim, act_dim=self.act_dim,
                chunk_size=self.chunk_size,
                hidden_dims=list(act_cfg.get("hidden_dims", [1024, 1024, 512])),
                dropout=0.0,
                num_domains=int(act_cfg.get("num_domains", 2)),
                cond_dim=int(act_cfg.get("cond_dim", 16)),
                default_domain_id=int(act_cfg.get("default_domain_id", 0)),
                proprio_dim=int(act_cfg.get("proprio_dim", 16)),
                d_model=int(act_cfg.get("d_model", 64)),
                nhead=int(act_cfg.get("nhead", 4)),
                attn_dropout=0.0,
                residual_init=float(act_cfg.get("residual_init", 0.01)),
            )
            self.seq_len = 1
            self._action_queue = []

        elif model_class == "ACTChunk" or self.chunk_size > 0:
            # ACT with Action Chunking (Transformer)
            self.model_type = "act_chunk"
            from train_act_chunk import ACTChunk
            act_cfg = self.config.get("act_chunk", self.config.get("act", {}))
            self.model = ACTChunk(
                obs_dim=self.obs_dim, act_dim=self.act_dim,
                chunk_size=self.chunk_size,
                d_model=int(act_cfg.get("d_model", 256)),
                nhead=int(act_cfg.get("nhead", 8)),
                num_encoder_layers=int(act_cfg.get("num_encoder_layers", 4)),
                num_decoder_layers=int(act_cfg.get("num_decoder_layers", 2)),
                dim_feedforward=int(act_cfg.get("dim_feedforward", 512)),
                dropout=0.0,
                use_cvae=bool(act_cfg.get("use_cvae", False)),
                latent_dim=int(act_cfg.get("latent_dim", 32)),
            )
            self.seq_len = int(self.config["data"].get("seq_len", 16))
            self._action_queue = []

        elif any("encoder" in k for k in state_dict):
            self.model_type = "act"
            from train_act import TaskACTEncoder
            c = self.config.get("act", {})
            model_kwargs = dict(
                obs_dim=self.obs_dim, act_dim=self.act_dim,
                d_model=int(c.get("d_model", 128)),
                nhead=int(c.get("nhead", 4)),
                num_layers=int(c.get("num_layers", 2)),
                dim_feedforward=int(c.get("dim_feedforward", 256)),
                dropout=0.0,
            )
            # Pass optional kwargs if the model class accepts them
            import inspect
            sig_params = inspect.signature(TaskACTEncoder.__init__).parameters
            if "use_residual_action" in sig_params:
                model_kwargs["use_residual_action"] = bool(c.get("use_residual_action", False))
            if "readout_mode" in sig_params:
                model_kwargs["readout_mode"] = str(c.get("readout_mode", "multi_scale"))
            if "readout_norm" in sig_params:
                model_kwargs["readout_norm"] = bool(c.get("readout_norm", True))
            self.model = TaskACTEncoder(**model_kwargs)
            self.seq_len = int(self.config["data"].get("seq_len", 16))
        else:
            self.model_type = "bc"
            from train_bc import TaskBCMLP
            c = self.config.get("bc", {})
            self.model = TaskBCMLP(
                obs_dim=self.obs_dim, act_dim=self.act_dim,
                hidden_dims=list(c.get("hidden_dims", [256, 256, 128])),
                dropout=0.0,
            )
            self.seq_len = 0

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.obs_buffer = []
        print(f"[OK] Loaded {self.model_type} model, epoch={ckpt.get('epoch','?')}, "
              f"obs_dim={self.obs_dim}, act_dim={self.act_dim}, "
              f"joints={len(self.joint_names)}"
              + (f", chunk={self.chunk_size}" if self.chunk_size > 0 else ""))

    def reset(self):
        self.obs_buffer = []
        self._prev_action = None
        if hasattr(self, '_action_queue'):
            self._action_queue = []

    def predict(self, obs, action_ema=0.0, action_scale=1.0):
        obs_z = np.clip((obs - self.obs_mean) / self.obs_std,
                        -self.obs_clip_z, self.obs_clip_z).astype(np.float32)
        with self.torch.no_grad():
            if self.model_type == "act_chunk":
                return self._predict_chunk(obs_z, action_ema, action_scale)
            elif self.model_type == "act":
                self.obs_buffer.append(obs_z)
                if len(self.obs_buffer) > self.seq_len:
                    self.obs_buffer = self.obs_buffer[-self.seq_len:]
                buf = list(self.obs_buffer)
                # Replicate first obs instead of zeros for padding
                while len(buf) < self.seq_len:
                    buf.insert(0, buf[0].copy())
                seq = self.torch.from_numpy(np.stack(buf)).unsqueeze(0)
                pred_z, _ = self.model(seq)
            else:
                pred_z, _ = self.model(self.torch.from_numpy(obs_z).unsqueeze(0))
        act_z = pred_z.squeeze(0).cpu().numpy()
        raw_action = (act_z * self.act_std + self.act_mean).astype(np.float32)
        if action_scale != 1.0:
            raw_action = raw_action * action_scale
        if action_ema > 0 and self._prev_action is not None:
            raw_action = action_ema * self._prev_action + (1 - action_ema) * raw_action
        self._prev_action = raw_action.copy()
        return raw_action

    def _predict_chunk(self, obs_z, action_ema, action_scale):
        """Temporal ensembling for ACT-Chunk model."""
        # Update obs buffer
        self.obs_buffer.append(obs_z)
        if len(self.obs_buffer) > self.seq_len:
            self.obs_buffer = self.obs_buffer[-self.seq_len:]

        buf = list(self.obs_buffer)
        while len(buf) < self.seq_len:
            buf.insert(0, buf[0].copy())  # replicate first obs instead of zeros
        seq = self.torch.from_numpy(np.stack(buf)).unsqueeze(0)

        # Get new chunk prediction
        out = self.model(seq)
        pred_chunk_z = out["actions"].squeeze(0).cpu().numpy()  # (K, act_dim)
        pred_chunk = pred_chunk_z * self.act_std + self.act_mean  # denormalize

        # Use first action from latest chunk (no ensembling)
        if getattr(self, 'no_ensemble', False) or self.chunk_size <= 1:
            raw_action = pred_chunk[0]
        else:
            # Add new chunk to queue with exponential weights
            self._action_queue.append(pred_chunk.copy())

            # Temporal ensembling: weighted average of overlapping predictions
            # Higher TEMPORAL_WEIGHT strongly favours the newest prediction.
            # 0.01 (old default) → weights nearly uniform, dilutes precision.
            # 1.0 → newest gets ~2.7× weight of age-1 prediction → much stronger recency bias.
            ensemble_weights = []
            ensemble_actions = []
            TEMPORAL_WEIGHT = getattr(self, 'temporal_weight', 1.0)

            for age, chunk in enumerate(reversed(self._action_queue)):
                if age < chunk.shape[0]:
                    w = np.exp(-TEMPORAL_WEIGHT * age)
                    ensemble_weights.append(w)
                    ensemble_actions.append(chunk[age])

            weights_arr = np.array(ensemble_weights)
            weights_arr /= weights_arr.sum()
            raw_action = np.zeros(self.act_dim, dtype=np.float32)
            for w, a in zip(weights_arr, ensemble_actions):
                raw_action += w * a

            if len(self._action_queue) > self.chunk_size:
                self._action_queue = self._action_queue[-self.chunk_size:]

        if action_scale != 1.0:
            raw_action = raw_action * action_scale
        if action_ema > 0 and self._prev_action is not None:
            raw_action = action_ema * self._prev_action + (1 - action_ema) * raw_action
        self._prev_action = raw_action.copy()
        return raw_action.astype(np.float32)


def run_evaluation(checkpoint_path, model_xml, num_trials, visualize, seed, action_ema=0.0, action_scale=1.0, no_ensemble=False, temporal_weight=1.0, no_target_reset=False):
    import mujoco

    policy = PolicyRunner(checkpoint_path)
    policy.no_ensemble = no_ensemble
    policy.temporal_weight = temporal_weight
    joint_names = policy.joint_names
    rng = np.random.default_rng(seed)
    mj_model = mujoco.MjModel.from_xml_path(str(model_xml))
    mj_data = mujoco.MjData(mj_model)

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

    # Target ball (mocap body) for visualization
    tgt_ball_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
    tgt_mocap_id = mj_model.body_mocapid[tgt_ball_bid] if tgt_ball_bid >= 0 else -1

    viewer = None
    if visualize:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        except Exception as e:
            print(f"[WARN] Could not launch viewer: {e}")

    total_targets, total_reached = 0, 0
    results = {"model_type": policy.model_type, "num_trials": num_trials,
               "obs_dim": policy.obs_dim, "act_dim": policy.act_dim,
               "joints": joint_names, "trials": []}

    for trial in range(1, num_trials + 1):
        mujoco.mj_resetData(mj_model, mj_data)
        for jname, adr in joint_qpos_adr.items():
            mj_data.qpos[adr] = policy.init_joint_pos.get(jname, 0.0)
        mujoco.mj_forward(mj_model, mj_data)
        policy.reset()

        prev_jp = {j: float(mj_data.qpos[a]) for j, a in joint_qpos_adr.items()}
        trial_res = {"targets": [], "total_success": 0, "total_timeout": 0}

        for tgt_num in range(1, TARGETS_PER_EPISODE + 1):
            tgt_mj, tgt_vr = sample_reachable_target(
                rng, mj_model, mj_data, left_shoulder_bid, right_shoulder_bid)
            # Reset obs buffer for each new target (ACT sequences should be per-target)
            if not no_target_reset:
                policy.reset()
            # Move target ball to sampled position
            if tgt_mocap_id >= 0:
                mj_data.mocap_pos[tgt_mocap_id] = tgt_mj
                mujoco.mj_forward(mj_model, mj_data)
            total_targets += 1
            sim_t, hold_t, min_d, reached = 0.0, 0.0, float("inf"), False
            steps_per_ctrl = max(1, int(CONTROL_DT / SIM_DT))
            ctrl_wall_t = time.perf_counter()

            while sim_t < EPISODE_TIMEOUT_S:
                obs, curr_jp = build_obs(
                    mj_data, joint_names, joint_qpos_adr, prev_jp,
                    tgt_vr, left_ee_bid, right_ee_bid)
                delta = policy.predict(obs, action_ema=action_ema, action_scale=action_scale)
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

                # Palm position = wrist_yaw_link xpos + R_wrist @ PALM_OFFSET_LOCAL.
                # Previously we used the wrist body origin, which sits ~9 cm behind the
                # palm — so an 0.16 m threshold on that origin meant the hand could still
                # be several cm away from the ball and still register success.
                if left_ee_bid >= 0:
                    R_l = mj_data.xmat[left_ee_bid].reshape(3, 3)
                    l_palm = mj_data.xpos[left_ee_bid] + R_l @ PALM_OFFSET_LOCAL
                else:
                    l_palm = np.zeros(3)
                if right_ee_bid >= 0:
                    R_r = mj_data.xmat[right_ee_bid].reshape(3, 3)
                    r_palm = mj_data.xpos[right_ee_bid] + R_r @ PALM_OFFSET_LOCAL
                else:
                    r_palm = np.zeros(3)
                d = min(float(np.linalg.norm(l_palm - tgt_mj)),
                        float(np.linalg.norm(r_palm - tgt_mj)))
                min_d = min(min_d, d)
                if d <= TARGET_REACH_THRESHOLD:
                    hold_t += CONTROL_DT
                    if hold_t >= HOLD_DURATION_S:
                        reached = True
                        break
                else:
                    hold_t = 0.0
                prev_jp = curr_jp
                if viewer and viewer.is_running():
                    viewer.sync()
                    # Real-time pacing: wait until wall-clock catches up
                    elapsed = time.perf_counter() - ctrl_wall_t
                    sleep_t = CONTROL_DT - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)
                    ctrl_wall_t = time.perf_counter()

            tag = "OK" if reached else "FAIL"
            trial_res["targets"].append({
                "target_num": tgt_num, "outcome": "success" if reached else "timeout",
                "time_s": round(sim_t, 2), "min_dist_m": round(min_d, 4),
            })
            if reached:
                trial_res["total_success"] += 1
                total_reached += 1
            else:
                trial_res["total_timeout"] += 1
            print(f"  Trial {trial} target {tgt_num}/{TARGETS_PER_EPISODE}: "
                  f"{tag}  t={sim_t:.1f}s  min_d={min_d:.3f}m")

        results["trials"].append(trial_res)
        rate = trial_res["total_success"] / TARGETS_PER_EPISODE
        print(f"[Trial {trial}/{num_trials}] "
              f"success={trial_res['total_success']}/{TARGETS_PER_EPISODE} ({rate:.0%})")

    overall = total_reached / max(total_targets, 1)
    results["overall_success_rate"] = round(overall, 4)
    results["total_targets"] = total_targets
    results["total_reached"] = total_reached
    print(f"\n{'='*50}")
    print(f"Model: {policy.model_type}  |  "
          f"Overall: {total_reached}/{total_targets} ({overall:.1%})")
    print(f"{'='*50}")
    if viewer:
        viewer.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument("--action-ema", type=float, default=0.0,
                        help="EMA smoothing for actions (0=off, 0.3=moderate, 0.5=heavy)")
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="Scale factor for predicted actions (>1 = larger steps)")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Disable temporal ensembling for ACT-Chunk (use first action only)")
    parser.add_argument("--temporal-weight", type=float, default=1.0,
                        help="Exponential decay weight for temporal ensembling (higher=prefer newer)")
    parser.add_argument("--no-target-reset", action="store_true",
                        help="Keep observation buffer across targets instead of resetting per-target")
    args = parser.parse_args()

    xml = Path(args.model) if args.model else Path(__file__).resolve().parent / "model" / "task_scene.xml"
    if not xml.exists():
        print(f"[ERROR] Model XML not found: {xml}\nRun setup_model.py first.")
        sys.exit(1)
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt}")
        sys.exit(1)

    res = run_evaluation(ckpt, xml, args.num_trials, args.visualize, args.seed,
                          action_ema=args.action_ema, action_scale=args.action_scale,
                          no_ensemble=args.no_ensemble, temporal_weight=args.temporal_weight,
                          no_target_reset=args.no_target_reset)

    out = Path(args.out) if args.out else ckpt.parent.parent / "mujoco_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
