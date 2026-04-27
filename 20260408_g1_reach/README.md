# 20260408 G1 Reach Task — Training Pipeline

## Overview

G1 dual-arm (14 DOF) reach task policy training with **43D observation** and **14D action**.

This pipeline reads VR demonstration data and maps VR bone euler angles to G1 joint positions
via axis decomposition, then trains BC (MLP) and ACT (Transformer) policies.

## Observation Space (43D)

| Range  | Dim | Feature                     | Deploy Source          |
|--------|-----|-----------------------------|------------------------|
| 0–13   | 14  | G1 joint positions (rad)    | Joint encoders         |
| 14–27  | 14  | G1 joint velocities (rad/s) | Encoder differentiation|
| 28–33  | 6   | EE positions (left+right)   | Forward kinematics     |
| 34–36  | 3   | Target rel to robot base    | Task command + FK      |
| 37–39  | 3   | Target rel to left hand     | Task command + FK      |
| 40–42  | 3   | Target rel to right hand    | Task command + FK      |

## Action Space (14D)

14 G1 joint position deltas (scalar per revolute joint):
- 7 left arm: shoulder (pitch/roll/yaw), elbow, wrist (roll/pitch/yaw)
- 7 right arm: same

## VR → G1 Joint Mapping

```
leftUpperArm  euler.y → left_shoulder_pitch
              euler.x → left_shoulder_roll
              euler.z → left_shoulder_yaw
leftLowerArm  euler.y → left_elbow
leftHand      euler.x → left_wrist_roll
              euler.y → left_wrist_pitch
              euler.z → left_wrist_yaw
(right side symmetric)
```

## Usage

```bash
# Train both BC and ACT, then analyze
python run_all.py

# Or individually
python train_bc.py --config config.json --out outputs
python train_act.py --config config.json --out outputs
python analyze_results.py
```

## Files

- `config.json` — Experiment configuration (data paths, hyperparameters)
- `common.py` — Data loading, VR→G1 mapping, normalization, dataset construction
- `train_bc.py` — Behavioral Cloning (MLP: 256→256→128, LayerNorm+GELU)
- `train_act.py` — ACT (Transformer encoder: 2 layers, 4 heads, d=128)
- `run_all.py` — Run full pipeline (BC → ACT → analyze)
- `analyze_results.py` — Compare BC vs ACT results
