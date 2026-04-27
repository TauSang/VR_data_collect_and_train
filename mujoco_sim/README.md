# MuJoCo G1 Dual-Arm Validation Environment

基于 MuJoCo 的 Unitree G1 双臂操作验证环境，用于：
1. **动作回放验证** — 将 VR 采集的动作数据在物理仿真中回放，确认 IK 动作在真实物理下可执行
2. **策略推理验证** — 加载训练好的 BC/ACT 策略，在仿真中执行推理并评估成功率
3. **VR→G1 关节映射验证** — 确认 VR 骨骼关节与 G1 真实关节的对应关系正确

## 环境要求

```bash
pip install mujoco numpy
# 训练后策略推理还需要:
pip install torch
```

## G1 Dual-Arm 关节配置

每条手臂 7 DOF，共 14 DOF：
| 序号 | 关节名称 | 运动轴 | 范围 (rad) |
|------|----------|--------|------------|
| 0 | left_shoulder_pitch_joint | Y | [-3.09, 2.67] |
| 1 | left_shoulder_roll_joint | X | [-1.59, 2.25] |
| 2 | left_shoulder_yaw_joint | Z | [-2.62, 2.62] |
| 3 | left_elbow_joint | Y | [-1.05, 2.09] |
| 4 | left_wrist_roll_joint | X | [-1.97, 1.97] |
| 5 | left_wrist_pitch_joint | Y | [-1.61, 1.61] |
| 6 | left_wrist_yaw_joint | Z | [-1.61, 1.61] |
| 7 | right_shoulder_pitch_joint | Y | [-3.09, 2.67] |
| 8 | right_shoulder_roll_joint | X | [-2.25, 1.59] |
| 9 | right_shoulder_yaw_joint | Z | [-2.62, 2.62] |
| 10 | right_elbow_joint | Y | [-1.05, 2.09] |
| 11 | right_wrist_roll_joint | X | [-1.97, 1.97] |
| 12 | right_wrist_pitch_joint | Y | [-1.61, 1.61] |
| 13 | right_wrist_yaw_joint | Z | [-1.61, 1.61] |

## 目录结构

```
mujoco_sim/
├── README.md
├── requirements.txt
├── setup_model.py          # 下载 G1 模型并生成 dual-arm scene
├── g1_dual_arm_scene.xml   # 双臂桌面任务场景定义
├── validate_replay.py      # VR 数据回放验证
├── validate_policy.py      # 训练策略推理验证
├── joint_mapping.py        # VR 骨骼 → G1 关节映射
└── model/                  # G1 MJCF 模型文件 (由 setup_model.py 下载)
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载 G1 模型
python setup_model.py

# 3. 验证关节映射 (无需数据)
python joint_mapping.py --visualize

# 4. 回放 VR 采集数据
python validate_replay.py --episodes ../data_collector/collector5/vr-demonstrations-episodes-*.jsonl

# 5. 策略推理验证 (需训练好的模型)
python validate_policy.py --checkpoint ../20260331_task_policy/outputs/bc/best_model.pt
```
