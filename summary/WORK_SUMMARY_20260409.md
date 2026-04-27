# G1 Reach 模仿学习全流程：训练、MuJoCo 部署与可视化验证（2026-04-08 ~ 04-09）

## 1. 本阶段总结

本阶段完成了从 VR 示教数据到 MuJoCo 仿真闭环验证的完整流程：

1. 采集新一批 VR 示教数据（collector6）
2. 构建专用 G1 Reach 训练管线 `20260408_g1_reach/`
3. 搭建 MuJoCo 验证环境 `mujoco_sim/`（模型下载、关节映射、场景构建）
4. 训练两轮模型（14 DOF → 8 DOF），诊断并修复 0% 成功率问题
5. 最终 BC 模型在 MuJoCo 物理仿真中达到 **28% reach 成功率**（最佳单轮 80%）
6. 实现 MuJoCo 离线渲染（截图）+ 实时交互式 Viewer 可视化

---

## 2. 新数据采集：collector6

**目录**：`data_collector/collector6/`

| 文件 | 描述 |
|------|------|
| `vr-demonstrations-session-20260408_203644.json` | 会话元数据，schema `v3_multi_repr_action` |
| `vr-demonstrations-episodes-20260408_203644.jsonl` | 逐帧 episode 数据 |
| `vr-demonstrations-events-20260408_203645.jsonl` | 事件日志 |

**数据统计**：
- 6 个 episode（ID: 1, 2, 3, 5, 6, 8），27 个 segment
- 结果分布：21 success / 4 timeout / 2 truncated
- 训练使用：25 segment（丢弃 2 个 truncated + 17 个无效帧）
- 训练集：5 个 episode (20 seg, 5698 帧) / 验证集：1 个 episode (5 seg, 1631 帧)

---

## 3. VR 前端 Bug 修复

### 3.1 任务超时逻辑修复（`src/components/RobotVR.vue`）

**问题**：原来超时会直接结束整个 episode，导致后续目标被跳过。

**修复**（第 747-785 行）：超时不再终止 episode，改为：
1. 记录 `episode_timeout` 事件
2. 递增 `completedTargets`（视为已尝试）
3. 若所有目标已尝试 → 以 `'timeout'` 结束 episode
4. 若还有剩余目标 → 生成下一个目标（`target_timeout_next` 事件），episode 继续

**意义**：允许采集含混合结果（部分成功/部分超时）的多目标 episode，提升数据质量。

### 3.2 任务手臂分配修复

- `right_only` 配置改为 `free`，允许双臂自由完成任务

---

## 4. 训练管线：`20260408_g1_reach/`

### 4.1 目录结构

```
20260408_g1_reach/
├── config.json          # 实验配置（关节、超参、数据路径）
├── common.py            # 数据加载、VR→G1 映射、归一化、obs/act 提取
├── train_bc.py          # BC MLP 训练 (TaskBCMLP)
├── train_act.py         # ACT Transformer 训练 (TaskACTEncoder)
├── run_all.py           # 一键运行：BC → ACT → 分析
├── analyze_results.py   # BC vs ACT 对比分析
├── README.md            # 完整文档
└── outputs/             # 训练输出
    ├── bc/
    │   ├── run_20260408_212553/   # Run 1 (14D)
    │   └── run_20260409_124525/   # Run 2 (8D) ← 当前最佳
    ├── act/
    │   ├── run_20260408_212637/   # Run 1 (14D)
    │   └── run_20260409_124600/   # Run 2 (8D) ← 当前最佳
    └── summary/                   # 对比分析
```

### 4.2 观测/动作空间设计

#### Run 1（14 DOF，43D obs / 14D action）— 已废弃

使用全部 14 个手臂关节（shoulder×3 + elbow + wrist×3，每侧 7 DOF）。

#### Run 2（8 DOF，31D obs / 8D action）— 当前版本

移除 6 个顺腕关节（reach 任务不需要），仅保留：

| 关节 | 描述 |
|------|------|
| `left_shoulder_pitch_joint` | 左肩俯仰 |
| `left_shoulder_roll_joint` | 左肩翻滚 |
| `left_shoulder_yaw_joint` | 左肩偏航 |
| `left_elbow_joint` | 左肘 |
| `right_shoulder_pitch_joint` | 右肩俯仰 |
| `right_shoulder_roll_joint` | 右肩翻滚 |
| `right_shoulder_yaw_joint` | 右肩偏航 |
| `right_elbow_joint` | 右肘 |

**观测向量** (31D)：
```
[8 关节位置] + [8 关节速度] + [3 左 EE 位置] + [3 右 EE 位置]
+ [3 目标相对基座] + [3 目标相对左 EE] + [3 目标相对右 EE]
= 8 + 8 + 6 + 3 + 3 + 3 = 31
```

**动作向量** (8D)：关节位置增量 $\Delta q$

### 4.3 VR→G1 关节映射

```
VR leftUpperArm  euler(x,y,z) → G1 shoulder(roll, pitch, yaw)
VR leftLowerArm  euler(y)     → G1 elbow
```

坐标变换：VR Y-up → MuJoCo Z-up
- 正向：$(x,y,z) \to (x, -z, y)$
- 逆向：$(x,y,z) \to (x, z, -y)$

### 4.4 模型架构

#### BC (TaskBCMLP)
- 隐藏层：[256, 256, 128]
- 激活：GELU
- 归一化：LayerNorm
- Dropout: 0.1
- 双头输出：action 预测 + success 分类

#### ACT (TaskACTEncoder)
- d_model=128, nhead=4, num_layers=2, dim_ff=256
- 位置编码 (positional encoding)
- norm_first=True (Pre-LN Transformer)
- 序列长度：16
- 双头输出：action 预测 + success 分类

### 4.5 训练超参

| 参数 | 值 |
|------|-----|
| Epochs | 80 |
| Batch size | 128 |
| Learning rate | 3e-4 |
| Weight decay | 1e-5 |
| Grad clip norm | 1.0 |
| Early stop patience | 12 |
| Success loss weight | 0.2 |
| Obs/act clip z | 8.0 |
| Seed | 42 |

#### 样本加权策略
```
success_bonus=0.5, near_target_bonus=0.3 (阈值 0.25m)
idle_discount=0.3, moving_bonus=0.15, approaching_bonus=0.25, max_weight=3.0
```

### 4.6 代码关键改动（Run 1 → Run 2）

1. **config.json**：`g1_joint_names` 从 14 → 8 关节，实验名改为 `g1_reach_31obs_8act`
2. **common.py**：文档更新为 `nJ*2 + 15` 动态维度说明
3. **train_bc.py / train_act.py**：断言从硬编码 `assert obs_dim == 43` 改为动态 `assert obs_dim == len(g1_joints) * 2 + 15`

---

## 5. MuJoCo 验证环境：`mujoco_sim/`

### 5.1 目录结构

```
mujoco_sim/
├── setup_model.py         # 下载 Unitree G1 模型（mujoco_menagerie）
├── joint_mapping.py       # VR 骨骼→G1 关节映射、关节限位、站立姿态
├── validate_replay.py     # VR 数据回放验证（检查物理可行性）
├── validate_policy.py     # 策略闭环评估（支持 --visualize 实时查看）
├── g1_dual_arm_scene.xml  # 带桌子+目标球的任务场景
├── README.md              # 文档
├── requirements.txt       # mujoco>=3.0.0, numpy>=1.24.0
├── model/
│   ├── task_scene.xml     # 简化任务场景（地板+天空+目标球）
│   ├── g1.xml             # G1 MJCF 模型
│   ├── g1_mjx.xml         # MJX 版本
│   └── assets/            # 网格、纹理
└── eval_frames/           # 离线渲染截图（34 张 PNG + composite.png）
```

### 5.2 validate_policy.py 关键设计

- **数据驱动**：关节名从 checkpoint config 读取，不硬编码
- **目标采样**：在 G1 手臂可达工作空间内（肩关节 0.28m 半径内），非 VR 人类手臂范围
- **动态观测构建**：`build_obs()` 支持任意关节数（nJ×2 + 15）
- **实时可视化**：`--visualize` 启动 MuJoCo 交互式 Viewer，30Hz 帧率同步
- **离线渲染**：通过 `mujoco.Renderer` 导出 PNG 截图

### 5.3 评估参数

| 参数 | 值 |
|------|-----|
| Reach 阈值 | 0.16m |
| 保持时间 | 0.25s |
| Episode 超时 | 20.0s |
| 每轮目标数 | 5 |
| 策略频率 | 30 Hz |
| 仿真步长 | 0.002s |

---

## 6. 训练与评估结果

### 6.1 训练指标对比

| 模型 | Run | obs/act | Best Epoch | Val MSE | Val MAE |
|------|-----|---------|-----------|---------|---------|
| BC | Run 1 (14D) | 43/14 | 67 | 0.0847 | — |
| **BC** | **Run 2 (8D)** | **31/8** | **79** | **0.1821** | **0.2259** |
| ACT | Run 1 (14D) | 43/14 | 79 | 0.0220 | — |
| **ACT** | **Run 2 (8D)** | **31/8** | **79** | **0.0416** | **0.0708** |

> ACT 离线指标远优于 BC（val_mse 低 77%），但闭环部署表现反转。

### 6.2 MuJoCo 闭环评估

| 模型 | Run | Trials | 目标数 | 成功 | 成功率 | 最小距离范围 |
|------|-----|--------|-------|------|--------|------------|
| ACT | Run 1 (14D) | 5 | 25 | 0 | **0.0%** | 0.62~1.38m |
| ACT | Run 2 (8D) | 1 | 5 | 0 | **0.0%** | 0.15~0.31m |
| **BC** | **Run 2 (8D)** | **5** | **25** | **7** | **28.0%** | **0.12~0.48m** |

**BC Run 2 逐轮详情**：

| Trial | 成功/目标 | 各目标最小距离 (m) |
|-------|----------|-------------------|
| 1 | **4/5** | 0.125, 0.155, 0.146, 0.153, 0.311 |
| 2 | **2/5** | 0.249, 0.263, 0.116, 0.127, 0.240 |
| 3 | **1/5** | 0.152, 0.248, 0.210, 0.481, 0.171 |
| 4 | 0/5 | 0.273, 0.179, 0.306, 0.226, 0.216 |
| 5 | 0/5 | 0.207, 0.190, 0.320, 0.459, 0.189 |

### 6.3 关键发现

1. **14 DOF → 8 DOF 降维**是突破 0% 的关键：
   - 6 个顺腕关节在训练数据中方差为零（reach 任务不用手腕），归一化后产生 NaN/极端值
   - 移除后 ACT 最小距离从 0.6m+ → 0.15m，BC 从不可用 → 28% 成功
2. **目标采样修复**：从 VR 人类手臂范围（0.73m）→ G1 可达范围（0.28m）
3. **BC > ACT 在闭环中**：ACT 离线 MSE 更低但闭环失败，原因可能是：
   - ACT 序列预测的累积误差在闭环中放大
   - BC 单步预测更稳健，对 distribution shift 更鲁棒
4. **VR↔MuJoCo FK 差距**（已知未修复）：相同关节角度产生 ~45cm 的末端位置差异，但关节空间策略在 eval 中使用 MuJoCo FK 保持一致性，所以可以工作

---

## 7. 可视化能力

### 7.1 离线渲染

通过 `mujoco.Renderer` 在 eval 过程中截取关键帧（start/mid/success/timeout），保存为 PNG。

输出目录：`mujoco_sim/eval_frames/`（34 帧 + composite.png 合成图）

### 7.2 实时交互式查看

```bash
cd mujoco_sim

# ACT 策略实时查看
python validate_policy.py \
  --checkpoint "../20260408_g1_reach/outputs/act/run_20260409_124600/checkpoints/best.pt" \
  --visualize --num-trials 1

# BC 策略实时查看
python validate_policy.py \
  --checkpoint "../20260408_g1_reach/outputs/bc/run_20260409_124525/checkpoints/best.pt" \
  --visualize --num-trials 1
```

- `--visualize`：打开 MuJoCo 交互式窗口（鼠标旋转/缩放）
- `--num-trials N`：运行 N 轮（每轮 5 个目标）
- `--seed S`：指定随机种子（换目标位置）

---

## 8. 诊断过程：从 0% 到 28%

### 8.1 第一轮部署（0/25 成功）

**现象**：手臂几乎不动，最小距离 0.6~1.4m。

**诊断三个根因**：
1. G1 手臂可达半径（0.31m）远小于 VR 人类手臂（0.73m），所有目标不可达
2. VR 关节角 → MuJoCo FK 存在 45cm 末端位置偏差
3. 6 个零方差顺腕关节导致归一化后观测爆炸

### 8.2 修复措施

| 问题 | 修复 |
|------|------|
| 零方差关节 | 从 config.json 移除 6 个顺腕关节（14D→8D） |
| 目标不可达 | `sample_reachable_target()` 在肩关节 0.28m 范围内采样 |
| 硬编码维度 | `validate_policy.py` 从 checkpoint 读取关节名，动态构建 obs |

### 8.3 第二轮部署（7/25 成功，28%）

BC 策略在 G1 可达工作空间内成功完成 reach 任务，最佳单轮 4/5 成功。

---

## 9. 完整文件变更清单

### 新建文件

| 文件 | 描述 |
|------|------|
| `20260408_g1_reach/config.json` | 实验配置 |
| `20260408_g1_reach/common.py` | 数据加载与特征提取 |
| `20260408_g1_reach/train_bc.py` | BC 训练代码 |
| `20260408_g1_reach/train_act.py` | ACT 训练代码 |
| `20260408_g1_reach/run_all.py` | 一键运行管线 |
| `20260408_g1_reach/analyze_results.py` | 对比分析 |
| `20260408_g1_reach/README.md` | 管线文档 |
| `mujoco_sim/setup_model.py` | G1 模型下载 |
| `mujoco_sim/joint_mapping.py` | 关节映射与限位 |
| `mujoco_sim/validate_replay.py` | VR 数据回放验证 |
| `mujoco_sim/validate_policy.py` | 策略闭环评估+可视化 |
| `mujoco_sim/g1_dual_arm_scene.xml` | 完整任务场景 |
| `mujoco_sim/model/task_scene.xml` | 简化任务场景 |
| `mujoco_sim/README.md` | 环境文档 |
| `mujoco_sim/requirements.txt` | 依赖 |
| `data_collector/collector6/*` | 新采集数据（3 文件） |

### 修改文件

| 文件 | 变更内容 |
|------|---------|
| `src/components/RobotVR.vue` | 超时逻辑修复 + 手臂分配修复 |
| `mujoco_sim/model/task_scene.xml` | 添加 `offwidth=1280 offheight=960`（离线渲染支持） |

### 生成产物

| 路径 | 内容 |
|------|------|
| `20260408_g1_reach/outputs/bc/run_20260409_124525/` | BC 8D 最佳模型 (28% MuJoCo 成功率) |
| `20260408_g1_reach/outputs/act/run_20260409_124600/` | ACT 8D 模型 |
| `20260408_g1_reach/outputs/summary/` | 对比分析报告 |
| `mujoco_sim/eval_frames/` | 34 张 MuJoCo 渲染截图 |
| `mujoco_sim/model/` | 下载的 G1 MJCF 模型 |

---

## 10. 下一步建议

1. **提升成功率**：
   - 采集更多 collector6 格式数据（当前仅 25 segment），增加到 100+ segment
   - 尝试 Domain Randomization（随机化目标位置、初始关节角）
   - 调整 reach 阈值（当前 0.16m 偏严格，许多 0.17~0.20m 的"近成功"被判失败）

2. **ACT 闭环改进**：
   - 降低 ACT 序列长度（16→8），减少累积误差
   - 尝试 action chunking / temporal ensemble（预测多步但只执行第一步）

3. **EE-space 策略**：
   - 输出末端位置增量 $\Delta x$ 而非关节角增量 $\Delta q$
   - 配合 IK 控制器，绕过 VR↔MuJoCo FK 偏差问题

4. **数据增强**：
   - 目标位置镜像翻转（左↔右）
   - 添加关节角噪声增强鲁棒性
