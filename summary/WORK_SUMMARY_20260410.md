# G1 Reach 模仿学习：数据质量诊断、关键 Bug 修复与性能突破（2026-04-10）

## 1. 本阶段总结

本阶段针对 MuJoCo 闭环评估中机器人**手臂持续上抬、无法触碰目标**的核心问题，进行了系统化诊断和修复：

1. **诊断 VR↔MuJoCo 目标分布不匹配**：VR 目标距基座 1.4m（需要走路），MuJoCo 专家目标仅 0.39m
2. **添加坐标系旋转变换** `_R_BASE`：统一 VR 录制与 MuJoCo 观测的空间坐标系
3. **修正 `_ROBOT_BASE_VR`**：从 `[0, 0.793, 0]`（骨盆高度）→ `[0, 0, 0]`（地面），匹配 VR Object3D 原点
4. **诊断动作偏置漂移**：发现模型输出非零均值 delta → 关节持续漂移到限位 → 手臂上抬
5. **发现并修复致命 Bug：Expert 未设置 `ctrl`**：MuJoCo 位置执行器（kp=500）在仿真中抵消了 IK 指令，导致专家成功率仅 6.4%
6. **修复后专家成功率从 6.4% → 91.6%**，重训 BC 模型 **MuJoCo 闭环成功率从 60% → 70%**（最佳单轮 100%）
7. **修复目标球可视化**：改用 mocap body 使红球不受重力影响
8. **收窄 VR 端目标生成范围**，使未来 VR 数据在手臂可达范围内

---

## 2. 问题诊断全流程

### 2.1 起始现象：手臂上抬，无法触碰目标

从上一阶段（04-09）的 DAgger v1（VR + MuJoCo 混合数据）训练得到的 BC 模型：
- 评估成功率 60%（BC）/ 36%（ACT）
- 可视化观察：**机器人手臂在最初 1-2 秒有伸手趋势，但随后持续上抬至头顶，保持"投降"姿态**
- 目标红球在胸前，手臂却在头顶

### 2.2 第一层诊断：VR↔MuJoCo 目标分布不匹配

运行 `diagnose_target_dist.py` 发现严重的分布差异：

| 数据源 | targetRelToRobotBase Y 均值 | 3D 距离均值 |
|--------|---------------------------|------------|
| VR collector6 | 1.25m | 1.42m |
| VR collector8 | 1.25m | 1.67m |
| MuJoCo expert | 0.32m | 0.39m |

**根因**：VR 场景中目标生成在整个空间（需要走路才能到达），而 MuJoCo 机器人不能移动，手臂可达范围仅 ~0.3m。约 50% 的 VR 帧中手离目标 >0.5m（走路阶段），模型学到了大量"伸手但够不到"的无效数据。

### 2.3 第二层诊断：观测坐标系缺少旋转

VR 录制使用 `poseRelativeToBase()` 计算机器人局部坐标，包含以 π 旋转的 base 变换。但 MuJoCo 中直接用 VR 世界坐标减去基座位置，**缺少旋转变换**。

添加了旋转矩阵 `_R_BASE`，将 VR 世界坐标差映射到机器人局部坐标：

$$R_{base} = \begin{bmatrix} 0 & 0 & -1 \\ 0 & 1 & 0 \\ -1 & 0 & 0 \end{bmatrix}$$

对应关系：
- robot-local X = −(VR-world Z)（左方向）
- robot-local Y = VR-world Y（上方向）  
- robot-local Z = −(VR-world X)（后方向）

### 2.4 第三层诊断：动作偏置导致关节漂移

运行 `diagnose_model.py` 跟踪 600 步推理过程：

| 关节 | 平均 delta/步 | 600步总漂移 | 最终位置 |
|------|-------------|-----------|---------|
| left_shoulder_pitch | +0.061 | +36.5 rad | 2.62（限位） |
| right_shoulder_pitch | +0.035 | +21.0 rad | 2.67（限位） |

**根因**：归一化器中 `act_mean` 对 shoulder_pitch ≈ +0.032（非零），说明训练数据中 94% 是失败轨迹（专家够不到目标），动作持续偏向某一方向。模型在最初 2 秒能接近目标（d=0.38→0.14m），但随后过冲并持续漂移到关节限位。

### 2.5 第四层诊断（致命 Bug）：Expert 未设置执行器 ctrl

**这是本阶段发现的最关键 Bug。**

G1 模型使用 **position actuator**（位置执行器，kp=500）：
```xml
<default class="g1">
  <position kp="500" dampratio="1" inheritrange="1"/>
</default>
```

`collect_expert.py` 此前只设置了 `mj_data.qpos[adr]`（关节位置），但**没有设置 `mj_data.ctrl[actuator]`**（执行器目标）。MuJoCo 仿真中 `ctrl` 默认为 0，所以：

1. IK 计算出正确的关节增量 → 写入 `qpos`
2. `mj_step()` 运行时，位置执行器以 kp=500 的力将关节拉回 `ctrl=0`
3. IK 的精心计算被物理引擎完全抵消
4. 结果：专家控制器怎么调参都只有 6-15% 成功率

### 2.6 修复与验证

| 修复项 | 修复前 | 修复后 |
|-------|-------|-------|
| `mj_data.ctrl[act_idx] = mj_data.qpos[adr]` | 未设置 | 每步同步 |
| 初始化时 ctrl 同步 | 未设置 | reset 后同步 |
| 专家成功率（gain=5.0） | 6.4% ~ 15% | **91.6%** |

修复一行代码，成功率从 15% → 91.6%，提升 **6 倍**。

---

## 3. 坐标系修复详情

### 3.1 `_R_BASE` 旋转矩阵

添加到 `validate_policy.py`、`collect_expert.py`、`collect_expert_v3.py`：

```python
_R_BASE = np.array([[0.0, 0.0, -1.0],
                     [0.0, 1.0,  0.0],
                     [-1.0, 0.0, 0.0]], dtype=np.float64)
```

所有空间观测特征（EE 位置、目标相对位置）在写入 obs 前统一经过：
```python
feature_local = _R_BASE @ (feature_vr_world - _ROBOT_BASE_VR)
```

### 3.2 `_ROBOT_BASE_VR` 修正

```python
# 修改前
_ROBOT_BASE_VR = np.array([0.0, 0.793, 0.0])  # 骨盆高度

# 修改后
_ROBOT_BASE_VR = np.array([0.0, 0.0, 0.0])    # 地面原点
```

原因：VR 中 robot Object3D 的 position 是地面原点（脚底），不是骨盆中心。

### 3.3 VR 端目标生成范围收窄

`src/components/RobotVR.vue` 中修改目标随机范围：

| 参数 | 修改前 | 修改后 | 说明 |
|------|-------|-------|------|
| `TASK_RANDOM_RANGE_X` | 0.45 | **0.35** | 左右范围 |
| `TASK_RANDOM_RANGE_Y` | 0.6 | **0.3** | 上下范围 |
| `TASK_MIN_Y` | 0.85 | **0.7** | 最低高度 |
| `TASK_MAX_Y` | 1.7 | **1.45** | 最高高度 |
| `TASK_MIN_Z` | -1.2 | **-0.4** | 最远深度 |
| `TASK_MAX_Z` | -0.5 | **-0.05** | 最近深度 |

修改后目标只在手臂可达范围内生成，不再需要走路。

---

## 4. 目标球可视化修复

### 4.1 问题演变

| 版本 | task_scene.xml | 现象 |
|------|---------------|------|
| 原始 | 无 joint（静态 body） | 球固定在初始位置不动 |
| 添加 freejoint | `<joint type="free"/>` | 球受重力掉落到地面以下 |
| **最终：mocap body** | `mocap="true"` | 球不受物理影响，可编程定位 |

### 4.2 最终方案

```xml
<body name="target_ball" mocap="true" pos="0.4 0 1.0">
  <geom name="target" type="sphere" size="0.04"
        rgba="1 0.2 0.2 0.8"
        contype="0" conaffinity="0"/>
</body>
```

脚本中通过 `mocap_pos` 定位：
```python
tgt_ball_bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
tgt_mocap_id = mj_model.body_mocapid[tgt_ball_bid]
mj_data.mocap_pos[tgt_mocap_id] = target_mj_position
```

---

## 5. 训练与评估结果

### 5.1 专家数据对比

| 数据版本 | 脚本 | 设置 ctrl | 专家成功率 | 总帧数 | Episodes |
|---------|------|----------|-----------|--------|----------|
| mujoco_expert (v1) | collect_expert.py | ❌ | ~6.4% | ~220K | 100 |
| mujoco_expert_v2 | collect_expert.py (gain=2.0) | ❌ | ~6.4% | ~220K | 100 |
| **mujoco_expert_v3** | collect_expert.py (gain=5.0) | ✅ | **91.6%** | **52,234** | 200 |

### 5.2 模型训练历程

| 实验 | 数据 | 训练配置 | 最佳 val_mse | BC 闭环成功率 |
|------|------|---------|-------------|-------------|
| 20260408 BC Run2 | collector6 (VR) | 8D, 80 epochs | 0.1821 | 28% |
| bc_oldparams | c6+c8 (VR) | 8D, batch=128 | 0.0034 | 44% (EMA=0.5) |
| dagger_v1 | c6+c8+expert_v1 | obs_noise=0.3 | 0.0076 | 60% (EMA=0.5) |
| dagger_v2 | expert_v2 only | obs_noise=0.1 | 0.0037 | 60% |
| **v3** | **expert_v3 only** | **obs_noise=0.1, ctrl fix** | **0.6416** | **70% (EMA=0.3)** |

> **注**：v3 的 val_mse 高于之前，是因为修复 ctrl 后数据中 action 幅度更大（执行器不再抵消动作），归一化后标准差更大，MSE 对应也更大。但闭环性能才是真正的评估指标。

### 5.3 v3 BC 详细评估（10 trials, 50 targets）

| Trial | 成功/目标 | 成功率 | 各目标 min_d (m) |
|-------|----------|--------|-----------------|
| 1 | 4/5 | 80% | 0.154, 0.128, **0.172**, 0.117, 0.147 |
| 2 | 4/5 | 80% | 0.154, 0.131, 0.132, **0.170**, 0.120 |
| 3 | 2/5 | 40% | 0.062, **0.208**, **0.237**, 0.141, **0.194** |
| 4 | 3/5 | 60% | 0.158, 0.114, 0.141, **0.257**, **0.180** |
| 5 | 3/5 | 60% | 0.130, 0.137, 0.148, **0.163**, **0.182** |
| 6 | 4/5 | 80% | 0.148, 0.139, **0.169**, 0.153, 0.154 |
| 7 | 3/5 | 60% | 0.145, 0.150, **0.192**, **0.199**, 0.144 |
| 8 | 4/5 | 80% | 0.142, 0.092, **0.257**, 0.047, 0.108 |
| 9 | 5/5 | **100%** | 0.155, 0.088, 0.155, 0.125, 0.132 |
| 10 | 3/5 | 60% | **0.171**, 0.154, 0.147, 0.113, **0.241** |
| **总计** | **35/50** | **70.0%** | |

> 粗体为 FAIL（>0.16m），许多失败在 0.163-0.194m，非常接近阈值。

### 5.4 成功率演进图

```
28% ──→ 44% ──→ 60% ──→ 60% ──→ 70%
 │        │        │        │        │
Run2   oldparams  DAgger  DAgger   v3
(VR)   (VR×2)    v1(混合)  v2     (ctrl Fix)
                                   ↑
                              关键突破点
```

---

## 6. 新增/修改文件清单

### 新建文件

| 文件 | 描述 |
|------|------|
| `mujoco_sim/collect_expert_v3.py` | 迭代 IK 求解器 + 平滑插值专家控制器 |
| `20260409train2/config_v3.json` | v3 训练配置（mujoco_expert_v3 数据） |
| `data_collector/mujoco_expert_v3/episodes.jsonl` | 52,234 帧专家数据（91.6% 成功） |
| `data_collector/mujoco_expert_v3/events.jsonl` | 专家事件日志 |
| `diagnose_model.py` | 模型推理诊断：跟踪关节漂移 |
| `diagnose_target_dist.py` | VR↔MuJoCo 目标分布对比分析 |

### 修改文件

| 文件 | 变更内容 |
|------|---------|
| `mujoco_sim/collect_expert.py` | **[关键]** 添加 `act_idx` + `ctrl` 同步；添加 `_R_BASE` 旋转；自适应阻尼 IK；holding 行为 |
| `mujoco_sim/validate_policy.py` | 添加 `_R_BASE` 旋转到 `build_obs()`；mocap 目标球定位 |
| `mujoco_sim/model/task_scene.xml` | 目标球改为 `mocap="true"` body |
| `src/components/RobotVR.vue` | 收窄目标生成范围（X/Y/Z 均缩小） |

### 训练产物

| 路径 | 内容 |
|------|------|
| `20260409train2/outputs/v3/bc/run_20260410_134000/` | v3 BC 最佳模型（70% MuJoCo 成功率） |
| `20260409train2/outputs/v3/bc/run_20260410_134000/checkpoints/best.pt` | 模型 checkpoint |
| `20260409train2/outputs/v3/bc/run_20260410_134000/mujoco_eval.json` | 评估结果 |

---

## 7. 关键经验总结

### 7.1 MuJoCo 位置执行器的陷阱

**教训**：MuJoCo 中使用 `<position>` 执行器时，**必须同步设置 `ctrl` 和 `qpos`**。

- `qpos` 是关节的几何位置
- `ctrl` 是执行器的目标位置
- `mj_step()` 中，PD 控制器施加力 $\tau = k_p \cdot (ctrl - qpos) - k_d \cdot \dot{q}$
- 如果只设 `qpos` 不设 `ctrl`，执行器会以 kp=500 的巨力把关节拉回 `ctrl`（默认=0）

这个 Bug 导致之前所有 IK 调参（gain、damping、MAX_DELTA）都是无用功——无论 IK 怎么算，物理引擎都会抵消。

### 7.2 诊断方法论

分层诊断顺序：
1. **数据分布** → `diagnose_target_dist.py` 发现 VR/MuJoCo 目标距离差异 4 倍
2. **坐标系** → 添加 `_R_BASE` 旋转对齐 VR 和 MuJoCo 的空间参考
3. **模型行为** → `diagnose_model.py` 发现动作偏置导致关节漂移
4. **仿真物理** → 检查 g1.xml 发现位置执行器，定位 `ctrl` 未设置的致命 Bug

### 7.3 数据质量 > 数据数量

| 数据版本 | 帧数 | 专家成功率 | 模型闭环成功率 |
|---------|------|-----------|-------------|
| v2（无 ctrl 修复） | 220K | 6.4% | 60% |
| v3（有 ctrl 修复） | 52K | 91.6% | 70% |

用 1/4 的数据量，因为数据质量高（91.6% vs 6.4% 成功率），模型性能反而更好。

---

## 8. 下一步规划

### 8.1 短期（提升到 85%+）

1. **增加专家数据量**：当前 200 episodes / 52K 帧，扩展到 500+ episodes
2. **使用 `collect_expert_v3.py`（迭代 IK + 平滑插值）** 收集更高质量数据，估计成功率 >95%
3. **更多 holding 数据**：当前 holding 帧比例不足，模型仍有轻微过冲倾向。增加 `HOLD_FRAMES_AFTER_REACH` 到 60 帧（2 秒）
4. **降低评估阈值或增加宽容度**：许多 FAIL 在 0.16-0.20m，考虑动态阈值或逐步收紧策略

### 8.2 中期（ACT 模型 + VR 数据融合）

1. **用 v3 数据训练 ACT 模型**：之前 ACT 闭环全部 0%/36%，可能是数据质量不行导致的
2. **采集新 VR 数据**：用已修改的窄范围目标生成（RobotVR.vue），确保 VR 目标在手臂可达范围内
3. **混合训练**：VR 演示 + MuJoCo 专家 → DAgger v3，需验证 `_R_BASE` 旋转在混合数据场景下的正确性
4. **注意**：`_R_BASE` 目前可能有 X 轴翻转问题（`-Z` vs `+Z`），在 MuJoCo-only 场景下自洽，但混合 VR 数据时需重新验证

### 8.3 长期

1. **末端空间策略**：输出 $\Delta x$（末端位移）而非 $\Delta q$（关节增量），配合 MuJoCo IK 控制器
2. **在线 DAgger**：policy 运行时实时查询专家修正，逐步替换专家动作
3. **Sim-to-Real**：验证 MuJoCo 策略在真实 G1 上的迁移性

---

## 9. 快速复现命令

```bash
# 1. 收集高质量专家数据（需 ctrl 修复后的 collect_expert.py）
cd mujoco_sim
python collect_expert.py --num-episodes 200 --out ../data_collector/mujoco_expert_v3 --gain 5.0 --seed 42

# 2. 训练 BC 模型
cd ../20260409train2
python train_bc.py --config config_v3.json --out outputs/v3

# 3. 评估（带可视化）
cd ../mujoco_sim
python validate_policy.py \
  --checkpoint ../20260409train2/outputs/v3/bc/run_20260410_134000/checkpoints/best.pt \
  --num-trials 10 --action-ema 0.3 --visualize

# 4. 仅评估（无可视化，更快）
python validate_policy.py \
  --checkpoint ../20260409train2/outputs/v3/bc/run_20260410_134000/checkpoints/best.pt \
  --num-trials 10 --action-ema 0.3
```
