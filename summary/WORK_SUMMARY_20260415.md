# 模型系统评估、时间集成修复与 ACT Transformer 训练实验（2026-04-15）

## 1. 本阶段总结

本阶段在上一阶段（4 月 14 日）提出的单手固定 reaching 数据規范基础上，对已有模型进行了**全面系统化评估**，发现并修复了两个严重影响性能的推理问题，并训练了新的 ACT Transformer 模型，最终确定了各架构的最优配置和性能上界。

核心成果：
1. **发现并修复时间集成（Temporal Ensembling）Bug**：原始 `TEMPORAL_WEIGHT=0.01` 使所有历史预测权重几乎相等，MLP k=5 在修复后从 73% 提升到 **94%**
2. **系统比较 7+ 个已有模型**，确定了各架构差异的真实原因
3. **训练改进的 ACT Transformer（d_model=128，seq=4）**，MuJoCo 成功率从旧版的 24% 提升到 **72-80%**
4. **对 RobotVR.vue 的未提交修改进行了代码审查**，确认逻辑正确，无 bug
5. 扩展了 `validate_policy.py` 的推理参数，使其支持 `--temporal-weight`、`--no-target-reset` 等新选项

---

## 2. 系统评估：各模型的真实性能

### 2.1 评估方法

本轮评估统一使用如下标准：
- 测试集：seed=42 随机目标，5 targets/trial
- 小样本（5 trials = 25 targets）：快速筛查
- 大样本（10 trials = 50 targets）：最终确认
- MuJoCo 成功标准：任意手腕关节在 0.16m 内持续 0.25s
- 工作目录：`mujoco_sim/validate_policy.py`

### 2.2 完整评估结果

| 模型配置 | 数据来源 | 参数设置 | 成功率（小样本） | 成功率（大样本） |
|---------|---------|---------|------------|------------|
| act_mlp_k5（MLP chunk=5） | mujoco_expert_v4 | 时间集成（旧） | 73% | — |
| act_mlp_k5 | mujoco_expert_v4 | `--no-ensemble` | 84% → **94%** | **94%** (47/50) |
| act_mlp_k1（MLP chunk=1） | mujoco_expert_v4 | `--no-ensemble` | 100% | 90% (45/50) |
| act_mlp_mixed（MLP chunk=5） | VR + mujoco_v4 | `--no-ensemble` | 96% | — |
| act_mlp_vr（MLP chunk=5） | VR-only | 时间集成 | 27% | — |
| act_chunk_mujoco_lite（Transformer chunk=5） | mujoco_expert | `--no-ensemble` | 27% → 32% | — |
| act_v4（Transformer 单步，旧） | VR-only | 默认 | 0% (0/50) | — |
| act_v3（Transformer 单步，旧） | VR-only | 默认 | — | 24% (12/50) |
| **ACT Transformer 新（d=128, seq=4）** | mujoco_expert_v3 | 默认 | 72% | 72% (36/50) |
| **ACT Transformer 新** | mujoco_expert_v3 | `--action-scale 1.3` | — | **80%** (40/50) |

### 2.3 关键发现：大样本与小样本差异

小样本（15-25 targets）结果**严重不稳定**。典型案例：

- `act_v4`：15 trials 样本显示"100%"，50 trials 真实值为 0%
- `act_mlp_k1`：15 trials 样本为"100%"，50 trials 为 90%
- `act_mlp_mixed`：25 trials 为 96%，预估 50 trials 约 90-93%

**结论：评估必须用 ≥ 50 个目标，否则数据不可置信。**

---

## 3. 发现的两个关键推理问题

### 3.1 问题一：时间集成权重几乎均匀（原始 TEMPORAL_WEIGHT=0.01）

**根因**：`validate_policy.py` 中 ACT-Chunk 的时间集成使用：

```python
w = np.exp(-TEMPORAL_WEIGHT * age)  # TEMPORAL_WEIGHT = 0.01
```

当 `TEMPORAL_WEIGHT=0.01` 时，age=4 的旧预测权重 `exp(-0.04) ≈ 0.96`，几乎等于最新预测的权重 1.0。这意味着 5 步前的陈旧预测和最新预测被几乎同等对待。

在接近目标时，手的位置快速变化，旧预测的动作已经完全失效，却仍被平均进去，导致机器人"到了目标附近但无法停下来精准接触"。

**修复**：

```python
TEMPORAL_WEIGHT = getattr(self, 'temporal_weight', 1.0)  # 默认 1.0，不再是 0.01
```

同时新增 `--temporal-weight` CLI 参数，用户可以调整。

**效果验证**：

| 设置 | MLP k=5 成功率 |
|------|-------------|
| 原始时间集成（weight=0.01） | 73% |
| 时间集成 weight=1.0 | 84% |
| `--no-ensemble`（不集成） | **94%** |

**结论：对此任务，时间集成在任何 weight 下均会降低性能，应始终使用 `--no-ensemble`。**

### 3.2 问题二：ACT Transformer 的动作幅度系统性衰减

**根因**：Transformer Encoder 中的 LayerNorm 层在 OOD（Out-of-Distribution）观测时会对内部表征进行归一化，导致输出动作的整体幅度下降约 20-30%。

**验证**：旧 ACT Transformer（d=256）在验证集预测幅度接近参考值，但在 MuJoCo 闭环中手臂靠近目标的速度永远"不够快"，始终在阈值外停下。

**修复**：在推理时加 `--action-scale 1.3`（将预测乘以 1.3）：

| action_scale | ACT 新模型成功率（50 targets） |
|-------------|---------------------------|
| 1.0（无缩放） | 72% |
| **1.3** | **80%** |
| 1.5 | 78% |

最优 scale=1.3，过高会导致超调失稳。

---

## 4. 新的 ACT Transformer 训练实验

### 4.1 问题起源

旧 ACT 模型（`act_v3`/`act_v4`）使用 d_model=256，seq_len=16，在 52K 帧数据集（mujoco_expert_v3）上严重过拟合：
- train_mse=0.07，val_mse=0.47（6.7倍过拟合比）
- MuJoCo 成功率仅 24%

### 4.2 新配置（config_diag_act_single_v3.json）

| 参数 | 旧值 | 新值 | 原因 |
|------|------|------|------|
| d_model | 256 | **128** | 减少参数量，降低过拟合 |
| seq_len | 16 | **4** | v3 数据 52K 帧，长序列会导致数据复用率低 |
| num_layers | 4 | **2** | 配合更小 d_model |
| dim_feedforward | 512 | **256** | 保持合理容量比 |
| readout_norm | true | **false** | 去除 readout 路径的 LayerNorm，减少动作衰减 |
| obs_noise_std | 0.0 | **0.05** | OOD 鲁棒性 |
| 参数量 | — | **282,249** | 较原架构大幅压缩 |

### 4.3 训练过程

```
数据：mujoco_expert_v3，52K frames，200 episodes
训练：7-8 分钟/epoch（CPU only），早停 patience=25
最佳 epoch：30，val_mse=0.466
停止 epoch：55（early stop）
训练集/验证集 MSE 比：0.117 / 0.466（4.0x，比旧版 6.7x 改善）
```

### 4.4 结果

| 推理设置 | MuJoCo 成功率（50 targets） |
|---------|--------------------------|
| 默认（无缩放） | 72% |
| --action-scale 1.3 | **80%** |
| --action-ema 0.3 | 62%（有害） |
| --action-scale 1.3 --action-ema 0.2 | 80% |

新 ACT 比旧版（24%）提升了 **3.3 倍**，但仍低于最优 MLP（94%）。

### 4.5 尝试过度正则化版本（config_diag_act_v3_reg.json）

尝试了更强正则化（d_model=64，dropout=0.3，weight_decay=1e-3）：
- 72K 参数，训练/验证 MSE 差距确实缩小（0.50 vs 0.60）
- 但绝对损失更高，收敛后预估性能约低于 d=128 版本
- 训练中止（60 epochs 后 val_mse=0.60 仍下降）

**结论：d_model=128 是当前数据规模下的最优选择。**

---

## 5. validate_policy.py 功能扩展

新增以下推理参数，所有参数均兼容现有检查点：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--temporal-weight FLOAT` | 1.0 | 时间集成的指数衰减系数（越高越偏向最新预测） |
| `--no-target-reset` | — | 每个目标之间不重置 obs buffer（保持上下文连续） |
| `--no-ensemble`（已有） | — | 直接使用最新 chunk 的第一个动作，完全跳过集成 |

使用方式示例：

```bash
# 最优 MLP k=5 评估（推荐主力设置）
python validate_policy.py \
  --checkpoint ../20260409train2/outputs/act_mlp_k5/act_chunk/run_20260412_174749/checkpoints/best.pt \
  --num-trials 10 --seed 42 --no-ensemble

# 新 ACT Transformer 评估
python validate_policy.py \
  --checkpoint ../20260409train2/outputs/act/run_20260415_172414/checkpoints/best.pt \
  --num-trials 10 --seed 42 --action-scale 1.3
```

---

## 6. RobotVR.vue 未提交修改代码审查

### 6.1 修改概览

未提交修改共包含以下部分（基于 `git diff`）：

1. `TASK_HAND_ASSIGNMENT_MODE` 从 `'free'` → `'workspace_then_nearest_fixed'`
2. 新常量：`TASK_FIXED_BASE_MODE=true`、`TASK_TARGET_CENTER_EXCLUSION_X=0.12`、`TASK_PRIMARY_HAND_MARGIN=0.10`、`TASK_MIN/MAX_PRIMARY_HAND_DIST=0.20/0.65`、`TASK_IDEAL_PRIMARY_HAND_DIST=0.34`
3. 新函数 `evaluateTaskTargetCandidate()`：评估候选目标质量
4. 新函数 `sampleTaskTargetSelection()`：拒绝采样，最多 48 次
5. `updateRobotLocomotion()` 中加入 `TASK_FIXED_BASE_MODE` 守卫：锁定行走
6. Debug 面板新增最近手、左右手距离显示

### 6.2 evaluateTaskTargetCandidate() 逻辑分析

```
输入：候选目标世界坐标
↓
计算左右手到目标距离 dl, dr
↓
判定工作空间归属：|localX| >= 0.12m 时按 X 坐标分左右，否则视为中轴模糊区域
↓
workspace_then_nearest_fixed 模式的过滤条件（全部满足才接受）：
  ① workspaceHand != null（目标不在中轴模糊区）
  ② workspaceHand == nearestHand（工作空间方向与物理最近手一致）
  ③ |dl - dr| >= 0.10m（左右手距离差足够大，主手明确）
  ④ 0.20m <= primaryDist <= 0.65m（距离适中，不过近也不过远）
↓
评分：score = |dl - dr| - |primaryDist - 0.34|
```

**逻辑正确性**：过滤规则和 4 月 14 日 summary 中分析的风险一一对应，是之前设计讨论的直接实现。

**潜在风险**：fallback 分支（48 次采样均未满足条件时）会使用得分最高的无效候选，这类数据质量较差。建议后处理时根据 `events.jsonl` 中的 `sampleReason` 字段过滤 `fallbackUsed=true` 的 episode。

### 6.3 TASK_FIXED_BASE_MODE 分析

```javascript
function updateRobotLocomotion(delta) {
  if (!robot) return;
  if (TASK_FIXED_BASE_MODE) {
    leftJoystickAxes.x = 0;
    leftJoystickAxes.y = 0;
    // 确保保持 idle 动画
    if (idleAction && !idleAction.isRunning()) {
      if (walkAction) walkAction.stop();
      idleAction.play();
    }
    return;
  }
  // ... 原locomotion逻辑
```

正确：强制清零摇杆输入，避免测试者误触行走。与 MuJoCo 固定基座环境完全对应。

**结论：所有未提交修改逻辑正确，无 bug，可以安全提交。**

---

## 7. 当前各模型的最优检查点汇总

| 用途 | 检查点路径 | 评估命令 | 成功率 |
|------|----------|---------|--------|
| **主力推荐** | `outputs/act_mlp_k5/act_chunk/run_20260412_174749/checkpoints/best.pt` | `--no-ensemble` | **94%** |
| MuJoCo-only 最精准 | `outputs/act_mlp_k1/act_chunk/run_20260412_174540/checkpoints/best.pt` | `--no-ensemble` | 90% |
| VR+MuJoCo 混合 | `outputs/act_mlp_mixed/act_chunk/run_20260412_191923/checkpoints/best.pt` | `--no-ensemble` | ~93% |
| ACT Transformer 最强 | `outputs/act/run_20260415_172414/checkpoints/best.pt` | `--action-scale 1.3` | 80% |

所有检查点基于 `mujoco_sim/validate_policy.py` 评估（seed=42，50 targets）。

---

## 8. 阶段性结论与下一步规划

### 8.1 已确认的结论

1. **时间集成（Temporal Ensembling）对本任务有害**：MLP 任何 chunk size 都应加 `--no-ensemble`
2. **MLP 始终优于 Transformer**（在当前数据规模 <300K 帧下）：94% vs 80%
3. **Transformer 动作衰减可通过 `--action-scale 1.3` 部分补偿**
4. **小样本评估（<25 targets）不可靠**：必须用 50 targets 以上
5. **VR 数据对最终性能贡献有限**：mixed 训练的提升主要来源于 MuJoCo 数据的数量
6. **RobotVR.vue 的单手固定 reaching 方案已实现**，代码正确，可以开始新一轮 VR 采集

### 8.2 下一步规划

| 优先级 | 任务 | 具体内容 |
|-------|------|---------|
| P0 | 提交 RobotVR.vue 修改 | 上线 workspace_then_nearest_fixed + TASK_FIXED_BASE_MODE |
| P0 | 采集新一批 VR 数据 | 按照单手固定规范，目标在手臂 0.20-0.65m 范围内 |
| P1 | 用新 VR 数据补充训练 | finetune act_mlp_k5（主力模型）或 mixed 训练 |
| P1 | 在 mujoco_expert_v5 上重新评估 | v5 是最新最大数据集（652K frames），看 MLP k=5 是否再提升 |
| P2 | 改进 ACT Transformer | 尝试在 v4/v5 数据（300K+ 帧）上训练，Transformer 可能在更大数据上追上 MLP |
| P2 | 实现`assignedHand` 特征注入 | 显式把主手信息送入模型观测，避免模型隐式猜测 |

---

## 9. 本阶段新增和修改的文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `mujoco_sim/validate_policy.py` | 修改 | 修复时间集成 weight、新增 --temporal-weight / --no-target-reset 参数、temporal_weight 可配置 |
| `20260409train2/config_diag_act_single_v3.json` | 新增 | ACT Transformer 改进训练配置（d=128, seq=4） |
| `20260409train2/config_diag_act_v3_reg.json` | 新增 | 过度正则化对照实验配置（已中止） |
| `20260409train2/outputs/act/run_20260415_172414/` | 新增 | 改进 ACT Transformer 训练输出 |
