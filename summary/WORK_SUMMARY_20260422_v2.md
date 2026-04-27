# 工作总结 v2 — 20260422 VR微调训练实验

## 目标

根据用户需求：
1. VR数据加入后模型在仿真中达到更高精度（目标 ≥90%）
2. 对ACT架构有创新性改进

本阶段重点验证目标（1）的可行性，为目标（2）提供实验依据。

## 实验设置

| 项 | 值 |
|----|----|
| 代码目录 | `20260422_train/` |
| 评估 | `mujoco_sim/validate_policy.py`，50 targets (10 trials × 5)，seed=42，`--no-ensemble` |
| 模型 | ACT-Chunk MLP，117K 参数，chunk_size=5，obs_dim=31，act_dim=8 |
| VR数据 | `data_collector/collector10/`，40 episodes，16466 frames，人类成功率 94.5% |
| 弱预训练数据 | `data_collector/mujoco_expert_v4_weak/`（v4的15%随机子集，seed=42），75 ep，45036 frames |
| 强预训练模型（参考） | `20260409train2/outputs/act_mlp_k5/act_chunk/run_20260412_174749/checkpoints/best.pt` |

## 关键代码改动

在 [20260422_train/train_act_chunk.py](20260422_train/train_act_chunk.py) 中新增：
- `--inherit-normalizer` 标志：加载 pretrain checkpoint 中保存的 `Normalizer`，跳过对当前数据集的重新 fit。跨域微调必需（据 `/memories/repo/vr-finetune-results.md` 记录：无此项 VR ft 会降至 0%）。
- `--freeze-backbone` 标志：冻结 MLP backbone，仅训练 action/success heads。

实现细节：pretrain checkpoint 在 dataset 构建前先行加载，以便在 `fit_normalizer()` 之前注入继承的 normalizer。

## 实验结果（全部 50 targets）

| # | 配置 | 预训练 | 微调数据 | MuJoCo 成功率 | Δ |
|---|------|--------|---------|---------------|---|
| 1 | 弱 MuJoCo 预训练 | — | — | **38/50 = 76%** | baseline |
| 2 | 强 MuJoCo 预训练（参考上限） | — | — | **47/50 = 94%** | — |
| 3 | +VR 混合微调（继承 norm, lr=3e-5） | 弱 MJ | VR+弱MJ | 38/50 = 76% | 0 |
| 4 | +VR 纯微调（继承 norm, lr=5e-5） | 弱 MJ | VR only | 36/50 = 72% | −4% |
| 5 | 混合 scratch（无预训练） | — | VR+弱MJ | 35/50 = 70% | −6% |
| 6 | 强预训练 + VR 纯微调（继承 norm） | 强 MJ | VR only | 44/50 = 88% | −6% |

对应 run 目录：
- 弱预训练：`outputs/act_chunk/run_20260422_171750/`
- VR 混合 ft：`outputs/act_chunk/run_20260422_174012/`
- VR 纯 ft：`outputs/act_chunk/run_20260422_174535/`
- 混合 scratch：`outputs/act_chunk/run_20260422_174919/`
- 强预训练 + VR ft：`outputs/act_chunk/run_20260422_175439/`

## 核心发现：VR数据对 MuJoCo eval 造成一致的 4–6% 回退

在四种引入 VR 的不同训练策略下（混合 ft / 纯 ft / 从 scratch 混合 / 强预训练 + ft），MuJoCo 成功率**始终等于或低于对应的纯 MuJoCo baseline**。没有一种配置让 VR 数据产生正增益。

## 根因分析

检查 [data_collector/collector10](data_collector/collector10/) 原始帧 vs [data_collector/mujoco_expert_v4_weak](data_collector/mujoco_expert_v4_weak/)：

1. **观测分布差异**：
   - VR obs 中的 `g1JointPositions` 来自前端 `RobotExpressive` 模型 → G1 关节的映射；
   - MuJoCo obs 中的 `g1JointPositions` 来自 G1 native 物理仿真；
   - 二者 mean/std 不同 → 即使继承 normalizer，潜在映射关系仍有 gap。

2. **动作空间差异**：
   - VR 数据：`jointTargetQuat`（RobotExpressive 的四元数关节目标）→ 通过 kinematic 解算生成 `g1JointDelta`；
   - MuJoCo 数据：策略直接输出 `g1JointDelta` 并被 PD 控制器执行；
   - 同一 obs 下两数据源的 "正确动作" 不一致。

3. **端执行器位置**：
   - VR：`RobotExpressive` avatar 的手腕位置；
   - MuJoCo：G1 机械臂末端位置；
   - 即使关节角完全相同，EE 位置也会有 offset。

这三重 domain gap 共同作用，使得拟合 VR 数据必然将模型从 MuJoCo optimum 拉离。

## 对论文叙事的影响

**不能照原计划讲 "VR → ≥90%" 的故事。** 当前 pipeline 下 VR 数据是 MuJoCo eval 的阻碍而非助力。

## 下一步可能方向（按投入产出排序）

### 方向 A — 架构创新桥接 domain gap（推荐作为论文核心贡献）

在 ACT 基础上引入 domain-adaptive 模块，例如：
- **Domain token + FiLM 调制**：给 obs 打 domain label（VR / MuJoCo），backbone 共享，action head 通过 FiLM 进行 per-domain 调制；inference 时强制使用 MuJoCo domain。
- **LoRA adapter**：MuJoCo backbone 冻结，VR 插 LoRA；inference 时可选启用以产生跨域鲁棒性，也可关闭保持 pure-MJ 行为。
- **Adversarial domain 对齐**：加 gradient-reverse domain classifier，让特征对 domain 不可辨。

若上述方法让 "弱MJ + VR" ≥ "弱MJ only"，即可论证 VR 数据价值。

### 方向 B — 改进 VR→G1 kinematics 映射

在数据采集端统一 VR 与 MuJoCo 的关节角/EE 几何，使 `g1JointPositions` 分布一致。工程量大，但可能根治 gap。

### 方向 C — 把故事重心从仿真改到部署

承认仿真 ≠ 真机，改以真机定量（例如机械臂实机 reach 成功率）作为 VR 价值的证据。脱离本 pipeline 的评估体系，需要硬件。

## 待办

- [ ] 选定方向 A / B / C 中的一项；若是 A，起草 domain-FiLM 模块 PR 并加入 `20260422_train/`
- [ ] 如果继续方向 A，定义 "VR 有价值" 的定量标准：弱MJ + VR (domain-FiLM) ≥ 弱MJ only + N%（建议 N ≥ 10）
- [ ] 更新 `/memories/repo/vr-finetune-results.md`（已完成）

## 附注

- 本次所有 eval 均遵循 SKILL：50 targets（10 trials × 5），seed=42，`--no-ensemble`。
- 评估脚本的 pipeline 自动检测需要 `train_act.py` / `train_bc.py` 存在，已在 `20260422_train/` 放空 stub。
- VR 目标生成已在 [frontend/src/components/RobotVR.vue](frontend/src/components/RobotVR.vue) 中修复：`TASK_MIN_Z=0.10, TASK_MAX_Z=0.32`，目标现在生成在机器人正前方（robot-local +Z）而非身后。
