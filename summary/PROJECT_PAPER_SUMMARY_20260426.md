# VR Robot Control 项目论文级总总结（截至 2026-04-26）

> 本文档用于论文写作素材整理，覆盖项目从早期 VR 数据接入、MuJoCo 评估修复、数据对齐、ACT/ACT-Chunk 架构迭代，到最终 v9 与 VR4 filtering 的全部关键工作。文中“可引用成功率”均来自 MuJoCo 严格闭环评估，至少 50 targets；最终主结论优先采用 5 seeds × 50 targets = 250 targets。

---

## 0. 一句话总览

本项目构建了一个 **VR 示教驱动的 Unitree G1 双臂 reaching 模仿学习系统**，完整链路为：

```text
VR 示教采集 → VR/G1 坐标与 FK 对齐 → 多源 imitation learning → ACT-Chunk 策略训练 → MuJoCo 严格闭环评估 → 本地可视化验证
```

最终最优模型为 **ACTChunkGatedCrossAttnFiLM**，使用 `mujoco_expert_v4 + VR1-3`，在严格多 seed MuJoCo 评估中达到：

```text
247/250 = 98.80% success rate, worst seed = 96.0%
```

相对于同架构 MJ-only baseline：

```text
230/250 = 92.00% → 247/250 = 98.80%，提升 +6.80pp
```

相对于 v2 MLP-FiLM 基线：

```text
238/250 = 95.20% → 247/250 = 98.80%，提升 +3.60pp
```

最重要的方法结论是：

1. **MuJoCo 预训练 + VR finetune 是有效的**，但 VR 数据必须对齐、筛选并控制分布；
2. **直接增加数据不一定提升性能**，VR4 直接加入会退化，过滤后恢复到 97.60%，但仍低于 v9；
3. **保留稳定 MLP-FiLM 主路径，并加入小幅 gated cross-attention residual** 是当前数据规模下最有效的架构方向；
4. **Transformer 替代主路径失败**，说明本任务在 <400K 帧数据下更偏向稳定低容量闭环控制器，而不是高容量序列模型。

---

## 1. 任务定义与系统设置

### 1.1 机器人与任务

| 项 | 内容 |
|---|---|
| 机器人 | Unitree G1 双臂人形机器人 |
| 任务 | 单手 reaching，触碰 MuJoCo 中随机目标球 |
| 控制对象 | 双臂 8 个关节 |
| 评估环境 | MuJoCo G1 仿真 |
| 控制频率 | 30 Hz |
| 成功判定 | 手掌中心到目标球中心距离低于阈值并保持 |
| 每 episode 目标数 | 5 targets |

任务的难点不在单个目标，而在闭环连续触达：策略需要根据当前关节、末端、目标相对位置持续输出稳定动作，避免抖动、过冲、陷入局部姿态或跨体不可达动作。

### 1.2 Observation / Action 空间

最终稳定使用的 observation 为 **31D**：

| 部分 | 维度 | 说明 |
|---|---:|---|
| joint positions | 8 | 左右肩 pitch/roll/yaw + elbow |
| joint velocities | 8 | 同 8 关节速度 |
| left EE position | 3 | 左末端在机器人局部坐标 |
| right EE position | 3 | 右末端在机器人局部坐标 |
| target rel base | 3 | 目标相对机器人基座 |
| target rel left | 3 | 目标相对左末端 |
| target rel right | 3 | 目标相对右末端 |
| 总计 | 31 | 机器人 proprioception + task geometry |

Action 为 **8D joint position delta**：

```text
[left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
 right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow]
```

最终推荐 `chunk_size=5`，即模型一次预测未来 5 步动作，但推理时不使用 temporal ensemble（本项目多次验证 temporal ensembling 有害）。

---

## 2. 数据体系

### 2.1 MuJoCo 专家数据

核心专家数据集为 `mujoco_expert_v4`：

| 指标 | 数值 |
|---|---:|
| segments | 2500 |
| frames | 300,716 |
| episodes | 500 |
| success segments | 2058 |
| timeout segments | 442 |

该数据集提供稳定的 MuJoCo 状态覆盖，是所有高性能模型的基础。历史上多次验证：**VR-only 无法替代 MJ 数据**，VR-only 在 MuJoCo 评估中约 24%–27% 或显著低于 mixed training。

### 2.2 VR 数据版本

| 数据 | segments | frames | episodes | 主要特征 |
|---|---:|---:|---:|---|
| VR1 | 200 | 16,466 | 40 | 早期对齐 VR 数据，有 timeout/困难样本 |
| VR2 | 250 | 23,659 | 50 | 更多中长轨迹，timeout 较多 |
| VR3 | 505 | 18,954 | 101 | 短一些，但与 VR1-2 合用有效 |
| VR4 | 605 | 19,257 | 122 | 几乎全 success，短段多，加入后退化 |

关键数据组合：

| 组合 | segments | frames | episodes | 结果倾向 |
|---|---:|---:|---:|---|
| MJ only | 2500 | 300,716 | 500 | 强基础，但缺少 VR 示教分布 |
| MJ + VR1-3 | 3455 | 359,795 | 691 | 当前最优组合 |
| MJ + VR1-4 | 4060 | 379,052 | 813 | 数据更多但性能下降 |

### 2.3 数据质量经验

本项目的核心数据教训：

> **数据量不是单调收益；VR 数据必须提供闭环鲁棒性所需的状态覆盖。**

有效 VR 数据往往包含：

- 更长的 reaching 轨迹；
- 接近边界目标；
- recovery/correction 过程；
- timeout 或 near-timeout 信号；
- 与 MuJoCo 可达目标分布一致的几何。

有害或弱增益 VR 数据往往是：

- 过短 easy-success snippet；
- 只记录最终接近目标的一小段；
- 几乎没有失败/纠错；
- 动作幅度偏大或 jerk 较多；
- 与 MuJoCo 目标采样分布不匹配。

---

## 3. 坐标系、数据对齐与评估环境修复

### 3.1 早期失败：直接混合 raw VR 有害

早期实验显示，将原始 VR 数据直接混入 MuJoCo 训练会退化：

| 设置 | MuJoCo 50-target |
|---|---:|
| weak MJ baseline | 76% |
| weak MJ + raw VR | 70% |
| strong MJ pretrain | 94% |
| strong MJ → raw VR finetune | 88% |

根因不是“VR 没用”，而是 VR 数据与 MuJoCo 训练数据在几何和动作统计上不一致。

### 3.2 坐标系统一

项目采用的关键坐标变换：

```text
VR Y-up → MuJoCo Z-up: (x, y, z) → (x, -z, y)
MuJoCo Z-up → VR Y-up: (x, y, z) → (x, z, -y)
```

机器人局部坐标使用 `_R_BASE` 旋转，使 VR 记录和 MuJoCo evaluator 构造 observation 的方式一致。

### 3.3 FK 对齐后处理

核心方法：对 VR JSONL 中每帧的 `g1JointPositions`，用 MuJoCo G1 模型做 forward kinematics，重新计算左右末端位置和 target-relative features。

作用：

- 不改 VR 前端；
- 旧数据可以离线重对齐；
- 末端位置与 G1 模型结构一致；
- 让 VR 数据进入与 MuJoCo expert 相同的 observation manifold。

早期定量结果：

| 数据处理 | MuJoCo 50-target |
|---|---:|
| 无 VR | 76% |
| raw VR 混合 | 70% |
| aligned VR 混合 | 80% |
| aligned VR + FiLM | 86% |

### 3.4 MuJoCo 目标采样修复

早期可视化发现 evaluator 会采到身后或跨体不可达目标。修复后加入约束：

- 目标必须在肩膀前方；
- 左手目标保持在左侧，右手目标保持在右侧；
- 不允许明显低于肩膀太多；
- 目标距离必须在 reachable sphere 内。

这一步非常关键，因为它让评估结果更接近“人类/机器人合理 reaching”任务，而不是把不可达目标当作策略失败。

---

## 4. 模型架构演进

### 4.1 BC / 单步 MLP 阶段

最早使用普通 BC MLP 和单步 action 预测，优点是稳定、简单；缺点是闭环动作容易抖动，难以表达短期动作序列。

### 4.2 ACT / Transformer 阶段

尝试 ACT Transformer 后发现：

- 小数据下 Transformer 容易动作幅度衰减；
- 低验证 MSE 不一定对应高闭环成功率；
- 需要 `action_scale=1.3` 才能部分补偿动作幅度；
- 总体不如 MLP-based ACT-Chunk 稳定。

本项目最终没有采用 full Transformer 作为主控制路径。

### 4.3 ACT-Chunk MLP 阶段

ACT-Chunk 将动作预测从单步变为未来 K 步 chunk，最终 `chunk_size=5` 成为主配置。

优点：

- 比单步 MLP 更稳定；
- 不需要复杂序列 encoder；
- 能提供短期动作一致性；
- 与 30 Hz 控制频率匹配。

### 4.4 ACT-DAFiLM / MLP-FiLM

为解决多源数据中的 MJ/VR 域偏差，引入 domain embedding + FiLM：

```text
h = LayerNorm(Wx)
gamma, beta = Linear(domain_embedding)
h' = GELU((1 + gamma) * h + beta)
```

关键设计：

| 设计 | 作用 |
|---|---|
| FiLM 零初始化 | 初始等价普通 MLP，便于热启动 |
| `domain_id=0` 推理 | 推理固定 MuJoCo 域，避免 VR 偏差进入前向路径 |
| 每层独立 FiLM | 按层调制多源差异 |
| checkpoint remap | 兼容旧 MLP checkpoint |

v2 基线即使用 `ACTChunkMLPFiLM`，在严格评估中达到：

```text
MJ-only: 199/250 = 79.60%
MJ+VR1-3 FT: 238/250 = 95.20%
提升: +15.60pp
```

这证明 **MJ-pretrain + VR-finetune** 是项目核心有效路径。

### 4.5 ACTChunkGatedCrossAttnFiLM

为了增加论文架构创新，同时避免 Transformer 主路径不稳定，最终采用：

```text
稳定 MLP-FiLM 主路径 + 小幅 gated cross-attention residual
```

结构思想：

1. 将 observation 分为 proprioception token 与 task geometry token；
2. 用小型 cross-attention 建模“自身姿态 ↔ 目标几何”的交互；
3. 将 attention 输出通过 `attn_to_obs` 映射回 observation residual；
4. residual gate 初始很小，保证初始行为接近稳定 MLP-FiLM。

关键超参：

| 参数 | 值 |
|---|---:|
| hidden dims | [1024, 1024, 512] |
| chunk size | 5 |
| d_model | 64 |
| nhead | 4 |
| cond_dim | 16 |
| residual_init | 0.01 |
| dropout | 0.1 |
| attn_dropout | 0.05 |
| params | ~1.806M |

该架构在 v9 中达到最终最优 **98.80%**。

---

## 5. 训练协议

### 5.1 两阶段训练

最终有效训练协议：

| 阶段 | 数据 | 目的 |
|---|---|---|
| Phase 1 | `mujoco_expert_v4` | 学稳定 MuJoCo prior 和 reachable workspace |
| Phase 2 | `mujoco_expert_v4 + VR aligned data` | 用 VR 示教补充策略分布，提高闭环鲁棒性 |

关键经验：

- Phase 1 质量很重要；弱 P1 会限制 Phase 2 上限；
- Phase 2 不能只用 VR，必须保留 MJ；
- Phase 2 batch=256 比 batch=512 更稳；
- `--inherit-normalizer` 很重要，保证 Phase 2 归一化不漂移；
- MLP/ACT-Chunk 系列评估必须 `--no-ensemble`。

### 5.2 评估协议

最终论文主表推荐使用：

```text
5 seeds × 10 trials × 5 targets = 250 targets
seeds = [42, 7, 123, 2024, 31415]
```

所有 MLP/ACT-Chunk 模型使用：

```text
--no-ensemble --action-scale 1.0
```

小于 50 targets 的结果只做调试或可视化，不作为论文成功率引用。

---

## 6. 关键实验结果总表

### 6.1 早期 50-target 结果

| 阶段 | 设置 | 结果 | 结论 |
|---|---|---:|---|
| raw VR 混合 | weak MJ + raw VR | 70% | raw VR 有害 |
| FK aligned | weak MJ + aligned VR | 80% | 对齐有效 |
| aligned + FiLM | weak MJ + aligned VR + FiLM | 86% | 域调制有效 |
| strong baseline | strong MJ | 94% | MJ 强 prior 很重要 |
| strong mixed FiLM + 采样修复 | MJ+VR+FiLM | 100% (50/50) | 早期 50-target 天花板 |

这些早期结果用于方法探索，但最终论文主结论应以 250-target 多 seed 结果为主。

### 6.2 v2：验证核心假设

| Arm | 数据 | SR | worst seed |
|---|---|---:|---:|
| v2 P1 | MJ-only | 199/250 = 79.60% | 76.0% |
| v2 P2 | MJ+VR1-3 | 238/250 = 95.20% | 92.0% |

核心结论：

```text
MJ-pretrain + VR-finetune 相比 MJ-only 提升 +15.60pp。
```

这是项目最重要的假设验证实验。

### 6.3 v7/v8/v9：架构迭代

| 模型 | P1 MJ-only | P2 MJ+VR | 结论 |
|---|---:|---:|---|
| v7 Transformer-lite | 182/250 = 72.80% | 134/250 = 53.60% | 替代 MLP 主路径失败 |
| v8 GatedCrossAttn-FiLM safe | 226/250 = 90.40% | 230/250 = 92.00% | attention residual 有潜力，但冻结太保守 |
| v9 GatedCrossAttn-FiLM | 230/250 = 92.00% | **247/250 = 98.80%** | 当前最优 |

v9 Phase 2 seed 明细：

| Seed | Success |
|---:|---:|
| 42 | 48/50 |
| 7 | 50/50 |
| 123 | 50/50 |
| 2024 | 50/50 |
| 31415 | 49/50 |

### 6.4 20260426：数据 scaling 与 VR4 诊断

| 实验 | 设置 | SR | 相对 v9 |
|---|---|---:|---:|
| 20260426 第一次 | 重训 P1 + MJ+VR1-4 + batch512 | 234/250 = 93.60% | -5.20pp |
| Scheme A | v9 P1 + MJ+VR1-4 + batch256 | 238/250 = 95.20% | -3.60pp |
| Scheme B | v9 P1 + VR4 独立 domain/降权 | 235/250 = 94.00% | -4.80pp |
| C1 VR4 len>=40 | v9 P1 + MJ+VR1-3 + filtered VR4 | 242/250 = 96.80% | -2.00pp |
| C2 VR4 quality_v1 | len>=40 + action_l2_mean<=0.06 | 244/250 = 97.60% | -1.20pp |

解释：

- 直接增加 VR4/data4 不提升，反而退化；
- 过滤短段和高动作幅度片段可以恢复一部分性能；
- 但即使最佳 filtering 也未超过 v9；
- 因此当前 VR4 不能作为最终最优数据集的一部分。

---

## 7. 最终最佳模型

| 项 | 内容 |
|---|---|
| 模型 | `ACTChunkGatedCrossAttnFiLM` |
| 数据 | `mujoco_expert_v4 + vr1_aligned + vr2_aligned + vr3_aligned` |
| Phase1 checkpoint | `20260425_train_v9/outputs/act_chunk/phase1_run_20260425_193808/checkpoints/best.pt` |
| Phase2 checkpoint | `20260425_train_v9/outputs/act_chunk/phase2_run_20260425_200159/checkpoints/best.pt` |
| strict SR | **247/250 = 98.80%** |
| worst seed | 96.0% |
| local visualize | 25/25 qualitative |

推荐论文中将 v9 作为最终方法模型，将 v2 作为强 baseline，将 v7/VR4 作为重要 negative/ablation。

---

## 8. 负结果与踩坑总结

这些负结果非常适合写入论文的 ablation 或 appendix，因为它们解释了方法选择的必要性。

### 8.1 Temporal ensembling 有害

对 MLP/ACT-Chunk 系列，temporal ensembling 会降低闭环表现。最终所有可引用评估均使用 `--no-ensemble`。

### 8.2 Transformer 主路径不适合当前数据规模

v7 Transformer-lite：P2 只有 **53.60%**。结论不是 attention 完全没用，而是 **不能用 Transformer 替代稳定 MLP 主控制器**。v9 的成功来自“MLP 主路径 + 小 attention residual”。

### 8.3 过滤 timeout 会伤害性能

历史实验中过滤 timeout / 只保留 success 会显著退化。Timeout 和 near-failure 包含边缘目标与纠错信息，是闭环鲁棒性的重要监督。

### 8.4 Huber loss / target jitter / 过强正则为负增益

历史验证：

- Huber loss 替代 MSE 后闭环变差；
- 对目标相关 observation 加噪会破坏任务几何；
- dropout/weight decay 过强会欠拟合；
- EMA 在本任务上没有带来正增益。

### 8.5 batch512 不是提升显存利用率的正确方向

20260426 第一次 batch512 结果 93.60%，低于 v9 98.80%。恢复 v9 P1 + batch256 后变为 95.20%，说明 batch512/recipe 改动不是无害优化。

### 8.6 local visualize 不能替代 strict eval

多次出现 25-target local visualize 很好，但 250-target 多 seed 评估下降的情况。因此论文中必须区分：

- local visualize：行为定性；
- 50-target：最低可引用；
- 250-target multi-seed：最终主结论。

---

## 9. 最终论文叙事建议

### 9.1 推荐标题方向

可围绕以下关键词组织：

- VR-guided imitation learning for humanoid reaching
- Domain-adaptive ACT-Chunk policy
- MuJoCo pretraining with VR finetuning
- Data quality over data quantity in VR demonstrations
- Gated proprioception-task cross-attention residual

### 9.2 核心 story arc

1. **问题**：VR 示教有真实人类控制风格，但直接混入 MuJoCo imitation learning 会因为坐标/分布差异导致退化。
2. **数据对齐**：通过 G1 FK 和统一机器人局部坐标，将 VR demonstration 投影到与 MuJoCo expert 一致的 observation manifold。
3. **多源学习**：引入 domain-adaptive FiLM，使 MJ 和 VR 在训练中共享主干但保留域条件调制，推理固定 MuJoCo 域。
4. **架构增强**：在稳定 MLP-FiLM 主路径上加入 gated cross-attention residual，建模 proprioception 与 task geometry 的交互。
5. **严格验证**：250-target multi-seed MuJoCo 评估证明 v9 达 98.80%，超过 v2 95.20%。
6. **数据质量发现**：更多 VR 数据不一定更好；VR4 直接加入退化，过滤后仍未超过 v9，说明数据分布和质量是关键。

### 9.3 可写成论文贡献的点

| 贡献 | 证据 |
|---|---|
| VR→G1 FK 对齐 pipeline | raw VR 混合 70%，aligned VR 80% |
| Domain-adaptive FiLM for multi-source imitation | aligned+FiLM 86%；v2 P2 95.20% |
| MJ pretrain + VR finetune 显著优于 MJ-only | v2 +15.60pp；v9 +6.80pp |
| Gated cross-attention residual architecture | v9 98.80%，超过 v2 95.20% |
| 数据质量分析 | VR4 direct 95.20%，filtered 97.60%，仍低于 v9 98.80% |

---

## 10. 推荐论文实验表

### 10.1 主结果表

| Method | Data | Architecture | Targets | SR |
|---|---|---|---:|---:|
| MJ-only baseline | MJ | ACTChunkMLPFiLM | 250 | 79.60% |
| MJ+VR FT baseline | MJ+VR1-3 | ACTChunkMLPFiLM | 250 | 95.20% |
| Transformer-lite | MJ+VR | ACTChunkTransformerLite | 250 | 53.60% |
| GatedCrossAttn safe | MJ+VR | Frozen GatedCrossAttn-FiLM | 250 | 92.00% |
| **Ours final** | MJ+VR1-3 | GatedCrossAttn-FiLM | 250 | **98.80%** |

### 10.2 数据 scaling / quality 表

| Data variant | Treatment | Targets | SR |
|---|---|---:|---:|
| MJ+VR1-3 | final v9 | 250 | **98.80%** |
| MJ+VR1-4 | direct append | 250 | 95.20% |
| MJ+VR1-4 | VR4 separate domain low weight | 250 | 94.00% |
| MJ+VR1-3 + VR4 len>=40 | filtered | 250 | 96.80% |
| MJ+VR1-3 + VR4 quality_v1 | filtered | 250 | 97.60% |

### 10.3 Ablation 解释

推荐论文正文强调：

- v2 表明 VR finetune 的必要性；
- v7 表明 full Transformer 替代主路径不可行；
- v9 表明 residual attention 可以作为安全增益；
- VR4 表明数据质量比数据数量重要。

---

## 11. 目前最可靠的结论

1. **MJ 数据是基础**：没有 MJ pretrain，VR-only 很难在 MuJoCo 上泛化。
2. **VR finetune 有显著价值**：v2 从 79.60% 到 95.20%，v9 从 92.00% 到 98.80%。
3. **数据对齐是前提**：raw VR 会退化，对齐后才有正增益。
4. **FiLM 是有效的域隔离机制**：能让 VR 梯度参与训练，同时推理保持 MuJoCo 域路径。
5. **Gated cross-attention residual 是当前最优架构创新**：不是替代 MLP，而是在 MLP 上增加可控 residual。
6. **更多数据不自动等于更好**：VR4 证明数据质量和分布更关键。
7. **严格评估必须多 seed**：小样本可视化结果不能作为论文主指标。

---

## 12. 后续数据采集建议

不建议继续采“同 VR4 分布”的短 easy-success 数据。下一批 VR 数据建议：

| 优先级 | 采集内容 | 原因 |
|---|---|---|
| P0 | recovery/correction 轨迹 | 提供闭环纠错监督 |
| P0 | 边界目标 reaching | 提升 worst seed 鲁棒性 |
| P0 | 长段轨迹，segment length ≥40 | 避免 easy snippet 污染 |
| P1 | near-timeout / timeout 保留 | 学失败边界和可达边界 |
| P1 | 控制动作幅度，过滤高 jerk | 避免动作风格污染 |
| P2 | 重复 v9 数据分布采样 | 用于复现实验稳定性 |

质量门槛建议：

```text
segment_len >= 40
mean(action_l2) <= 0.06 作为初始筛选线
必须保留一定比例 near-failure / recovery 样本
每批新数据先做 data-only 统计，再进入 HPC 训练
```

---

## 13. 最终产物清单

| 类型 | 路径 / 内容 |
|---|---|
| 最优 checkpoint | `20260425_train_v9/outputs/act_chunk/phase2_run_20260425_200159/checkpoints/best.pt` |
| v9 严格评估 | `20260425_train_v9/eval_ablation.json` |
| v9 本地可视化 | `20260425_train_v9/outputs/act_chunk/phase2_run_20260425_200159/visual_eval_v9_seed42_5trials.json` |
| v2 核心 baseline | `20260424_train_v2/` |
| VR4 filtering 实验 | `20260426_train_v4/` |
| MuJoCo evaluator | `mujoco_sim/validate_policy.py` |
| 数据采集目录 | `data_collector/` |
| VR 前端 | `frontend/src/components/RobotVR.vue` |
| 论文前期总结 | `summary/PAPER_SUMMARY_20260423.md` |
| v9 总结 | `summary/WORK_SUMMARY_20260425_v7_to_v9.md` |
| 20260426 退化分析 | `summary/WORK_SUMMARY_20260426_schemeB_and_drop_analysis.md` |
| VR4 filtering 总结 | `summary/WORK_SUMMARY_20260426_vr4_filtering.md` |

---

## 14. 可直接写进论文的精炼结论

> We found that VR demonstrations can significantly improve a MuJoCo-pretrained humanoid reaching policy only when the demonstrations are geometrically aligned and distributionally controlled. A two-stage MJ-pretrain + VR-finetune ACT-Chunk policy improved from 79.60% to 95.20% in strict multi-seed MuJoCo evaluation. Replacing the stable MLP-FiLM controller with a Transformer-lite backbone failed, but adding a small gated cross-attention residual on top of the MLP-FiLM path achieved the best result, 98.80% over 250 targets. Finally, adding more VR data did not monotonically improve performance: a fourth VR batch degraded performance when appended directly, and filtering recovered part of the loss but still did not surpass the VR1-3 model. This indicates that VR demonstration quality and coverage, especially recovery and boundary-reaching behaviors, are more important than raw frame count.

中文对应：

> 本项目验证了：VR 示教数据能显著提升 MuJoCo 预训练的 G1 reaching 策略，但前提是示教数据经过几何对齐，并且数据分布能提供真实闭环鲁棒性。两阶段 MJ 预训练 + VR finetune 的 ACT-Chunk 策略从 79.60% 提升到 95.20%；在保留 MLP-FiLM 稳定主路径的基础上加入 gated cross-attention residual 后，最终达到 98.80%。同时，VR4 数据实验表明，更多 VR 帧数不必然提升性能，数据质量、长轨迹、纠错过程和边界样本比 raw quantity 更关键。
