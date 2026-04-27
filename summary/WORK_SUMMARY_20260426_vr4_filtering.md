# 20260426 VR4 Filtering 实验总结

## 1. 本阶段总结

本阶段按用户要求完成了 **VR4 filtering → 重新训练 → 严格评估 → 本地 MuJoCo 可视化** 的完整闭环。核心问题是：VR4 直接加入 `MJ+VR1-4` 后会使 v9 从 98.80% 下降，因此需要判断 **VR4 是否经过质量过滤后还能成为有效增量数据**。

最终结论：

- VR4 过滤后确实比直接追加更好；最佳过滤方案 `vr4_quality_v1` 达到 **244/250 = 97.60%**。
- 但它仍低于当前最优 v9 的 **247/250 = 98.80%**，差 **-1.20pp**。
- 因此，**当前 VR4 不能纳入最终最优训练 recipe**；过滤说明 VR4 内部有一部分有效片段，但继续采集“同类型 easy-success/短段数据”不值得。
- 后续如果要继续采集，应采 **更长轨迹、边缘目标、失败/near-timeout、recovery/correction**，而不是继续增加同分布 VR4 数据。

---

## 2. 问题诊断

### 2.1 已知退化现象

| 实验 | 配置 | 严格 SR | worst seed |
|---|---|---:|---:|
| v9 | v9 P1 + MJ+VR1-3 | **247/250 = 98.80%** | 96.0% |
| 20260426 第一次 | 重训 P1 + MJ+VR1-4 + batch512 | 234/250 = 93.60% | 88.0% |
| Scheme A | v9 P1 + MJ+VR1-4 + batch256 | 238/250 = 95.20% | 94.0% |
| Scheme B | v9 P1 + MJ+VR1-4，VR4 独立 domain/降权 | 235/250 = 94.00% | 82.0% |

诊断结论：

1. 第一次训练下降的一部分来自弱 P1 / batch512 recipe：从 93.60% 恢复到 Scheme A 的 95.20%。
2. 但即使恢复 v9 P1 + batch256，加入 VR4 仍低于 v9：95.20% vs 98.80%。
3. 将 VR4 单独设 domain 并降权也没有恢复性能：Scheme B 仅 94.00%。

### 2.2 VR4 数据特征

VR4 原始数据：

| 指标 | 数值 |
|---|---:|
| segments | 605 |
| frames | 19,257 |
| episodes | 122 |
| success | 604 |
| timeout | 1 |
| segment length median | 25 frames |
| segment length p75 | 32 frames |
| segment length p90 | 54.2 frames |
| mean action L2 median | 0.0463 |
| mean action L2 p75 | 0.0606 |
| mean action L2 p90 | 0.0749 |

主要风险：

- 短段过多，容易变成 easy-success snippet；
- 几乎全 success，缺少 timeout / near-failure / recovery 信号；
- 动作幅度整体高于早期 VR 数据，可能改变闭环控制风格；
- 局部看似成功，但会削弱多 seed 评估下的鲁棒性。

---

## 3. 解决方法

### 3.1 过滤策略

本次没有修改共享 loader，而是将 VR4 materialize 成两个 filtered JSONL 数据集，确保实验可复现、可回溯。

| 过滤集 | 规则 | 保留 segments | 保留 frames | outcome |
|---|---|---:|---:|---|
| `vr4_len40` | `segment_len >= 40` | 104/605 | 7,590/19,257 | 103 success + 1 timeout |
| `vr4_quality_v1` | `segment_len >= 40 && mean(action_l2) <= 0.06` | 84/605 | 5,486/19,257 | 84 success |

对应文件：

| 文件 | 作用 |
|---|---|
| `20260426_train_v4/filter_vr4_quality.py` | 从 VR4 原始 JSONL 生成过滤数据集 |
| `20260426_train_v4/filtered_data/vr4_len40/` | C1 过滤数据 |
| `20260426_train_v4/filtered_data/vr4_quality_v1/` | C2 过滤数据 |
| `20260426_train_v4/filter_report.json` | 过滤统计报告 |

### 3.2 训练配置

两个 arm 均使用相同主 recipe：

- 架构：`ACTChunkGatedCrossAttnFiLM`
- 预训练：v9 Phase1 `MJ-only` checkpoint
- Phase2 数据：`MJ + VR1-3 + filtered VR4`
- VR4 domain：并入 VR domain/source，不单独开 domain
- `batch_size=256`
- `lr=5e-5`
- `chunk_size=5`
- `--inherit-normalizer`
- `--no-ensemble` 评估

### 3.3 工程修复

HPC 首次 eval 作业失败，原因是 `validate_policy.py` 原本只通过 `train_bc.py` / `train_act.py` 向上查找训练目录；而 `20260426_train_v4/` 只 re-export 了 `train_act_chunk.py`，导致 checkpoint 加载时找不到 `train_act_chunk` 模块。

已修复：

- 修改 `mujoco_sim/validate_policy.py` 的 pipeline root 检测逻辑；
- 现在当父目录含 `train_act_chunk.py` 时，也会加入 `sys.path`；
- 本地 import smoke：`5/5` 成功；
- 重新提交 eval-only job 后 strict eval 完成。

---

## 4. 实验结果汇总

### 4.1 本地 smoke test

本地 smoke 使用 `config_smoke_filter_quality.json`：

| 检查项 | 结果 |
|---|---|
| 数据加载 | `3539 segments, 365281 frames, 731 episodes` |
| train/val samples | `328255 / 37026` |
| domain histogram | `{0: 267983, 1: 60272}` |
| pretrained loading | `Loaded 39/39 pretrained params` |
| 2-epoch smoke | 正常完成 |

### 4.2 HPC 训练

| job | id | 状态 | 耗时 |
|---|---:|---|---:|
| train | 1722490 | COMPLETED | 00:48:01 |
| eval 第一次 | 1722491 | FAILED | 00:00:05 |
| eval-only 修复后 | 1723082 | COMPLETED | 00:02:10 |

训练指标：

| Arm | 数据规模 | best epoch | best val MSE | params |
|---|---:|---:|---:|---:|
| C1 `vr4_len40` | 3559 segments / 367385 frames | 3 | 0.139726 | 1,806,697 |
| C2 `vr4_quality_v1` | 3539 segments / 365281 frames | 3 | 0.137748 | 1,806,697 |

### 4.3 严格 MuJoCo 多 seed 评估

评估协议：5 seeds × 10 trials × 5 targets = **250 targets/arm**，seeds = `[42, 7, 123, 2024, 31415]`，全部 `--no-ensemble`。

| Arm | seed=42 | seed=7 | seed=123 | seed=2024 | seed=31415 | 聚合 SR | worst seed |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 `vr4_len40` | 49/50 | 50/50 | 49/50 | 49/50 | 45/50 | **242/250 = 96.80%** | 90.0% |
| C2 `vr4_quality_v1` | 49/50 | 50/50 | 48/50 | 47/50 | 50/50 | **244/250 = 97.60%** | 94.0% |

对比基线：

| 对比对象 | SR | C2 差值 |
|---|---:|---:|
| 20260426 第一次 batch512 | 93.60% | +4.00pp |
| Scheme B VR4 独立 domain/降权 | 94.00% | +3.60pp |
| Scheme A VR4 直接追加 batch256 | 95.20% | +2.40pp |
| v9 MJ+VR1-3 当前最优 | 98.80% | **-1.20pp** |

严格评估已通过 `validate_eval_artifacts.py` 校验：

- 每个 arm `agg_total=250`；
- 每个 seed 都是 50 targets；
- seed 列表完全为 `[42, 7, 123, 2024, 31415]`；
- checkpoint 和 per-seed eval JSON 均存在。

### 4.4 本地 MuJoCo 可视化

本地 Windows 运行，不是在超算上：

| 项 | 值 |
|---|---|
| checkpoint | `20260426_train_v4/outputs/act_chunk/phase2_filter_quality_run_20260426_162522/checkpoints/best.pt` |
| visual command | `--visualize --num-trials 5 --seed 42 --no-ensemble --action-scale 1.0` |
| qualitative result | 24/25 = 96.0% |

注意：本地 25-target 可视化仅用于定性行为检查，不作为最终 SR 引用；最终结论使用 250-target strict eval。

---

## 5. 阶段结论与下一步

### 5.1 已确认结论

1. **过滤有效但不够**：C2 从 Scheme A 的 95.20% 提升到 97.60%，说明 VR4 中有一部分可用片段。
2. **当前 VR4 仍不能进入最终最优 recipe**：C2 仍低于 v9 的 98.80%，差 -1.20pp。
3. **短段确实是污染因素之一**：只保留 `len>=40` 就能达到 96.80%，比 Scheme A +1.60pp。
4. **再过滤动作幅度有额外收益**：`len>=40 && act_l2_mean<=0.06` 提升到 97.60%，比 C1 +0.80pp。
5. **更多同分布 VR4 数据不一定有益**：VR4 的主要缺陷不是数量不足，而是分布偏 easy-success、缺少 recovery/timeout。

### 5.2 VR4 能不能用？

结论分层：

| 用途 | 判断 |
|---|---|
| 作为当前最优主训练数据 | **不能**，因为未超过 v9 98.80% |
| 作为诊断/消融数据 | 可以，已证明 filtering 能恢复一部分性能 |
| 作为继续采集模板 | 不建议照搬，应改变采集分布 |
| 作为论文负结果 | 非常有价值，可说明“数据质量 > 数据数量” |

### 5.3 还要不要增加数据？

建议：**要增加，但不要增加同类型数据**。

下一批数据应满足：

- 每段更长：目标 segment length 尽量 ≥40 frames；
- 包含纠错：故意采集 overshoot 后回拉、接近失败后的 recovery；
- 包含边界目标：左右臂同侧边界、远近边界、较高/较低但可达目标；
- 保留困难样本：不要只保留 success，near-timeout 和 timeout 对闭环鲁棒性有价值；
- 控制动作幅度：避免大量高速 jerk 段，粗略目标 `mean(action_l2) <= 0.06`；
- 采集后先跑质量筛选，再进入训练。

### 5.4 最终推荐 recipe

截至本阶段，最优仍是：

| 项 | 推荐 |
|---|---|
| 最优 checkpoint | `20260425_train_v9/outputs/act_chunk/phase2_run_20260425_200159/checkpoints/best.pt` |
| 最优数据 | `mujoco_expert_v4 + vr1_aligned + vr2_aligned + vr3_aligned` |
| 最优架构 | `ACTChunkGatedCrossAttnFiLM` |
| 最优 strict SR | **247/250 = 98.80%** |
| 是否加入当前 VR4 | 否 |
