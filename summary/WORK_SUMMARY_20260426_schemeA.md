# 20260426 Scheme A 工作总结 — 复用 v9 P1 + data1-4 + v9 Phase2 recipe

## 1. 本阶段总结

本次按用户指定执行 **方案 A**：复用 v9 Phase1 checkpoint，不重新训练 P1；Phase2 使用 `MJ + VR1-4/data1-4`，并完全恢复 v9 Phase2 recipe（`batch_size=256`、`lr=5e-5`、`patience=120`、`weighted_source={0:1.5,1:1.0}`）。

核心结果：

| 模型 | 数据/方案 | 严格评估 SR | worst seed |
|---|---|---:|---:|
| v9 P1 | MJ-only | 230/250 = **92.00%** | 84.0% |
| v9 P2 | MJ + VR1-3 | 247/250 = **98.80%** | 96.0% |
| Scheme A | v9 P1 + MJ + VR1-4, batch256 | 238/250 = **95.20%** | 94.0% |

结论：Scheme A 恢复到 v2 baseline 水平（95.20%），并显著好于 20260426 batch512 结果（93.60%），但仍低于 v9 当前最优（98.80%）。因此 `batch512/弱 P1` 确实是上一轮退化的重要因素，但 `VR4` 在当前同域混合方式下没有带来超过 v9 的增益。

---

## 2. 问题诊断

### 2.1 原问题

上一轮 `20260426_train` 的 `data1-4 + batch512` 结果：

- P1 MJ-only：202/250 = 80.80%
- P2 MJ+data1-4：234/250 = 93.60%
- 低于 v9 P2：98.80%，差 -5.20pp

这个结果存在两个主要混杂因素：

1. P1 重新训练且显著弱于 v9 P1；
2. Phase2 batch 从 v9 的 256 改为 512。

### 2.2 Scheme A 隔离变量

Scheme A 固定：

- P1 checkpoint：`20260425_train_v9/outputs/act_chunk/phase1_run_20260425_193808/checkpoints/best.pt`
- Phase2 recipe：完全恢复 v9 配方
- 唯一核心数据变化：VR1-3 → VR1-4

因此它主要回答：**在干净 v9 recipe 下，加入 VR4 是否优于 v9。**

---

## 3. 实施方法

### 3.1 新目录

创建：`20260426_train_v2/`

关键文件：

| 文件 | 作用 |
|---|---|
| `config_phase2_mj_vr1_4_batch256_v9recipe.json` | Scheme A 主训练配置 |
| `config_smoke_phase2.json` | 本地 smoke test 配置 |
| `eval_ablation.py` | 3-arm strict eval：v9 P1 / v9 P2 / Scheme A |
| `validate_eval_artifacts.py` | strict eval artifact 验证 |
| `run_local_visualize.py` | 本地 MuJoCo 可视化入口 |
| `train_act_chunk.py` | wrapper，供 `validate_policy.py` 动态加载 v9 架构类 |

### 3.2 本地 smoke test

本地 CPU smoke test 通过：

- 数据成功加载：4060 segments / 379052 frames / 813 episodes
- 训练样本：340839
- 验证样本：38213
- domain histogram：`{0: 268209, 1: 72630}`
- 成功继承 v9 P1 normalizer
- 成功加载 pretrained：`Loaded 39/39 pretrained params`

### 3.3 HPC job

| 项 | 值 |
|---|---|
| Train job | `1721745` |
| Eval retry job | `1722046` |
| Train elapsed | 00:25:14 |
| Eval elapsed | 00:03:37 |
| Train node | g0153 |

第一次 eval job `1721747` 因 `validate_policy.py` 输出 JSON 字段名假设错误失败；已修复为解析 `Overall: x/y` 输出行，并只重提 eval，不重复训练。

---

## 4. 实验结果汇总

### 4.1 训练结果

Scheme A run：`20260426_train_v2/outputs/act_chunk/phase2_schemeA_run_20260426_142318/`

| 指标 | 数值 |
|---|---:|
| best_epoch | 3 |
| best_val_mse | 0.1436836520 |
| history_len | 123 |
| params | 1,806,697 |

Top-k checkpoint：

| rank | epoch | val_mse |
|---:|---:|---:|
| 1 | 3 | 0.143684 |
| 2 | 2 | 0.145154 |
| 3 | 1 | 0.152327 |

### 4.2 数据

| Source | segments | source_id | outcomes |
|---|---:|---:|---|
| mujoco_expert_v4 | 2500 | 0 | success 2058 / timeout 442 |
| vr1_aligned | 200 | 1 | success 189 / timeout 11 |
| vr2_aligned | 250 | 1 | success 230 / timeout 20 |
| vr3_aligned | 505 | 1 | success 505 |
| vr4_aligned | 605 | 1 | success 604 / timeout 1 |

合计：4060 segments / 379052 frames / 813 episodes。

### 4.3 严格评估

评估标准：

- seeds = `[42, 7, 123, 2024, 31415]`
- 每 seed = 10 trials × 5 targets = 50 targets
- 每 arm = 250 targets
- ACT-Chunk 使用 `--no-ensemble`

Per-seed：

| 模型 | seed 42 | seed 7 | seed 123 | seed 2024 | seed 31415 | 聚合 |
|---|---:|---:|---:|---:|---:|---:|
| v9 P1 MJ-only | 49/50 | 46/50 | 46/50 | 42/50 | 47/50 | 230/250 = 92.00% |
| v9 P2 MJ+VR1-3 | 48/50 | 50/50 | 50/50 | 50/50 | 49/50 | 247/250 = 98.80% |
| Scheme A MJ+VR1-4 | 50/50 | 47/50 | 47/50 | 47/50 | 47/50 | 238/250 = 95.20% |

对比：

| 对比 | 差值 |
|---|---:|
| Scheme A vs v9 P1 | +3.20pp |
| Scheme A vs v2 baseline 95.20% | +0.00pp |
| Scheme A vs v9 P2 | -3.60pp |
| Scheme A vs 20260426 batch512 data1-4 | +1.60pp |

### 4.4 本地 MuJoCo 可视化

已在本地 Windows 运行，不是在超算上：

- checkpoint：`20260426_train_v2/outputs/act_chunk/phase2_schemeA_run_20260426_142318/checkpoints/best.pt`
- seed：42
- trials：5
- targets：25
- `--visualize`
- `--no-ensemble`

输出：

```text
Overall: 25/25 (100.0%)
```

注意：本地 25-target 可视化只作为定性观察；最终数字仍以 250-target strict eval 为准。

---

## 5. 阶段结论与下一步

### 5.1 已确认结论

1. `batch512/弱 P1` 是上一轮退化的重要因素：
   - 20260426 batch512：93.60%
   - Scheme A batch256 + v9 P1：95.20%
   - 提升 +1.60pp

2. `data1-4` 在干净 v9 recipe 下能达到 v2 baseline，但不能超过 v9：
   - Scheme A：95.20%
   - v9：98.80%
   - 仍低 -3.60pp

3. VR4 不是完全不可用，但当前把 VR1-4 全部合并为同一个 `source_id=1` 的方式会稀释 v9 中有效的 VR1-3 信号。

### 5.2 推荐下一步

P0：跑 **方案 B**。

- 复用 v9 P1；
- MJ：`source_id=0`；
- VR1-3：`source_id=1`；
- VR4：`source_id=2`；
- `num_domains=3`；
- source ratio 建议：`{0:1.5, 1:1.2, 2:0.3}`；
- batch 保持 256；
- 目的：保留 VR4，但把 VR4 降权并用独立 FiLM domain 吸收偏差。

P1：如方案 B 仍不超过 v9，再做 VR4 过滤/重加权：

- 过滤过短 segment；
- 降权 easy success；
- 提高 timeout/长轨迹/边缘目标权重。

P2：暂不建议继续盲目采更多同分布数据；如果采集，优先采 recovery/correction 和边缘目标覆盖，而不是继续 easy success。
