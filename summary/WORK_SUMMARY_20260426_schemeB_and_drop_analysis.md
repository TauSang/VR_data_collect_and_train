# 20260426 Scheme B 与第一次训练退化分析

## 1. 本阶段总结

按用户要求继续执行下一步：在 Scheme A 之后跑 **Scheme B**，即复用 v9 Phase1 checkpoint，Phase2 使用 `MJ + VR1-4`，但将 VR4 单独设为 `source_id=2/domain=2` 并降权。

Scheme B 严格评估结果：

| 模型 | 配置 | 严格 SR | worst seed |
|---|---|---:|---:|
| v9 P1 | MJ-only | 230/250 = **92.00%** | 84.0% |
| v9 P2 | MJ + VR1-3 | 247/250 = **98.80%** | 96.0% |
| 20260426 第一次 | 重训 P1 + MJ+VR1-4 + batch512 | 234/250 = **93.60%** | 88.0% |
| Scheme A | v9 P1 + MJ+VR1-4 + batch256，同一 VR domain | 238/250 = **95.20%** | 94.0% |
| Scheme B | v9 P1 + MJ+VR1-4 + batch256，VR4 独立 domain/降权 | 235/250 = **94.00%** | 82.0% |

结论：Scheme B 没有改善 Scheme A，反而下降到 94.00%。这说明 VR4 不是简单通过“独立 domain + 降权”就能转化为增益。今天第一次训练下降的根因基本可以分解为两部分：

1. **训练 recipe/P1 起点问题**：重训 P1 明显弱于 v9 P1，batch512 也比 batch256 recipe 差；这部分从 93.60% → 95.20% 得到验证。
2. **VR4/data4 本身或其混合方式问题**：即使用 v9 P1 + batch256，加入 VR4 后仍从 v9 的 98.80% 降到 95.20%；Scheme B 进一步验证 VR4 低权独立 domain 也没有恢复性能。

---

## 2. Scheme B 实施细节

### 2.1 新目录

创建：`20260426_train_v3/`

核心文件：

| 文件 | 作用 |
|---|---|
| `config_phase2_schemeB_vr4_domain_lowweight.json` | Scheme B 主配置 |
| `config_smoke_phase2.json` | 本地 smoke test 配置 |
| `eval_ablation.py` | Scheme B strict eval |
| `validate_eval_artifacts.py` | strict eval 校验 |
| `run_local_visualize.py` | 本地 MuJoCo 可视化 |

### 2.2 配置变化

Scheme B 相对 Scheme A 的唯一核心变化：

| Source | source_id | sampler mass |
|---|---:|---:|
| MJ | 0 | 1.5 |
| VR1-3 | 1 | 1.2 |
| VR4 | 2 | 0.3 |

有效采样质量大约为：

| 数据 | 有效占比 |
|---|---:|
| MJ | 50% |
| VR1-3 | 40% |
| VR4 | 10% |

模型配置：

- `num_domains=3`
- `default_domain_id=0`
- batch=256
- lr=5e-5
- epochs=600
- patience=120
- 复用 v9 P1 checkpoint
- `--inherit-normalizer`

### 2.3 关键实现修复

由于 v9 P1 checkpoint 的 `domain_emb.weight` 是 2-domain，而 Scheme B 模型是 3-domain，直接加载会导致整块 `domain_emb.weight` 被跳过。这样 MJ/VR1-3 的 domain embedding 也会随机重置，破坏对照实验。

已修改 `20260425_train_v9/train_act_chunk.py` 的 pretrained loading 逻辑：

- 如果 `domain_emb.weight` 维度从 2 扩到 3：
  - 前 2 行继承 checkpoint；
  - 新增 domain 行复制最后一个 pretrained domain embedding；
  - 其余参数正常加载。

本地 smoke test 验证：

```text
Train domain histogram: {0: 268209, 1: 55560, 2: 17070}
Loaded 39/39 pretrained params
```

---

## 3. Scheme B 训练与评估结果

HPC job：

| job | id | 状态 | 耗时 |
|---|---:|---|---:|
| train | 1722070 | COMPLETED | 00:24:31 |
| eval | 1722071 | COMPLETED | 00:01:08 |

训练结果：

| 指标 | 数值 |
|---|---:|
| best_epoch | 2 |
| best_val_mse | 0.1446335994 |
| params | 1,806,713 |
| history_len | 122 |

Top-k：

| rank | epoch | val_mse |
|---:|---:|---:|
| 1 | 2 | 0.144634 |
| 2 | 1 | 0.153128 |

严格评估：

| seed | success |
|---:|---:|
| 42 | 50/50 |
| 7 | 50/50 |
| 123 | 47/50 |
| 2024 | 41/50 |
| 31415 | 47/50 |

聚合：235/250 = **94.00%**，worst seed = 82.0%。

对比：

| 对比 | 差值 |
|---|---:|
| Scheme B vs 20260426 第一次 batch512 | +0.40pp |
| Scheme B vs Scheme A | -1.20pp |
| Scheme B vs v2 baseline | -1.20pp |
| Scheme B vs v9 best | -4.80pp |

### 3.1 本地 MuJoCo 可视化

已在本地 Windows 运行，不是在超算上：

- checkpoint：`20260426_train_v3/outputs/act_chunk/phase2_schemeB_run_20260426_150549/checkpoints/best.pt`
- seed：42
- trials：5
- targets：25
- `--visualize`
- `--no-ensemble`

结果：

```text
Overall: 25/25 (100.0%)
```

注意：本地 25-target 结果只作为定性可视化，不作为最终 SR；最终结论必须使用 250-target strict eval。

---

## 4. 今天第一次训练为什么下降

今天第一次训练指 `20260426_train/`：

- P1：MJ-only 重训
- P2：MJ + VR1-4/data1-4
- Phase2 batch=512
- `samples_per_epoch=681678`
- P2 strict SR：234/250 = 93.60%

相比 v9：

- v9 P2：247/250 = 98.80%
- 下降：-5.20pp

### 4.1 直接证据 1：P1 起点明显变弱

| P1 | best_val_mse | strict SR |
|---|---:|---:|
| v9 P1 | 0.300248 | 230/250 = 92.00% |
| 20260426 第一次 P1 | 0.315063 | 202/250 = 80.80% |

同样是 MJ-only，20260426 第一次 P1 比 v9 P1 低 **11.20pp**。因此第一次 P2 是从更弱的 MuJoCo prior 开始 finetune。

注意：第一次 P2 从它自己的 P1 提升了：

- 20260426 第一次 P1：80.80%
- 20260426 第一次 P2：93.60%
- 提升：+12.80pp

这说明 VR finetune 仍然有效；问题是 P1 起点过低，最终追不上 v9。

### 4.2 直接证据 2：batch512/recipe 不是无害改动

| 实验 | P1 | data | batch | P2 SR | P2 best_val_mse |
|---|---|---|---:|---:|---:|
| 20260426 第一次 | 当天重训弱 P1 | MJ+VR1-4 | 512 | 93.60% | 0.165902 |
| Scheme A | v9 P1 | MJ+VR1-4 | 256 | 95.20% | 0.143684 |

Scheme A 只恢复 v9 P1 + batch256 recipe 后，SR 从 93.60% 提到 95.20%，val MSE 也从 0.165902 降到 0.143684。

因此第一次训练下降中至少有一部分来自：

- 弱 P1；
- batch512；
- 为了 batch512 增加 `samples_per_epoch`；
- DataLoader worker/pin/persistent 等非原始 v9 recipe 变化。

目前不能精确区分“弱 P1”和“batch512”各占多少，但 Scheme A 已证明这些 recipe/P1 因素合计造成约 **+1.60pp 可恢复损失**。

### 4.3 直接证据 3：VR4/data4 加入后没有带来增益

干净对照是 v9 vs Scheme A：

| 实验 | P1 | data | batch | SR |
|---|---|---|---:|---:|
| v9 | v9 P1 | MJ+VR1-3 | 256 | 98.80% |
| Scheme A | v9 P1 | MJ+VR1-4 | 256 | 95.20% |

这说明即使排除了弱 P1 和 batch512，加入 VR4 后仍然低于 v9 **-3.60pp**。

Scheme B 再进一步测试“VR4 独立 domain + 降权”：

| 实验 | VR4 处理 | SR |
|---|---|---:|
| Scheme A | VR4 与 VR1-3 同 source/domain | 95.20% |
| Scheme B | VR4 独立 source/domain，约 10% 权重 | 94.00% |

Scheme B 仍然不如 Scheme A，说明问题不是简单的“VR4 与 VR1-3 混在一个 domain 里”这么单一。

### 4.4 数据层面的可能原因

VR4 数据统计此前已验证：

| Source | segments | frames | frames/segment | action L2 mean | timeout |
|---|---:|---:|---:|---:|---:|
| VR1 | 200 | 16466 | 82.3 | 0.0238 | 11 |
| VR2 | 250 | 23659 | 94.6 | 0.0276 | 20 |
| VR3 | 505 | 18954 | 37.5 | 0.0423 | 0 |
| VR4 | 605 | 19257 | 31.8 | 0.0479 | 1 |

VR4 的特点：

1. segment 更多，但每段更短；
2. action 幅度更大；
3. 几乎全是 success，timeout/困难样本极少；
4. 可能更偏 easy-success / fast-reach 风格。

这类数据可能降低验证 MSE 或局部行为看起来很好，但不一定提升多 seed 闭环鲁棒性。Scheme B 的 worst seed 只有 82.0%，说明它在某些 target distribution/seed 下明显不稳。

### 4.5 总体分解

v9 → 第一次 20260426 的总下降：

- v9：98.80%
- 第一次：93.60%
- 总下降：-5.20pp

可由当前实验分解为：

| 因素 | 证据 | 影响 |
|---|---|---:|
| 弱 P1 + batch512/recipe 改动 | 第一次 93.60% → Scheme A 95.20% | 约 -1.60pp |
| VR4/data1-4 当前混合不如 VR1-3 | v9 98.80% → Scheme A 95.20% | 约 -3.60pp |
| VR4 独立 domain/降权未解决 | Scheme A 95.20% → Scheme B 94.00% | -1.20pp |

因此今天第一次训练下降的主因不是单一 bug，而是：

> **为了提升显存/吞吐而改了 recipe，导致 P1/P2 起点和收敛变差；同时 VR4/data4 的分布没有提供 v9 所需的闭环鲁棒性，反而稀释或扰动了原本有效的 VR1-3 信号。**

---

## 5. 阶段结论与下一步

### 5.1 已确认结论

1. 不要再用 batch512 作为默认 recipe；它没有带来性能收益。
2. 当前最优仍是 v9：MJ + VR1-3，98.80%。
3. VR4 不能直接追加进训练集；无论同 domain 还是独立低权，都没有超过 v9。
4. 如果继续利用 VR4，应先做数据过滤/质量选择，而不是继续调 source ratio。

### 5.2 推荐下一步方案

P0：跑 **Scheme C：VR4 filtered**。

建议只保留 VR4 中更像“有价值监督”的片段，例如：

- 去掉过短 segment，例如 `< 40 frames`；
- 降权或过滤过快/动作过大的片段；
- 保留接近 timeout、长轨迹、边缘目标；
- VR1-3 仍按 v9 原权重；
- 复用 v9 P1；
- batch=256；
- VR4 权重先低，例如 5%–10%。

P1：如果 Scheme C 仍不超过 v9，应回退到 v9 数据配置，停止把 VR4 用作训练数据，只把 VR4 作为诊断数据集。

P2：若要继续采集，不建议继续采 easy-success；应采 recovery/correction、边缘目标、长路径、失败边界数据。
