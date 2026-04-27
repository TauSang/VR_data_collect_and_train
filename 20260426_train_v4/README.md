# 20260426_train_v4 — VR4 filtering 实验

目标：验证 VR4 是否能在过滤后重新变成正增益数据。

## 背景

已完成的 data1-4 实验结论：

| 实验 | 设置 | 严格 SR |
|---|---|---:|
| v9 | v9 P1 + MJ+VR1-3 | 247/250 = 98.80% |
| 20260426 第一次 | 重训 P1 + MJ+VR1-4 + batch512 | 234/250 = 93.60% |
| Scheme A | v9 P1 + MJ+VR1-4 + batch256 | 238/250 = 95.20% |
| Scheme B | v9 P1 + MJ+VR1-4，VR4 独立 domain/降权 | 235/250 = 94.00% |

VR4 原始数据：605 segments / 19257 frames / 122 episodes，604 success / 1 timeout。主要问题是短段过多、动作幅度偏大、几乎全是 easy-success，缺少 recovery/timeout 监督。

## 本轮过滤方案

生成两个 filtered VR4 数据集：

1. `vr4_len40`：保留 `segment_len >= 40` 的段。
   - 用于测试“短段/easy snippet 是否是主要污染源”。
2. `vr4_quality_v1`：保留 `segment_len >= 40` 且 `mean(||action||_2) <= 0.06` 的段。
   - 用于测试“短段 + 高动作幅度段”是否共同导致退化。

训练时两者都使用：

- v9 Phase1 checkpoint；
- v9 Phase2 batch256 recipe；
- VR4 filtered 数据并入 VR domain（source_id=1），不再新开 domain；
- `--inherit-normalizer`；
- ACTChunkGatedCrossAttnFiLM；
- strict eval：5 seeds × 50 targets = 250 targets。

## 判断标准

| 结果 | 解释 |
|---|---|
| filtered VR4 ≥ 98.80% | VR4 过滤后可用，可继续按过滤标准采集/扩充 |
| 95.20% < filtered VR4 < 98.80% | 过滤有帮助但未超过 v9，VR4 暂不能作为主训练增量 |
| filtered VR4 ≤ 95.20% | VR4 当前不可用，应排除或改采数据分布 |
