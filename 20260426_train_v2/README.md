# 20260426_train_v2 — Scheme A

目的：隔离 `data1-4` 的影响。

方案 A：复用 v9 Phase1 checkpoint，Phase2 使用 `MJ + VR1-4`，并完全恢复 v9 Phase2 recipe（batch=256、lr=5e-5、patience=120、weighted_source 1.5:1.0、不使用 batch512）。

固定预训练 checkpoint：

- `20260425_train_v9/outputs/act_chunk/phase1_run_20260425_193808/checkpoints/best.pt`

对照：

- v9 P1 MJ-only: 230/250 = 92.00%
- v9 P2 MJ+VR1-3: 247/250 = 98.80%
- 20260426 data1-4 batch512: 234/250 = 93.60%

训练后必须运行：

1. HPC strict eval: 5 seeds × 50 targets
2. 本地 MuJoCo `--visualize`
3. 写入 summary
