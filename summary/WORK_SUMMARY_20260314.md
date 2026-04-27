# VR 机器人示教数据平台改造与训练工作总结（截至 2026-03-14）

## 1. 本日目标

在 3 月 13 日已打通“采集 → 训练”基础闭环的前提下，进一步完成：

1. 训练流程规范化（可复现实验目录、结果不覆盖）
2. 跨批次实验固定化（collector3 训练，collector2 测试）
3. 新建 ACT 目录并完成首轮 50 epoch 训练
4. 产出可用于论文写作的图像与结果汇总

---

## 2. 训练流程与目录规范化

### 2.1 跨批次训练配置与脚本落地

新增配置：
- `vr-control-robot/configs/cross_c3_train_c2_test.yaml`

新增脚本：
- `vr-control-robot/scripts/train_cross_dataset.py`
- `vr-control-robot/scripts/data_health_check.py`

能力：
- 固定数据划分：`collector3 -> train`, `collector2 -> test`
- 每次运行自动新建时间戳目录（避免覆盖）
- 自动输出 `metrics.json/csv` 与 `result_summary.json/csv`

### 2.2 旧训练脚本统一输出策略

更新：
- `vr-control-robot/scripts/train.py`

调整后：
- 默认采用 `artifacts/experiments/<exp>/run_YYYYMMDD_HHMMSS/`
- 保留 `--legacy-static-output` 兼容旧固定目录模式

---

## 3. BC 与 ACT 工程结构拆分

### 3.1 BC 内容整理到独立目录（兼容旧实现）

新建：
- `vr-control-robot/bc/`
  - `configs/`
  - `scripts/`
  - `README.md`

说明：
- `bc/scripts/` 采用转发包装方式，复用现有根目录脚本实现，先保证稳定迁移。

### 3.2 ACT 新框架目录搭建

新建：
- `vr-control-robot/act/`
  - `configs/base_act.yaml`
  - `scripts/convert_jsonl_to_seq_hdf5.py`
  - `scripts/train_act.py`
  - `scripts/evaluate_act.py`
  - `scripts/export_policy_act.py`
  - `scripts/generate_paper_figures.py`
  - `README.md`

ACT 当前实现形态：
- 时序输入（窗口 `seq_len=16`）
- Transformer Encoder 策略（ACT 风格基线）
- 输出动作为 `jointDelta`

---

## 4. 数据体检与跨批次实验执行

### 4.1 数据体检执行

已执行：
- `scripts/data_health_check.py --config configs/cross_c3_train_c2_test.yaml`

产物：
- `vr-control-robot/artifacts/health_reports/health_summary_20260313_173028.json`
- 以及对应单数据集体检文件

### 4.2 BC 跨批次实验（collector3 训，collector2 测）

已执行完成，目录：
- `vr-control-robot/artifacts/experiments/c3_train_c2_test/run_20260313_173033/`

关键指标：
- `best_val_loss = 0.0014036210`
- `cross_test_mse = 0.0005376475`
- baseline（test）：
  - `zero_action_mse = 0.0040513435`
  - `mean_action_mse = 0.0040512425`
  - `prev_action_mse = 0.0062848232`

结论：
- BC 在跨批次测试上明显优于简单 baseline。

---

## 5. ACT 首轮 50 轮训练与评估

### 5.1 训练配置

配置文件：
- `vr-control-robot/act/configs/base_act.yaml`

本次设置：
- `epochs = 50`
- train: collector3（序列化后）
- test: collector2（序列化后）

### 5.2 数据转换结果

- train 序列样本：`16297`
- test 序列样本：`1787`
- `seq_len=16`, `obs_dim=48`, `act_dim=24`

### 5.3 训练结果

运行目录：
- `vr-control-robot/artifacts/experiments_act/act_c3_train_c2_test/run_20260313_180439/`

关键指标：
- `best_epoch = 42`
- `best_val_loss = 0.0021401328`
- `test_mse = 0.0009135309`
- baseline（test）：
  - `zero_action_mse = 0.0043885289`
  - `mean_action_mse = 0.0043884409`

结论：
- ACT 跨批次显著优于 baseline；
- 在本次设置下，BC 指标仍优于 ACT（后续可继续调参与扩数）。

### 5.4 模型导出

导出文件：
- `vr-control-robot/artifacts/exported/policy_act_c3_train_c2_test_20260313_180439.pt`

修复项：
- 解决 Transformer 导出 trace 校验失败问题：
  - `torch.jit.trace(..., check_trace=False)`

---

## 6. 论文图像产出能力补齐

新增脚本：
- `vr-control-robot/act/scripts/generate_paper_figures.py`

已生成图像（ACT run 目录下）：
- `figures/act_training_curve.png`
- `figures/act_vs_baseline_bar.png`
- `figures/act_joint_rmse.png`
- `figures/act_pred_vs_gt_scatter.png`
- `figures/act_error_hist.png`
- `figures/bc_vs_act_bar.png`

并自动生成：
- `result_summary.json`
- `result_summary.csv`

说明：
- 该套图像已可直接用于论文实验章节初稿。

---

## 7. 当前状态判断（截至 3/14）

1. 采集与训练工程化程度明显提升，具备可复现实验能力。
2. BC 与 ACT 两条线均已形成“配置-训练-评估-导出-可视化”闭环。
3. 目前结论仍以离线指标为主，尚需 Isaac Sim 闭环轨迹验证来确认在线可用性。

---

## 8. 下一步建议

1. 在 Isaac Sim 完成两阶段验证：
   - 示教动作重放基线（不经模型）
   - 模型闭环控制（BC/ACT）
2. 固化在线指标：
   - 关节轨迹 RMSE
   - 末端轨迹误差
   - 稳定性（漂移/抖动/失败率）
3. 扩充数据到 100~200 episodes，并增加动作多样性。
4. 继续 ACT 调参：
   - `seq_len`、`d_model`、`num_layers`、学习率与正则策略
5. 形成 BC vs ACT 的统一实验表格（多 run 聚合），支撑论文结论。

---

## 9. 3/14 关键产物索引

- 新 summary：
  - `WORK_SUMMARY_20260314.md`
- 跨批次 BC run：
  - `vr-control-robot/artifacts/experiments/c3_train_c2_test/run_20260313_173033/`
- ACT run：
  - `vr-control-robot/artifacts/experiments_act/act_c3_train_c2_test/run_20260313_180439/`
- ACT 导出模型：
  - `vr-control-robot/artifacts/exported/policy_act_c3_train_c2_test_20260313_180439.pt`
- ACT 图像：
  - `vr-control-robot/artifacts/experiments_act/act_c3_train_c2_test/run_20260313_180439/figures/`
