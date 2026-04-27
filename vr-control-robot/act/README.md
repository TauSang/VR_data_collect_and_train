# ACT 子目录（新建框架）

该目录用于集中管理 ACT（Action Chunking Transformer）相关实验。

## 目录

- `configs/`：ACT 配置
- `scripts/`：ACT 数据转换、训练、评估、导出

## 当前状态

已提供可运行的最小框架：

1. `convert_jsonl_to_seq_hdf5.py`：将逐帧数据转换为序列窗口数据
2. `train_act.py`：Transformer 编码器版本的 ACT 风格基线训练
3. `evaluate_act.py`：加载 checkpoint 评估
4. `export_policy_act.py`：导出 TorchScript 模型

## 示例流程

在 `vr-control-robot/` 下运行：

1) 转换数据

- `python act/scripts/convert_jsonl_to_seq_hdf5.py --config act/configs/base_act.yaml`

2) 训练

- `python act/scripts/train_act.py --config act/configs/base_act.yaml`

3) 评估

- `python act/scripts/evaluate_act.py --config act/configs/base_act.yaml --ckpt <best.pt路径>`

4) 导出

- `python act/scripts/export_policy_act.py --config act/configs/base_act.yaml --ckpt <best.pt路径>`

5) 生成论文图像与结果汇总

- `python act/scripts/generate_paper_figures.py --config act/configs/base_act.yaml --run-dir <ACT run目录> --bc-summary <可选: BC result_summary.json>`

会在 `<ACT run目录>/figures/` 下生成：

- `act_training_curve.png`
- `act_vs_baseline_bar.png`
- `act_joint_rmse.png`
- `act_pred_vs_gt_scatter.png`
- `act_error_hist.png`
- `bc_vs_act_bar.png`（传入 BC summary 时生成）
