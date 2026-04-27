# collector5_baseline

独立实验目录：仅基于 `collector5` 数据，分别训练简单 `BC` 和 `ACT`。

## 设计原则

- 不依赖项目里其它训练代码。
- 数据、代码、模型、图像、总结全部放在本目录。
- 默认训练 50 轮。

## 文件说明

- `config.json`：数据路径与超参数
- `common.py`：数据解析与公共工具
- `train_bc.py`：简单 MLP BC
- `train_act.py`：简单 Transformer-Encoder ACT
- `analyze_results.py`：对比 BC/ACT，产出汇总图和报告
- `run_all.py`：一键执行 BC + ACT + 分析

## 运行

在本目录下执行：

```bash
python run_all.py
```

或分别执行：

```bash
python train_bc.py --config config.json --out outputs
python train_act.py --config config.json --out outputs
python analyze_results.py
```

## 产出

- `outputs/bc/run_*/`
  - `checkpoints/best.pt`
  - `metrics.json`
  - `loss_curve.png`
  - `pred_scatter.png`
- `outputs/act/run_*/`
  - 同上
- `outputs/summary/`
  - `bc_vs_act_val_mse.png`
  - `summary.json`
  - `summary.md`
