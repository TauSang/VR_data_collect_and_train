# VR Data → PyTorch → Isaac Sim 训练框架

这个目录是你后续训练与仿真对接的独立工作区，面向当前导出的示教数据（session/episodes/events）。

## 目录结构

- `configs/`：训练配置（数据路径、模型、超参数）
- `scripts/`：数据转换、训练、评估、导出
- `bc/`：BC（Behavior Cloning）整理目录（配置与脚本入口）
- `act/`：ACT（Action Chunking Transformer）实验目录（新框架）
- `src/vrtrain/`：核心训练代码
  - `data/`：数据读取与向量化
  - `models/`：策略网络
  - `trainers/`：行为克隆训练器
  - `sim/`：Isaac Sim 对接接口（桥接层）
  - `utils/`：工具函数
- `artifacts/`：训练产物（数据集、模型、日志）

## 当前支持

- 将 `episodes.jsonl` 转为 HDF5 训练集
- PyTorch 行为克隆（BC）训练
- 导出 TorchScript 模型（用于部署/仿真）
- Isaac Sim 推理桥接接口骨架

## BC / ACT 入口说明

- BC（已完成主流程）：
  - 推荐入口：`bc/configs/` + `bc/scripts/`
  - 例如：`python bc/scripts/train.py --config bc/configs/collector3_50.yaml`

- ACT（新框架，当前为可运行基线）：
  - 配置：`act/configs/base_act.yaml`
  - 流程：
    1) `python act/scripts/convert_jsonl_to_seq_hdf5.py --config act/configs/base_act.yaml`
    2) `python act/scripts/train_act.py --config act/configs/base_act.yaml`
    3) `python act/scripts/evaluate_act.py --config act/configs/base_act.yaml --ckpt <best.pt>`
    4) `python act/scripts/export_policy_act.py --config act/configs/base_act.yaml --ckpt <best.pt>`

## 标准流程（建议）

1. 把多轮采集数据放在 `data_collector/collector*`。
2. 修改 `configs/base.yaml` 中的数据输入路径。
3. 运行 `scripts/convert_jsonl_to_hdf5.py` 转换数据。
4. 运行 `scripts/train.py` 开始训练。
5. 运行 `scripts/export_policy.py` 导出模型。
6. 在 Isaac Sim 侧调用 `src/vrtrain/sim/isaac_bridge.py` 的桥接接口。

## 输出目录策略（已统一）

`scripts/train.py` 现已默认使用“每次训练新建 run 目录”策略，避免覆盖历史结果。

示例：

- `artifacts/experiments/base/run_20260313_210501/checkpoints/best.pt`
- `artifacts/experiments/base/run_20260313_210501/metrics.json`
- `artifacts/experiments/base/run_20260313_210501/metrics.csv`

并在实验根目录维护：

- `artifacts/experiments/<run_name>/latest_run.txt`

如需沿用旧版“固定目录覆盖”行为，可使用：

```bash
python scripts/train.py --config configs/collector3_50.yaml --legacy-static-output
```

## 跨批次实验（collector3 训练，collector2 测试）

已新增配置与脚本：

- 配置：`configs/cross_c3_train_c2_test.yaml`
- 训练+跨集评估：`scripts/train_cross_dataset.py`
- 数据体检：`scripts/data_health_check.py`

### 1) 先跑数据体检（建议每次训练前执行）

```bash
python scripts/data_health_check.py --config configs/cross_c3_train_c2_test.yaml
```

会在 `artifacts/health_reports/` 下生成带时间戳的新文件，便于历史对比。

### 2) 跑跨批次训练与测试

```bash
python scripts/train_cross_dataset.py --config configs/cross_c3_train_c2_test.yaml
```

默认行为：

- 训练集：collector3
- 测试集：collector2
- 每次运行自动创建新目录：
  - `artifacts/experiments/c3_train_c2_test/run_YYYYMMDD_HHMMSS/`
- 目录内包含：
  - `checkpoints/best.pt`, `checkpoints/last.pt`
  - `metrics.json`, `metrics.csv`
  - `result_summary.json`, `result_summary.csv`

这样每次训练都会新增一份独立结果，不会覆盖旧实验，方便直接横向对比。

## 你开始训练前还需要做的事

1. 安装 Python 环境与依赖（见 `requirements.txt`）。
2. 确认训练数据路径有效（`configs/base.yaml`）。
3. 决定 observation/action 维度是否包含你不需要的字段（目前默认不使用抓夹）。
4. 规划 train/val 划分与 episode 过滤规则（可按 episode 范围筛）。
5. 先跑一次小规模训练验证 loss 收敛，再放全量数据。
6. 明确 Isaac Sim 控制频率与动作接口（关节增量/目标角度）一致性。
