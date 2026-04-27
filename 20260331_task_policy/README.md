# 20260331_task_policy

本实验是针对 `collector5` 的新一轮独立训练，目标是学习更贴近实际部署的 policy：

- 不使用 human pose / controller 输入。
- 使用 `机器人状态 + 任务状态` 作为 observation。
- 使用 `jointDelta` 作为主输出，学习真正可执行的动作 policy。
- 使用 `segment_success` 作为辅助监督，而不是把“任务完成情况”直接当作 policy 输出。

## 设计原则

- 主任务始终是动作预测；仅预测成功/失败本身不能直接得到 policy。
- 使用 events 文件给 target segment 打标签，避免仅靠 frame 内弱标签。
- 以 target segment 为基本单位建样本，排除 `auto_end_by_new_start` 截断段。
- 加入标准化、异常值过滤和更合理的 episode-level 切分。

## 目录说明

- `config.json`：数据路径与超参数
- `common.py`：数据解析、segment 标签、标准化、dataset 构建
- `train_bc.py`：任务条件 MLP BC
- `train_act.py`：任务条件 Transformer ACT
- `analyze_results.py`：汇总训练结果并输出总结
- `run_all.py`：一键执行 BC + ACT + 分析

## 使用

```bash
python run_all.py
```

或分别执行：

```bash
python train_bc.py --config config.json --out outputs
python train_act.py --config config.json --out outputs
python analyze_results.py
```

## 关键 observation

- 机器人关节位置 / 速度
- 左右 end-effector 位姿
- target 世界坐标与相对左右手坐标
- 距离、接触、保持时长、进度、phase 等任务状态

## 关键输出

- 主输出：`jointDelta`
- 辅助输出：`segment_success`（仅作 representation shaping 和分析）
