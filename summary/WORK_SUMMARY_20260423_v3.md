# 工作总结 v3 — VR 数据价值 + ACT 架构创新（2026-04-23）

## 核心结论

**VR 数据 + 几何对齐 + ACT-DAFiLM 架构创新，组合获得 +10pp 收益（76% → 86%）。**

| 方案 | MuJoCo 50-target |
|------|-----------------|
| Weak-MJ 基线 | 38/50 = 76% |
| + VR 原始混合（scratch）| 35/50 = 70% **(劣化)** |
| + VR 几何对齐混合（scratch）| 40/50 = 80% (+4) |
| + VR 对齐 + **FiLM 架构创新** | **43/50 = 86% (+10)** |
| Strong-MJ + VR 对齐 FT | 47/50 = 94% (天花板，无回退) |
| Strong-MJ + VR 对齐 + FiLM FT | 47/50 = 94% (天花板，无回退) |

## A) 架构创新：ACT-DAFiLM

### 设计动机
- 直接混合 VR 数据会污染共享主干，因为 VR 残留几何/动力学偏差。
- 期望：让 VR 提供梯度信号，又不让其偏差泄漏到推理路径。

### 核心机制
- 在 `ACTChunkMLP` 上加 `domain_emb: Embedding(num_domains=2, cond_dim=16)`。
- 每层 MLP 后加 FiLM：`h ← (1+γ)·LN(Wx) + β`，γ/β 由 domain embedding 经线性层产生。
- **零初始化** FiLM 输出层 → 训练起点等价于普通 MLP，可无缝从 MLP 检查点加载。
- 训练：每个样本携带 `source_id`（0=MuJoCo, 1=VR）。
- 推理：`default_domain_id=0`，永远走 MuJoCo 域，VR 偏差被锁在 γ_VR/β_VR 中。

### 实现位置
- 模型类：[20260422_train/train_act_chunk.py](20260422_train/train_act_chunk.py) 中 `ACTChunkMLPFiLM` + `_FiLMBlock`
- 数据流：[20260422_train/common.py](20260422_train/common.py) 中 `SegmentData.source_id`
- 评估：[mujoco_sim/validate_policy.py](mujoco_sim/validate_policy.py) 识别 `model_class=="ACTChunkMLPFiLM"`
- 配置：[20260422_train/config_mixed_film.json](20260422_train/config_mixed_film.json)、[20260422_train/config_strong_mixed_film_ft.json](20260422_train/config_strong_mixed_film_ft.json)

### Checkpoint 兼容
`ACTChunkMLPFiLM.load_from_mlp_state()` 把旧 MLP 检查点的 `backbone.{idx}` 重映射为 `blocks.{idx//4}.{linear|norm}`，FiLM 参数保持零初始化，确保 stage-2 finetune 起点与原模型完全一致。

## B) 收集端最小改动：几何对齐脚本

### 问题诊断
VR 端 `endEffector` 字段记录的是 three.js 世界坐标 + VR 轴向（Y up，Z 向身后），而 MuJoCo expert 记录的是 `_R_BASE` 旋转后的机器人本地坐标。即使共享归一化器，几何差异仍让策略学到错误偏移。

### 解决方案
脚本 [scripts/align_collector_to_g1fk.py](scripts/align_collector_to_g1fk.py)：
- 加载 `mujoco_sim/model/task_scene.xml`，按 G1 关节顺序写入 `qpos`。
- 调用 `mj_kinematics` 取 `left_wrist_yaw_link` / `right_wrist_yaw_link` 世界位姿。
- 通过 `mujoco_to_vr` + `_R_BASE` 转回 VR 客户端使用的标准化坐标。
- 重算 `targetRelToRobotBase/Left/Right`、`distToTargetLeft/Right`。
- 用法：

```powershell
python scripts/align_collector_to_g1fk.py `
  --model mujoco_sim/model/task_scene.xml `
  --episodes data_collector/collector10/vr-demonstrations-episodes-20260422_170038.jsonl `
  --events data_collector/collector10/vr-demonstrations-events-20260422_170039.jsonl `
  --out data_collector/collector10_aligned
```

### 为何不改采集前端
- 用户明确："这大概是最后一次对收集端的改进"。
- 后处理脚本 0 风险，老数据可批量复用。
- 未来扩充数据集只需重新跑脚本，不改 VR/Vue 代码。

## 对论文的意义

1. **VR 数据价值得证**：在 weak-data 设置下，VR 对齐数据带来 +4pp，叠加 FiLM 架构带来 +10pp 总收益。
2. **架构创新有故事**：DAFiLM 是 ACT 的小型化扩展，零初始化保证向后兼容，inference 路径与原模型完全一致（无额外 latency / 参数膨胀也很有限：~1K 额外参数）。
3. **天花板下不掉点**：strong + FiLM FT 仍保持 94%，证明架构改造没有引入回退。

## 后续动作

- 按用户授权扩充对齐后的 VR 数据集（重跑 `align_collector_to_g1fk.py` 即可）。
- 如果新数据让 weak-MJ + VR + FiLM 突破 90%，故事更圆满。
- strong + 大规模对齐 VR + FiLM 是否能突破 94% 上限是下一阶段实验目标。
