# Copilot Instructions — VR Robot Control

## 项目概要

这是一个 VR 机器人模仿学习项目。核心链路：VR 示教采集 → 策略训练（BC/ACT） → MuJoCo 仿真评估 → 部署。
目标机器人：Unitree G1 双臂人形机器人，任务：单手 reaching（触碰目标球）。

## 语言偏好

- 正文用**中文**，代码和命令用英文
- Summary 文档和技术讨论全部使用中文

## 关键约束

### 评估标准
- MuJoCo 评估必须 ≥ **50 targets**（10 trials × 5 targets），否则结果**不可引用**
- MLP 模型必须使用 `--no-ensemble`
- 评估 seed 固定为 42

### 训练规范
- 新训练实验必须在根目录创建 `YYYYMMDD_train/` 新目录
- 不要在旧日期目录下叠加新实验
- 从 `20260409train2/` 复制最新代码作为起点

### 已知结论（避免重复踩坑）
- 时间集成（Temporal Ensembling）对此任务**有害**，永远用 `--no-ensemble`
- MLP 优于 Transformer（在 <300K 帧数据下）
- `action_scale=1.3` 补偿 Transformer 动作衰减
- VR-only 模型 MuJoCo 成功率约 24-27%，需要 finetune 或 mixed training
- 小样本评估结果不可信（act_v4 小样本 "100%"，大样本 0%）

## 项目关键路径

| 组件 | 路径 |
|------|------|
| 最新训练代码 | `20260409train2/` |
| MuJoCo 评估脚本 | `mujoco_sim/validate_policy.py` |
| 最优模型 | `20260409train2/outputs/act_mlp_k5/act_chunk/run_20260412_174749/checkpoints/best.pt` |
| 数据采集目录 | `data_collector/` |
| VR 前端 | `frontend/src/components/RobotVR.vue` |
| Summary 文档 | `summary/` |
| 工作流 Skill | `.github/skills/vr-robot-workflow/SKILL.md` |
