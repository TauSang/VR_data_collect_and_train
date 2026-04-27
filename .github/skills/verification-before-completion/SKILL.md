---
name: verification-before-completion
description: 'VERIFICATION SKILL adapted from obra/superpowers (156k★). USE FOR: verifying model training results, confirming MuJoCo evaluation numbers, validating data pipeline output, ensuring evaluation meets minimum target count (≥50), checking model checkpoint integrity. TRIGGERS: before claiming any success rate, before writing summary conclusions, before committing model results, before declaring a fix works.'
---

# 验证后再声明完成（Verification Before Completion）

> 改编自 [obra/superpowers](https://github.com/obra/superpowers)（156k★ MIT License），针对本项目模型训练与评估场景优化。

## 核心铁律

```
没有运行验证命令就声称结果 = 说谎，不是省事。
```

## 验证门控

```
在声明任何状态或表达满意之前：

1. 识别：什么命令能证明这个声明？
2. 运行：执行完整命令（新鲜的，完整的）
3. 阅读：完整输出，检查退出码，数成功/失败数
4. 确认：输出是否支持这个声明？
   - 否 → 说明实际状态和证据
   - 是 → 带证据声明结果
5. 然后才能说"完成"

跳过任何步骤 = 不可接受
```

## 本项目特定验证规则

### 模型评估

| 声明 | 需要什么证据 | 不够的证据 |
|------|-------------|----------|
| "成功率 X%" | `validate_policy.py` 输出 ≥50 targets | 小样本、之前的运行、"应该差不多" |
| "Loss 下降了" | 训练日志显示 val_loss 对比 | 只看 train_loss |
| "Bug 修复了" | 修复前后对比评估数据 | "代码改了应该行" |
| "MLP 优于 Transformer" | 同数据集、同参数、各 ≥50 targets | 不同数据集的对比 |
| "专家数据质量好" | 专家成功率 + 数据统计 | "跑了一下看起来没问题" |

### 标准评估命令

```bash
# MLP 模型评估（必须 --no-ensemble）
cd mujoco_sim
python validate_policy.py \
  --checkpoint <path_to_best.pt> \
  --num-trials 10 --seed 42 --no-ensemble

# Transformer 模型评估
python validate_policy.py \
  --checkpoint <path_to_best.pt> \
  --num-trials 10 --seed 42 --action-scale 1.3

# 最少 50 targets = 10 trials × 5 targets
# 低于此数的结果 **不可引用**
```

### 训练完成验证清单

- [ ] `best.pt` 检查点存在且非空
- [ ] `val_loss` 优于 baseline
- [ ] MuJoCo 闭环评估已运行（≥50 targets）
- [ ] 成功率数字来自命令输出，不是估算
- [ ] 结果已写入 `mujoco_eval.json`

### Summary 文档验证清单

- [ ] 每个成功率数字都有对应的评估命令输出
- [ ] 结论中 "已确认" 的条目都有 ≥50 targets 支撑
- [ ] 失败的实验也记录在案
- [ ] 对比表格中的数据来源一致（同 seed、同 targets 数）

## 红旗信号

- 使用 "应该"、"大概"、"差不多"
- 在验证前表达满意（"太好了！"、"Done！"）
- 引用上次运行的结果而不是当前运行
- 认为 "loss 低" 等于 "MuJoCo 闭环好"
- "小样本 100%" → **这是本项目反复验证过的陷阱**

## 已知教训

| 错误信念 | 实际结果 | Summary 来源 |
|---------|---------|-------------|
| "act_v4 百分百成功" | 50 targets 实测 0% | 20260415 |
| "act_mlp_k1 百分百成功" | 50 targets 实测 90% | 20260415 |
| "专家成功率够用" | ctrl 未同步导致只有 6.4% | 20260410 |
| "时间集成会帮助" | 在此任务上一直有害 | 20260415 |
