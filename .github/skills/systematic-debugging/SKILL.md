---
name: systematic-debugging
description: 'DEBUGGING SKILL adapted from obra/superpowers (156k★). USE FOR: diagnosing model training failures, MuJoCo simulation bugs, VR-MuJoCo distribution mismatch, policy rollout anomalies, data pipeline issues, coordinate system errors, 0% success rate diagnosis. TRIGGERS: user reports bug, unexpected behavior, model performance regression, "手臂上抬", "成功率为0", spinning, action drift, observation mismatch, or any diagnostic task.'
---

# 系统化调试流程（Systematic Debugging）

> 改编自 [obra/superpowers](https://github.com/obra/superpowers)（156k★ MIT License），针对本项目 VR→MuJoCo 模仿学习场景优化。

## 核心铁律

```
在尝试任何修复之前，必须先找到根因。
对症治疗不是调试，是掩盖问题。
```

## 何时使用

- 模型 MuJoCo 闭环成功率低于预期
- 手臂出现异常行为（上抬、旋转、漂移、抖动）
- 训练 loss 正常但闭环失败
- 专家数据质量可疑
- VR 数据与 MuJoCo 分布不一致
- 坐标系疑似错误
- 评估结果不稳定或不可复现

## 四阶段流程

**必须按顺序完成每个阶段，不可跳过。**

### 阶段 1：根因调查

**在尝试任何修复之前：**

1. **仔细阅读错误信息**
   - 不要跳过 stack trace 或 warning
   - 记录关键数值（loss、成功率、距离均值）

2. **可靠复现**
   - 用固定 seed（`--seed 42`）复现问题
   - 确保评估至少 50 targets（`--num-trials 10`）
   - 小样本结果**不可信**（项目多次验证）

3. **检查最近改动**
   - `git diff` 看哪些文件变了
   - 配置文件（`config_*.json`）是否有意外修改
   - 归一化参数是否匹配当前数据集

4. **追踪数据流（本项目关键路径）**
   ```
   VR 录制 → JSONL → common.py load_segments()
      → normalize_obs/act() → 模型训练
      → checkpoint → validate_policy.py → MuJoCo 闭环
   ```
   在每个阶段打印关键值，确认数据流到哪里出了问题。

5. **常见根因速查表**

   | 症状 | 历史根因 | 参见 Summary |
   |------|---------|-------------|
   | 手臂持续上抬到头顶 | 动作偏置漂移 + 数据分布偏移 | 20260410 |
   | 专家成功率 <15% | `ctrl` 未同步到执行器 | 20260410 |
   | 闭环成功率 0% | 坐标系缺少 `_R_BASE` 旋转 | 20260410 |
   | 目标球不动/掉落 | body 类型问题（需 mocap） | 20260410 |
   | 时间集成导致性能下降 | TEMPORAL_WEIGHT 太小 | 20260415 |
   | Transformer 动作幅度不足 | LayerNorm 衰减，需 scale | 20260415 |
   | 小样本评估显示 100% | 样本不足，≥50 targets 才可信 | 20260415 |
   | VR 目标分布与 MuJoCo 不匹配 | VR 目标需要走路，MuJoCo 只伸手 | 20260410 |

### 阶段 2：模式分析

1. **找工作的参照物**
   - 哪个模型/配置是成功的？（MLP k=5 → 94%）
   - 成功模型用了什么数据集？什么推理参数？

2. **对比差异**
   - 成功 vs 失败配置的每一个参数
   - 数据集是否相同？归一化参数是否相同？
   - 推理参数（`--no-ensemble`、`--action-scale`）是否一致？

3. **检查关键诊断脚本**
   ```bash
   # 数据分布诊断
   python diagnose_target_dist.py
   
   # 模型推理行为诊断
   python diagnose_model.py
   
   # 旋转行为诊断
   python diagnose_spinning.py
   ```

### 阶段 3：假设与测试

1. **形成单一假设**
   - 明确写出："我认为 X 是根因，因为 Y"
   - 具体、可验证，不要模糊

2. **最小改动测试**
   - 一次只改一个变量
   - 不要同时改多个东西
   - 用标准评估验证：
     ```bash
     cd mujoco_sim
     python validate_policy.py \
       --checkpoint <path> \
       --num-trials 10 --seed 42 --no-ensemble
     ```

3. **判断结果**
   - 有效 → 进入阶段 4
   - 无效 → 形成新假设，**不在旧修复上叠加新修复**

4. **≥3 次修复失败 → 质疑架构**
   - 不是简单 bug，可能是设计问题
   - 讨论后再继续，不要盲目尝试第 4 次

### 阶段 4：实施修复

1. **最小化修复**
   - 只修根因，不做顺手改进
   - 一次一个改动

2. **验证修复**
   - 必须跑完整评估（≥50 targets）
   - 对比修复前后数据
   - 结果记录到当天训练目录的 `outputs/` 中

3. **记录到 Summary**
   - 失败的尝试也要记录
   - 每个结论必须有具体数据支撑

## 红旗信号 — 立即停下

如果你发现自己在想：
- "这个应该行" → 运行验证命令
- "随便改改试试看" → 回到阶段 1
- "小样本够了" → **不够，必须 ≥50 targets**
- "loss 很低所以模型没问题" → loss 低 ≠ 闭环成功率高
- "之前那个配置差不多的" → 差不多 ≠ 一样

**以上任何一条 → 停下，回到阶段 1。**
