# VR 机器人示教项目深度分析报告（2026-04-01）

## 目录

1. [项目目标与当前状态总览](#1-项目目标与当前状态总览)
2. [各模块完成度评估](#2-各模块完成度评估)
3. [20260331_task_policy 训练深度分析](#3-20260331_task_policy-训练深度分析)
4. [关键矛盾与问题](#4-关键矛盾与问题)
5. [目标可行性评估](#5-目标可行性评估)
6. [建议与改进路线](#6-建议与改进路线)
7. [最终结论](#7-最终结论)

---

## 1. 项目目标与当前状态总览

### 1.1 目标定义

> 从 VR 采集示教数据 → 训练策略模型 → 部署到 Isaac Sim → 机器人自主伸手触碰目标

这个目标可以分解为四个子任务：

| 子任务 | 描述 | 当前状态 |
|--------|------|----------|
| **VR 数据采集** | WebXR 采集任务驱动的示教数据 | ✅ 已完成，schema 稳定 |
| **数据预处理** | JSONL → 可训练格式，含特征工程 | ✅ 已完成，多套方案并存 |
| **策略训练** | BC / ACT / GRU 等模型训练 | ⚠️ 已跑通，但数据不足、结论不稳定 |
| **Isaac Sim 部署** | TorchScript 加载 + 闭环控制 | ❌ 仅有骨架代码，未实际验证 |

### 1.2 项目结构现状

当前存在 **三套独立训练管线**，定位不同但有功能重叠：

```
imitation-learning/     → Human→Robot 映射（GRU），输入人类姿态
vr-control-robot/       → Robot-only BC/ACT，输入机器人状态
20260331_task_policy/   → Robot+Task conditioned BC/ACT，输入机器人+任务状态
```

### 1.3 数据资产

| 采集批次 | Episodes | 任务化 | 用途 |
|----------|----------|--------|------|
| collector1 | ~5 | ❌ | 早期验证 |
| collector2 | 10 | ❌ | 跨批次测试集 |
| collector3 | 50 | ❌ | 主训练集（旧） |
| collector4 | 少量 | ❌ | 补充 |
| collector5 | 13 | ✅ | 任务化训练（当前主力） |

**核心问题：任务化数据（collector5）仅 13 个 episode，远不够支撑稳定训练。**

---

## 2. 各模块完成度评估

### 2.1 VR 前端采集系统 — 完成度 95%

**已实现（稳固）：**
- WebXR 双手控制 + IK 手臂跟随
- 任务球随机生成（5 目标/episode）
- 双手独立判定（接触/握持/成功）
- 三文件导出（session.json + episodes.jsonl + events.jsonl）
- 帧级任务观测（`obs.task`）含距离/接触/阶段/进度
- 自动归档脚本（ingest-exports.mjs）
- 配置页面（自定义骨骼模型映射）

**缺口：**
- 未实现视觉输入（相机图像），当前纯关节级
- 采集效率评估工具缺失（无法快速统计每批次质量）

### 2.2 imitation-learning 管线 — 完成度 80%

**已实现：**
- JSONL → H5 转换（含 label_shift、motion_weight）
- GRU 时序策略网络
- 跨批次评估 + 基线对比
- TorchScript 导出（含内嵌归一化）

**关键发现（问题严重）：**
- 扰动实验显示 **Human 输入几乎无效**（zeroing human → 仅 1.05× 退化）
- Robot state 是关键（zeroing robot state → 22.6× 退化）
- Human↔Robot 相关性极低（< 0.015）
- **结论：Human→Robot 映射在当前 schema 下几乎退化为"自回归 robot state"，不是真正的模仿学习。**

**根因分析：** 当前 VR 端的动作标签 `jointDelta` 是由 IK 解算得到的，不是人直接控制的。人的手柄姿态经过 IK 间接影响 `jointDelta`，导致人类观测与动作标签之间的映射非常间接，MLP/GRU 难以学到稳定关系。

### 2.3 vr-control-robot 管线 — 完成度 70%

**已实现：**
- BC（MLP）+ ACT（Transformer Encoder）训练
- Reach-oriented 分段训练（按 target segment 切分）
- Isaac Sim bridge 骨架（`IsaacPolicyBridge`）
- TorchScript 导出
- collector5_baseline 对比实验

**collector5_baseline 关键结论：**
- BC MSE = 0.0027，ACT MSE = 0.0075
- **BC 显著优于 ACT（obs_dim=48，无任务特征）**

### 2.4 20260331_task_policy 管线 — 完成度 60%

**已实现：**
- Robot+Task conditioned 特征（obs_dim=93）
- 基于 target segment 的精细数据划分
- 样本加权（成功/近距离/接触/阶段加权）
- BC vs ACT 对比训练

**结论与 collector5_baseline 矛盾：**
- ACT MSE = 0.0091，BC MSE = 0.0341
- **ACT 大幅优于 BC（73% 降低），完全相反的结论**

---

## 3. 20260331_task_policy 训练深度分析

### 3.1 实验设计评估

**优点：**
- 去除 human pose 输入的决策正确 — 部署时不会有人类输入
- 丰富的 93 维观测空间包含了任务状态信息
- 样本加权机制合理（强调关键帧）
- 基于 segment 的数据组织更符合任务逻辑

**问题清单：**

#### 问题 1：数据量严重不足

| 统计项 | 数值 | 评价 |
|--------|------|------|
| 总 episodes | 13 | 远低于 100+ 的最低推荐值 |
| 可用 segments | 22 | segment 级别很少 |
| 训练帧 | 5,300 | 对 93 维输入偏少 |
| 验证帧 | 2,187 | 仅 2 个 episode |
| 验证 episodes | 2 | **统计上不可靠** |
| 成功/失败比 | 18:4 | 严重不平衡 |

**影响：** 验证集仅有 2 个 episode（ep11、ep13），任何结论的置信度很低。切换验证 episode 可能导致完全不同的结论。

#### 问题 2：ACT 散点图出现明显的 "竖直条纹"

从 ACT 的 `pred_scatter.png` 中可以清晰看到：当 ground truth 的 jointDelta 接近 0 时，ACT 的预测值出现一个明显的 **竖直分布带**（在 x≈0 处 y 值散布在 -0.07 到 +0.06 之间）。

**原因分析：**
- 大量静止/小幅运动帧的 ground truth delta ≈ 0
- ACT 的序列 context 中积累了非零的 task state 变化（距离、阶段变化等）
- 模型在静止帧上"过度反应"，产生虚假动作预测

**部署风险：** 这意味着即使机器人应该静止时，ACT 也可能输出抖动动作，导致闭环控制不稳定。

#### 问题 3：Success 分类头效果极差

| 模型 | val_success_acc |
|------|----------------|
| BC | 26.6% |
| ACT | 36.4% |

对于二分类任务，这些数值接近甚至低于随机猜测（如果 success 占比较高，全猜 success 就能达到 ~73%）。说明辅助监督头并没有学到有意义的 representation。

**原因：** 18 个 success 段 vs 4 个 timeout 段，类别不平衡严重。`success_loss_weight=0.2` 的权重也偏低。

#### 问题 4：BC 的 val_action_mse = 0.034 远高于之前的实验

对比不同实验的 BC 结果：

| 实验 | obs_dim | val_action_mse | 数据 |
|------|---------|---------------|------|
| vr-control-robot 跨批次 (c3→c2) | 48 | 0.00140 | collector3→2 |
| collector5_baseline | 48 | 0.00271 | collector5 |
| **20260331_task_policy** | **93** | **0.03411** | **collector5** |

**同一份 collector5 数据，BC 从 0.0027 退化到 0.0341（12.6 倍退化！）**

原因分析：
1. 观测维度从 48 增加到 93，特征空间变大但数据量未增加
2. 新增的 task 特征（one-hot phase、距离、接触标志等）引入了高维稀疏信号
3. MLP 对高维稀疏输入拟合效率低，需要更多数据
4. **ACT 的 Transformer 通过注意力机制更好地捕捉了这些结构化特征**

#### 问题 5：归一化后的 action 空间分析

action 进行了 z-score 归一化 + clip（±8.0），但 `jointDelta` 的原始幅度很小（大多在 ±0.2 之间，从散点图观察）。这意味着：
- 归一化的 std 本身就很小
- clip_z=8.0 基本不起作用
- 归一化可能放大了噪声帧的权重

### 3.2 代码质量评估

**common.py — 质量较高：**
- 数据解析鲁棒，有完善的类型检查和默认值
- `_frame_is_valid` 过滤异常帧（velocity/delta 超限）
- segment 划分逻辑清晰
- 归一化与反归一化流程完整

**train_bc.py / train_act.py — 质量尚可：**
- 训练循环标准，有 early stopping
- 加权损失实现正确
- 可视化输出完善
- **缺少 learning rate scheduler**（可能影响收敛后期）
- **缺少 cross-validation**（仅单次随机划分）

**analyze_results.py — 功能完整但结论自动生成可能误导：**
- 自动比较 BC vs ACT 并给出建议
- 但未考虑上述统计可靠性问题

---

## 4. 关键矛盾与问题

### 4.1 BC vs ACT 结论不一致

| 实验 | 优胜者 | obs_dim | 条件 |
|------|--------|---------|------|
| collector5_baseline | BC | 48 | 纯 robot state |
| 20260331_task_policy | ACT | 93 | robot + task state |

**解读：**
- 当观测空间简单（48维）时，MLP 已足以在少量数据上拟合，序列模型反而过拟合
- 当观测空间复杂（93维，含结构化 task 特征）时，Transformer 的注意力机制优于 MLP
- **这不代表 ACT 一定更好，而是说 task 特征改变了问题的最优模型选择**

### 4.2 三套管线各自为战

- `imitation-learning/` 发现 human 输入无效 → 结论应引导后续放弃 human→robot 路线
- `vr-control-robot/` 和 `20260331_task_policy/` 使用不同的 obs 构造、不同的数据划分方式、不同的代码库
- 结果无法直接横向对比，增加了混乱

### 4.3 Isaac Sim 部署完全空白

- `IsaacPolicyBridge` 仅有接口定义
- 没有任何 Isaac Sim 场景配置、机器人 URDF/USD、任务环境
- 没有验证 VR 虚拟人的骨骼关节是否与 Isaac Sim 机器人关节对齐
- **这是最大的未知风险**

---

## 5. 目标可行性评估

### 5.1 "机器人自主伸手触碰目标" — 技术可行性评估

| 维度 | 评估 | 说明 |
|------|------|------|
| **数据采集** | ✅ 可行 | 采集系统成熟，schema 稳定 |
| **数据量** | ❌ 不足 | 13 episodes 远不够，需 100+ |
| **策略训练（offline MSE）** | ⚠️ 有基础但不稳定 | ACT 在 task-conditioned 下有潜力 |
| **策略训练（闭环验证）** | ❌ 未开始 | MSE 低≠部署成功，需闭环测试 |
| **Isaac Sim 部署** | ❌ 未开始 | 骨架代码，未验证 |
| **关节空间对齐** | ❓ 未验证 | VR avatar 关节 ≠ Isaac Sim 机器人关节 |

### 5.2 总体判断

**目标在技术原理上是可行的**，但当前进展距离实际达成还有显著差距。主要瓶颈不是算法或代码，而是：

1. **数据量不足** — 最紧迫
2. **未做闭环验证** — 最关键的未知
3. **Isaac Sim 落地为零** — 最大的工作量

---

## 6. 建议与改进路线

### 6.1 立即执行（本周）

#### A. 大规模数据采集

- **目标：至少 100 个 task 化 episode（collector6+）**
- 按 WORK_SUMMARY_20260325 的准则执行
- 成功率目标：50-60% success_5of5、20-30% timeout、10-20% auto_end
- 左右手平衡 4:6 ~ 6:4
- 空间多样性（左中右、远中近、高中低）

#### B. 统一训练管线

**建议合并为一套管线**，不再维护三套独立代码：
- 保留 `20260331_task_policy/` 的 Robot+Task conditioned 方案作为主线
- 从 `vr-control-robot/` 迁移 Isaac Sim bridge
- 从 `imitation-learning/` 借鉴 GRU model 作为第三对比方案
- **废弃** `imitation-learning/` 的 Human→Robot 路线（已证明 human 输入无效）

#### C. 修复验证方法

- 实现 **K-fold episode-level cross-validation**（建议 5-fold）
- 报告均值 ± 标准差，而非单次划分结果
- 这需要在 `common.py` 中修改 `split_segments_by_episode` 支持多次不同划分

### 6.2 短期执行（1-2 周）

#### D. 改进 ACT 模型

针对散点图中的 "竖直条纹" 问题，建议：
1. 增加 **action smoothing loss**：相邻帧预测的一致性约束
2. 对静止帧（`|jointDelta| < threshold`）降低 sample weight，减少 zero-dominant 样本对模型的干扰
3. 尝试 **action chunking**（预测未来多帧动作，取平均），而非仅预测当前帧
4. 添加 **learning rate cosine scheduler**

#### E. 增加 baseline 对比

当前缺少关键 baseline：
- **Oracle replay**：回放 ground truth 动作序列，测量 Isaac Sim 中的实际效果上限
- **PID controller 简单基线**：用目标位置和末端位置差做比例控制，作为 "不需要学习" 的下限
- **Random policy**：随机动作，验证指标不是 trivially easy

#### F. Isaac Sim 最小可行集成

1. 选定一个具体机器人模型（如 Franka Panda 或 UR5，或自定义人形臂）
2. 建立关节映射表：VR avatar joint names ↔ Isaac Sim joint names
3. 加载 TorchScript 模型到 Isaac Sim，在单目标 reach 任务上测试
4. 指标：成功率（末端到达目标 ±5cm 范围内）、平均到达用时、超时率

### 6.3 中期执行（3-4 周）

#### G. 迭代优化

- 根据 Isaac Sim 闭环结果反向调优训练
- 可能需要添加 domain randomization（目标位置、初始关节状态）
- 考虑训练时加 action noise 增强鲁棒性

#### H. 扩展任务复杂度

- 从单 arm reach 扩展到双 arm reach
- 增加障碍物避让
- 增加抓取（需要在 action space 加入 gripper 控制）

---

## 7. 最终结论

### 7.1 当前状态评价

项目已经完成了 **数据采集端的全套基础设施** 和 **离线训练的初步验证**，这是扎实的基础。但存在三个卡点：

1. **数据规模瓶颈**：任务化数据仅 13 个 episode，离可靠训练的 100+ episode 差距很大。当前所有 BC vs ACT 对比的结论在统计上都不可信——换一种 train/val 划分方式可能完全反转。

2. **管线碎片化**：三套独立训练管线导致精力分散、结论矛盾、维护成本高。`imitation-learning/` 的核心发现（human 输入无效）应及时反哺到决策中，不应继续投入该路线。

3. **部署验证真空**：Isaac Sim 集成尚未开始。即使离线 MSE 降到极低，也不等于闭环部署能成功。Real-time inference latency、累积误差漂移、关节空间不匹配等问题只有在 sim 中才能暴露。

### 7.2 对 20260331_task_policy 的专项结论

- **设计方向正确**：Robot+Task conditioned 输入排除了无效的 human pose，直接面向部署
- **ACT 优于 BC 的结论需谨慎接受**：验证集仅 2 个 episode，不具备统计显著性
- **ACT 散点图暴露闭环风险**：静止帧预测抖动可能导致部署时机器人颤抖
- **代码质量较高**：common.py 的数据处理逻辑鲁棒，train 脚本规范

### 7.3 "触碰目标"目标是否可实现

**可以实现，但需要按以下优先级补齐短板：**

```
[紧急] 扩充数据到 100+ episode
         ↓
[紧急] 统一管线 + 修复验证方法（K-fold CV）
         ↓
[关键] Isaac Sim 最小闭环验证（哪怕用 ground truth replay）
         ↓
[重要] 基于闭环结果迭代训练
         ↓
[可选] 模型优化（ACT action chunking / 抖动抑制）
```

**预估：如果全力推进数据采集 + Isaac Sim 集成，"reach & touch" 任务在合理的工程投入下是可以达成的。** 核心前提是确保 VR 虚拟人的关节空间和 Isaac Sim 机器人关节空间可以对齐，这一点需要尽早验证。

---

*本报告基于截至 2026-04-01 的项目全部代码、数据、训练结果和工作总结进行分析。*
