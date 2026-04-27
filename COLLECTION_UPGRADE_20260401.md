# 数据采集系统升级方案 — 2026.04.01

> **目标**：将数据采集代码升级为 Schema v3，支持多种动作表示、末端执行器速度、帧级重要性标注，  
> 为后续 IROS 论文级模型训练（Action Chunking Transformer / Diffusion Policy）打好数据基础。

---

## 一、修改内容总览

### 1. recordingManager.js — 数据采集引擎

| 修改项 | 文件位置 | 说明 |
|--------|----------|------|
| **旋转向量动作 (Rotation Vector)** | `action.jointDeltaRotVec` | 从四元数差计算 axis×angle 形式的旋转向量（3D），**无万向锁**，是现代机器人学习的标准动作表示 |
| **绝对目标四元数** | `action.jointTargetQuat` | 记录当前帧各关节四元数作为上一帧的"目标位姿"，支持绝对位置控制训练范式 |
| **末端执行器速度** | `obs.endEffectorVelocity` | 左/右手相对机器人基座的线速度 + 角速度（有限差分），为策略提供动态信息 |
| **机器人基座世界位姿** | `obs.robotBaseWorldPose` | 明确记录 `{p, q}`，便于后处理坐标变换和 Isaac Sim 对齐 |
| **帧活动标签** | `frameLabel` | 自动分类为 `idle` / `moving` / `approaching` / `contacting` / `holding`，用于训练时的重要性采样 |
| **动作幅度** | `actionMagnitude` | 所有关节旋转向量 L2 范数，快速判断该帧是否有实际运动 |
| **Schema 升级** | `schemaVersion` | `v2_human_motion_input` → `v3_multi_repr_action` |

#### 新增工具函数

```javascript
// 四元数差 → 旋转向量（axis × angle）
quatDeltaToRotVec(curQuatArr, prevQuatArr) → [rx, ry, rz]

// 帧活动分类
classifyFrameActivity(actionMagnitude, taskObs, prevDistToTarget) → string
```

#### 新的帧数据结构

```jsonc
{
  "index": 123,
  "episodeId": 1,
  "timestamp": 12345.678,
  "dt": 0.033,

  "obs": {
    // ... 原有字段不变 ...
    "endEffectorVelocity": {
      "left":  { "linearVelocity": [vx, vy, vz], "angularVelocity": [wx, wy, wz] },
      "right": { "linearVelocity": [vx, vy, vz], "angularVelocity": [wx, wy, wz] }
    },
    "robotBaseWorldPose": { "p": [x, y, z], "q": [qx, qy, qz, qw] }
  },

  "action": {
    "jointDelta":       { /* Euler delta (legacy) */ },
    "jointDeltaRotVec": { /* Rotation vector delta (preferred) */ },
    "jointTargetQuat":  { /* Absolute target quaternion */ },
    "gripperCommand":   { "left": 0.0, "right": 0.0 }
  },

  "actionMagnitude": 0.0234,
  "frameLabel": "approaching"
}
```

### 2. common.py — 训练数据处理

| 修改项 | 说明 |
|--------|------|
| **多动作表示支持** | 新增 `action_repr` 参数：`"euler_delta"` (默认/向后兼容) / `"rot_vec"` (推荐) / `"target_quat"` (绝对控制) |
| **末端执行器速度特征** | 新增 `include_ee_velocity` 参数，为每只手添加 6 维速度特征 (3 线速度 + 3 角速度) |
| **帧级重要性采样** | 利用 `frameLabel` 字段：`idle` 帧权重 ×0.3 衰减，`approaching` 帧 +0.15 奖励 |
| **'hold' 相位** | PHASES 列表增加 `hold`（8 维 one-hot），obs_dim 相应变化 |
| **目标坐标系修正** | `targetPose`(世界坐标) → `targetRelToRobotBase`(机器人基座相对坐标)，保留 fallback |
| **feature name 升级** | `target_pose.p.*` → `target_rel_base.p.*`，新增 `ee_vel.*` 特征名 |

### 3. config.json — 训练配置

```json
{
  "data": {
    "include_ee_velocity": true,
    "action_repr": "rot_vec",
    "sample_weighting": {
      "idle_discount": 0.3,
      "moving_bonus": 0.05,
      "approaching_bonus": 0.15,
      "max_weight": 3.0
    }
  }
}
```

### 4. train_bc.py / train_act.py — 训练脚本

- 更新 `build_feature_names()` 和 `build_action_names()` 调用，透传新参数
- obs_dim / act_dim 仍然从数据 shape 动态推导，无需手动修改

---

## 二、数据维度变化

### 新 Observation 维度（默认配置）

| 分组 | 特征 | 维度 |
|------|------|------|
| 关节位置 (Euler) | 8 joints × 3 | 24 |
| 关节速度 (Euler) | 8 joints × 3 | 24 |
| 末端执行器位姿 | 2 hands × (3 pos + 4 quat) | 14 |
| **末端执行器速度** ★ | 2 hands × (3 linear + 3 angular) | **12** |
| 目标相对基座位置 | 3 | 3 |
| 目标相对左/右手位置 | 2 × 3 | 6 |
| 距离特征 | 3 | 3 |
| 接触标志 | 3 | 3 |
| 保持时间 | 3 | 3 |
| 进度 | 3 | 3 |
| 最近手 one-hot | 3 | 3 |
| 相位 one-hot | 8 (增加 hold) | 8 |
| **总计** | | **106** |

### Action 维度

| 表示 | 维度 | 说明 |
|------|------|------|
| `euler_delta` | 24 | 8 joints × 3 (Euler 差分，向后兼容) |
| `rot_vec` ★ | 24 | 8 joints × 3 (旋转向量，**推荐**) |
| `target_quat` | 32 | 8 joints × 4 (绝对目标四元数) |

---

## 三、为什么选择旋转向量 (Rotation Vector)？

1. **无万向锁 (Gimbal Lock)**：Euler 角在 ±90° 附近有奇异性，旋转向量完全避免
2. **紧凑**：3 维表示，与 Euler delta 相同维度
3. **连续**：小旋转时近似线性，数值稳定
4. **Isaac Sim 友好**：Isaac Sim 内部使用四元数，旋转向量可直接转换
5. **标准**：ACT、Diffusion Policy 等 SOTA 方法均推荐此表示

---

## 四、数据采集操作指南

### 准备工作

1. 确保已部署最新代码（包含本次修改）
2. 使用支持 WebXR 的 VR 设备（Meta Quest 3 推荐）
3. 启动开发服务器：`npm run dev`

### 采集流程

#### 单次采集会话

1. 佩戴 VR 设备，打开浏览器进入 VR 界面
2. 进入 VR 模式后，系统自动开始录制
3. 按下 **左手柄 Trigger** 开始新 Episode
4. 每个 Episode 包含 5 个目标球，依次触达即可
5. 操作要点：
   - **自然、流畅地移动**双手去接触目标球
   - 避免突然静止不动（减少 idle 帧比例）
   - 尽量用不同路径到达目标（增加数据多样性）
   - 超时 (20s) 自动结束当前 Episode
6. 完成后，系统提示下载数据集，确认即可

#### 采集数量目标

| 阶段 | 最少 Episodes | 用途 |
|------|--------------|------|
| 验证阶段 | 3-5 | 确认新数据格式正确，无异常值 |
| 初步训练 | 50-80 | 初步验证模型可行性 |
| 正式训练 | **150-300** | IROS 论文级实验数据量 |
| 充分训练 | 500+ | 最佳性能（如时间允许） |

#### 数据质量检查要点

采集 3-5 个 Episode 后，用以下 Python 命令快速验证：

```python
import json, math

f = open('vr-demonstrations-episodes-XXXXXX.jsonl', 'r', encoding='utf-8')
frames = [json.loads(l) for l in f if l.strip()]
f.close()

# 1. 检查是否有 v3 字段
print('Has rotVec?', 'jointDeltaRotVec' in frames[10].get('action', {}))
print('Has eeVelocity?', 'endEffectorVelocity' in frames[10].get('obs', {}))
print('Has frameLabel?', 'frameLabel' in frames[10])
print('Has robotBaseWorldPose?', 'robotBaseWorldPose' in frames[10].get('obs', {}))
print('Has targetRelToRobotBase?', 'targetRelToRobotBase' in frames[10].get('obs',{}).get('task',{}))

# 2. 检查 frameLabel 分布
from collections import Counter
labels = Counter(f.get('frameLabel','?') for f in frames)
print('Frame labels:', dict(labels))
# 目标：idle < 40%, moving+approaching > 50%

# 3. 检查旋转向量范围
import numpy as np
mags = []
for fr in frames:
    rv = fr.get('action',{}).get('jointDeltaRotVec',{})
    for j in rv.values():
        mags.append(np.linalg.norm(j))
mags = np.array(mags)
print(f'RotVec mag: mean={mags.mean():.4f}, max={mags.max():.4f}, >0.5={np.sum(mags>0.5)}')
# 如果 max > 1.0 说明有异常大的运动
```

---

## 五、后续训练路线图

### 第一阶段：数据验证 + 基线复现

- [ ] 采集 3-5 个 Episode，验证新数据格式
- [ ] 用 `action_repr: "rot_vec"` 重跑 BC/ACT 基线
- [ ] 对比旧 Euler delta 和新 RotVec 在相同数据上的训练效果

### 第二阶段：扩大数据 + 模型创新

- [ ] 采集 150+ Episodes（5-8 小时采集时间）
- [ ] 实现 **Phase-Aware Action Chunking Transformer (PA-ACT)**:
  - 创新点 1：**层次化任务阶段引导** — 将相位信息作为 cross-attention 条件注入 Transformer 解码层，而非简单拼接到输入
  - 创新点 2：**自适应动作块长度** — 根据当前相位动态调整 action chunk 长度（reach 阶段用长 chunk 做粗规划，contact 阶段用短 chunk 做精细控制）
  - 创新点 3：**接触预测辅助任务** — 联合预测"距离接触还有多少帧"，提供梯度信号引导策略学习接触时机
- [ ] 实现 **Task-Conditioned Diffusion Policy** 对比实验

### 第三阶段：Isaac Sim 部署 + 仿真验证

- [ ] 将训练好的策略导出为 TorchScript
- [ ] 在 Isaac Sim 中搭建相同的机器人 + 目标球环境
- [ ] 实现 `IsaacPolicyBridge`：obs 构建 → 策略推理 → 关节控制
- [ ] 记录成功率、到达时间、轨迹平滑度指标

### 第四阶段：论文撰写

- [ ] 实验：BC vs ACT vs PA-ACT vs Diffusion Policy（四方法对比）
- [ ] 消融实验：动作表示（Euler vs RotVec vs TargetQuat）、帧重要性采样、相位引导
- [ ] 可视化：attention map、动作轨迹、成功/失败案例分析
- [ ] 投稿目标：**IROS 2027** 或 **CoRL 2027**

---

## 六、模型创新方向详解

### PA-ACT: Phase-Aware Action Chunking Transformer

**核心思想**：传统 ACT 将所有观测扁平拼接后送入 Transformer，忽略了任务阶段信息的结构化特性。我们提出将相位作为显式的层次化条件：

```
                    ┌─────────────────┐
                    │  Phase Encoder  │ ← one-hot phase + distance features
                    │  (learnable)    │
                    └────────┬────────┘
                             │ phase embedding
                             ▼
┌──────────┐     ┌──────────────────────┐     ┌──────────────┐
│ Obs       │────▶│ Causal Transformer   │────▶│ Action Chunk │
│ Encoder   │     │ + Phase Cross-Attn   │     │ Decoder      │
└──────────┘     └──────────────────────┘     └──────────────┘
     ▲                                              │
     │                                              ▼
 [joint_pos, joint_vel,         [a_{t}, a_{t+1}, ..., a_{t+K-1}]
  ee_pose, ee_vel,               K = f(phase)  ← adaptive chunk length
  target_rel, ...]
```

**论文卖点**：
1. 首次在 VR 遥操作模仿学习中引入相位感知的动作分块
2. 自适应 chunk 长度直接建模了"粗到精"的运动规划策略
3. 接触预测辅助任务提供了明确的奖励信号，解决了稀疏奖励问题

### 与现有工作的区别

| 方法 | Action Repr | Temporal | Phase-Aware | Contact Pred |
|------|-------------|----------|-------------|--------------|
| BC (baseline) | single-step | ✗ | ✗ | ✗ |
| ACT (Zhao et al.) | fixed chunk | ✓ | ✗ | ✗ |
| Diffusion Policy | fixed chunk | ✓ | ✗ | ✗ |
| **PA-ACT (ours)** | **adaptive chunk** | **✓** | **✓** | **✓** |

---

## 七、关于旧数据 (collector5)

collector5 的 13 个 Episode 使用旧 schema (v2) 采集，存在以下限制：

1. **无 `jointDeltaRotVec`** — 但可从 `jointQuaternions` 在训练时后处理计算
2. **无 `endEffectorVelocity`** — 可从连续帧的 `endEffector` 位姿差分计算  
3. **无 `frameLabel`** — 可根据 `jointDelta` 幅度和 task 状态重建
4. **无 `targetRelToRobotBase`** — training code 已做 fallback 到 `targetPose`
5. **Euler wrapping bug** — 17 帧有 >3.0 rad 的异常 delta

**建议**：collector5 数据仅用于初步格式验证和测试流程。正式训练应全部使用新采集的 v3 数据。

---

## 修改文件清单

| 文件 | 修改类型 | 主要变更 |
|------|----------|----------|
| `src/utils/recordingManager.js` | 新增功能 | 旋转向量动作、EE 速度、帧标签、基座位姿 |
| `20260331_task_policy/common.py` | 扩展接口 | 多动作表示、EE 速度特征、帧重要性采样 |
| `20260331_task_policy/config.json` | 更新配置 | `action_repr`、`include_ee_velocity`、新权重参数 |
| `20260331_task_policy/train_bc.py` | 适配调用 | 透传新参数 |
| `20260331_task_policy/train_act.py` | 适配调用 | 透传新参数 |
