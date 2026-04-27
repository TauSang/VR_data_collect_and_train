# 训练模型详细介绍 — 2026.04.01

> 本文档详细说明 `20260331_task_policy/` 中两个训练模型（**BC** 和 **ACT**）的输入输出、网络结构、训练策略及数据处理全流程。

---

## 一、整体架构概览

```
VR 采集 (30Hz)
    │ episodes.jsonl / events.jsonl
    ▼
common.py —— 数据加载 & 特征提取
    │  SegmentData { obs, act, weights }
    ├── train_bc.py ──→ TaskBCMLP      (单帧 → 动作)
    └── train_act.py ──→ TaskACTEncoder (时序 → 动作)
```

两个模型均为**模仿学习（Imitation Learning）**框架，从 VR 人类示范数据中学习机器人手臂控制策略。

---

## 二、数据来源与预处理

### 2.1 原始数据

| 属性 | 值 |
|------|----|
| 采集设备 | Meta Quest 3 VR 头显 |
| 采样频率 | 30 Hz |
| 任务 | `reach_hold` / `reach_touch_grasp`（末端执行器触达目标球） |
| 数据格式 | Schema v3，JSONL 格式 |
| 关节数量 | 8 个（左右肩 / 上臂 / 前臂 / 手） |

### 2.2 数据分段

原始 JSONL 按 `(episodeId, targetIndex, targetId)` 三元组切分为**片段（Segment）**，每个片段对应一次"接近目标球"的尝试。

- `allowed_outcomes`：只保留 `success` 和 `timeout` 的片段（丢弃截断数据）
- **训练/验证切分**：按 **Episode** 级别 80/20 随机划分（避免同一 Episode 跨集）

### 2.3 帧级重要性权重

不同帧的质量差异显著，训练时使用**加权损失**而非等权 MSE：

| 条件 | 权重调整 |
|------|----------|
| `frameLabel == "idle"` | × 0.3（降低静止帧权重） |
| `frameLabel == "moving"` | + 0.05 |
| `frameLabel == "approaching"` | + 0.15（靠近目标，重要） |
| 接触状态 (`contactFlag`) | + 0.35（接触帧极重要） |
| 末端执行器距目标 < 0.25m | + 0.25（近距重要） |
| 片段结果为 `success` | + 0.1 |
| 上限 | max_weight = 3.0 |

---

## 三、模型输入：Observation 向量（106 维）

> 配置：`use_joint_velocities=true`，`include_end_effector_quat=true`，`include_phase=true`，`include_ee_velocity=true`

所有特征经 **Z-score 归一化**（均值/标准差从训练集拟合），推断时超出 ±8σ 的值截断。

| 特征组 | 特征名示例 | 维度 | 说明 |
|--------|-----------|------|------|
| **关节位置** (Euler) | `joint_pos.leftShoulder.x/y/z` | 8×3 = **24** | 8 个关节的局部 Euler 角（弧度） |
| **关节速度** (Euler) | `joint_vel.leftUpperArm.x/y/z` | 8×3 = **24** | (当前 − 上帧) / dt，弧度/秒 |
| **末端执行器位姿** | `ee.left.p.x`，`ee.left.q.x/y/z/w` | 2×(3+4) = **14** | 左/右手相对机器人基座的位置(3) + 四元数(4) |
| **末端执行器速度** ★ | `ee_vel.left.linear.x`，`ee_vel.left.angular.x` | 2×(3+3) = **12** | 有限差分线速度(3) + 角速度(3)，Schema v3 新增 |
| **目标相对基座** | `target_rel_base.p.x/y/z` | **3** | 目标球相对机器人基座的位置 |
| **目标相对左/右手** | `target_rel_left.p.x/y/z` | 2×3 = **6** | 目标在两只手坐标系下的位置 |
| **距离特征** | `dist.target`，`dist.target_left/right` | **3** | 目标到最近手/左手/右手的距离 |
| **接触标志** | `contact.any/left/right` | **3** | 当前帧是否接触目标（0/1） |
| **保持时间** | `hold.any_sec/left_sec/right_sec` | **3** | 接触持续时间（秒） |
| **进度** | `progress.completed_targets`，`ratio` | **3** | 已完成目标数 / 总目标数 / 比例 |
| **最近手 one-hot** | `nearest_hand.none/left/right` | **3** | 哪只手离目标最近 |
| **任务相位 one-hot** | `phase.idle/reach/align/grasp/hold/...` | **8** | 当前任务阶段 (8 类) |
| **合计** | | **106** | |

---

## 四、模型输出：Action 向量（24 维）

### 4.1 动作表示：旋转向量（Rotation Vector）★ 推荐

> 当前配置 `action_repr: "rot_vec"`

```
动作 = 8 个关节的旋转向量（axis × angle）
维度：8 关节 × 3 = 24
```

**旋转向量的计算**（相邻帧四元数差）：

$$\Delta q = q_{t}^{-1} \otimes q_{t+1}$$

$$\text{rotVec} = \text{axis} \times \theta, \quad \theta = 2 \arctan2(|\vec{v}|, w)$$

**为什么不用 Euler Delta？**

| 对比 | Euler Delta | 旋转向量 (rot_vec) |
|------|------------|-------------------|
| 万向锁 | 有奇异性（±90° 附近） | **无** |
| 连续性 | 不连续跳变 | **小角度近似线性** |
| 维度 | 3 | 3（相同） |
| Isaac Sim 对齐 | 需二次转换 | **可直接转四元数** |

### 4.2 其他可选动作表示

| 表示 | 维度 | 适用场景 |
|------|------|---------|
| `euler_delta` | 24 | 旧版，向后兼容 |
| `rot_vec` | 24 | **当前推荐**，无万向锁 |
| `target_quat` | 32 (8×4) | 绝对目标控制，适合位置控制器 |

### 4.3 辅助输出：成功概率

两个模型均同时输出一个**标量**成功概率（sigmoid 激活），用于辅助监督：

```
success_prob ∈ [0, 1]
含义：策略预测当前片段是否以成功结束
```

---

## 五、模型一：TaskBCMLP（行为克隆 BC）

### 5.1 策略：单帧行为克隆（Behavior Cloning）

最基础的模仿学习：直接从**单帧观测**映射到**当前帧动作**，无时序上下文。

### 5.2 网络结构

```
输入: obs [B, 106]
  │
  ├─ Linear(106 → 256) → LayerNorm → GELU → Dropout(0.1)
  ├─ Linear(256 → 256) → LayerNorm → GELU → Dropout(0.1)
  └─ Linear(256 → 128) → LayerNorm → GELU → Dropout(0.1)
       │
       ├── action_head: Linear(128 → 24) ── 输出: 预测动作 [B, 24]
       └── success_head: Linear(128 → 1) ─ 输出: 成功概率 logit [B]
```

**参数量估算**：约 **130K** 参数

### 5.3 输入输出总结

```
输入:  obs_t         形状 [B, 106]  — 当前帧观测（归一化后）
输出:  pred_act_t    形状 [B, 24]   — 预测的旋转向量动作（归一化空间）
       pred_success  形状 [B]       — 预测成功概率 logit
```

### 5.4 训练细节

| 超参数 | 值 |
|--------|----|
| 损失函数 | `weighted_MSE(动作) + 0.2 × weighted_BCE(成功)` |
| 优化器 | AdamW，lr=3e-4，weight_decay=1e-5 |
| Batch size | 128 |
| 最大 Epoch | 80 |
| 早停 patience | 12 epoch（监控 val_action_mse） |
| 梯度裁剪 | 最大范数 1.0 |

---

## 六、模型二：TaskACTEncoder（动作分块 Transformer / ACT）

### 6.1 策略：时序 Transformer（Action Chunking Transformer）

ACT 是截止当前机器人操控模仿学习的 **SOTA 方法**。

核心思想：

- **时序感知**：输入过去 `seq_len=16` 帧的观测序列（~533ms @ 30Hz），利用 Transformer 建模时序依赖
- **Action Chunking**：预测**当前时刻**（最后一帧）的动作，而非在输出序列上展开（此处为 Encoder-only 简化版）
- **位置编码**：正弦余弦 Positional Encoding 注入帧的时序位置信息

### 6.2 为什么比 BC 更好？

| 能力 | BC (MLP) | ACT (Transformer) |
|------|---------|-------------------|
| 时序上下文 | ✗（只看当前帧） | ✓（看过去 16 帧） |
| 速度感知 | 部分（joint_vel 特征） | 全上下文轨迹感知 |
| 预测稳定性 | 帧间波动大 | Attention 平滑输出 |
| 复杂运动理解 | 弱 | 强 |

### 6.3 网络结构

```
输入: obs_seq [B, 16, 106]   — 16 帧观测序列（归一化后）
  │
  ├─ input_proj: Linear(106 → 128)         → [B, 16, 128]
  ├─ PositionalEncoding(d_model=128)        → [B, 16, 128]
  └─ TransformerEncoder (2 层)
       每层: Pre-LN + MultiHeadAttention(4头) + FFN(256) + GELU + Dropout(0.1)
           → [B, 16, 128]
  
  取最后一个 Token: z = encoder_output[:, -1, :]  → [B, 128]
  │
  ├── action_head:  LayerNorm(128) → Linear(128 → 24)  输出: 预测动作 [B, 24]
  └── success_head: LayerNorm(128) → Linear(128 → 1)   输出: 成功概率 logit [B]
```

**Transformer 配置**：

| 参数 | 值 |
|------|----|
| `d_model` | 128 |
| `nhead` | 4 |
| `num_layers` | 2 |
| `dim_feedforward` | 256 |
| `activation` | GELU |
| 归一化策略 | Pre-LayerNorm（更稳定） |
| `seq_len` | 16 帧（约 533ms） |

**参数量估算**：约 **300K** 参数

### 6.4 输入输出总结

```
输入:  obs_seq_{t-15:t}  形状 [B, 16, 106]  — 过去 16 帧观测序列
输出:  pred_act_t        形状 [B, 24]        — 当前帧预测动作（归一化空间）
       pred_success      形状 [B]            — 成功概率 logit
```

### 6.5 训练细节

| 超参数 | 值 |
|--------|----|
| 损失函数 | `weighted_MSE(动作) + 0.2 × weighted_BCE(成功)` |
| 优化器 | AdamW，lr=3e-4，weight_decay=1e-5 |
| Batch size | 128 |
| 最大 Epoch | 80 |
| 早停 patience | 12 epoch（监控 val_action_mse） |
| 梯度裁剪 | 最大范数 1.0 |
| 序列构建 | 滑动窗口，步长 1 帧 |

---

## 七、损失函数详解

### 7.1 动作损失（加权 MSE）

$$\mathcal{L}_{\text{action}} = \frac{\sum_i w_i \cdot \text{MSE}(\hat{a}_i, a_i)}{\sum_i w_i}$$

- $w_i$：帧级重要性权重（见第二节）
- $a_i$：归一化后的旋转向量动作（24 维）
- 评估指标：action_mse、action_mae

### 7.2 成功预测损失（加权 BCE）

$$\mathcal{L}_{\text{success}} = \frac{\sum_i w_i \cdot \text{BCE}(\hat{s}_i, s_i)}{\sum_i w_i}$$

- $s_i \in \{0, 1\}$：该片段是否成功
- 评估指标：success_bce、success_acc

### 7.3 总损失

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{action}} + 0.2 \times \mathcal{L}_{\text{success}}$$

---

## 八、归一化与推断

### 8.1 归一化

- **方法**：Z-score（均值=0，标准差=1）
- **拟合来源**：仅在训练集上计算 obs_mean、obs_std、act_mean、act_std
- **存储**：随检查点一起保存为 `normalizer` 字典
- **异常值裁剪**：超出 obs_clip_z=8.0σ / act_clip_z=8.0σ 的值截断

### 8.2 推断流程

```python
# 1. 将实时观测构造为同样的 106 维向量
obs = frame_to_robot_task_obs_act(frame, ...)

# 2. Z-score 归一化
obs_z = (obs - normalizer.obs_mean) / normalizer.obs_std

# ACT: 维护滑动窗口
obs_window.append(obs_z)              # 保持 16 帧
obs_seq = np.stack(obs_window)        # [16, 106]

# 3. 模型推断
pred_act_z, _ = model(obs_seq)        # [24]

# 4. 反归一化 → 真实旋转向量
pred_act = pred_act_z * normalizer.act_std + normalizer.act_mean

# 5. 旋转向量 → 四元数 → 关节控制指令
```

---

## 九、模型对比总结

| 维度 | TaskBCMLP (BC) | TaskACTEncoder (ACT) |
|------|---------------|----------------------|
| **策略类型** | 单帧行为克隆 | 时序 Transformer 编码器 |
| **输入** | `[B, 106]` 单帧 | `[B, 16, 106]` 16帧序列 |
| **输出** | `[B, 24]` + success | `[B, 24]` + success |
| **参数量** | ~130K | ~300K |
| **时序感知** | ✗ | ✓（约 533ms 上下文） |
| **优点** | 简单快速，易调试 | 时序平滑，性能更强 |
| **缺点** | 帧间抖动，无历史感知 | 需要维护滑动窗口 |
| **动作表示** | 旋转向量 (rot_vec) | 旋转向量 (rot_vec) |
| **训练文件** | `train_bc.py` | `train_act.py` |
| **输出目录** | `outputs/bc/` | `outputs/act/` |

---

## 十、文件结构

```
20260331_task_policy/
├── config.json          # 所有超参数配置（数据/模型/训练）
├── common.py            # 数据加载、特征提取、归一化工具
├── train_bc.py          # BC 训练入口
├── train_act.py         # ACT 训练入口
├── run_all.py           # 同时运行 BC + ACT
└── outputs/
    ├── bc/run_XXXXXX/
    │   ├── checkpoints/best.pt   # 最优模型（含 normalizer）
    │   └── metrics.json
    └── act/run_XXXXXX/
        ├── checkpoints/best.pt
        └── metrics.json
```

---

*生成日期：2026-04-01 | 项目：vr-robot-control | Schema v3*
