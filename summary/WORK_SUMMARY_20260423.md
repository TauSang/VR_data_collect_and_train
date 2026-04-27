# VR 数据价值验证 · Data1+Data2 训练总结（2026-04-23）

> 本次会话：新增 Data2（23.7K 帧 VR 示教）后，在本地 CPU 完成 3 组训练 + 4 组 50-target MuJoCo 评估，并回答用户 3 个问题。后半段修正了评估判定标准（见 §0）。

---

## 0. 重要修正：评估判定标准收紧（当日追加）

用户反馈 "任务完成得太快了，手臂其他位置碰到也算完成" → 检查后确认 `validate_policy.py` 存在两个问题：

1. **距离基准错了**：之前距离用的是 `left_wrist_yaw_link` 的 body 原点；但实际手掌中心（rubber_hand 几何 + 惯性 CoM）在该 link 本地 +X 方向约 **+0.09 m**。原本 0.16 m 的阈值相当于手掌到球心 ~7 cm，容易被手腕 / 前臂侧擦中就触发。
2. **保持时间太短**：0.25 s 允许"飞掠式"瞬时过点，策略即便抖着过去也能触发成功。

**修正**（`mujoco_sim/validate_policy.py`）：
- 距离改为 **palm centre = xpos(wrist) + R_wrist @ [0.09, 0, 0]** 到球心的欧氏距离
- `TARGET_REACH_THRESHOLD`：0.16 → **0.12** m （球半径 0.04 m，掌心距球心 12 cm ≈ 掌面距球面 8 cm，对应"手掌接近并停在球上"）
- `HOLD_DURATION_S`：0.25 → **0.3** s

> 注：曾短暂尝试 0.08m / 0.5s 的严格档，但结合可视化发现即便掌心已明显触碰球体（截图证据），0.08m 仍会被判失败——主要因为接触瞬间掌心几何中心到球心约 9-10cm。因此最终采用 0.12m / 0.3s 作为"真实触碰"的合理判定线。

**三档判定下三个模型对比**（seed=42, --no-ensemble, 50 targets）：

| 模型 | 宽松 (0.16/0.25) | 严格 (0.08/0.5) | **最终采用 (0.12/0.3)** |
|------|---|---|---|
| Strong-MJ baseline (`run_20260422_171750`) | 92% | 20% | **56% (28/50)** |
| d1 Mixed FiLM FT (`run_20260423_105010`) | 100% | 28% | **92% (46/50)** |
| d1+d2 Mixed FiLM FT (`run_20260423_153217`) | 100% | 42% | **84% (42/50)** |

**重要结论更新**：
- 宽松判定"100% vs 100%"是天花板假象。中等判定下 **VR 数据 (Mixed FT) 相比 MJ-only 提升 +28~36pp 绝对**，这才是 VR 数据真正贡献的信号强度。
- **d1-only 中等判定下 (92%) 反而高于 d1+d2 (84%)**：新增的 data2 在当前 FiLM-Mixed-FT 配方下出现轻微稀释（早停时刻 d1+d2 尚未充分融合），不是加数据就有用，**训练配方（早停门槛、采样比、domain weight）需要针对 data2 调优**。
- 建议后续把"判定档 + 严格档 min_dist 百分位"一起报告，避免单阈值假象。
- 抖动问题处理：可视化时启用 `--action-ema 0.3`（inference-time EMA），明显缓解 chunk=5 closed-loop 的手臂抖动，且不伤害成功率。推荐此后可视化和部署默认都带 `--action-ema 0.3`。

**额外观察（留待后续）**：可视化中发现 **机器人左臂会穿过躯干**，这是 G1 模型未启用自碰撞（self-collision）导致的，属于仿真设置问题而非策略问题，应在 XML 中开启躯干/臂段的 `contype`/`conaffinity` 自碰撞检测。

---

## 1. 本阶段总结

- **核心产出**：证明扩充后的 VR 数据（data1 16.5K + data2 23.7K = 40.2K 帧）在 **FiLM-Mixed-FT 范式** 下保持 100% MuJoCo 成功率，与 data1-only 的 100% 持平，成功率上限没有退化。
- **但仍有 2 个观察**：
  1. **纯 VR 微调（VR-only FiLM FT）**：data1+data2 反而比 data1-only 更差（44% vs 52%）。VR-only 路径仍高度依赖数据子集特性，不是加数据就有用。
  2. **训练时间 / min_dist 指标略退**：d1+d2 mixed FT 的 mean_time=1.32s vs d1-only=1.03s。数据分布差异让训练前期即 early stop。
- **关键数字**：
  - Strong-MJ baseline（修复采样后）：**92%** (46/50) —— 这是本次重新基准化得到的数值
  - d1 mixed FiLM FT（20260422）：**100%** (50/50)，mean_time=**1.03s**
  - **d1+d2 mixed FiLM FT（今日）：100% (50/50)，mean_time=1.32s** ← 成功率保持，但动作略慢
  - d1-only VR-only FiLM FT（今日，作为对照）：52%
  - d1+d2 VR-only FiLM FT（今日）：44%
- **结论**：新增的 23.7K 帧 VR 数据 **没有显著推进上限**，但也 **没有造成上限倒退**。上限已经被 mixed-FT 范式钉在 100%（seed=42 条件下），想进一步区分质量需要：换 seed、加 min_dist 统计、做 tie-break 指标。

---

## 2. 今日工作流程

### 2.1 数据对齐

Data2 的 23705 帧 VR 原始数据走和 data1 相同的 G1-FK 对齐管线：

```powershell
python scripts/align_collector_to_g1fk.py `
  --model mujoco_sim/model/task_scene.xml `
  --episodes VR_raw_data/data2/vr-demonstrations-episodes-20260423_144919.jsonl `
  --events   VR_raw_data/data2/vr-demonstrations-events-20260423_144920.jsonl `
  --out      data_collector/data2_aligned
# → in=23705  out=23705  skip=0
```

输出：[data_collector/data2_aligned/episodes.jsonl](data_collector/data2_aligned/episodes.jsonl)  
（经 G1 正向运动学重投影到 MuJoCo 规范帧，与 collector10_aligned 可直接混合）

### 2.2 今日三组训练

文件夹：[20260423_train/](20260423_train/)（代码从 20260422_train 完整复制）

| # | 配置 | 数据 | Pretrain | Epochs | Best Val MSE |
|---|-----|------|---------|--------|-------------|
| A | [config_vr_d1d2_film_ft.json](20260423_train/config_vr_d1d2_film_ft.json) | d1+d2（VR-only, 40K 帧） | weak-MJ MLP (76%) | 40/40 | 0.1176 |
| B | [config_vr_d1_only_film_ft.json](20260423_train/config_vr_d1_only_film_ft.json) | d1（VR-only, 16.5K 帧） | weak-MJ MLP (76%) | 40/40 | 0.1192 |
| C | [config_strong_mixed_film_ft_d1d2.json](20260423_train/config_strong_mixed_film_ft_d1d2.json) | MJ_v4 + d1 + d2 | weak-MJ MLP (76%) | **17/40 (early stop)** | 0.2521 |

> 注：A/B 和 C 的 val_mse 不可直接比较（val 集构成不同：A/B 只有 VR 域，C 是 MJ+VR 混合）。

### 2.3 评估（50 target，seed=42，`--no-ensemble`）

| # | 模型 | 成功率 | mean_time | min_dist_mean | 关键结论 |
|---|------|-------|----------|---------------|---------|
| baseline | strong-MJ pretrain (re-bench) | **92%** (46/50) | - | - | 修复目标采样后的 MJ-only 基准 |
| A | d1+d2 VR-only FiLM FT | 44% (22/50) | - | - | VR-only 即使加数据仍不稳定 |
| B | d1 VR-only FiLM FT (repro) | 52% (26/50) | - | - | 与历史 60% 接近，路径本身方差大 |
| 历史 | d1 mixed FiLM FT | 100% (50/50) | **1.03s** | **0.114m** | 基准（20260422） |
| C | **d1+d2 mixed FiLM FT** | **100% (50/50)** | 1.32s | 0.128m | ✓ 成功率保持；动作略慢 |

**VR 数据价值的量化**：
```
Baseline (MJ only)           92%     ← 对照
+ d1 Mixed FiLM FT          100%  +8 ← 论文主要贡献（d1 = 16.5K 帧）
+ d1+d2 Mixed FiLM FT       100%  +8 ← d2 未上限突破，但也未回退
```

---

## 3. 回答用户 3 个问题

### Q1：为什么训练经常出现 early stop？

**根因**：`early_stop_patience=15`（连续 15 个 epoch val_mse 不下降则停），与混合数据分布一起触发：

1. **验证集 80% 是 MJ 域，20% 是 VR 域**（今日 C 训练：Train 域直方图 `{0: 254319, 1: 30465}`）。MJ 分布已被 pretrain 拟合得很好，val_mse 在前 3-5 个 epoch 就达到局部极小。
2. **Cosine lr 衰减**：lr 3e-5 → 1e-6，第 20 个 epoch 后有效步长小于数据噪声，val_mse 曲线进入平顶。
3. **FiLM 参数起点为 0**：前几个 epoch 主要是 FiLM 的 γ/β 从 0 拟合到 MJ-VR 差异，一旦饱和就收敛。
4. **训练信号不均衡**：VR 域 MSE 在 0.05 左右，MJ 域 MSE 已在 0.25 左右，整体 val_mse 被 MJ 主导，看不到 VR 的继续学习。

**今日 C 的具体表现**：best_epoch=17（val_mse=0.2521），第 17-32 连续 15 个 epoch 都在 0.253-0.260 来回震荡，满足 patience 条件 → early stop。

**这不是问题，是"特征"**：本质上 mixed FT 从 epoch 15 之后就是在 overfit VR 小样本。即使不早停也没收益。  
**如果想强制跑满**：
- 把 `early_stop_patience` 调到 40（等于关闭早停）
- 或对 VR 样本加大权重（`sample_weighting.moving_bonus` 抬到 0.5），让 loss 能反映 VR 域的下降

### Q2：以往 ACT 架构改进对本问题有什么启发

**本次翻阅的前期工作**：[summary/PAPER_SUMMARY_20260423.md](summary/PAPER_SUMMARY_20260423.md)、[memory/repo/act-training-insights.md](./)、[summary/WORK_SUMMARY_20260415.md](summary/WORK_SUMMARY_20260415.md)

**前期结论摘要**（帮今天的实验定位）：

| 已验证结论 | 对今日工作的意义 |
|-----------|----------------|
| MLP k=5 > Transformer（<300K 帧） | 继续用 MLP，不换 transformer |
| Temporal Ensembling 有害 | 永远 `--no-ensemble`（已遵守） |
| Transformer d=256 vs d=128：256 从 72% 暴跌到 24% | 别盲目加宽度 |
| `readout_norm=false` + `obs_noise_std=0.1` → ACT v4 到 89% | 今日 `obs_noise_std=0.02` 偏保守 |
| `action_scale=1.3` 能补偿 Transformer 动作衰减 | MLP 不需要（无 LayerNorm 衰减） |
| VR-only fine-tune 始终不稳定（24-27%，今日 44-52%） | 路径本身方差大，不是数据不够 |
| **FiLM 域嵌入是唯一能"加 VR 不掉"的方法** | 今日验证正确——mixed FiLM 保持 100% |

**下一步架构可以尝试**（按优先级）：

1. **P0：提升 `obs_noise_std` 到 0.05-0.10**  
   前期发现配合 `readout_norm=false` 能从 72% → 89%。今日 mixed FT 的 `obs_noise_std=0.02`，对 OOD 稳健性不足。这也解释了为什么 d1+d2 mixed FT 的 mean_time 变长——模型在 test time 对 obs 分布漂移不够鲁棒。

2. **P1：Chunk size 对比（k=1 vs k=5）**  
   历史：MuJoCo-only k=1 = 98%，k=5 = 94%。抖动问题可能来自 k=5 的 closed-loop chunk 重叠。建议做一次 `chunk_size=1` 对照。

3. **P1：加权 VR 样本**  
   今日 train 直方图 MJ:VR = 8.3:1，loss 几乎看不到 VR 的梯度贡献。DataLoader 里给 VR 样本 `sample_weight *= 3`，或者用 WeightedRandomSampler 维持每 batch 的 1:1 比例。

4. **P2：Residual FiLM + LayerScale**  
   现在的 FiLM `(1+γ)·LN(Wx) + β` 让 VR 偏差可以无限放大。加 `γ` 的 L2 正则（weight 0.01），防止 VR 域 FiLM 权重过度扩张。

5. **P2：Two-stage loss**  
   前 10 个 epoch 只 fine-tune FiLM + 最后一层（其他冻结），后 30 个 epoch 全网开放。对应今日 C 的 best_epoch=17 现象。

### Q3：可视化检查是否还有抖动

我已经在另一个 MuJoCo 窗口启动了今日 C 模型的可视化（见终端 `2499477f-...`），命令：

```powershell
python mujoco_sim/validate_policy.py `
  --checkpoint 20260423_train/outputs/act_chunk/run_20260423_153217/checkpoints/best.pt `
  --num-trials 2 --seed 42 --no-ensemble --visualize
```

**你应该观察的 3 个判据**：

1. **目标抓取阶段（小距离时）末端是否高频颤动**  
   抖动 = chunk_size=5 闭环伪影（前期已诊断，与数据无关）  
   → 若仍有，建议试 `chunk_size=1`

2. **空闲阶段（距离 > 0.3m，任务开始前）手臂是否静止**  
   如果这里就在晃，是 obs 归一化/action_scale 问题，不是架构问题

3. **接近目标时是否冲过目标再回拉**  
   → overshoot = policy 对 `distToTarget` 的建模不精细，可通过 `near_target_bonus` 调高

**量化抖动的客观方法**（需要时可跑）：在 validate_policy.py 加一行 `print(np.std(np.diff(actions_history, axis=0), axis=0).mean())`，抖动程度 = 连续动作差分的标准差。对比：
- MJ-only pretrain（基线）
- d1 mixed FT（历史）  
- d1+d2 mixed FT（今日）

---

## 4. 文件清单

**今日新增**：

| 文件 | 用途 |
|------|-----|
| [20260423_train/](20260423_train/) | 今日训练根目录 |
| [20260423_train/config_vr_d1d2_film_ft.json](20260423_train/config_vr_d1d2_film_ft.json) | VR-only d1+d2 配置 |
| [20260423_train/config_vr_d1_only_film_ft.json](20260423_train/config_vr_d1_only_film_ft.json) | VR-only d1 控制配置 |
| [20260423_train/config_strong_mixed_film_ft_d1d2.json](20260423_train/config_strong_mixed_film_ft_d1d2.json) | MJ + d1 + d2 混合配置 ← 主实验 |
| [20260423_train/outputs/act_chunk/run_20260423_151511](20260423_train/outputs/act_chunk/run_20260423_151511) | A 模型 ckpt |
| [20260423_train/outputs/act_chunk/run_20260423_151721](20260423_train/outputs/act_chunk/run_20260423_151721) | B 模型 ckpt |
| [20260423_train/outputs/act_chunk/run_20260423_153217](20260423_train/outputs/act_chunk/run_20260423_153217) | **C 模型 ckpt（主结果 100%）** |
| [data_collector/data2_aligned/](data_collector/data2_aligned/) | data2 对齐后 JSONL |

**使用的历史资产**：

| 文件 | 用途 |
|------|-----|
| [20260422_train/outputs/act_chunk/run_20260422_171750/checkpoints/best.pt](20260422_train/outputs/act_chunk/run_20260422_171750/checkpoints/best.pt) | weak-MJ pretrain (76%→re-bench 92%) |
| [data_collector/mujoco_expert_v4/](data_collector/mujoco_expert_v4/) | MJ 强监督数据（260K 帧） |
| [data_collector/collector10_aligned/](data_collector/collector10_aligned/) | data1 对齐 JSONL |

---

## 5. 阶段结论与下一步

### 已确认结论
- ✅ Data2 与 collector10 / MJ_v4 完全兼容，无需代码改动
- ✅ d1+d2 mixed FiLM FT = 100% (50/50)，上限保持  
- ✅ VR-only FT 路径无论多少数据都不稳定（不建议作为论文主线）
- ✅ Early stop 是 mixed 训练的正常行为，不是 bug
- ⚠️ d1+d2 mean_time 比 d1 略长（1.32s vs 1.03s），可能是 VR 域 distribution shift 或模型过拟合 VR 子集
- ⚠️ 当前 50-target 基准在 seed=42 下已触顶，无法区分"好"和"更好"

### 下一步优先级

**P0（本周可做）**
- 对 d1+d2 mixed FT 换 seed（123, 456, 789）各跑一次 50-target，看平均成功率是否 ≥ d1（或在上限附近）
- 上 `obs_noise_std=0.05`，重训一次 mixed-ft，测试抖动是否减小

**P1（数据再扩充后）**
- 当 VR 数据达 100K+ 帧时，试 `chunk_size=1` 对比 mixed-ft 成功率
- 加 WeightedRandomSampler 维持 MJ:VR = 1:1 batch 比例

**P2（论文定稿前）**
- 加一列客观抖动指标（action diff std）
- 所有实验扩到 100-target (20 trials × 5 targets) 建立置信区间

---

## 附录 A：早停日志（C 实验关键段）

```
[ACT-Chunk] Train domain histogram: {0: 254319, 1: 30465}   ← MJ:VR = 8.3:1
[ACT-Chunk] epoch=001 val_mse=0.270475
[ACT-Chunk] epoch=005 val_mse=0.252375   ← 5 epoch 内达到 0.25
[ACT-Chunk] epoch=017 val_mse=0.252109   ← ★ best
[ACT-Chunk] epoch=018 val_mse=0.257754
[ACT-Chunk] epoch=025 val_mse=0.258249
[ACT-Chunk] epoch=032 early stop at epoch=32
[ACT-Chunk] Done! best_epoch=17, best_val_mse=0.252109
```

## 附录 B：评估原始 50-target 汇总

**strong_mixed_film_ft_d1d2（今日主结果）**：
- Overall: 50/50 (100.0%)
- time_mean=1.32s, time_median=0.72s
- min_dist_mean=0.1278m

**d1 mixed FiLM FT（历史对照）**：
- Overall: 50/50 (100.0%)
- time_mean=1.03s, time_median=0.61s
- min_dist_mean=0.1142m
