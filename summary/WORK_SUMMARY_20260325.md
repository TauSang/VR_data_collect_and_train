# VR 任务化示教采集与训练前准备总结（截至 2026-03-25）

## 1. 当前阶段结论

项目已从“采集功能可用”进入“按统一准则扩充高质量数据”阶段。

本阶段目标：
- 冻结当前采集 schema（保持兼容）
- 按约束准则持续收集高质量任务化示教数据
- 达到规模与分布要求后启动 ACT 训练与 Isaac Sim 闭环验证

---

## 2. collector5 数据状态判断

本批次数据可用于训练，不需要返工重采：
- 目录：data_collector/collector5/
- 三文件齐全：session / episodes / events
- 统计：totalEpisodes=13，totalFrames=10139，totalEvents=160
- recordedChannels 包含 taskObservation=true
- 事件中同时包含：
  - 成功回合（task_success_5of5）
  - 超时失败回合（timeout，含 failureLabel）
  - 中断回合（auto_end_by_new_start）

结论：collector5 属于“可直接纳入训练池”的有效批次。

---

## 3. 训练数据收集约束准则（执行版）

### 3.1 回合类型允许范围

采集过程中以下行为都允许：
- A. 中断当前回合并开启新回合（auto_end_by_new_start）
- B. 等待系统自然超时（timeout）
- C. 完整完成 5/5 目标（task_success_5of5）

注意：允许中断和超时，但必须控制整体占比（见 3.3）。

### 3.2 单回合最小质量要求

每个回合应满足以下基本条件：
- episode_start / episode_end 成对存在
- target_spawned 与 targetIndex 逻辑一致
- 若为 timeout，应有 episode_timeout 与 episode_task_summary
- 若为 success，应有最终 episode_task_summary，且 completedTargets=5
- taskObservation 保持每帧可读（不缺字段）

### 3.3 批次配比约束（每 30~50 回合）

推荐目标配比：
- 40% ~ 60%：task_success_5of5（完整成功回合）
- 20% ~ 40%：timeout（失败样本，含 near/far）
- 剩余：auto_end_by_new_start（中断回合）

解释：
- 成功回合保证策略学到“完整达成任务链”
- timeout 回合提供困难负样本，提升鲁棒性
- 中断回合可保留，但不应成为主流

### 3.4 手侧平衡约束

按批次统计 successHand：
- 左/右成功比例尽量维持在 4:6 ~ 6:4
- 避免长期单手偏置（会导致策略偏手）

### 3.5 目标空间覆盖约束

每批次需覆盖多样目标分布：
- 左/中/右位置都出现
- 近/中/远深度都出现
- 高/中/低高度都出现

避免只在“舒适区位置”采集。

### 3.6 失败类型覆盖约束

失败样本优先保留以下类别：
- timeout_near_no_squeeze
- timeout_far_no_contact
- 其他 timeout_*（若出现）

目的：帮助模型学习“接近但失败”与“远离目标失败”的不同恢复策略。

### 3.7 批次验收清单（采完即检）

每批次结束后确认：
1) 三文件齐全且计数一致（session vs events/episodes）
2) success / timeout / auto_end 三类都存在（非全单一类型）
3) 左右手成功有覆盖
4) 目标分布非单一区域
5) 无明显 schema 漂移（字段缺失或重命名）

---

## 4. 下一阶段任务

1. 继续按本准则采集更多批次（优先提升完整 5/5 成功回合数量）
2. 当累计规模达到可训练阈值后，启动 Human+Task conditioned ACT 训练
3. 在 Isaac Sim 做闭环验证：
   - 触碰成功率
   - 到达耗时
   - 超时率
   - 轨迹稳定性

---

## 5. 本文档用途

本文件作为 2026-03-25 起的数据采集执行标准。后续新批次以此作为统一验收基线。