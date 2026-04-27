# 跨机器人泛化实验 & 项目结构整理（2026-04-16）

## 1. 本阶段总结

本次工作主要围绕三个任务展开：

1. **建立项目工作流 Skill** —— 创建了 `.github/skills/vr-robot-workflow/SKILL.md`，将项目的训练命名规范、评估标准、已知结论等固化为 AI Agent 的可加载技能文件。
2. **跨机器人零样本迁移实验** —— 不修改模型权重，将最优 G1 策略（MLP k=5，94%）直接跨机器人测试：
   - G1 带灵巧手版本（44 DOF）：**92%** （仅下降 2%）
   - 全新自制紧凑型双臂机器人 CompArm-8（完全不同运动学结构）：**2%** （完全失败）
3. **前端目录整理** —— 将前端代码迁移至 `frontend/`，部署配置迁移至 `deploy/`，根目录大幅精简。

**核心结论：G1 策略对同系列不同配置（灵巧手）高度鲁棒，但对完全不同运动学结构的机器人完全失效（零样本迁移不可行）。**

---

## 2. 跨机器人泛化实验

### 2.1 实验背景

研究问题：在不修改任何模型权重的前提下，将为 G1 机器人训练的到达策略，直接运用于结构不同的机器人，能否完成任务？

最优模型：`20260409train2/outputs/act_mlp_k5/act_chunk/run_20260412_174749/checkpoints/best.pt`
- 架构：ACT Chunk MLP（chunk_size=5）
- 训练数据：mujoco_expert_v4（300K frames）
- G1 标准测试：**47/50 = 94%**

### 2.2 实验方案

| 机器人 | 描述 | 关节数（总） | 策略控制关节 |
|--------|------|------------|------------|
| G1 标准（基线） | 29 DOF 两足行走机器人 | 29 | 8（双臂）|
| G1 带灵巧手 | 44 DOF（含多指手） | 44 | 8（双臂）|
| CompArm-8（自制） | 台式紧凑型双臂机器人 | 8 | 8（双臂）|

**CompArm-8 与 G1 的关键差异：**

| 参数 | G1 | CompArm-8 |
|------|-----|-----------|
| 形态 | 两足行走人形 | 底座固定台式双臂 |
| 肩部高度 | ~1.3m | ~0.78m |
| 上臂长度 | ~0.25m | 0.16m |
| 前臂长度 | ~0.18m | 0.22m（比例对调）|
| 肩中心距 | ~0.40m | 0.30m |
| 整机质量 | ~35kg | ~12kg |
| 控制增益 kp | 500 | 300 |

**唯一共同接口**（零样本迁移的桥梁）：
- 相同的 8 个关节名称（`*_shoulder_pitch/roll/yaw_joint`, `*_elbow_joint`）
- 相同的执行器命名
- 相同的身体名称（`*_shoulder_pitch_link`, `*_wrist_yaw_link`）
- 相同的观测维度（31D）和动作维度（8D）

### 2.3 实验结果

| 机器人配置 | 成功/总目标 | 成功率 | min_d 均值 |
|-----------|------------|--------|-----------|
| G1 标准（基线） | 47/50 | **94%** | 0.14m |
| G1 带灵巧手（+2kg/arm） | 46/50 | **92%** | 0.14m |
| CompArm-8（不同运动学） | 1/50 | **2%** | 0.36m |

全部测试：`--num-trials 10 --seed 42 --no-ensemble`（50 targets 满足置信度要求）

结果文件：
- `mujoco_eval.json` — G1 标准
- `mujoco_eval_with_hands.json` — G1 带灵巧手
- `mujoco_eval_compact_arm.json` — CompArm-8

### 2.4 根因分析

**G1 带灵巧手 → 92%（成功）：**
- 灵巧手只改变了手腕/肘部的惯性属性
- 8 个被控关节（臂部）的运动学结构完全相同
- 观测空间（末端执行器位置）的分布几乎不变
- 额外质量仅导致响应速度略慢，策略可适应

**CompArm-8 → 2%（失败）：**

根本原因是**观测分布偏移（Observation Distribution Shift）**：

```
G1 训练时：  末端执行器在 [x=±0.2m, y, z≈1.1～1.4m] 的区域活动
CompArm-8：  末端执行器在 [x=±0.15m, y, z≈0.2～0.8m] 的区域活动
```

策略输入的 31D 观测向量：
- 前 16D（关节位置/速度）：分布基本相同（同名关节，同限位）
- 后 15D（空间特征：末端位置、目标位置、相对距离）：**分布完全不同**

策略从未见过末端执行器在这样的笛卡尔坐标下，输出的关节增量指向错误方向，无法收敛。

**为什么 1/50 偶然成功：**
随机采样恰好将目标放在了策略碰巧让末端抵达的位置。

### 2.5 结论

1. **策略不可直接跨异构机器人迁移**：即使关节接口完全兼容，运动学结构差异导致笛卡尔空间分布偏移，策略完全失效。
2. **同系列变体有较强鲁棒性**：G1 带灵巧手（动力学参数改变）→ 92%，几乎无影响。
3. **实现跨机器人迁移的方案**：
   - **P0 方案**：在新机器人上重新采集专家数据并重新训练（最可靠）
   - **P1 方案**：域随机化训练（同时在多种机器人配置上训练）
   - **P2 方案**：基于机器人运动学参数的策略调适（参数化策略）

---

## 3. 工作流 Skill 建设

### 3.1 GitHub Karpathy Skill 搜索

搜索结论：**Andrej Karpathy 没有发布任何 SKILL.md 文件**（三次 GitHub API 独立确认）
- 其公开 repo：nanoGPT (56.7k★)、llm.c (29.6k★)、llama2.c (19.4k★)、micrograd (15.5k★)
- 均无 Copilot skill 文件

找到的相关 repo：`arpitg1304/robotics-agent-skills`（173★）：
- 包含 ROS1/ROS2/robot-perception 等技能
- 定位于 ROS 框架开发，与本项目 MuJoCo 直驱方案差异较大，未引入

### 3.2 创建项目工作流 Skill

创建 `.github/skills/vr-robot-workflow/SKILL.md`，包含：
- 训练目录命名规范（`YYYYMMDD_train/`）
- Summary 文档结构模板
- 评估标准（≥50 targets，MLP 必须 `--no-ensemble`）
- 项目关键路径速查表
- 已知结论备忘录（防止重复踩坑）

---

## 4. 新增文件

| 文件 | 说明 |
|------|------|
| `mujoco_sim/model/task_scene_with_hands.xml` | G1 带灵巧手任务场景 |
| `mujoco_sim/model/task_scene_compact_arm.xml` | CompArm-8 完全自制紧凑双臂机器人场景 |
| `.github/skills/vr-robot-workflow/SKILL.md` | 项目工作流 Skill |
| `deploy/nginx.conf` | （迁移自根目录）|
| `deploy/ssl.conf` | （迁移自根目录）|
| `deploy/server.crt` | （迁移自根目录）|
| `deploy/server.key` | （迁移自根目录）|

---

## 5. 项目结构整理

### 5.1 前端代码迁移

将散落在根目录的前端文件集中到 `frontend/` 目录：

```
根目录（整理前）：
  index.html config.html src/ public/  ← 混在 ML 文件之间

根目录（整理后）：
  frontend/
    index.html          ← 主应用入口
    config.html         ← Avatar 配置页入口
    src/                ← Vue 源代码
    public/             ← 静态资源（vite.svg, models/, avatars.json）
```

**部署配置迁移到 `deploy/`：**
```
根目录（整理前）：nginx.conf ssl.conf server.crt server.key
deploy/（整理后）：nginx.conf ssl.conf server.crt server.key
```

### 5.2 Vite 配置更新

`vite.config.js` 修改：
- 新增 `root: 'frontend'` —— Vite 项目根从项目根目录改为 `frontend/`
- 新增 `import path from 'path'`
- 移除 `xrSmoke` 入口（对应文件不存在）
- HTML 入口改为绝对路径（`path.resolve(__dirname, 'frontend/xxx.html')`）
- `outDir` 显式设为 `path.resolve(__dirname, 'dist')`，保持构建输出在项目根目录

| 配置项 | 修改前 | 修改后 |
|--------|--------|--------|
| root | （默认 = 项目根） | `'frontend'` |
| input main | `'index.html'` | `path.resolve(__dirname, 'frontend/index.html')` |
| input config | `'config.html'` | `path.resolve(__dirname, 'frontend/config.html')` |
| input xrSmoke | `'xr-smoke.html'` | 已移除 |
| outDir | （默认） | `path.resolve(__dirname, 'dist')` |

**未移动的文件（保持在根目录）：**
- `scripts/` —— 数据导入工具，依赖 `process.cwd()` 找到 `data_collector/`
- `package.json`, `node_modules/` —— npm 根目录，不可移动
- `vite.config.js` —— Vite 配置入口，需在项目根

---

## 6. 阶段结论与下一步

### 已确认的结论（带数据支持）

| 结论 | 支持数据 |
|------|---------|
| MLP k=5 是当前最优策略 | G1标准94% vs Transformer 80% |
| 策略对同系列机器人参数变化有强鲁棒性 | G1带手92%（仅-2%）|
| 策略无法零样本迁移至异构机器人 | CompArm-8 仅2%（随机水平）|
| 跨机器人迁移需要数据适配 | 运动学变化导致观测分布偏移 |

### 下一步任务

| 优先级 | 任务 |
|--------|------|
| P0 | 在 CompArm-8 上采集专家数据并重新训练，验证"同架构不同机器人能训练出好策略"的假设 |
| P1 | 域随机化训练：同时使用 G1 + G1-with-hands 数据，测试泛化提升 |
| P2 | 研究机器人无关的策略表示（如基于任务空间/笛卡尔空间的策略）|
