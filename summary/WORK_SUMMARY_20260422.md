# VR 采集链路排查与 collector9 修复（2026-04-22）

## 1. 本阶段总结

- 本次重点排查了 `collector9` 最新数据中的两个问题：目标球生成半空间错误，以及 VR 采集过程中控制器 tracking loss 导致的机器人手臂滞后。
- 基于 `data_collector/collector9/vr-demonstrations-events-20260422_121448.jsonl` 和 `data_collector/collector9/vr-demonstrations-episodes-20260422_121448.jsonl` 的统计，确认目标采样仍前后混采，且活跃任务中仍存在 tracking loss / stale pose。
- 已在 `frontend/src/components/RobotVR.vue` 中完成针对性修复：目标深度只采样操作者侧工作区、优先使用 WebXR grip pose、固定基座模式下关闭无用摇杆读取、限制 render-loop 调试输出、降低 VR 调试面板刷新频率、单手任务仅跟随当前指定手。

关键数字：

| 指标 | 数值 | 含义 |
|------|------|------|
| `target_spawned` 数 | 45 | 最新 collector9 有 45 个目标生成事件 |
| `sampleLocalZ > 0` | 26 | 目标落到机器人远离操作者的一侧 |
| `sampleLocalZ < 0` | 19 | 目标落到操作者侧工作区 |
| `controller_tracking_lost` | 17 | 本批次 tracking 丢失事件数 |
| `controller_tracking_restored` | 15 | 本批次 tracking 恢复事件数 |
| `leftPoseValid=false` | 58 / 5880 帧 | 左手失效帧 |
| `rightPoseValid=false` | 1078 / 5880 帧 | 右手失效帧，明显偏高 |

## 2. 问题诊断

### 2.1 目标球生成半空间仍然不对

观察：

- 最新 `collector9` 中，`target_spawned.payload.sampleLocalZ` 同时存在正值和负值。
- 由于当前放置关系是“机器人在操作者前方，且与操作者同向”，操作者实际面对的是机器人的背面，因此操作者侧可见工作区对应 `robot-local -Z`，不应该继续前后混采。

数据证据：

- 45 个 `target_spawned` 中，`sampleLocalZ > 0` 有 26 个，`sampleLocalZ < 0` 有 19 个，说明此前逻辑仍在错误半空间生成了大量目标。
- 典型错误样本：`sampleLocalZ = 0.368, 0.223, 0.341, 0.277 ...`，这些目标更靠机器人远离操作者的一侧。

根因：

- 之前为了“增加前方目标概率”引入了双半空间采样，但在“机器人与操作者同向”的关系下，把“机器人前方”和“操作者侧工作区”混为一谈，导致半空间语义错位。

### 2.2 卡顿不是纯主观体感，数据里存在 stale pose

观察：

- `collector9` 中有 17 次 `controller_tracking_lost`，且并非只发生在闲置手。
- 其中活跃手的 tracking loss 也存在：`('left', 'left') = 8`，`('right', 'right') = 5`。
- 在 `rightPoseValid=false` 的片段中，`obs.user.rightController.p` 会在多个连续帧里完全不变，表现为“控制器位姿冻结在旧位置”。

数据证据：

- 最新批次共有 5880 帧，`rightPoseValid=false` 达到 1078 帧，远高于左手的 58 帧。
- 典型冻结片段中，右手控制器位置连续多帧保持同一值 `[0.258, 0.777, 0.107]`，而 `trackingState` 已经是 `lost`。

根因分析：

- 控制器位姿源原先优先走 `renderer.xr.getController()`（target-ray space），不够贴近手柄 grip pose，且更容易在 tracking 不稳定时留下旧变换。
- 固定基座采集模式下，代码仍每帧读取摇杆输入，并在 render loop 中持续输出高频调试日志。
- VR 3D 调试面板每帧重绘整张 canvas，也会增加浏览器与串流链路的 CPU 压力。

## 3. 解决方法

### 3.1 目标采样改回操作者侧工作区

修改文件：`frontend/src/components/RobotVR.vue`

- 明确“机器人在操作者前方且与操作者同向”时，操作者侧工作区是 `robot-local -Z`。
- `randomTaskTargetWorldPosition()` 中不再前后混采，而是只在 `-Z` 半空间采样目标深度。
- 同时保留已有的左右手工作区与可见性约束。

### 3.2 手柄位姿改为优先使用 grip pose

修改文件：`frontend/src/components/RobotVR.vue`

- 新增 `controllerGrip1 / controllerGrip2` 和 `controllerGripsByHand`。
- 在 `setupControllers()` 中通过 `renderer.xr.getControllerGrip()` 绑定 grip-space 对象。
- `getControllerByHand()` 现在优先返回 grip-space controller，找不到时才回退到原来的 target-ray controller。
- 会话开始/结束和输入源移除时同步清空 grip-space 缓存，避免残留旧引用。

### 3.3 降低 render-loop 负载

修改文件：`frontend/src/components/RobotVR.vue`

- 固定基座模式下，`updateJoystickInput()` 直接返回，不再每帧读取摇杆输入。
- 引入 `showTimedDebug()`，将摇杆/移动相关的高频 debug 改为限频输出。
- `updateVRDebugPanel()` 现在只有在文本内容变化，或距离上次绘制超过 250ms 时才重绘 canvas。

### 3.4 单手任务只跟随当前指定手

修改文件：`frontend/src/components/RobotVR.vue`

- `handleLeftHandFollow()` 在任务指定右手时直接跳过。
- `handleRightHandFollow()` 在任务指定左手时直接跳过。

这条修改直接对应 `collector9` 中“单手任务但另一只手频繁 tracking lost”的模式，可以减少不必要的 IK 计算和无效控制器依赖。

## 4. 实验结果汇总

### 4.1 collector9 诊断结果

| 检查项 | 结果 | 结论 |
|------|------|------|
| 目标半空间分布 | `sampleLocalZ > 0`: 26, `sampleLocalZ < 0`: 19 | 仍存在明显前后混采 |
| tracking loss 事件 | lost 17, restored 15 | tracking 不稳定真实存在 |
| 活跃手 loss | 左手任务丢失 8 次，右手任务丢失 5 次 | 不是只有闲置手有问题 |
| 右手 pose 失效帧 | 1078 / 5880 | 右手 tracking 明显不稳定 |
| stale pose 现象 | 连续多帧右手位置完全相同 | 数据中已写入滞后片段 |

### 4.2 本次修复项与验证

| 修复项 | 文件 | 验证结果 |
|------|------|------|
| 目标只采样 `robot-local -Z` | `frontend/src/components/RobotVR.vue` | 文件无语法错误 |
| 优先使用 `getControllerGrip()` | `frontend/src/components/RobotVR.vue` | 文件无语法错误 |
| 固定基座关闭摇杆读取 | `frontend/src/components/RobotVR.vue` | 文件无语法错误 |
| render-loop debug 限频 | `frontend/src/components/RobotVR.vue` | 文件无语法错误 |
| VR debug 面板限频重绘 | `frontend/src/components/RobotVR.vue` | 文件无语法错误 |
| 单手任务仅跟随指定手 | `frontend/src/components/RobotVR.vue` | 文件无语法错误 |
| 前端启动 | `npm run dev` | Vite 正常启动，`http://localhost:5174/` |

说明：

- 本次没有做 MuJoCo 闭环评估，因为问题集中在 VR 采集链路而非模型训练或推理。
- 当前“修复有效”仅能确认到代码层和开发服务器启动层；是否解决头显内体感卡顿，仍需要新采一批数据验证。

## 5. 阶段结论与下一步

### 5.1 已确认结论

- `collector9` 最新数据已经足够证明：目标半空间曾被错误混采，且 tracking loss / stale pose 在录制数据中真实存在，不是纯主观体感。
- 在“机器人位于操作者前方且与操作者同向”的关系下，操作者侧工作区应当固定为 `robot-local -Z`。
- 对当前任务来说，固定基座模式下每帧摇杆读取、高频日志和每帧 VR 面板重绘，都是可以直接删除或限频的额外负担。
- 单手任务没有必要持续驱动另一只手的 IK，这会放大 tracking 不稳定对 render-loop 的影响。

### 5.2 下一步任务

- P0：完全重启 VBS、SteamVR、Chrome 后，重新采一个小批次（建议新目录 `collector10`），只验证两件事：
  1. `sampleLocalZ` 是否已全部落在负值半空间；
  2. 活跃手是否还出现明显 `controller_tracking_lost` 或长时间 stale pose。

- P1：如果 `collector10` 仍存在活跃手 tracking loss，继续对 fresh 的 `rrServer` / `vrserver` 日志做会话级排查，不再优先怀疑采样逻辑。

- P2：只有在 `collector10` 的 tracking 和目标分布都稳定后，才考虑把这批 VR 数据纳入后续 finetune / mixed training。