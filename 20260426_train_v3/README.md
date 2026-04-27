# 20260426_train_v3 — Scheme B

目的：在 Scheme A 基础上验证 VR4 独立 domain + 降权是否能恢复/超过 v9。

方案 B：

- 复用 v9 Phase1 checkpoint
- Phase2 使用 `MJ + VR1-4`
- batch=256，保持 v9 Phase2 recipe
- MJ: `source_id=0`
- VR1-3: `source_id=1`
- VR4: `source_id=2`
- `num_domains=3`
- `source_ratio={0:1.5, 1:1.2, 2:0.3}`

对照：

- v9 P2 MJ+VR1-3: 247/250 = 98.80%
- Scheme A MJ+VR1-4 all-VR-same-domain: 238/250 = 95.20%

训练后必须运行：strict eval、local MuJoCo visualize、summary。
