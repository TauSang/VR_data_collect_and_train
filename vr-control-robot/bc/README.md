# BC 子目录（整理后）

该目录用于集中管理 Behavior Cloning（BC）相关内容。

## 目录

- `configs/`：BC 配置（从根目录 `configs/` 同步而来）
- `scripts/`：BC 脚本入口（包装器，转发到根目录 `scripts/`）

## 推荐用法

在 `vr-control-robot/` 目录执行：

- `python bc/scripts/train.py --config bc/configs/collector3_50.yaml`
- `python bc/scripts/train_cross_dataset.py --config bc/configs/cross_c3_train_c2_test.yaml`
- `python bc/scripts/data_health_check.py --config bc/configs/cross_c3_train_c2_test.yaml`

说明：当前核心实现仍在根目录 `scripts/` 与 `src/vrtrain/`，本目录先提供统一入口，便于后续继续拆分重构。
