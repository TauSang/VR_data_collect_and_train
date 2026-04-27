import json
from pathlib import Path

import matplotlib.pyplot as plt


def _latest_run_dir(model_dir: Path) -> Path:
    runs = sorted([p for p in model_dir.glob("run_*") if p.is_dir()])
    if not runs:
        raise RuntimeError(f"No runs found in {model_dir}")
    return runs[-1]


def _load_metrics(run_dir: Path) -> dict:
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _best_row(history: list[dict]) -> dict:
    return min(history, key=lambda x: x["val_action_mse"])


def main():
    root = Path(__file__).resolve().parent
    out_root = root / "outputs"

    bc_run = _latest_run_dir(out_root / "bc")
    act_run = _latest_run_dir(out_root / "act")

    bc = _load_metrics(bc_run)
    act = _load_metrics(act_run)
    bc_best = _best_row(bc["history"])
    act_best = _best_row(act["history"])

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot([h["epoch"] for h in bc["history"]], [h["val_action_mse"] for h in bc["history"]], label="BC val_action_mse")
    plt.plot([h["epoch"] for h in act["history"]], [h["val_action_mse"] for h in act["history"]], label="ACT val_action_mse")
    plt.xlabel("epoch")
    plt.ylabel("val action MSE")
    plt.title("G1 Reach — BC vs ACT (43D obs → 14D act)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    compare_png = summary_dir / "bc_vs_act_val_action_mse.png"
    plt.savefig(compare_png, dpi=140)
    plt.close()

    ratio = float((act_best["val_action_mse"] - bc_best["val_action_mse"]) / max(abs(bc_best["val_action_mse"]), 1e-12))
    report = {
        "bc_run": str(bc_run),
        "act_run": str(act_run),
        "bc_best_epoch": int(bc_best["epoch"]),
        "bc_best_val_action_mse": float(bc_best["val_action_mse"]),
        "bc_best_val_action_mae": float(bc_best["val_action_mae"]),
        "bc_best_val_success_acc": float(bc_best["val_success_acc"]),
        "act_best_epoch": int(act_best["epoch"]),
        "act_best_val_action_mse": float(act_best["val_action_mse"]),
        "act_best_val_action_mae": float(act_best["val_action_mae"]),
        "act_best_val_success_acc": float(act_best["val_success_acc"]),
        "act_minus_bc_ratio_on_val_action_mse": ratio,
        "data_summary": bc["data_summary"],
        "train_summary": bc["train_summary"],
        "val_summary": bc["val_summary"],
        "split_summary": bc["split_summary"],
    }

    with (summary_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# G1 Reach Task — 训练总结",
        "",
        "## 设计要点",
        "",
        "- 机器人: Unitree G1 dual-arm (14 DOF)",
        "- Observation: 43D = 14 joint_pos + 14 joint_vel + 6 ee_pos + 9 target_rel",
        "- Action: 14D = G1 joint position delta (scalar per joint)",
        "- VR bone euler → G1 joint 通过轴映射 (pitch=Y, roll=X, yaw=Z)",
        "- 移除了部署不可用的特征: contact_flags, hold_time, progress, phase",
        "- 保留了部署可用的特征: joint pos/vel (编码器), ee pos (FK), target_rel (任务指令+FK)",
        "",
        "## 数据构成",
        "",
        f"- 可用 segment 数: {report['data_summary']['num_kept_segments']}",
        f"- 原始 outcome: {report['data_summary']['raw_segment_outcomes']}",
        f"- 保留 outcome: {report['data_summary']['kept_segment_outcomes']}",
        f"- 丢弃异常帧: {report['data_summary']['num_invalid_frames_dropped']}",
        f"- train episodes: {report['split_summary']['train_episodes']}",
        f"- val episodes: {report['split_summary']['val_episodes']}",
        "",
        "## 最优验证结果",
        "",
        f"- BC:  epoch={bc_best['epoch']}, val_mse={bc_best['val_action_mse']:.6f}, val_mae={bc_best['val_action_mae']:.6f}, succ_acc={bc_best['val_success_acc']:.4f}",
        f"- ACT: epoch={act_best['epoch']}, val_mse={act_best['val_action_mse']:.6f}, val_mae={act_best['val_action_mae']:.6f}, succ_acc={act_best['val_success_acc']:.4f}",
        f"- ACT vs BC val_mse 变化: {ratio * 100:.2f}%",
        "",
        "## 下一步",
        "",
        "- 在 MuJoCo G1 模型中回放预测轨迹验证物理合理性",
        "- 接入真机 G1 做闭环测试",
        "- 如数据不足，重新用 VR 采集更多 reach 示教",
        "",
        f"- 对比曲线: {compare_png}",
    ]
    with (summary_dir / "SUMMARY.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[analyze] done")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
