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

    # ── Comparison plot ──
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([h["epoch"] for h in bc["history"]], [h["val_action_mse"] for h in bc["history"]], label="BC val_action_mse")
    plt.plot([h["epoch"] for h in act["history"]], [h["val_action_mse"] for h in act["history"]], label="ACT val_action_mse")
    plt.xlabel("epoch"); plt.ylabel("val action MSE")
    plt.title("G1 Reach — BC vs ACT (31D obs → 8D act)")
    plt.grid(True, alpha=0.3); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([h["epoch"] for h in bc["history"]], [h["val_action_mae"] for h in bc["history"]], label="BC val_action_mae")
    plt.plot([h["epoch"] for h in act["history"]], [h["val_action_mae"] for h in act["history"]], label="ACT val_action_mae")
    plt.xlabel("epoch"); plt.ylabel("val action MAE")
    plt.title("G1 Reach — BC vs ACT MAE")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    compare_png = summary_dir / "bc_vs_act_val_action_mse.png"
    plt.savefig(compare_png, dpi=140); plt.close()

    # ── LR curve ──
    if "lr" in bc["history"][0]:
        plt.figure(figsize=(6, 3))
        plt.plot([h["epoch"] for h in bc["history"]], [h["lr"] for h in bc["history"]], label="BC")
        plt.plot([h["epoch"] for h in act["history"]], [h["lr"] for h in act["history"]], label="ACT")
        plt.xlabel("epoch"); plt.ylabel("LR"); plt.title("Learning Rate Schedule")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(summary_dir / "lr_schedule.png", dpi=140); plt.close()

    ratio = float((act_best["val_action_mse"] - bc_best["val_action_mse"]) / max(abs(bc_best["val_action_mse"]), 1e-12))

    # Extract combined data summary
    data_summary = bc.get("data_summary", {})

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
        "data_summary": data_summary,
        "train_summary": bc.get("train_summary", {}),
        "val_summary": bc.get("val_summary", {}),
        "split_summary": bc.get("split_summary", {}),
    }

    with (summary_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Format data source info
    source_info = ""
    if "per_source" in data_summary:
        for src in data_summary["per_source"]:
            source_info += f"  - {src.get('source_name', '?')}: {src.get('num_kept_segments', '?')} segments kept\n"

    lines = [
        "# G1 Reach Task — 训练总结 (Combined c6+c7+c8)",
        "",
        "## 设计要点",
        "",
        "- 机器人: Unitree G1 dual-arm (8 DOF: L/R shoulder pitch/roll/yaw + elbow)",
        "- Observation: 31D = 8 joint_pos + 8 joint_vel + 6 ee_pos + 9 target_rel",
        "- Action: 8D = G1 joint position delta",
        "- 数据源: collector6 + collector7 + collector8 (合并训练)",
        "- 学习率: Cosine annealing, 200 epochs",
        "- ACT改进: d_model=256, 8头, 4层, multi-scale readout",
        "",
        "## 数据构成",
        "",
        f"- 总 segment 数: {data_summary.get('total_segments', '?')}",
        f"- 总帧数: {data_summary.get('total_frames', '?')}",
        f"- 总 episode 数: {data_summary.get('total_episodes', '?')}",
        f"- Outcome 分布: {data_summary.get('outcome_totals', '?')}",
        "",
        "### 各数据源",
        "",
        source_info,
        "",
        "## 最优验证结果",
        "",
        f"- BC:  epoch={bc_best['epoch']}, val_mse={bc_best['val_action_mse']:.6f}, val_mae={bc_best['val_action_mae']:.6f}, succ_acc={bc_best['val_success_acc']:.4f}",
        f"- ACT: epoch={act_best['epoch']}, val_mse={act_best['val_action_mse']:.6f}, val_mae={act_best['val_action_mae']:.6f}, succ_acc={act_best['val_success_acc']:.4f}",
        f"- ACT vs BC val_mse 变化: {ratio * 100:.2f}%",
        "",
        "## 下一步",
        "",
        "- 在 MuJoCo G1 模型中回放最优checkpoint验证效果",
        "- 接入真机 G1 做闭环测试",
        "",
        f"- 对比曲线: {compare_png.name}",
    ]
    with (summary_dir / "SUMMARY.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[analyze] done")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
