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
    plt.title("Task-Conditioned BC vs ACT on collector5")
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
        "# 20260331 task-conditioned 训练总结",
        "",
        "## 设计结论",
        "",
        "- 本轮训练没有使用 human pose / controller 输入。",
        "- observation 改为：机器人状态 + 任务状态。",
        "- 主输出仍是 `jointDelta`，`segment_success` 只作为辅助监督。",
        "- 这比“直接输出任务完成情况”更接近真正可部署的 policy。",
        "",
        "## 数据构成",
        "",
        f"- 可用 segment 总数: {report['data_summary']['num_kept_segments']}",
        f"- 原始 segment outcome: {report['data_summary']['raw_segment_outcomes']}",
        f"- 保留 segment outcome: {report['data_summary']['kept_segment_outcomes']}",
        f"- 丢弃异常帧数: {report['data_summary']['num_invalid_frames_dropped']}",
        f"- train episodes: {report['split_summary']['train_episodes']}",
        f"- val episodes: {report['split_summary']['val_episodes']}",
        "",
        "## 最优验证结果",
        "",
        f"- BC: epoch={bc_best['epoch']}, val_action_mse={bc_best['val_action_mse']:.6f}, val_action_mae={bc_best['val_action_mae']:.6f}, val_success_acc={bc_best['val_success_acc']:.4f}",
        f"- ACT: epoch={act_best['epoch']}, val_action_mse={act_best['val_action_mse']:.6f}, val_action_mae={act_best['val_action_mae']:.6f}, val_success_acc={act_best['val_success_acc']:.4f}",
        f"- ACT 相对 BC 的 val_action_mse 变化比例: {ratio * 100:.2f}%",
        "",
        "## 判断",
        "",
        "- 先看 `val_action_mse/mae`，因为这直接决定动作回归质量。",
        "- `val_success_acc` 主要看 representation 是否学到任务可完成性，不直接等于 policy 好坏。",
        "- 如果 BC 继续优于 ACT，通常意味着当前数据规模仍偏小，或序列建模收益还不足以覆盖模型复杂度。",
        "",
        "## 建议的下一步",
        "",
        "- 如果 BC 明显更稳：优先把 BC 接到闭环控制上做离线回放和在线验证。",
        "- 如果 ACT 接近或超过 BC：继续扩大成功 segment、调长序列、加入更强的 phase/progress 建模。",
        "- 无论谁更好，都建议下一步补充真正的 gripper / squeeze 到机器人动作标签，否则抓取阶段仍有监督缺口。",
        "",
        f"- 对比曲线图: {compare_png}",
    ]
    with (summary_dir / "SUMMARY_20260331.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[analyze] done")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
