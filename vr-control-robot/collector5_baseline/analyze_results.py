import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _latest_run_dir(model_dir: Path) -> Path:
    runs = sorted([p for p in model_dir.glob("run_*") if p.is_dir()])
    if not runs:
        raise RuntimeError(f"No runs found in {model_dir}")
    return runs[-1]


def _load_metrics(run_dir: Path):
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    root = Path(__file__).resolve().parent
    out_root = root / "outputs"

    bc_run = _latest_run_dir(out_root / "bc")
    act_run = _latest_run_dir(out_root / "act")

    bc = _load_metrics(bc_run)
    act = _load_metrics(act_run)

    bc_h = bc["history"]
    act_h = act["history"]

    bc_best = min(bc_h, key=lambda x: x["val_mse"])
    act_best = min(act_h, key=lambda x: x["val_mse"])

    # combined curve
    plt.figure(figsize=(8, 5))
    plt.plot([h["epoch"] for h in bc_h], [h["val_mse"] for h in bc_h], label="BC val_mse")
    plt.plot([h["epoch"] for h in act_h], [h["val_mse"] for h in act_h], label="ACT val_mse")
    plt.xlabel("epoch")
    plt.ylabel("val MSE")
    plt.title("BC vs ACT on collector5")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    summary_dir = out_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    compare_png = summary_dir / "bc_vs_act_val_mse.png"
    plt.savefig(compare_png, dpi=140)
    plt.close()

    # markdown report
    report = {
        "bc_run": str(bc_run),
        "act_run": str(act_run),
        "bc_best_epoch": int(bc_best["epoch"]),
        "bc_best_val_mse": float(bc_best["val_mse"]),
        "bc_best_val_mae": float(bc_best["val_mae"]),
        "act_best_epoch": int(act_best["epoch"]),
        "act_best_val_mse": float(act_best["val_mse"]),
        "act_best_val_mae": float(act_best["val_mae"]),
        "improve_ratio_mse": float((bc_best["val_mse"] - act_best["val_mse"]) / max(bc_best["val_mse"], 1e-12)),
    }

    with (summary_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# collector5 基线结果（BC vs ACT）",
        "",
        f"- BC run: {bc_run}",
        f"- ACT run: {act_run}",
        "",
        "## 最优验证结果",
        f"- BC: epoch={bc_best['epoch']}, val_mse={bc_best['val_mse']:.6f}, val_mae={bc_best['val_mae']:.6f}",
        f"- ACT: epoch={act_best['epoch']}, val_mse={act_best['val_mse']:.6f}, val_mae={act_best['val_mae']:.6f}",
        "",
        f"- ACT 相对 BC 的 val_mse 改善比例: {report['improve_ratio_mse'] * 100:.2f}%",
        "",
        f"- 对比曲线图: {compare_png}",
    ]
    with (summary_dir / "summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[analyze] done")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
