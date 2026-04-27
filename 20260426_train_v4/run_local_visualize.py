"""Run local MuJoCo visualization for the best VR4 filtering checkpoint."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "20260426_train_v4"
VALIDATE = ROOT / "mujoco_sim" / "validate_policy.py"


def _best_from_eval() -> tuple[str, Path]:
    eval_path = TRAIN_DIR / "eval_ablation.json"
    if eval_path.exists():
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        arm = data[data["best_arm"]]
        return arm["label"], ROOT / arm["ckpt"]
    candidates = sorted((TRAIN_DIR / "outputs" / "act_chunk").glob("phase2_filter_*_run_*/checkpoints/best.pt"))
    if not candidates:
        raise FileNotFoundError("No filtering best.pt found")
    return "latest_filter", candidates[-1]


def main() -> int:
    label, ckpt = _best_from_eval()
    out_path = ckpt.parent.parent / f"visual_eval_{label}_seed42_5trials.json"
    cmd = [
        sys.executable,
        str(VALIDATE),
        "--checkpoint",
        str(ckpt),
        "--visualize",
        "--num-trials",
        "5",
        "--seed",
        "42",
        "--no-ensemble",
        "--action-scale",
        "1.0",
        "--out",
        str(out_path),
    ]
    print("[visualize] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    print(f"[visualize] saved {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
