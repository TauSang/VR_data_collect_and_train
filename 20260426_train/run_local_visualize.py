"""Launch local MuJoCo visualization for the latest 20260426 Phase 2 checkpoint."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "20260426_train"
VALIDATE = ROOT / "mujoco_sim" / "validate_policy.py"


def latest_phase2_best() -> Path:
    runs = sorted(
        (TRAIN_DIR / "outputs" / "act_chunk").glob("phase2_run_*/checkpoints/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError("No 20260426 phase2 checkpoint found under outputs/act_chunk")
    return runs[0]


def main() -> int:
    ckpt = latest_phase2_best()
    out = ckpt.parent.parent / "visual_eval_seed42_5trials.json"
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
        str(out),
    ]
    print("[visualize]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
