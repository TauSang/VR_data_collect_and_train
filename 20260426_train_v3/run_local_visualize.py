"""Launch local MuJoCo visualization for the latest Scheme B checkpoint."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "20260426_train_v3"
VALIDATE = ROOT / "mujoco_sim" / "validate_policy.py"


def latest_best() -> Path:
    candidates = sorted((TRAIN_DIR / "outputs" / "act_chunk").glob("phase2_schemeB_run_*/checkpoints/best.pt"))
    if not candidates:
        candidates = sorted((TRAIN_DIR / "outputs" / "act_chunk").glob("run_*/checkpoints/best.pt"))
    if not candidates:
        raise FileNotFoundError("No Scheme B best.pt found")
    return candidates[-1]


def main() -> int:
    ckpt = latest_best()
    out = ckpt.parents[1] / "visual_eval_seed42_5trials.json"
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
    print("[visualize] " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
