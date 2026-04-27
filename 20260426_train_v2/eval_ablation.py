"""Strict MuJoCo evaluation for 20260426 Scheme A.

Compares:
  A) v9 Phase1 MJ-only baseline
  B) v9 Phase2 MJ+VR1-3 current best baseline
  C) Scheme A: v9 P1 + MJ+VR1-4 finetune, exact v9 Phase2 recipe, batch256

Each arm uses 5 seeds × 10 trials × 5 targets = 250 targets.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "20260426_train_v2"
VALIDATE = ROOT / "mujoco_sim" / "validate_policy.py"
SEEDS = [42, 7, 123, 2024, 31415]

V9_P1 = ROOT / "20260425_train_v9" / "outputs" / "act_chunk" / "phase1_run_20260425_193808" / "checkpoints" / "best.pt"
V9_P2 = ROOT / "20260425_train_v9" / "outputs" / "act_chunk" / "phase2_run_20260425_200159" / "checkpoints" / "best.pt"


def latest_best() -> Path:
    candidates = sorted((TRAIN_DIR / "outputs" / "act_chunk").glob("phase2_schemeA_run_*/checkpoints/best.pt"))
    if not candidates:
        candidates = sorted((TRAIN_DIR / "outputs" / "act_chunk").glob("phase2_*/checkpoints/best.pt"))
    if not candidates:
        candidates = sorted((TRAIN_DIR / "outputs" / "act_chunk").glob("run_*/checkpoints/best.pt"))
    if not candidates:
        raise FileNotFoundError("No Scheme A best.pt found under 20260426_train_v2/outputs/act_chunk")
    return candidates[-1]


def eval_one(label: str, ckpt: Path, scale: float = 1.0) -> dict:
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))
    per_seed = []
    for seed in SEEDS:
        out_path = TRAIN_DIR / f"eval_{label}_seed{seed}.json"
        cmd = [
            sys.executable,
            str(VALIDATE),
            "--checkpoint",
            str(ckpt),
            "--num-trials",
            "10",
            "--seed",
            str(seed),
            "--no-ensemble",
            "--action-scale",
            f"{scale:.3f}",
            "--out",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            print(out[-2000:], flush=True)
            raise RuntimeError(f"validate_policy failed for {label} seed={seed}")
        m = re.search(r"Overall:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", out)
        if not m:
            print(out[-2000:], flush=True)
            raise RuntimeError(f"parse failed for {label} seed={seed}")
        succ = int(m.group(1))
        total = int(m.group(2))
        print(f"  [{label}] seed={seed}: {succ}/{total}", flush=True)
        if total != 50:
            raise RuntimeError(f"{label} seed={seed} expected 50 targets, got {total}")
        per_seed.append({"seed": seed, "succ": succ, "total": total})
    agg_succ = sum(x["succ"] for x in per_seed)
    agg_total = sum(x["total"] for x in per_seed)
    if agg_total != 250:
        raise RuntimeError(f"{label} expected 250 targets, got {agg_total}")
    agg_sr = 100.0 * agg_succ / agg_total
    worst_sr = min(100.0 * x["succ"] / x["total"] for x in per_seed)
    print(f"  => {label}: {agg_succ}/{agg_total} = {agg_sr:.2f}%  worst={worst_sr:.1f}%", flush=True)
    return {
        "label": label,
        "ckpt": str(ckpt.relative_to(ROOT)),
        "scale": scale,
        "per_seed": per_seed,
        "agg_succ": agg_succ,
        "agg_total": agg_total,
        "agg_sr": agg_sr,
        "worst_sr": worst_sr,
    }


def main() -> int:
    scheme_ckpt = latest_best()
    print("=== Scheme A strict eval: v9 P1 baseline ===", flush=True)
    v9_p1 = eval_one("v9_phase1_mj_only_baseline", V9_P1, 1.0)
    print("=== Scheme A strict eval: v9 P2 current best baseline ===", flush=True)
    v9_p2 = eval_one("v9_phase2_mj_vr1_3_baseline", V9_P2, 1.0)
    print("=== Scheme A strict eval: v9 P1 + data1-4 batch256 ===", flush=True)
    scheme = eval_one("schemeA_v9p1_mj_vr1_4_batch256", scheme_ckpt, 1.0)

    result = {
        "arm_A_v9_p1_mj_only": v9_p1,
        "arm_B_v9_p2_mj_vr1_3": v9_p2,
        "arm_C_schemeA_mj_vr1_4": scheme,
        "delta_scheme_vs_p1_pp": scheme["agg_sr"] - v9_p1["agg_sr"],
        "delta_scheme_vs_v9_p2_pp": scheme["agg_sr"] - v9_p2["agg_sr"],
        "seeds": SEEDS,
    }
    out = TRAIN_DIR / "eval_ablation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
