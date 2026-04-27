"""Strict MuJoCo evaluation for 20260426 Scheme B.

Scheme B: v9 P1 + MJ/VR1-4, VR4 as independent source/domain and lower sampler mass.
Runs 5 seeds × 10 trials × 5 targets = 250 targets.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "20260426_train_v3"
VALIDATE = ROOT / "mujoco_sim" / "validate_policy.py"
SEEDS = [42, 7, 123, 2024, 31415]


def latest_best() -> Path:
    candidates = sorted((TRAIN_DIR / "outputs" / "act_chunk").glob("phase2_schemeB_run_*/checkpoints/best.pt"))
    if not candidates:
        candidates = sorted((TRAIN_DIR / "outputs" / "act_chunk").glob("run_*/checkpoints/best.pt"))
    if not candidates:
        raise FileNotFoundError("No Scheme B best.pt found")
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
    print("=== Scheme B strict eval: v9 P1 + VR4 independent domain/low weight ===", flush=True)
    scheme = eval_one("schemeB_vr4_domain_lowweight", scheme_ckpt, 1.0)
    result = {
        "arm_schemeB_vr4_domain_lowweight": scheme,
        "reference_v9_p1_sr": 92.00,
        "reference_v9_p2_sr": 98.80,
        "reference_schemeA_sr": 95.20,
        "reference_20260426_batch512_sr": 93.60,
        "delta_schemeB_vs_v9_p1_pp": scheme["agg_sr"] - 92.00,
        "delta_schemeB_vs_v9_p2_pp": scheme["agg_sr"] - 98.80,
        "delta_schemeB_vs_schemeA_pp": scheme["agg_sr"] - 95.20,
        "delta_schemeB_vs_batch512_pp": scheme["agg_sr"] - 93.60,
        "seeds": SEEDS,
    }
    out = TRAIN_DIR / "eval_ablation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
