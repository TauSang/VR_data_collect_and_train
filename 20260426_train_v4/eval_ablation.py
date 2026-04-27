"""Strict MuJoCo evaluation for VR4 filtering variants.

Runs 2 filtered arms × 5 seeds × 50 targets = 500 targets total.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "20260426_train_v4"
VALIDATE = ROOT / "mujoco_sim" / "validate_policy.py"
SEEDS = [42, 7, 123, 2024, 31415]

ARMS = {
    "filter_len40": {
        "glob": "phase2_filter_len40_run_*/checkpoints/best.pt",
        "label": "filter_len40",
    },
    "filter_quality": {
        "glob": "phase2_filter_quality_run_*/checkpoints/best.pt",
        "label": "filter_quality_v1",
    },
}


def latest_best(pattern: str) -> Path:
    candidates = sorted((TRAIN_DIR / "outputs" / "act_chunk").glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint matching {pattern}")
    return candidates[-1]


def eval_one(label: str, ckpt: Path, scale: float = 1.0) -> dict:
    if not ckpt.exists() or ckpt.stat().st_size <= 0:
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
            print(out[-2500:], flush=True)
            raise RuntimeError(f"validate_policy failed for {label} seed={seed}")
        m = re.search(r"Overall:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", out)
        if not m:
            print(out[-2500:], flush=True)
            raise RuntimeError(f"parse failed for {label} seed={seed}")
        succ = int(m.group(1))
        total = int(m.group(2))
        if total != 50:
            raise RuntimeError(f"{label} seed={seed} expected 50 targets, got {total}")
        print(f"  [{label}] seed={seed}: {succ}/{total}", flush=True)
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
    print("=== 20260426 VR4 filtering strict eval ===", flush=True)
    results = {}
    for arm_name, spec in ARMS.items():
        ckpt = latest_best(spec["glob"])
        print(f"=== {arm_name}: {ckpt.relative_to(ROOT)} ===", flush=True)
        results[f"arm_{arm_name}"] = eval_one(spec["label"], ckpt, 1.0)

    best_key = max((k for k in results if k.startswith("arm_")), key=lambda k: results[k]["agg_sr"])
    best = results[best_key]
    result = {
        **results,
        "best_arm": best_key,
        "best_label": best["label"],
        "best_sr": best["agg_sr"],
        "reference_v9_p1_sr": 92.00,
        "reference_v9_p2_sr": 98.80,
        "reference_schemeA_sr": 95.20,
        "reference_schemeB_sr": 94.00,
        "reference_20260426_batch512_sr": 93.60,
        "delta_best_vs_v9_p2_pp": best["agg_sr"] - 98.80,
        "delta_best_vs_schemeA_pp": best["agg_sr"] - 95.20,
        "delta_best_vs_schemeB_pp": best["agg_sr"] - 94.00,
        "delta_best_vs_batch512_pp": best["agg_sr"] - 93.60,
        "seeds": SEEDS,
    }
    out = TRAIN_DIR / "eval_ablation.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
