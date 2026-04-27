"""v6 head-to-head eval: P1 vs P2."""
from __future__ import annotations
import json, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATE = ROOT / "mujoco_sim" / "validate_policy.py"

P1_BEST = ROOT / "20260425_train_v6/outputs/act_chunk/phase1_run_20260425_134437/checkpoints/best.pt"
P2_BEST = ROOT / "20260425_train_v6/outputs/act_chunk/phase2_run_20260425_140348/checkpoints/best.pt"

SEEDS = [42, 7, 123, 2024, 31415]

def run(ckpt, scale, seed):
    cmd = [sys.executable, str(VALIDATE), "--checkpoint", str(ckpt),
           "--num-trials", "10", "--seed", str(seed),
           "--no-ensemble", "--action-scale", f"{scale:.3f}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    out = proc.stdout + proc.stderr
    m = re.search(r"Overall:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", out)
    if not m:
        print(out[-1500:]); raise RuntimeError("parse failed")
    return int(m.group(1)), int(m.group(2))

def eval_arm(label, ckpt, scale=1.0):
    rows = []
    for s in SEEDS:
        succ, total = run(ckpt, scale, s)
        rows.append({"seed": s, "succ": succ, "total": total})
        print(f"  [{label}] seed={s}: {succ}/{total}", flush=True)
    tot = sum(r["succ"] for r in rows); tall = sum(r["total"] for r in rows)
    worst = min(r["succ"]/r["total"] for r in rows) * 100
    print(f"  => {label}: {tot}/{tall} = {100*tot/tall:.2f}%  worst={worst:.1f}%", flush=True)
    return {"label": label, "ckpt": str(ckpt.relative_to(ROOT)), "scale": scale,
            "per_seed": rows, "agg_succ": tot, "agg_total": tall,
            "agg_sr": 100 * tot / tall, "worst_sr": worst}

print("=== v6 Arm A (MJ-only, P1) ===")
a = eval_arm("v6_phase1_mj_only", P1_BEST)
print("=== v6 Arm B (MJ + VR finetune, P2) ===")
b = eval_arm("v6_phase2_mj_vr_ft", P2_BEST)

delta = b["agg_sr"] - a["agg_sr"]
print(f"\n=== v6 ABLATION ===")
print(f"  Arm A (MJ-only):     {a['agg_sr']:.2f}%")
print(f"  Arm B (MJ + VR):     {b['agg_sr']:.2f}%")
print(f"  Delta (VR helps?):   {delta:+.2f}pp")

results = {"arm_A_mj_only": a, "arm_B_mj_vr_ft": b,
           "delta_pp": delta, "seeds": SEEDS}
Path(__file__).with_suffix(".json").write_text(json.dumps(results, indent=2))
