"""v3 ablation eval."""
import json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
VALIDATE = ROOT / "mujoco_sim" / "validate_policy.py"
P1 = ROOT / "20260424_train_v3/outputs/act_chunk/phase1_run_20260424_153729/checkpoints/best.pt"
P2 = ROOT / "20260424_train_v3/outputs/act_chunk/phase2_run_20260424_155738/checkpoints/best.pt"
SEEDS = [42, 7, 123, 2024, 31415]

def run(ckpt, seed):
    cmd = [sys.executable, str(VALIDATE), "--checkpoint", str(ckpt),
           "--num-trials", "10", "--seed", str(seed), "--no-ensemble"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    m = re.search(r"Overall:\s+(\d+)/(\d+)", proc.stdout + proc.stderr)
    return int(m.group(1)), int(m.group(2))

def arm(label, ckpt):
    rows = []
    for s in SEEDS:
        a, b = run(ckpt, s); rows.append([s, a, b])
        print(f"  [{label}] seed={s}: {a}/{b}", flush=True)
    tot = sum(r[1] for r in rows); tall = sum(r[2] for r in rows)
    worst = min(r[1]/r[2] for r in rows)*100
    print(f"  => {label}: {tot}/{tall} = {100*tot/tall:.2f}%  worst={worst:.1f}%", flush=True)
    return {"label": label, "agg_sr": 100*tot/tall, "agg": f"{tot}/{tall}", "worst": worst, "rows": rows}

r1 = arm("v3_phase1_mj_only", P1)
r2 = arm("v3_phase2_mj_vr_ft", P2)
print(f"\n=== v3 ABLATION ===\n  A: {r1['agg_sr']:.2f}%  B: {r2['agg_sr']:.2f}%  Δ={r2['agg_sr']-r1['agg_sr']:+.2f}pp")
Path(__file__).with_suffix(".json").write_text(json.dumps({"A": r1, "B": r2}, indent=2))
