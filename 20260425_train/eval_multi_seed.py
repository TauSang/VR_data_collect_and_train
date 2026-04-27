"""Multi-seed evaluation of the 20260425 improved checkpoint.

Reports aggregated SR over 5 seeds x 50 targets = 250 targets.
"""
from __future__ import annotations
import json, re, subprocess, sys, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATE = ROOT / "mujoco_sim" / "validate_policy.py"
CKPT = ROOT / "20260425_train/outputs/act_chunk/improved_run_20260424_124340/checkpoints/best.pt"

SEEDS = [42, 7, 123, 2024, 31415]

def run(ckpt, scale, ema, seed):
    cmd = [sys.executable, str(VALIDATE), "--checkpoint", str(ckpt),
           "--num-trials", "10", "--seed", str(seed),
           "--no-ensemble", "--action-scale", f"{scale:.3f}"]
    if ema > 0:
        cmd += ["--action-ema", f"{ema:.3f}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    out = proc.stdout + proc.stderr
    m = re.search(r"Overall:\s+(\d+)/(\d+)\s+\(([\d.]+)%\)", out)
    if not m:
        print(out[-1500:])
        raise RuntimeError("parse failed")
    return int(m.group(1)), int(m.group(2)), out


def eval_config(label, ckpt, scale, ema):
    rows = []
    for s in SEEDS:
        succ, total, _ = run(ckpt, scale, ema, s)
        rows.append({"seed": s, "succ": succ, "total": total})
        print(f"  [{label}] scale={scale} ema={ema} seed={s}: {succ}/{total}", flush=True)
    tot = sum(r["succ"] for r in rows); tall = sum(r["total"] for r in rows)
    print(f"  => {label} scale={scale} ema={ema}: {tot}/{tall} = {100*tot/tall:.2f}%  "
          f"worst={min(r['succ']/r['total'] for r in rows)*100:.1f}%", flush=True)
    return {"label": label, "scale": scale, "ema": ema, "per_seed": rows,
            "agg_succ": tot, "agg_total": tall, "agg_sr": 100 * tot / tall,
            "worst_sr": 100 * min(r["succ"]/r["total"] for r in rows)}


results = []
# 1) baseline of improved model
results.append(eval_config("improved_best", CKPT, 1.0, 0.0))
# 2) try small scale bump
results.append(eval_config("improved_scale1.05", CKPT, 1.05, 0.0))
# 3) with EMA
results.append(eval_config("improved_scale1.05_ema0.2", CKPT, 1.05, 0.2))

# also evaluate top3 ensemble-independent: pick top1/top2/top3 at scale=1.0
for i in (1, 2, 3):
    p = ROOT / f"20260425_train/outputs/act_chunk/improved_run_20260424_124340/checkpoints/top{i}.pt"
    if p.exists():
        results.append(eval_config(f"top{i}", p, 1.0, 0.0))

Path(__file__).with_suffix(".json").write_text(json.dumps(results, indent=2))

results_sorted = sorted(results, key=lambda r: (-r["agg_sr"], -r["worst_sr"]))
print("\n=== RANKED ===")
for r in results_sorted:
    print(f"  {r['label']:30s} scale={r['scale']} ema={r['ema']}: "
          f"{r['agg_succ']}/{r['agg_total']} = {r['agg_sr']:.2f}%  worst={r['worst_sr']:.1f}%")
