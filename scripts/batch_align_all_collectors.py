#!/usr/bin/env python
"""Batch-align all VR collector directories to G1 canonical frame.

Scans data_collector/collectorN/ (N numeric, no _aligned suffix), runs
scripts/align_collector_to_g1fk.py on each, writes to collectorN_aligned/.

Usage (from project root):
    python scripts/batch_align_all_collectors.py
    python scripts/batch_align_all_collectors.py --only 10 11        # specific ids
    python scripts/batch_align_all_collectors.py --force              # overwrite existing
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_collector"
MODEL = ROOT / "mujoco_sim" / "model" / "task_scene.xml"
ALIGN = ROOT / "scripts" / "align_collector_to_g1fk.py"


def find_jsonls(collector_dir: Path):
    eps = sorted(collector_dir.glob("vr-demonstrations-episodes-*.jsonl"))
    evs = sorted(collector_dir.glob("vr-demonstrations-events-*.jsonl"))
    if not eps or not evs:
        return None, None
    # pair by timestamp prefix (strip prefix and keep YYYYMMDD_HHMMSS)
    return eps[-1], evs[-1]  # use latest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="Only process these collector ids (e.g. --only 10 11)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing collectorN_aligned directories")
    args = ap.parse_args()

    pat = re.compile(r"^collector(\d+)$")
    candidates = []
    for d in sorted(DATA.iterdir()):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        cid = int(m.group(1))
        if args.only and cid not in args.only:
            continue
        candidates.append((cid, d))

    if not candidates:
        print("No collector directories found matching pattern.")
        return 1

    print(f"[batch-align] candidates: {[c[0] for c in candidates]}")

    for cid, src in candidates:
        out = DATA / f"collector{cid}_aligned"
        if out.exists() and not args.force:
            print(f"[skip] {out.name} exists (use --force to overwrite)")
            continue

        eps, evs = find_jsonls(src)
        if eps is None:
            print(f"[warn] {src.name}: no episodes/events jsonl, skip")
            continue

        cmd = [
            sys.executable, str(ALIGN),
            "--model", str(MODEL),
            "--episodes", str(eps),
            "--events", str(evs),
            "--out", str(out),
        ]
        print(f"[run ] collector{cid} -> {out.name}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[fail] collector{cid} returned {r.returncode}")
            return r.returncode

    print("[batch-align] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
