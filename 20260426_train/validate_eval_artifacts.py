"""Validate strict 20260426 eval artifacts and print comparison against v2/v9 baselines."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "20260426_train"
EXPECTED_SEEDS = [42, 7, 123, 2024, 31415]
BASELINE_V2_SR = 95.20
BASELINE_V9_SR = 98.80


def _check_arm(name: str, arm: dict) -> None:
    seeds = [row.get("seed") for row in arm.get("per_seed", [])]
    totals = [row.get("total") for row in arm.get("per_seed", [])]
    if seeds != EXPECTED_SEEDS:
        raise AssertionError(f"{name}: unexpected seeds {seeds}; expected {EXPECTED_SEEDS}")
    if totals != [50] * len(EXPECTED_SEEDS):
        raise AssertionError(f"{name}: every seed must evaluate 50 targets, got {totals}")
    if arm.get("agg_total") != 250:
        raise AssertionError(f"{name}: agg_total must be 250, got {arm.get('agg_total')}")
    ckpt = ROOT / arm["ckpt"]
    if not ckpt.exists():
        raise FileNotFoundError(f"{name}: checkpoint missing: {ckpt}")


def main() -> int:
    path = TRAIN_DIR / "eval_ablation.json"
    if not path.exists():
        raise FileNotFoundError(path)
    results = json.loads(path.read_text(encoding="utf-8"))
    if results.get("seeds") != EXPECTED_SEEDS:
        raise AssertionError(f"top-level seeds mismatch: {results.get('seeds')}")
    _check_arm("arm_A_mj_only", results["arm_A_mj_only"])
    _check_arm("arm_B_mj_vr_ft", results["arm_B_mj_vr_ft"])

    p1 = results["arm_A_mj_only"]
    p2 = results["arm_B_mj_vr_ft"]
    print("=== 20260426 strict eval verified ===")
    print(f"P1 MJ-only: {p1['agg_succ']}/{p1['agg_total']} = {p1['agg_sr']:.2f}% worst={p1['worst_sr']:.1f}%")
    print(f"P2 MJ+data1-4: {p2['agg_succ']}/{p2['agg_total']} = {p2['agg_sr']:.2f}% worst={p2['worst_sr']:.1f}%")
    print(f"Delta P2-P1: {results['delta_pp']:+.2f}pp")
    print(f"vs v2 baseline 95.20%: {p2['agg_sr'] - BASELINE_V2_SR:+.2f}pp")
    print(f"vs v9 best 98.80%: {p2['agg_sr'] - BASELINE_V9_SR:+.2f}pp")
    if p2["agg_sr"] > BASELINE_V9_SR:
        print("Decision: data1-4 improves over v9; more data collection is supported.")
    else:
        print("Decision: data1-4 does not exceed v9; inspect VR4 quality/weighting before collecting more.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
