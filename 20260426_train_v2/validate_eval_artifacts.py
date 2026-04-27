"""Validate strict eval artifacts for 20260426 Scheme A before citing results."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "20260426_train_v2" / "eval_ablation.json"
SEEDS = [42, 7, 123, 2024, 31415]
V2_BASELINE = 95.20
V9_BEST = 98.80


def _check_arm(name: str, arm: dict) -> None:
    seeds = [int(x["seed"]) for x in arm.get("per_seed", [])]
    if seeds != SEEDS:
        raise AssertionError(f"{name}: bad seeds {seeds}")
    for x in arm["per_seed"]:
        if int(x["total"]) != 50:
            raise AssertionError(f"{name}: seed {x['seed']} total={x['total']} != 50")
    if int(arm["agg_total"]) != 250:
        raise AssertionError(f"{name}: agg_total={arm['agg_total']} != 250")
    ckpt = ROOT / arm["ckpt"]
    if not ckpt.exists() or ckpt.stat().st_size <= 0:
        raise AssertionError(f"{name}: missing checkpoint {ckpt}")


def main() -> int:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    if data.get("seeds") != SEEDS:
        raise AssertionError(f"top-level seeds mismatch: {data.get('seeds')}")
    arms = [
        ("v9_p1", data["arm_A_v9_p1_mj_only"]),
        ("v9_p2", data["arm_B_v9_p2_mj_vr1_3"]),
        ("schemeA", data["arm_C_schemeA_mj_vr1_4"]),
    ]
    for name, arm in arms:
        _check_arm(name, arm)
    p1 = data["arm_A_v9_p1_mj_only"]
    v9 = data["arm_B_v9_p2_mj_vr1_3"]
    scheme = data["arm_C_schemeA_mj_vr1_4"]
    print("=== 20260426 Scheme A strict eval verified ===")
    print(f"v9 P1 MJ-only: {p1['agg_succ']}/{p1['agg_total']} = {p1['agg_sr']:.2f}% worst={p1['worst_sr']:.1f}%")
    print(f"v9 P2 MJ+VR1-3: {v9['agg_succ']}/{v9['agg_total']} = {v9['agg_sr']:.2f}% worst={v9['worst_sr']:.1f}%")
    print(f"Scheme A MJ+VR1-4: {scheme['agg_succ']}/{scheme['agg_total']} = {scheme['agg_sr']:.2f}% worst={scheme['worst_sr']:.1f}%")
    print(f"Delta Scheme A vs v9 P1: {scheme['agg_sr'] - p1['agg_sr']:+.2f}pp")
    print(f"Delta Scheme A vs v2 baseline {V2_BASELINE:.2f}%: {scheme['agg_sr'] - V2_BASELINE:+.2f}pp")
    print(f"Delta Scheme A vs v9 best {V9_BEST:.2f}%: {scheme['agg_sr'] - V9_BEST:+.2f}pp")
    if scheme["agg_sr"] >= V9_BEST:
        print("Decision: Scheme A matches/exceeds v9; data4 is useful under the clean v9 recipe.")
    elif scheme["agg_sr"] >= V2_BASELINE:
        print("Decision: Scheme A beats v2 but not v9; data4 likely needs weighting/domain isolation.")
    else:
        print("Decision: Scheme A does not beat v2/v9; prioritize VR4 quality or source-domain ablation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
