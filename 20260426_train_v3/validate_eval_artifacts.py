"""Validate strict eval artifacts for 20260426 Scheme B before citing results."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "20260426_train_v3" / "eval_ablation.json"
SEEDS = [42, 7, 123, 2024, 31415]
V2_BASELINE = 95.20
V9_P1 = 92.00
V9_BEST = 98.80
SCHEME_A = 95.20
FIRST_BATCH512 = 93.60


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
    scheme = data["arm_schemeB_vr4_domain_lowweight"]
    _check_arm("schemeB", scheme)
    sr = float(scheme["agg_sr"])
    print("=== 20260426 Scheme B strict eval verified ===")
    print(f"Scheme B: {scheme['agg_succ']}/{scheme['agg_total']} = {sr:.2f}% worst={scheme['worst_sr']:.1f}%")
    print(f"Delta Scheme B vs v9 P1 {V9_P1:.2f}%: {sr - V9_P1:+.2f}pp")
    print(f"Delta Scheme B vs first batch512 {FIRST_BATCH512:.2f}%: {sr - FIRST_BATCH512:+.2f}pp")
    print(f"Delta Scheme B vs Scheme A {SCHEME_A:.2f}%: {sr - SCHEME_A:+.2f}pp")
    print(f"Delta Scheme B vs v2 baseline {V2_BASELINE:.2f}%: {sr - V2_BASELINE:+.2f}pp")
    print(f"Delta Scheme B vs v9 best {V9_BEST:.2f}%: {sr - V9_BEST:+.2f}pp")
    if sr >= V9_BEST:
        print("Decision: Scheme B matches/exceeds v9; keep VR4 independent-domain low-weight recipe.")
    elif sr >= SCHEME_A:
        print("Decision: Scheme B improves over Scheme A but not v9; tune VR4 ratio/filters next.")
    else:
        print("Decision: Scheme B does not improve over Scheme A; VR4 likely needs filtering or exclusion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
