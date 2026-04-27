"""Validate strict eval artifacts for 20260426_train_v4."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "20260426_train_v4"
EXPECTED_SEEDS = [42, 7, 123, 2024, 31415]


def _check_arm(name: str, arm: dict) -> None:
    assert arm["agg_total"] == 250, f"{name}: agg_total={arm['agg_total']}"
    assert arm["agg_succ"] == sum(x["succ"] for x in arm["per_seed"]), name
    assert [x["seed"] for x in arm["per_seed"]] == EXPECTED_SEEDS, name
    assert all(x["total"] == 50 for x in arm["per_seed"]), name
    ckpt = ROOT / arm["ckpt"]
    assert ckpt.exists() and ckpt.stat().st_size > 0, f"missing checkpoint: {ckpt}"
    for row in arm["per_seed"]:
        eval_json = TRAIN_DIR / f"eval_{arm['label']}_seed{row['seed']}.json"
        assert eval_json.exists() and eval_json.stat().st_size > 0, f"missing eval JSON: {eval_json}"


def main() -> int:
    path = TRAIN_DIR / "eval_ablation.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    arms = {k: v for k, v in data.items() if k.startswith("arm_")}
    assert set(arms) == {"arm_filter_len40", "arm_filter_quality"}, sorted(arms)
    for name, arm in arms.items():
        _check_arm(name, arm)

    best_key = data["best_arm"]
    best = arms[best_key]
    print("=== 20260426 VR4 filtering strict eval verified ===")
    for name, arm in arms.items():
        print(f"{name}: {arm['agg_succ']}/{arm['agg_total']} = {arm['agg_sr']:.2f}% worst={arm['worst_sr']:.1f}%")
    print(f"Best: {best_key} = {best['agg_sr']:.2f}%")
    print(f"Delta best vs v9 98.80%: {best['agg_sr'] - 98.80:+.2f}pp")
    print(f"Delta best vs Scheme A 95.20%: {best['agg_sr'] - 95.20:+.2f}pp")
    if best["agg_sr"] >= 98.80:
        print("Decision: VR4 is usable after filtering; collect more data with this quality profile.")
    elif best["agg_sr"] > 95.20:
        print("Decision: filtering helps, but VR4 still does not beat v9; use only after more targeted collection/filtering.")
    else:
        print("Decision: filtered VR4 still does not improve over Scheme A/v9; current VR4 should be excluded from best recipe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
