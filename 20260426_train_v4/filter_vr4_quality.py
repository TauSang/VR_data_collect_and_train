"""Create filtered VR4 datasets for 20260426_train_v4.

The training code consumes JSONL datasets, so this script materializes filtered
VR4 subsets under 20260426_train_v4/filtered_data/ instead of changing the
shared loader. This keeps the experiment reproducible and easy to compare.
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
V9_DIR = ROOT / "20260425_train_v9"
if str(V9_DIR) not in sys.path:
    sys.path.insert(0, str(V9_DIR))

from common import load_segments  # noqa: E402

JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]

SRC_DIR = ROOT / "data_collector" / "vr4_aligned"
OUT_ROOT = Path(__file__).resolve().parent / "filtered_data"


def _base_config() -> dict:
    return {
        "data": {
            "data_sources": [
                {
                    "name": "vr4_aligned",
                    "episodes_jsonl": "../data_collector/vr4_aligned/episodes.jsonl",
                    "events_jsonl": "../data_collector/vr4_aligned/events.jsonl",
                    "source_id": 1,
                }
            ],
            "g1_joint_names": JOINTS,
            "allowed_outcomes": ["success", "timeout"],
            "max_abs_g1_velocity": 10.0,
            "max_abs_g1_delta": 1.0,
            "sample_weighting": {
                "success_bonus": 0.5,
                "near_target_bonus": 0.3,
                "near_target_threshold": 0.25,
                "idle_discount": 0.3,
                "moving_bonus": 0.15,
                "approaching_bonus": 0.25,
                "max_weight": 3.0,
            },
        }
    }


def _segment_stats(seg) -> dict:
    act_l2 = np.linalg.norm(seg.act, axis=1)
    return {
        "episode_id": int(seg.episode_id),
        "target_index": int(seg.target_index),
        "target_id": int(seg.target_id),
        "outcome": str(seg.outcome),
        "length": int(seg.act.shape[0]),
        "frames": int(seg.act.shape[0]),
        "act_l2_mean": float(act_l2.mean()),
        "act_l2_p95": float(np.percentile(act_l2, 95)),
        "act_l2_max": float(act_l2.max()),
    }


def _keep(stats: dict, spec: dict) -> bool:
    if stats["length"] < int(spec.get("min_segment_len", 0)):
        return False
    max_mean = spec.get("max_act_l2_mean")
    if max_mean is not None and stats["act_l2_mean"] > float(max_mean):
        return False
    max_p95 = spec.get("max_act_l2_p95")
    if max_p95 is not None and stats["act_l2_p95"] > float(max_p95):
        return False
    return True


def _frame_key(frame: dict) -> tuple[int, int, int] | None:
    obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
    task = obs.get("task", {}) if isinstance(obs.get("task", {}), dict) else {}
    ep = int(frame.get("episodeId", 0) or 0)
    ti = int(task.get("targetIndex", 0) or 0)
    tid = int(task.get("targetId", 0) or 0)
    if ep <= 0 or ti <= 0 or tid <= 0:
        return None
    return ep, ti, tid


def _event_key(ev: dict) -> tuple[int, int, int] | None:
    payload = ev.get("payload", {}) if isinstance(ev.get("payload", {}), dict) else {}
    ep = int(ev.get("episodeId", 0) or 0)
    ti = int(payload.get("targetIndex", 0) or 0)
    tid = int(payload.get("targetId", 0) or 0)
    if ep <= 0 or ti <= 0 or tid <= 0:
        return None
    return ep, ti, tid


def _write_subset(name: str, kept_keys: set[tuple[int, int, int]]) -> dict:
    out_dir = OUT_ROOT / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_episodes = {k[0] for k in kept_keys}
    frame_count = 0
    frame_segments = Counter()
    with (SRC_DIR / "episodes.jsonl").open("r", encoding="utf-8") as fin, \
            (out_dir / "episodes.jsonl").open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            frame = json.loads(line)
            key = _frame_key(frame)
            if key in kept_keys:
                fout.write(line if line.endswith("\n") else line + "\n")
                frame_count += 1
                frame_segments[key] += 1

    event_count = 0
    target_event_types = {"target_spawned", "target_success", "target_reached", "episode_timeout"}
    episode_event_types = {"episode_start", "episode_end"}
    with (SRC_DIR / "events.jsonl").open("r", encoding="utf-8") as fin, \
            (out_dir / "events.jsonl").open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            ev = json.loads(line)
            ev_type = str(ev.get("type", ""))
            key = _event_key(ev)
            ep = int(ev.get("episodeId", 0) or 0)
            keep = (ev_type in target_event_types and key in kept_keys) or \
                (ev_type in episode_event_types and ep in kept_episodes)
            if keep:
                fout.write(line if line.endswith("\n") else line + "\n")
                event_count += 1

    return {
        "name": name,
        "segments": len(frame_segments),
        "frames": frame_count,
        "events": event_count,
        "path": str(out_dir.relative_to(ROOT)),
    }


def main() -> int:
    cfg = _base_config()
    segments, summary = load_segments(cfg)
    stats = [_segment_stats(s) for s in segments]

    specs = {
        "vr4_len40": {
            "description": "Keep VR4 segments with length >= 40 frames.",
            "min_segment_len": 40,
        },
        "vr4_quality_v1": {
            "description": "Keep VR4 segments with length >= 40 and mean action L2 <= 0.06.",
            "min_segment_len": 40,
            "max_act_l2_mean": 0.06,
        },
    }

    report = {
        "source": "data_collector/vr4_aligned",
        "input_summary": summary,
        "input_segments": len(stats),
        "input_frames": int(sum(x["frames"] for x in stats)),
        "input_outcomes": dict(Counter(x["outcome"] for x in stats)),
        "variants": {},
    }

    for name, spec in specs.items():
        kept = [x for x in stats if _keep(x, spec)]
        kept_keys = {(x["episode_id"], x["target_index"], x["target_id"]) for x in kept}
        write_summary = _write_subset(name, kept_keys)
        report["variants"][name] = {
            "filter": spec,
            "kept_segments": len(kept),
            "kept_frames": int(sum(x["frames"] for x in kept)),
            "kept_outcomes": dict(Counter(x["outcome"] for x in kept)),
            "dropped_segments": len(stats) - len(kept),
            "dropped_frames": int(sum(x["frames"] for x in stats) - sum(x["frames"] for x in kept)),
            "materialized": write_summary,
            "kept_examples_head": kept[:10],
        }
        print(f"[{name}] kept {len(kept)}/{len(stats)} segments, "
              f"{sum(x['frames'] for x in kept)}/{sum(x['frames'] for x in stats)} frames, "
              f"outcomes={dict(Counter(x['outcome'] for x in kept))}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = Path(__file__).resolve().parent / "filter_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
