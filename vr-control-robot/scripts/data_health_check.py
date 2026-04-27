import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vrtrain.utils.config import load_yaml


def _safe_get3(d: dict, key: str):
    v = d.get(key)
    if not isinstance(v, list) or len(v) != 3:
        return [0.0, 0.0, 0.0]
    return [float(v[0]), float(v[1]), float(v[2])]


def analyze_jsonl(jsonl_path: Path, joint_names: list[str], large_delta_threshold: float = 1.5):
    episode_counts = defaultdict(int)
    dt_values = []

    total_frames = 0
    bad_episode_frames = 0
    missing_obs_frames = 0
    missing_action_frames = 0
    missing_joint_pos_count = 0
    missing_joint_delta_count = 0
    zero_action_frames = 0
    large_action_frames = 0
    max_abs_joint_delta = 0.0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_frames += 1

            frame = json.loads(line)
            ep = int(frame.get("episodeId", 0))
            if ep <= 0:
                bad_episode_frames += 1
            else:
                episode_counts[ep] += 1

            dt_s = frame.get("dt", None)
            if isinstance(dt_s, (int, float)) and float(dt_s) > 0:
                dt_values.append(float(dt_s))

            obs = frame.get("obs")
            action = frame.get("action")
            if not isinstance(obs, dict):
                missing_obs_frames += 1
                obs = {}
            if not isinstance(action, dict):
                missing_action_frames += 1
                action = {}

            jp = obs.get("jointPositions", {}) if isinstance(obs.get("jointPositions", {}), dict) else {}
            jd = action.get("jointDelta", {}) if isinstance(action.get("jointDelta", {}), dict) else {}

            all_deltas = []
            for j in joint_names:
                if j not in jp:
                    missing_joint_pos_count += 1
                if j not in jd:
                    missing_joint_delta_count += 1

                d3 = _safe_get3(jd, j)
                all_deltas.extend(d3)
                local_abs = max(abs(v) for v in d3)
                max_abs_joint_delta = max(max_abs_joint_delta, local_abs)

            if all(abs(v) < 1e-8 for v in all_deltas):
                zero_action_frames += 1
            if any(abs(v) > large_delta_threshold for v in all_deltas):
                large_action_frames += 1

    episode_lengths = list(episode_counts.values())
    episode_num = len(episode_lengths)

    dt_np = np.asarray(dt_values, dtype=np.float64) if dt_values else np.asarray([], dtype=np.float64)
    dt_stats = {
        "count": int(dt_np.size),
        "mean": float(np.mean(dt_np)) if dt_np.size else None,
        "std": float(np.std(dt_np)) if dt_np.size else None,
        "p50": float(np.percentile(dt_np, 50)) if dt_np.size else None,
        "p95": float(np.percentile(dt_np, 95)) if dt_np.size else None,
        "min": float(np.min(dt_np)) if dt_np.size else None,
        "max": float(np.max(dt_np)) if dt_np.size else None,
    }

    # 以 30Hz 目标采样为参考，dt > 0.1s 视为异常慢帧
    slow_dt_ratio = float(np.mean(dt_np > 0.1)) if dt_np.size else None

    report = {
        "dataset": str(jsonl_path),
        "checked_at": dt.datetime.now().isoformat(),
        "joint_names": joint_names,
        "total_frames": total_frames,
        "episode_count": episode_num,
        "episode_length": {
            "min": int(min(episode_lengths)) if episode_lengths else 0,
            "max": int(max(episode_lengths)) if episode_lengths else 0,
            "mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "median": float(np.median(episode_lengths)) if episode_lengths else 0.0,
        },
        "bad_episode_frames": bad_episode_frames,
        "missing_obs_frames": missing_obs_frames,
        "missing_action_frames": missing_action_frames,
        "missing_joint_position_entries": missing_joint_pos_count,
        "missing_joint_delta_entries": missing_joint_delta_count,
        "zero_action_frames": zero_action_frames,
        "zero_action_ratio": float(zero_action_frames / total_frames) if total_frames > 0 else None,
        "large_action_frames": large_action_frames,
        "large_action_ratio": float(large_action_frames / total_frames) if total_frames > 0 else None,
        "max_abs_joint_delta": float(max_abs_joint_delta),
        "dt_stats": dt_stats,
        "slow_dt_ratio_dt_gt_0_1": slow_dt_ratio,
    }

    # 简单健康判定
    report["health_pass"] = bool(
        total_frames > 0
        and episode_num > 0
        and bad_episode_frames == 0
        and (report["zero_action_ratio"] is None or report["zero_action_ratio"] < 0.2)
        and (slow_dt_ratio is None or slow_dt_ratio < 0.1)
    )

    return report


def collect_jsonl_paths_from_config(cfg: dict):
    data_cfg = cfg.get("data", {})
    cands = [
        data_cfg.get("raw_jsonl"),
        data_cfg.get("train_raw_jsonl"),
        data_cfg.get("test_raw_jsonl"),
    ]
    return [Path(p) for p in cands if isinstance(p, str) and p.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cross_c3_train_c2_test.yaml")
    parser.add_argument("--jsonl", action="append", default=[], help="可多次传入，覆盖 config 中的数据路径")
    parser.add_argument("--out-dir", default="./artifacts/health_reports")
    parser.add_argument("--large-delta-threshold", type=float, default=1.5)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    joint_names = list(cfg.get("data", {}).get("joint_names", []))
    if not joint_names:
        raise RuntimeError("config.data.joint_names 为空，无法体检")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.jsonl:
        jsonl_paths = [Path(p) for p in args.jsonl]
    else:
        jsonl_paths = collect_jsonl_paths_from_config(cfg)

    if not jsonl_paths:
        raise RuntimeError("未找到可检查的 JSONL 路径，请传 --jsonl 或在 config 中配置 data.raw_jsonl/train_raw_jsonl/test_raw_jsonl")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    all_reports = []
    for p in jsonl_paths:
        if not p.exists():
            print(f"[health] skip missing: {p}")
            continue

        report = analyze_jsonl(p, joint_names, large_delta_threshold=float(args.large_delta_threshold))
        all_reports.append(report)

        single_out = out_dir / f"health_{p.stem}_{ts}.json"
        with single_out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[health] saved: {single_out}")

    if not all_reports:
        raise RuntimeError("没有生成任何体检报告（可能路径不存在）")

    merged_out = out_dir / f"health_summary_{ts}.json"
    with merged_out.open("w", encoding="utf-8") as f:
        json.dump({"reports": all_reports, "created_at": dt.datetime.now().isoformat()}, f, ensure_ascii=False, indent=2)
    print(f"[health] merged summary: {merged_out}")


if __name__ == "__main__":
    main()
