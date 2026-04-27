from common import load_config, load_segments
from pathlib import Path
cfg = load_config(Path("config_act_chunk_vr.json"))
segments, summary = load_segments(cfg)
print(f"Segments: {summary['total_segments']}")
print(f"Frames: {summary['total_frames']}")
print(f"Episodes: {summary['total_episodes']}")
print(f"Outcomes: {summary['outcome_totals']}")
for src in summary["per_source"]:
    name = src["source_name"]
    kept = src["num_kept_segments"]
    raw = src["raw_segment_outcomes"]
    keptout = src["kept_segment_outcomes"]
    print(f"  {name}: {kept} segs, raw={raw}, kept={keptout}")
