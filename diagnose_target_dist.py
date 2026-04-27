"""
Diagnose target distribution mismatch between VR data and MuJoCo evaluation.

VR targets: generated in robot-local frame, X[-0.45,+0.45], Y[0.85,1.7], Z[-1.2,-0.5]
MuJoCo targets: 0.28m radius sphere around shoulder

The observation feature "targetRelToRobotBase" has very different distributions!
"""
import json, math, sys
from pathlib import Path
import numpy as np

def load_episodes_jsonl(path):
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    return frames

def extract_target_rel(frame):
    """Get targetRelToRobotBase xyz from a frame."""
    obs = frame.get("obs", {})
    if not isinstance(obs, dict):
        return None
    task = obs.get("task", {})
    if not isinstance(task, dict):
        return None
    rel = task.get("targetRelToRobotBase", None)
    if rel is None:
        return None
    p = rel.get("p", None) if isinstance(rel, dict) else None
    if p is None:
        return None
    if isinstance(p, list) and len(p) == 3:
        return p
    if isinstance(p, dict):
        return [p.get("x", 0), p.get("y", 0), p.get("z", 0)]
    return None

def extract_hand_dists(frame):
    obs = frame.get("obs", {})
    if not isinstance(obs, dict):
        return None, None
    task = obs.get("task", {})
    if not isinstance(task, dict):
        return None, None
    dl = task.get("distToTargetLeft", None)
    dr = task.get("distToTargetRight", None)
    return dl, dr

def analyze_file(path, label):
    print(f"\n{'='*60}")
    print(f"  {label}: {Path(path).name}")
    print(f"{'='*60}")
    
    frames = load_episodes_jsonl(path)
    print(f"  Total frames: {len(frames)}")
    
    rels = []
    dists_left = []
    dists_right = []
    dists_min = []
    
    for f in frames:
        r = extract_target_rel(f)
        if r is not None:
            rels.append(r)
        dl, dr = extract_hand_dists(f)
        if dl is not None and math.isfinite(dl):
            dists_left.append(dl)
        if dr is not None and math.isfinite(dr):
            dists_right.append(dr)
        if dl is not None and dr is not None:
            dists_min.append(min(dl, dr))
    
    if not rels:
        print("  No targetRelToRobotBase data found!")
        return
    
    rels = np.array(rels)
    print(f"  Frames with target data: {len(rels)}")
    
    print(f"\n  targetRelToRobotBase (VR coords: X=lateral, Y=up, Z=forward):")
    for i, axis in enumerate(["X (lateral)", "Y (up)    ", "Z (forward)"]):
        vals = rels[:, i]
        print(f"    {axis}: mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"range=[{vals.min():.3f}, {vals.max():.3f}]")
    
    # Distance from robot base in 3D
    base_dists = np.linalg.norm(rels, axis=1)
    print(f"\n  3D distance target-to-base: mean={base_dists.mean():.3f}  "
          f"std={base_dists.std():.3f}  range=[{base_dists.min():.3f}, {base_dists.max():.3f}]")

    if dists_min:
        dists_min = np.array(dists_min)
        print(f"  Min hand distance: mean={dists_min.mean():.3f}  "
              f"std={dists_min.std():.3f}  range=[{dists_min.min():.3f}, {dists_min.max():.3f}]")
        
        # How many frames have target within arm reach?
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
            pct = (dists_min < thresh).mean() * 100
            print(f"    Frames with min_hand_dist < {thresh}m: {pct:.1f}%")
    
    # Distribution of the Z component (forward/backward) - critical for reaching
    z_vals = rels[:, 2]
    print(f"\n  Z (forward) distribution:")
    for edge in [-1.5, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.5]:
        pct = (z_vals < edge).mean() * 100
        print(f"    Z < {edge:+.1f}m: {pct:.1f}%")

# ── MuJoCo expected distribution ──
def show_mujoco_expected():
    print(f"\n{'='*60}")
    print(f"  MuJoCo evaluation target distribution (expected)")
    print(f"{'='*60}")
    print(f"  Targets sampled within 0.28m sphere of a shoulder")
    print(f"  robot_base_vr = [0.0, 0.793, 0.0]")
    print(f"  Shoulder in VR coords: ~[±0.15, 1.2, -0.08]")
    print(f"  targetRelToBase = target_vr - robot_base_vr")
    print(f"  Expected ranges:")
    print(f"    X (lateral):  roughly [-0.43, +0.43]")
    print(f"    Y (up):       roughly [0.13, 0.69]")
    print(f"    Z (forward):  roughly [-0.36, +0.20]")
    print(f"  3D distance target-to-base: ~0.4-0.7m")
    print(f"  Min hand distance: always < 0.28m (by design)")

# ── Run ──
data_dir = Path("e:/XT/vr-robot-control/data_collector")

vr_files = [
    (data_dir / "collector6" / "vr-demonstrations-episodes-20260408_203644.jsonl", "VR collector6"),
    (data_dir / "collector8" / "vr-demonstrations-episodes-20260409_183018.jsonl", "VR collector8"),
]

mujoco_files = [
    (data_dir / "mujoco_expert" / "episodes.jsonl", "MuJoCo expert"),
]

for path, label in vr_files:
    if path.exists():
        analyze_file(str(path), label)
    else:
        print(f"  {label}: FILE NOT FOUND at {path}")

for path, label in mujoco_files:
    if path.exists():
        analyze_file(str(path), label)
    else:
        print(f"  {label}: FILE NOT FOUND at {path}")

show_mujoco_expected()
