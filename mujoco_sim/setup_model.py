"""
Download Unitree G1 model from mujoco_menagerie and set up for dual-arm tasks.

Two download strategies:
  1. Fast: download individual files via raw.githubusercontent.com (default)
  2. Full: download entire menagerie zip (fallback, ~200MB)
"""
import io
import os
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

RAW_BASE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_g1"
MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip"
MODEL_DIR = Path(__file__).resolve().parent / "model"
NEEDED_PREFIX = "mujoco_menagerie-main/unitree_g1/"

# Files needed for G1 model (g1.xml + assets)
G1_FILES = [
    "g1.xml",
    "scene.xml",
    "LICENSE",
]

# STL mesh assets referenced in g1.xml
G1_ASSETS = [
    "pelvis.STL", "pelvis_contour_link.STL",
    "left_hip_pitch_link.STL", "left_hip_roll_link.STL", "left_hip_yaw_link.STL",
    "left_knee_link.STL", "left_ankle_pitch_link.STL", "left_ankle_roll_link.STL",
    "right_hip_pitch_link.STL", "right_hip_roll_link.STL", "right_hip_yaw_link.STL",
    "right_knee_link.STL", "right_ankle_pitch_link.STL", "right_ankle_roll_link.STL",
    "waist_yaw_link_rev_1_0.STL", "waist_roll_link_rev_1_0.STL",
    "torso_link_rev_1_0.STL", "logo_link.STL", "head_link.STL",
    "left_shoulder_pitch_link.STL", "left_shoulder_roll_link.STL",
    "left_shoulder_yaw_link.STL", "left_elbow_link.STL",
    "left_wrist_roll_link.STL", "left_wrist_pitch_link.STL",
    "left_wrist_yaw_link.STL", "left_rubber_hand.STL",
    "right_shoulder_pitch_link.STL", "right_shoulder_roll_link.STL",
    "right_shoulder_yaw_link.STL", "right_elbow_link.STL",
    "right_wrist_roll_link.STL", "right_wrist_pitch_link.STL",
    "right_wrist_yaw_link.STL", "right_rubber_hand.STL",
]


def download_file(url: str, target: Path) -> bool:
    """Download a single file."""
    try:
        resp = urlopen(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            f.write(resp.read())
        return True
    except (URLError, OSError) as e:
        print(f"  [FAIL] {url}: {e}")
        return False


def download_individual_files():
    """Download G1 model files individually (faster)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    assets_dir = MODEL_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    total = len(G1_FILES) + len(G1_ASSETS)
    done = 0

    for fname in G1_FILES:
        target = MODEL_DIR / fname
        if target.exists():
            done += 1
            continue
        url = f"{RAW_BASE}/{fname}"
        print(f"  [{done + 1}/{total}] {fname} ...")
        if download_file(url, target):
            done += 1

    for fname in G1_ASSETS:
        target = assets_dir / fname
        if target.exists():
            done += 1
            continue
        url = f"{RAW_BASE}/assets/{fname}"
        print(f"  [{done + 1}/{total}] assets/{fname} ...")
        if download_file(url, target):
            done += 1

    return done, total


def download_and_extract():
    if (MODEL_DIR / "g1.xml").exists():
        print(f"[OK] G1 model already exists at {MODEL_DIR / 'g1.xml'}")
        return

    print("Downloading G1 model files individually...")
    done, total = download_individual_files()
    print(f"Downloaded {done}/{total} files")

    if (MODEL_DIR / "g1.xml").exists():
        print("[OK] G1 model ready.")
        return

    # Fallback: download full zip
    print(f"\nFallback: downloading full mujoco_menagerie zip...")
    resp = urlopen(MENAGERIE_URL)
    data = resp.read()
    print(f"Downloaded {len(data) / 1024 / 1024:.1f} MB")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        extracted = 0
        for info in zf.infolist():
            if not info.filename.startswith(NEEDED_PREFIX):
                continue
            rel = info.filename[len(NEEDED_PREFIX):]
            if not rel:
                continue
            target = MODEL_DIR / rel
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
        print(f"Extracted {extracted} files to {MODEL_DIR}")

    if (MODEL_DIR / "g1.xml").exists():
        print("[OK] G1 model ready.")
    else:
        print("[ERROR] g1.xml not found after extraction!")


def verify_model():
    """Verify the model loads in MuJoCo."""
    try:
        import mujoco
    except ImportError:
        print("[WARN] mujoco not installed, skipping verification. Run: pip install mujoco")
        return False

    xml_path = MODEL_DIR / "g1.xml"
    if not xml_path.exists():
        print(f"[ERROR] {xml_path} does not exist")
        return False

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        print(f"[OK] Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators, {model.nbody} bodies")

        # Print arm joint info
        arm_joints = []
        for i in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and ("shoulder" in name or "elbow" in name or "wrist" in name):
                arm_joints.append((i, name))
        print(f"\nArm joints ({len(arm_joints)}):")
        for idx, name in arm_joints:
            qpos_adr = model.jnt_qposadr[idx]
            limited = model.jnt_limited[idx]
            if limited:
                lo = model.jnt_range[idx, 0]
                hi = model.jnt_range[idx, 1]
                print(f"  [{idx:2d}] {name:<35s}  qpos[{qpos_adr}]  range=[{lo:.3f}, {hi:.3f}]")
            else:
                print(f"  [{idx:2d}] {name:<35s}  qpos[{qpos_adr}]  unlimited")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return False


if __name__ == "__main__":
    download_and_extract()
    verify_model()
