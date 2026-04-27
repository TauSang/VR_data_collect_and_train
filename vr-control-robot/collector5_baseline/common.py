import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class EpisodeData:
    episode_id: int
    obs: np.ndarray  # [T, obs_dim]
    act: np.ndarray  # [T, act_dim]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path) -> Dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get3(d: Dict, key: str) -> List[float]:
    v = d.get(key, [0.0, 0.0, 0.0])
    if not isinstance(v, list) or len(v) != 3:
        return [0.0, 0.0, 0.0]
    out = []
    for x in v:
        try:
            out.append(float(x))
        except Exception:
            out.append(0.0)
    return out


def frame_to_obs_act(frame: Dict, joint_names: List[str], use_joint_velocities: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
    act = frame.get("action", {}) if isinstance(frame.get("action", {}), dict) else {}

    joint_pos = obs.get("jointPositions", {}) if isinstance(obs.get("jointPositions", {}), dict) else {}
    joint_vel = obs.get("jointVelocities", {}) if isinstance(obs.get("jointVelocities", {}), dict) else {}
    joint_delta = act.get("jointDelta", {}) if isinstance(act.get("jointDelta", {}), dict) else {}

    obs_vec: List[float] = []
    act_vec: List[float] = []

    for j in joint_names:
        obs_vec.extend(_get3(joint_pos, j))
        if use_joint_velocities:
            obs_vec.extend(_get3(joint_vel, j))
        act_vec.extend(_get3(joint_delta, j))

    return np.asarray(obs_vec, dtype=np.float32), np.asarray(act_vec, dtype=np.float32)


def load_episodes(episodes_jsonl: Path, joint_names: List[str], use_joint_velocities: bool) -> List[EpisodeData]:
    by_ep: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    with episodes_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            fr = json.loads(s)
            ep = int(fr.get("episodeId", 0) or 0)
            if ep <= 0:
                continue
            obs, act = frame_to_obs_act(fr, joint_names, use_joint_velocities)
            by_ep.setdefault(ep, []).append((obs, act))

    episodes: List[EpisodeData] = []
    for ep_id in sorted(by_ep.keys()):
        pairs = by_ep[ep_id]
        if not pairs:
            continue
        obs = np.stack([p[0] for p in pairs], axis=0)
        act = np.stack([p[1] for p in pairs], axis=0)
        episodes.append(EpisodeData(episode_id=ep_id, obs=obs, act=act))
    return episodes


def split_by_episode(episodes: List[EpisodeData], train_split: float) -> Tuple[List[EpisodeData], List[EpisodeData]]:
    n = len(episodes)
    if n < 2:
        raise RuntimeError("Need at least 2 episodes for split.")
    train_n = max(1, int(round(n * train_split)))
    train_n = min(train_n, n - 1)
    return episodes[:train_n], episodes[train_n:]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
