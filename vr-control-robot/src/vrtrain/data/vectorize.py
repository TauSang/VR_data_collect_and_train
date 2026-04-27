import numpy as np


def _get3(d: dict, key: str):
    v = d.get(key, [0.0, 0.0, 0.0])
    if v is None:
        return [0.0, 0.0, 0.0]
    return v


def frame_to_obs_action(frame: dict, joint_names: list[str], use_joint_velocities: bool, include_gripper: bool):
    obs = frame.get("obs", {})
    act = frame.get("action", {})

    joint_pos = obs.get("jointPositions", {})
    joint_vel = obs.get("jointVelocities", {})
    joint_delta = act.get("jointDelta", {})

    obs_vec = []
    act_vec = []

    for j in joint_names:
        obs_vec.extend(_get3(joint_pos, j))
        if use_joint_velocities:
            obs_vec.extend(_get3(joint_vel, j))
        act_vec.extend(_get3(joint_delta, j))

    if include_gripper:
        gs = obs.get("gripperState", {})
        gc = act.get("gripperCommand", {})
        obs_vec.extend([float(gs.get("left", 0.0)), float(gs.get("right", 0.0))])
        act_vec.extend([float(gc.get("left", 0.0)), float(gc.get("right", 0.0))])

    return np.asarray(obs_vec, dtype=np.float32), np.asarray(act_vec, dtype=np.float32)
