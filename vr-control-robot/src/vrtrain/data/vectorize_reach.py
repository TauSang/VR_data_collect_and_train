import numpy as np


PHASE_TO_INDEX = {
    "idle": 0,
    "reach": 1,
    "align": 2,
    "hold": 3,
    "grasp": 4,
    "success": 5,
    "timeout": 6,
}


def _get3(d: dict, key: str):
    v = d.get(key, [0.0, 0.0, 0.0])
    if not isinstance(v, list) or len(v) != 3:
        return [0.0, 0.0, 0.0]
    return [float(v[0]), float(v[1]), float(v[2])]


def _get_scalar(d: dict, key: str, default: float = 0.0):
    try:
        return float(d.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _get_pose_p(node, fallback=None):
    if isinstance(node, dict):
        p = node.get("p", None)
        if isinstance(p, list) and len(p) == 3:
            return [float(p[0]), float(p[1]), float(p[2])]
    if fallback is not None:
        return list(fallback)
    return [0.0, 0.0, 0.0]


def infer_assigned_hand(task: dict, fallback: str | None = None):
    for key in ("assignedHand", "successHand", "nearestHand"):
        hand = task.get(key, None)
        if hand in ("left", "right"):
            return hand
    return fallback


def task_distance(task: dict, assigned_hand: str | None = None):
    if assigned_hand == "left":
        return _get_scalar(task, "distToTargetLeft", _get_scalar(task, "distToTarget", 0.0))
    if assigned_hand == "right":
        return _get_scalar(task, "distToTargetRight", _get_scalar(task, "distToTarget", 0.0))
    left = _get_scalar(task, "distToTargetLeft", _get_scalar(task, "distToTarget", 0.0))
    right = _get_scalar(task, "distToTargetRight", _get_scalar(task, "distToTarget", 0.0))
    vals = [v for v in (left, right) if np.isfinite(v)]
    if not vals:
        return 0.0
    return float(min(vals))


def frame_to_reach_obs_action(
    frame: dict,
    joint_names: list[str],
    use_joint_velocities: bool,
    include_phase: bool,
    default_assigned_hand: str | None = None,
):
    obs = frame.get("obs", {}) if isinstance(frame.get("obs", {}), dict) else {}
    act = frame.get("action", {}) if isinstance(frame.get("action", {}), dict) else {}
    task = obs.get("task", {}) if isinstance(obs.get("task", {}), dict) else {}

    joint_pos = obs.get("jointPositions", {}) if isinstance(obs.get("jointPositions", {}), dict) else {}
    joint_vel = obs.get("jointVelocities", {}) if isinstance(obs.get("jointVelocities", {}), dict) else {}
    end_eff = obs.get("endEffector", {}) if isinstance(obs.get("endEffector", {}), dict) else {}
    joint_delta = act.get("jointDelta", {}) if isinstance(act.get("jointDelta", {}), dict) else {}

    assigned_hand = infer_assigned_hand(task, fallback=default_assigned_hand)
    hand_one_hot = [0.0, 0.0]
    if assigned_hand == "left":
        hand_one_hot = [1.0, 0.0]
    elif assigned_hand == "right":
        hand_one_hot = [0.0, 1.0]

    obs_vec = []
    act_vec = []

    for joint_name in joint_names:
        obs_vec.extend(_get3(joint_pos, joint_name))
        if use_joint_velocities:
            obs_vec.extend(_get3(joint_vel, joint_name))
        act_vec.extend(_get3(joint_delta, joint_name))

    obs_vec.extend(_get_pose_p(end_eff.get("left", None)))
    obs_vec.extend(_get_pose_p(end_eff.get("right", None)))
    obs_vec.extend(_get_pose_p(task.get("targetRelToRobotBase", None), fallback=_get_pose_p(task.get("targetPose", None))))
    obs_vec.extend(_get_pose_p(task.get("targetRelToLeftHand", None)))
    obs_vec.extend(_get_pose_p(task.get("targetRelToRightHand", None)))
    obs_vec.extend([
        _get_scalar(task, "distToTargetLeft", 0.0),
        _get_scalar(task, "distToTargetRight", 0.0),
    ])
    obs_vec.extend(hand_one_hot)

    if include_phase:
        phase = task.get("phaseLabel", "idle")
        phase_one_hot = [0.0] * len(PHASE_TO_INDEX)
        phase_idx = PHASE_TO_INDEX.get(phase, 0)
        phase_one_hot[phase_idx] = 1.0
        obs_vec.extend(phase_one_hot)

    return np.asarray(obs_vec, dtype=np.float32), np.asarray(act_vec, dtype=np.float32)
