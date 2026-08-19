#!/usr/bin/env python3
"""Reference implementation of the S10 observation / action convention.

Drop this into the IsaacLab side so the training environment builds its
observation exactly the way rl_deploy does. Hand-transcribing the layout is the
most common way a policy ends up useless after transfer: nothing errors, the
robot just does not walk, and the cause is a permuted joint or a missing scale.

Every constant here mirrors src/S10_sdk_deploy/run_policy/s10_policy_runner.hpp.
Running this file as a script re-reads that header and asserts they still agree,
so the two cannot drift apart silently:

    /usr/bin/python3 src/S10_sdk_deploy/scripts/s10_obs_reference.py

See doc/policy_interface.md for the prose version.
"""

from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
CPP_HEADER = (CURRENT_DIR / ".." / "run_policy" / "s10_policy_runner.hpp").resolve()

# ---------------------------------------------------------------- constants --

BASE_OBS_DIM = 57
ACTION_DIM = 16
MOTOR_NUM = 16

OMEGA_SCALE = 0.25     # omega_scale_
DOF_VEL_SCALE = 0.05   # dof_vel_scale_

# Hardware / simulator ordering: each leg is hipx, hipy, knee, wheel
ROBOT_ORDER = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint",
]

# Policy ordering: all twelve leg joints first, then the four wheels
POLICY_ORDER = [
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
]

DOF_DEFAULT_POLICY = np.array([
    0.0, -0.3, 0.6,
    0.0, -0.3, 0.6,
    0.0, 0.3, -0.6,
    0.0, 0.3, -0.6,
    0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)

DOF_DEFAULT_ROBOT = np.array([
    0.0, -0.3, 0.6, 0.0,
    0.0, -0.3, 0.6, 0.0,
    0.0, 0.3, -0.6, 0.0,
    0.0, 0.3, -0.6, 0.0,
], dtype=np.float32)

# Per leg: hipx, hipy, knee, wheel
ACTION_SCALE_ROBOT = np.array([0.125, 0.25, 0.25, 5.0] * 4, dtype=np.float32)
KP = np.array([80.0, 80.0, 80.0, 0.0] * 4, dtype=np.float32)
KD = np.array([2.0, 2.0, 2.0, 0.6] * 4, dtype=np.float32)

GRAVITY_DIRECTION = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _permutation(src_names, dst_names):
    """perm[i] = index in src_names of dst_names[i].

    Mirrors generate_permutation() in the C++: gather src-ordered data into
    dst order with ``dst_data[i] = src_data[perm[i]]``.
    """
    idx = {name: i for i, name in enumerate(src_names)}
    return np.array([idx[n] for n in dst_names], dtype=np.int64)


ROBOT2POLICY = _permutation(ROBOT_ORDER, POLICY_ORDER)
POLICY2ROBOT = _permutation(POLICY_ORDER, ROBOT_ORDER)


# ------------------------------------------------------------------- helpers --

def quat_to_rotmat(quat_wxyz) -> np.ndarray:
    """World-from-body rotation matrix. Quaternion in (w, x, y, z), MuJoCo order."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def projected_gravity(rot_world_from_body: np.ndarray) -> np.ndarray:
    """Gravity direction expressed in the body frame."""
    return (rot_world_from_body.T @ GRAVITY_DIRECTION).astype(np.float32)


def normalize_lidar(ranges, range_max: float = 10.0) -> np.ndarray:
    """inf / out-of-range -> range_max, then scale to [0, 1].

    Must match LidarInterface + S10PolicyRunner::SetLidarRanges on the deploy
    side, or the network sees a different distribution than it trained on.
    """
    r = np.asarray(ranges, dtype=np.float32).copy()
    r[~np.isfinite(r)] = range_max
    np.clip(r, 0.0, range_max, out=r)
    return r / range_max


# --------------------------------------------------------------- observation --

def build_observation(base_ang_vel_body,
                      base_quat_wxyz,
                      command,
                      joint_pos_robot,
                      joint_vel_robot,
                      last_action,
                      lidar_ranges=None,
                      lidar_range_max: float = 10.0) -> np.ndarray:
    """Assemble the observation exactly as S10PolicyRunner::getRobotAction does.

    Args:
        base_ang_vel_body: (3,) body-frame angular velocity, rad/s, unscaled
        base_quat_wxyz:    (4,) base orientation, MuJoCo (w, x, y, z) order
        command:           (3,) [forward, side, yaw]; deploy clamps to
                           +-1.0 / +-0.6 / +-1.0
        joint_pos_robot:   (16,) joint positions in ROBOT order, rad
        joint_vel_robot:   (16,) joint velocities in ROBOT order, rad/s
        last_action:       (16,) previous raw network output, POLICY order
        lidar_ranges:      optional (N,) metric ranges; appended normalised

    Returns:
        (57,) or (57 + N,) float32
    """
    ang = np.asarray(base_ang_vel_body, dtype=np.float32) * OMEGA_SCALE
    grav = projected_gravity(quat_to_rotmat(base_quat_wxyz))
    cmd = np.asarray(command, dtype=np.float32)

    jp = np.asarray(joint_pos_robot, dtype=np.float32)[ROBOT2POLICY]
    jv = np.asarray(joint_vel_robot, dtype=np.float32)[ROBOT2POLICY] * DOF_VEL_SCALE

    # Wheels spin continuously, so their absolute angle carries no information
    # and is zeroed before the default offset is removed.
    jp[12:16] = 0.0
    jp = jp - DOF_DEFAULT_POLICY

    la = np.asarray(last_action, dtype=np.float32)

    for name, arr, want in (("base_ang_vel", ang, 3), ("command", cmd, 3),
                            ("joint_pos", jp, 16), ("joint_vel", jv, 16),
                            ("last_action", la, 16)):
        if arr.shape != (want,):
            raise ValueError(f"{name} must have shape ({want},), got {arr.shape}")

    obs = np.concatenate([ang, grav, cmd, jp, jv, la]).astype(np.float32)
    assert obs.shape == (BASE_OBS_DIM,), obs.shape

    if lidar_ranges is not None:
        obs = np.concatenate([obs, normalize_lidar(lidar_ranges, lidar_range_max)])
    return obs.astype(np.float32)


# -------------------------------------------------------------------- action --

def decode_action(action_policy):
    """Turn a raw network output into the joint targets rl_deploy will send.

    Returns:
        (goal_joint_pos, goal_joint_vel), both (16,) in ROBOT order.
        Leg joints are position-controlled; the wheel of each leg is
        velocity-controlled. Entries that are not used for a given joint
        are left at zero, matching the C++.
    """
    a = np.asarray(action_policy, dtype=np.float32)
    if a.shape != (ACTION_DIM,):
        raise ValueError(f"action must have shape ({ACTION_DIM},), got {a.shape}")

    tmp = a[POLICY2ROBOT] * ACTION_SCALE_ROBOT + DOF_DEFAULT_ROBOT

    goal_pos = np.zeros(MOTOR_NUM, dtype=np.float32)
    goal_vel = np.zeros(MOTOR_NUM, dtype=np.float32)
    for leg in range(4):
        goal_pos[leg * 4: leg * 4 + 3] = tmp[leg * 4: leg * 4 + 3]
        goal_vel[leg * 4 + 3] = tmp[leg * 4 + 3]
    return goal_pos, goal_vel


# ---------------------------------------------------------------- self-check --

def _cpp_floats(text, marker, count):
    """Pull `count` float literals following `marker` in the C++ source."""
    import re
    i = text.index(marker)
    nums = re.findall(r"-?\d+\.?\d*", text[i + len(marker): i + len(marker) + 900])
    return [float(x) for x in nums[:count]]


def self_check() -> int:
    """Re-read the C++ header and verify every constant still matches."""
    import re

    if not CPP_HEADER.is_file():
        print(f"FAIL  cannot find {CPP_HEADER}")
        return 1
    src = CPP_HEADER.read_text(encoding="utf-8", errors="replace")
    fails = []

    def check(label, got, want, tol=0.0):
        same = (np.abs(np.asarray(got, dtype=float)
                       - np.asarray(want, dtype=float)) <= tol).all()
        print(f"  {'ok  ' if same else 'FAIL'}  {label}")
        if not same:
            fails.append(f"{label}: header={got} python={want}")

    m = re.search(r"base_observation_dim\s*=\s*(\d+)", src)
    check("base observation dim", [int(m.group(1))] if m else [-1], [BASE_OBS_DIM])

    m = re.search(r"action_dim\s*=\s*(\d+)", src)
    check("action dim", [int(m.group(1))] if m else [-1], [ACTION_DIM])

    m = re.search(r"omega_scale_\s*=\s*([\d.]+)", src)
    check("omega scale", [float(m.group(1))] if m else [-1], [OMEGA_SCALE])

    m = re.search(r"dof_vel_scale_\s*=\s*([\d.]+)", src)
    check("dof vel scale", [float(m.group(1))] if m else [-1], [DOF_VEL_SCALE])

    check("dof_default_eigen_policy",
          _cpp_floats(src, "dof_default_eigen_policy <<", 16), DOF_DEFAULT_POLICY, 1e-6)
    check("dof_default_eigen_robot",
          _cpp_floats(src, "dof_default_eigen_robot <<", 16), DOF_DEFAULT_ROBOT, 1e-6)
    check("action_scale_robot",
          _cpp_floats(src, "action_scale_robot = {", 16), ACTION_SCALE_ROBOT, 1e-6)

    m = re.search(r"kp_\s*=\s*Vec4f\(([^)]*)\)", src)
    check("kp", [float(x) for x in m.group(1).split(",")] if m else [], KP[:4], 1e-6)
    m = re.search(r"kd_\s*=\s*Vec4f\(([^)]*)\)", src)
    check("kd", [float(x) for x in m.group(1).split(",")] if m else [], KD[:4], 1e-6)

    for label, want in (("robot_order", ROBOT_ORDER), ("policy_order", POLICY_ORDER)):
        i = src.index(f"{label} = {{")
        got = re.findall(r'"([a-z_]+_joint)"', src[i:i + 900])[:16]
        same = got == want
        print(f"  {'ok  ' if same else 'FAIL'}  {label}")
        if not same:
            fails.append(f"{label} mismatch: {got}")

    # Round-trip: policy order -> robot order -> policy order must be identity
    rt = np.arange(ACTION_DIM)[POLICY2ROBOT][ROBOT2POLICY]
    same = (rt == np.arange(ACTION_DIM)).all()
    print(f"  {'ok  ' if same else 'FAIL'}  permutation round-trip")
    if not same:
        fails.append("permutation round-trip is not the identity")

    # Shape smoke test
    obs = build_observation(np.zeros(3), [1, 0, 0, 0], np.zeros(3),
                            np.zeros(16), np.zeros(16), np.zeros(16))
    print(f"  {'ok  ' if obs.shape == (57,) else 'FAIL'}  build_observation -> {obs.shape}")
    obs_l = build_observation(np.zeros(3), [1, 0, 0, 0], np.zeros(3),
                              np.zeros(16), np.zeros(16), np.zeros(16),
                              lidar_ranges=np.full(72, np.inf))
    print(f"  {'ok  ' if obs_l.shape == (129,) else 'FAIL'}  with 72 lidar beams -> {obs_l.shape}")

    # An upright robot at the default pose must produce zeroed joint terms
    upright = build_observation(np.zeros(3), [1, 0, 0, 0], np.zeros(3),
                               DOF_DEFAULT_ROBOT, np.zeros(16), np.zeros(16))
    ok = np.allclose(upright[3:6], [0, 0, -1]) and np.allclose(upright[9:21], 0.0)
    print(f"  {'ok  ' if ok else 'FAIL'}  default pose -> gravity (0,0,-1), leg joint terms 0")
    if not ok:
        fails.append("default pose observation is not neutral")

    print()
    if fails:
        print(f"{len(fails)} mismatch(es) against {CPP_HEADER.name}:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print(f"All constants agree with {CPP_HEADER.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_check())
