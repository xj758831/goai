#!/usr/bin/env python3
"""Measure the official S10 MuJoCo speed and wheel-contact baseline.

This is an isolated diagnostic.  It loads the unmodified official
``S10_track.xml`` and never writes XML, friction, robot, or ROS2 simulator
parameters.  Two control paths are measured independently:

* ``cmd_vel``: the shipped ``policy.onnx`` receives the official 57-D
  observation with ``[forward, 0, 0]`` command at 50 Hz.
* ``wheel_velocity``: leg posture is held with the official MuJoCo PD gains;
  all four wheel velocity targets are set directly in MuJoCo joint units.

Examples:

    # Required visible one-robot smoke test.
    /usr/bin/python3 scripts/mujoco/s10_speed_baseline.py --gui-smoke

    # Complete 3 cmd_vel + 4 direct-wheel baseline (5 s each).
    /usr/bin/python3 scripts/mujoco/s10_speed_baseline.py

The script writes CSV samples, summary.json, plots, frames and MP4 files under
``logs/mujoco/s10_speed_baseline_<timestamp>/`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SDK_ROOT = PROJECT_ROOT / "src" / "S10_sdk_deploy"
MJCF_PATH = SDK_ROOT / "S10_description" / "s10_mjcf" / "mjcf" / "S10_track.xml"
POLICY_PATH = SDK_ROOT / "policy" / "policy.onnx"

sys.path.insert(0, str(SDK_ROOT / "scripts"))
from s10_obs_reference import KD, KP, build_observation, decode_action  # noqa: E402


DT = 0.001
POLICY_HZ = 50
POLICY_STEPS = int(round(1.0 / (DT * POLICY_HZ)))
WHEEL_NAMES = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")
WHEEL_JOINT_NAMES = tuple(f"{name}_joint" for name in WHEEL_NAMES)
BASE_BODY_NAME = "base_link"

# These are read from the unmodified ROS2 MuJoCo simulator.  They are only
# copied here to reset this independent diagnostic to the official pose.
TRACK_START_BASE_POS = np.array([0.0, -2.5, 0.2], dtype=np.float64)
RAW_JOINT_INIT = np.array(
    [
        -0.438, -1.16, 2.76, 0.0,
        0.438, -1.16, 2.76, 0.0,
        -0.438, 1.16, -2.76, 0.0,
        0.438, 1.16, -2.76, 0.0,
    ],
    dtype=np.float64,
)

# The official state machine performs a two-stage 3.0 s stand-up before it
# enters RL mode.  The policy's [0, -0.3, 0.6] posture is this standing pose,
# in the internal robot/MuJoCo coordinate system.  The ROS bridge's offset and
# sign transform is applied once on transmit and once on receive, so it
# cancels before S10PolicyRunner sees the robot state.
THIGH_LENGTH_M = 0.25
SHANK_LENGTH_M = 0.25
PRE_STAND_HEIGHT_M = 0.12
STAND_HEIGHT_M = 0.48
STAND_DURATION_S = 1.5
STAND_LEG_KP = np.array([120.0, 120.0, 120.0, 0.0] * 4, dtype=np.float64)
STAND_LEG_KD = np.array([2.0, 2.0, 2.0, 0.0] * 4, dtype=np.float64)

DIRECT_WHEEL_TARGETS = (-1.75, -3.0, -5.0, -6.75)
CMD_VEL_TARGETS = (0.2, 0.4, 0.6)
SAMPLE_FIELDS = (
    "step",
    "sim_time_s",
    "mode",
    "target_cmd_vel_mps",
    "target_wheel_velocity_rad_s",
    "base_x_m",
    "base_y_m",
    "base_z_m",
    "base_x_displacement_m",
    "base_x_delta_m",
    "base_vx_body_mps",
    "base_vy_body_mps",
    "base_vz_body_mps",
    "base_speed_world_mps",
    "roll_rad",
    "pitch_rad",
    "fl_wheel_omega_rad_s",
    "fr_wheel_omega_rad_s",
    "hl_wheel_omega_rad_s",
    "hr_wheel_omega_rad_s",
    "fl_contact_force_N",
    "fr_contact_force_N",
    "hl_contact_force_N",
    "hr_contact_force_N",
    "fl_contact_normal_N",
    "fr_contact_normal_N",
    "hl_contact_normal_N",
    "hr_contact_normal_N",
    "fl_contact_count",
    "fr_contact_count",
    "hl_contact_count",
    "hr_contact_count",
    "wheel_radius_m",
    "wheel_kinematic_speed_mps",
    "actual_to_theoretical_speed_ratio",
    "wheel_slip_proxy",
    "wheel_spin_no_progress",
    "sim_diverged",
)


@dataclass(frozen=True)
class Case:
    mode: str
    target: float

    @property
    def name(self) -> str:
        sign = "m" if self.target < 0.0 else "p"
        compact = f"{abs(self.target):.2f}".replace(".", "p")
        prefix = "cmd_vel" if self.mode == "cmd_vel" else "wheel_velocity"
        return f"{prefix}_{sign}{compact}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0, help="seconds per full-baseline case")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory; defaults to logs/mujoco/s10_speed_baseline_<timestamp>",
    )
    parser.add_argument(
        "--only",
        choices=("all", "cmd_vel", "wheel_velocity"),
        default="all",
        help="select a subset of the full baseline",
    )
    parser.add_argument("--gui-smoke", action="store_true", help="run only a visible 1-robot -1.75 rad/s smoke test")
    parser.add_argument("--smoke-duration", type=float, default=2.0, help="seconds for --gui-smoke")
    parser.add_argument("--frame-period", type=float, default=0.10, help="seconds between saved rendered frames")
    parser.add_argument("--no-capture", action="store_true", help="skip MP4 and frame output")
    parser.add_argument("--policy", type=Path, default=POLICY_PATH, help="official ONNX policy path")
    return parser.parse_args()


def id_for(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise KeyError(f"missing MuJoCo {obj_type.name}: {name}")
    return int(obj_id)


def euler_from_xmat(xmat: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch, yaw from MuJoCo's world-from-body 3x3 matrix."""
    r = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def leg_pose_for_height(height_m: float) -> np.ndarray:
    """Match StandUpState::GetHipYPosByHeight/GetKneePosByHeight."""
    if not 0.0 < height_m < THIGH_LENGTH_M + SHANK_LENGTH_M:
        raise ValueError(f"invalid S10 stand height: {height_m}")
    hipy = -math.acos(
        (THIGH_LENGTH_M * THIGH_LENGTH_M + height_m * height_m - SHANK_LENGTH_M * SHANK_LENGTH_M)
        / (2.0 * height_m * THIGH_LENGTH_M)
    )
    knee = math.pi - math.acos(
        (THIGH_LENGTH_M * THIGH_LENGTH_M + SHANK_LENGTH_M * SHANK_LENGTH_M - height_m * height_m)
        / (2.0 * THIGH_LENGTH_M * SHANK_LENGTH_M)
    )
    return np.array(
        [
            0.0, hipy, knee, 0.0,
            0.0, hipy, knee, 0.0,
            0.0, -hipy, -knee, 0.0,
            0.0, -hipy, -knee, 0.0,
        ],
        dtype=np.float64,
    )


PRE_STAND_JOINT_TARGET = leg_pose_for_height(PRE_STAND_HEIGHT_M)
STANDING_JOINT_TARGET = leg_pose_for_height(STAND_HEIGHT_M)


def cubic_blend(progress: float) -> tuple[float, float]:
    """Cubic position blend and unit-time derivative used by the state machine."""
    u = float(np.clip(progress, 0.0, 1.0))
    return 3.0 * u * u - 2.0 * u * u * u, 6.0 * u * (1.0 - u)


def body_velocity(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = np.zeros(6, dtype=np.float64)
    world = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, local, 1)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body_id, world, 0)
    return local[:3].copy(), local[3:].copy(), world[3:].copy()


def describe_model(model: mujoco.MjModel) -> dict[str, Any]:
    base_body_id = id_for(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME)
    wheel_body_ids = [id_for(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in WHEEL_NAMES]
    wheel_joint_ids = [id_for(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in WHEEL_JOINT_NAMES]
    wheel_dof_addresses = [int(model.jnt_dofadr[joint_id]) for joint_id in wheel_joint_ids]
    wheel_geom_ids: list[list[int]] = []
    radii: list[float] = []
    for body_id in wheel_body_ids:
        geom_ids = [geom_id for geom_id in range(model.ngeom) if model.geom_bodyid[geom_id] == body_id]
        if not geom_ids:
            raise RuntimeError(f"wheel body {body_id} has no geometry")
        wheel_geom_ids.append(geom_ids)
        cylinder_radii = [
            float(model.geom_size[geom_id, 0])
            for geom_id in geom_ids
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        ]
        if not cylinder_radii:
            raise RuntimeError(f"cannot read cylinder radius for wheel body {body_id}")
        radii.append(max(cylinder_radii))
    if not np.allclose(radii, radii[0], atol=1e-9):
        raise RuntimeError(f"wheel radius mismatch in official MJCF: {radii}")
    return {
        "base_body_id": base_body_id,
        "wheel_body_ids": wheel_body_ids,
        "wheel_dof_addresses": wheel_dof_addresses,
        "wheel_geom_ids": wheel_geom_ids,
        "wheel_radius_m": float(radii[0]),
    }


def reset_official_start(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Match the stock MuJoCo ROS2 simulator reset without changing its code."""
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = TRACK_START_BASE_POS
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qpos[7:23] = RAW_JOINT_INIT
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def run_official_standup(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, str | None]:
    """Reproduce the stock stand-up trajectory before starting a policy case.

    StandUpState first moves to ``pre_height_`` for 1.5 s, then extends to
    ``stand_height_`` for another 1.5 s.  This warm-up is intentionally
    excluded from the speed measurement window.
    """
    steps_per_stage = int(round(STAND_DURATION_S / DT))
    for step in range(2 * steps_per_stage):
        if step < steps_per_stage:
            blend, blend_rate = cubic_blend(step / steps_per_stage)
            start, goal = RAW_JOINT_INIT, PRE_STAND_JOINT_TARGET
            wheel_kd = 0.0
        else:
            blend, blend_rate = cubic_blend((step - steps_per_stage) / steps_per_stage)
            start, goal = PRE_STAND_JOINT_TARGET, STANDING_JOINT_TARGET
            wheel_kd = 1.0
        target_pos = start + blend * (goal - start)
        target_vel = blend_rate * (goal - start) / STAND_DURATION_S
        gains_d = STAND_LEG_KD.copy()
        gains_d[[3, 7, 11, 15]] = wheel_kd
        torque = STAND_LEG_KP * (target_pos - data.qpos[7:23]) + gains_d * (target_vel - data.qvel[6:22])
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
        diverged, reason = has_diverged(data)
        if diverged:
            return diverged, reason
    return False, None


def wheel_contact_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    wheel_geom_ids: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum contact-force magnitudes for collision contacts of each wheel body."""
    geom_to_wheel = {
        geom_id: wheel_index
        for wheel_index, geom_ids in enumerate(wheel_geom_ids)
        for geom_id in geom_ids
    }
    magnitudes = np.zeros(4, dtype=np.float64)
    normals = np.zeros(4, dtype=np.float64)
    counts = np.zeros(4, dtype=np.int32)
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        wheel_index = geom_to_wheel.get(int(contact.geom1))
        if wheel_index is None:
            wheel_index = geom_to_wheel.get(int(contact.geom2))
        if wheel_index is None:
            continue
        mujoco.mj_contactForce(model, data, contact_index, contact_force)
        magnitudes[wheel_index] += float(np.linalg.norm(contact_force[:3]))
        normals[wheel_index] += abs(float(contact_force[0]))
        counts[wheel_index] += 1
    return magnitudes, normals, counts


class OfficialController:
    """Controller matching the stock MuJoCo torque law for either test mode."""

    def __init__(self, mode: str, target: float, policy_path: Path | None):
        self.mode = mode
        self.target = float(target)
        self.pos_target = STANDING_JOINT_TARGET.copy()
        self.vel_target = np.zeros(16, dtype=np.float64)
        self.last_action = np.zeros(16, dtype=np.float32)
        self.session: ort.InferenceSession | None = None
        if mode == "cmd_vel":
            if policy_path is None or not policy_path.is_file():
                raise FileNotFoundError(f"official policy is required for cmd_vel tests: {policy_path}")
            self.session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
            model_input = self.session.get_inputs()[0]
            if model_input.name != "obs" or model_input.shape != [1, 57]:
                raise RuntimeError(
                    f"expected official 57-D policy input 'obs', got {model_input.name} {model_input.shape}"
                )

    def update(self, step: int, model: mujoco.MjModel, data: mujoco.MjData, ids: dict[str, Any]) -> None:
        if self.mode == "wheel_velocity":
            self.pos_target[:] = STANDING_JOINT_TARGET
            self.vel_target[:] = 0.0
            self.vel_target[[3, 7, 11, 15]] = self.target
            return

        if step % POLICY_STEPS != 0:
            return
        assert self.session is not None
        angular_local, _, _ = body_velocity(model, data, ids["base_body_id"])
        observation = build_observation(
            angular_local,
            data.qpos[3:7].copy(),
            np.array([self.target, 0.0, 0.0], dtype=np.float32),
            data.qpos[7:23].copy(),
            data.qvel[6:22].copy(),
            self.last_action,
        )
        # Keep the raw ONNX action exactly as S10PolicyRunner does.  The deploy
        # C++ path applies its action scale directly and does not clip here.
        raw_action = np.asarray(self.session.run(["actions"], {"obs": observation[None]})[0][0], dtype=np.float32)
        if not np.isfinite(raw_action).all():
            raise RuntimeError("official policy produced a non-finite action")
        self.last_action = raw_action.copy()
        self.pos_target, self.vel_target = (target.astype(np.float64) for target in decode_action(raw_action))

    def apply(self, data: mujoco.MjData) -> None:
        joint_pos = data.qpos[7:23]
        joint_vel = data.qvel[6:22]
        torque = KP.astype(np.float64) * (self.pos_target - joint_pos) + KD.astype(np.float64) * (
            self.vel_target - joint_vel
        )
        data.ctrl[:] = torque


class CaseCapture:
    def __init__(self, model: mujoco.MjModel, case_dir: Path, enabled: bool, frame_period: float):
        self.enabled = enabled
        self.case_dir = case_dir
        self.frame_period = frame_period
        self.next_frame_s = 0.0
        self.frame_index = 0
        self.renderer: mujoco.Renderer | None = None
        self.camera: mujoco.MjvCamera | None = None
        self.video: cv2.VideoWriter | None = None
        if not enabled:
            return
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 3.5
        self.camera.azimuth = 135.0
        self.camera.elevation = -20.0
        self.video = cv2.VideoWriter(
            str(self.case_dir / "rollout.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, 1.0 / frame_period),
            (640, 480),
        )

    def maybe_capture(self, data: mujoco.MjData, base_body_id: int, sim_time: float) -> None:
        if not self.enabled or sim_time + 1e-12 < self.next_frame_s:
            return
        assert self.renderer is not None and self.camera is not None
        self.next_frame_s += self.frame_period
        base_pos = data.xpos[base_body_id]
        self.camera.lookat[:] = base_pos
        self.camera.lookat[2] += 0.15
        self.renderer.update_scene(data, camera=self.camera)
        rgb = self.renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frame_path = self.case_dir / f"frame_{self.frame_index:04d}_t{sim_time:05.2f}s.png"
        cv2.imwrite(str(frame_path), bgr)
        if self.video is not None and self.video.isOpened():
            self.video.write(bgr)
        self.frame_index += 1

    def close(self) -> None:
        if self.video is not None:
            self.video.release()
        if self.renderer is not None:
            self.renderer.close()


def one_row(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict[str, Any],
    case: Case,
    step: int,
    measurement_start_time: float,
    start_x: float,
    previous_x: float,
    diverged: bool,
) -> dict[str, Any]:
    _, local_linear, world_linear = body_velocity(model, data, ids["base_body_id"])
    base_pos = data.xpos[ids["base_body_id"]]
    roll, pitch, _ = euler_from_xmat(data.xmat[ids["base_body_id"]])
    wheel_velocities = data.qvel[ids["wheel_dof_addresses"]].copy()
    forces, normals, counts = wheel_contact_forces(model, data, ids["wheel_geom_ids"])
    kinematic_speed = ids["wheel_radius_m"] * float(np.mean(np.abs(wheel_velocities)))
    actual_forward = abs(float(local_linear[0]))
    speed_ratio = actual_forward / kinematic_speed if kinematic_speed > 0.05 else 0.0
    slip_proxy = float(np.clip(1.0 - speed_ratio, 0.0, 1.0)) if kinematic_speed > 0.05 else 0.0
    speed_threshold = max(1.0, abs(case.target) * 0.4) if case.mode == "wheel_velocity" else 1.0
    no_progress = int(float(np.mean(np.abs(wheel_velocities))) > speed_threshold and actual_forward < 0.02)
    return {
        "step": step,
        "sim_time_s": float(data.time - measurement_start_time),
        "mode": case.mode,
        "target_cmd_vel_mps": case.target if case.mode == "cmd_vel" else 0.0,
        "target_wheel_velocity_rad_s": case.target if case.mode == "wheel_velocity" else 0.0,
        "base_x_m": float(base_pos[0]),
        "base_y_m": float(base_pos[1]),
        "base_z_m": float(base_pos[2]),
        "base_x_displacement_m": float(base_pos[0] - start_x),
        "base_x_delta_m": float(base_pos[0] - previous_x),
        "base_vx_body_mps": float(local_linear[0]),
        "base_vy_body_mps": float(local_linear[1]),
        "base_vz_body_mps": float(local_linear[2]),
        "base_speed_world_mps": float(np.linalg.norm(world_linear)),
        "roll_rad": roll,
        "pitch_rad": pitch,
        "fl_wheel_omega_rad_s": float(wheel_velocities[0]),
        "fr_wheel_omega_rad_s": float(wheel_velocities[1]),
        "hl_wheel_omega_rad_s": float(wheel_velocities[2]),
        "hr_wheel_omega_rad_s": float(wheel_velocities[3]),
        "fl_contact_force_N": float(forces[0]),
        "fr_contact_force_N": float(forces[1]),
        "hl_contact_force_N": float(forces[2]),
        "hr_contact_force_N": float(forces[3]),
        "fl_contact_normal_N": float(normals[0]),
        "fr_contact_normal_N": float(normals[1]),
        "hl_contact_normal_N": float(normals[2]),
        "hr_contact_normal_N": float(normals[3]),
        "fl_contact_count": int(counts[0]),
        "fr_contact_count": int(counts[1]),
        "hl_contact_count": int(counts[2]),
        "hr_contact_count": int(counts[3]),
        "wheel_radius_m": float(ids["wheel_radius_m"]),
        "wheel_kinematic_speed_mps": kinematic_speed,
        "actual_to_theoretical_speed_ratio": speed_ratio,
        "wheel_slip_proxy": slip_proxy,
        "wheel_spin_no_progress": no_progress,
        "sim_diverged": int(diverged),
    }


def has_diverged(data: mujoco.MjData) -> tuple[bool, str | None]:
    arrays = (data.qpos, data.qvel, data.qacc, data.ctrl)
    if not all(np.isfinite(array).all() for array in arrays):
        return True, "non_finite_state"
    max_velocity = float(np.max(np.abs(data.qvel)))
    max_acceleration = float(np.max(np.abs(data.qacc)))
    if max_velocity > 1.0e4 or max_acceleration > 1.0e7:
        return True, f"state_limit(qvel={max_velocity:.3g},qacc={max_acceleration:.3g})"
    return False, None


def summarize_rows(case: Case, rows: list[dict[str, Any]], diverged: bool, divergence_reason: str | None) -> dict[str, Any]:
    if not rows:
        return {"case": case.name, "error": "no samples"}
    tail = rows[max(0, int(len(rows) * 0.8)) :]

    def series(key: str, source: list[dict[str, Any]] = rows) -> np.ndarray:
        return np.asarray([float(row[key]) for row in source], dtype=np.float64)

    def stats(key: str) -> dict[str, float]:
        values = series(key)
        tail_values = series(key, tail)
        return {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "final": float(values[-1]),
            "tail_mean": float(tail_values.mean()),
        }

    force_keys = [f"{wheel}_contact_force_N" for wheel in ("fl", "fr", "hl", "hr")]
    wheel_keys = [f"{wheel}_wheel_omega_rad_s" for wheel in ("fl", "fr", "hl", "hr")]
    return {
        "case": case.name,
        "mode": case.mode,
        "target": case.target,
        "samples": len(rows),
        "duration_s": float(rows[-1]["sim_time_s"]),
        "diverged": diverged,
        "divergence_reason": divergence_reason,
        "base_x_displacement_m": stats("base_x_displacement_m"),
        "base_vx_body_mps": stats("base_vx_body_mps"),
        "base_speed_world_mps": stats("base_speed_world_mps"),
        "pitch_rad": stats("pitch_rad"),
        "roll_rad": stats("roll_rad"),
        "wheel_kinematic_speed_mps": stats("wheel_kinematic_speed_mps"),
        "actual_to_theoretical_speed_ratio": stats("actual_to_theoretical_speed_ratio"),
        "wheel_slip_proxy": stats("wheel_slip_proxy"),
        "wheel_spin_no_progress_fraction": float(series("wheel_spin_no_progress").mean()),
        "wheel_angular_velocity_rad_s": {
            key.removesuffix("_omega_rad_s"): stats(key) for key in wheel_keys
        },
        "wheel_contact_force_N": {key.removesuffix("_contact_force_N"): stats(key) for key in force_keys},
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(case: Case, rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    values = {
        name: np.asarray([float(row[name]) for row in rows])
        for name in SAMPLE_FIELDS
        if name != "mode" and name in rows[0]
    }
    t = values["sim_time_s"]
    figure, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True, constrained_layout=True)
    axes[0].plot(t, values["base_vx_body_mps"], label="base vx body")
    if case.mode == "cmd_vel":
        axes[0].axhline(case.target, linestyle="--", color="black", label="cmd_vel target")
    axes[0].plot(t, values["wheel_kinematic_speed_mps"], label="r * mean(|omega|)")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    for wheel in ("fl", "fr", "hl", "hr"):
        axes[1].plot(t, values[f"{wheel}_wheel_omega_rad_s"], label=wheel)
    if case.mode == "wheel_velocity":
        axes[1].axhline(case.target, linestyle="--", color="black", label="target")
    axes[1].set_ylabel("wheel omega (rad/s)")
    axes[1].legend(loc="best", ncol=5)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, values["pitch_rad"], label="pitch")
    axes[2].plot(t, values["roll_rad"], label="roll")
    axes[2].set_ylabel("angle (rad)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, values["wheel_slip_proxy"], label="slip proxy")
    axes[3].plot(t, values["wheel_spin_no_progress"], label="spin/no progress")
    axes[3].plot(t, values["base_x_displacement_m"], label="base x displacement")
    axes[3].set_xlabel("simulation time (s)")
    axes[3].set_ylabel("proxy / displacement")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)
    figure.suptitle(f"S10 MuJoCo speed baseline: {case.name}")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def configure_gui(viewer: Any, data: mujoco.MjData, base_body_id: int) -> None:
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = base_body_id
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[:] = data.xpos[base_body_id]


def run_case(
    model: mujoco.MjModel,
    case: Case,
    output_dir: Path,
    duration_s: float,
    policy_path: Path,
    capture_enabled: bool,
    frame_period: float,
    gui: bool,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    reset_official_start(model, data)
    ids = describe_model(model)
    stand_diverged, stand_reason = run_official_standup(model, data)
    if stand_diverged:
        raise RuntimeError(f"official stand-up warm-up diverged: {stand_reason}")
    controller = OfficialController(case.mode, case.target, policy_path)
    case_dir = output_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    capture = CaseCapture(model, case_dir, capture_enabled, frame_period)
    viewer = mujoco.viewer.launch_passive(model, data) if gui else None
    if viewer is not None:
        configure_gui(viewer, data, ids["base_body_id"])

    measurement_start_time = float(data.time)
    start_x = float(data.xpos[ids["base_body_id"], 0])
    previous_x = start_x
    rows: list[dict[str, Any]] = []
    diverged = False
    divergence_reason: str | None = None
    max_steps = int(round(duration_s / DT))
    wall_start = time.perf_counter()
    try:
        for step in range(max_steps):
            if viewer is not None and not viewer.is_running():
                break
            controller.update(step, model, data, ids)
            controller.apply(data)
            mujoco.mj_step(model, data)
            diverged, divergence_reason = has_diverged(data)
            row = one_row(
                model,
                data,
                ids,
                case,
                step + 1,
                measurement_start_time,
                start_x,
                previous_x,
                diverged,
            )
            rows.append(row)
            previous_x = row["base_x_m"]
            capture.maybe_capture(data, ids["base_body_id"], row["sim_time_s"])
            if viewer is not None and step % 10 == 0:
                viewer.sync()
                target_wall_time = (step + 1) * DT
                wait_s = target_wall_time - (time.perf_counter() - wall_start)
                if wait_s > 0.0:
                    time.sleep(wait_s)
            if diverged:
                break
    finally:
        if viewer is not None:
            viewer.close()
        capture.close()

    save_csv(case_dir / "samples.csv", rows)
    save_plot(case, rows, case_dir / "curves.png")
    summary = summarize_rows(case, rows, diverged, divergence_reason)
    summary["paths"] = {
        "csv": str(case_dir / "samples.csv"),
        "plot": str(case_dir / "curves.png"),
        "frames": str(case_dir),
        "video": str(case_dir / "rollout.mp4"),
    }
    print(
        f"[{case.name}] dx={summary['base_x_displacement_m']['final']:.3f} m "
        f"vx_tail={summary['base_vx_body_mps']['tail_mean']:.3f} m/s "
        f"wheel_kin_tail={summary['wheel_kinematic_speed_mps']['tail_mean']:.3f} m/s "
        f"slip_tail={summary['wheel_slip_proxy']['tail_mean']:.3f} "
        f"diverged={diverged}",
        flush=True,
    )
    return summary


def requested_cases(which: str) -> list[Case]:
    cases: list[Case] = []
    if which in ("all", "cmd_vel"):
        cases.extend(Case("cmd_vel", value) for value in CMD_VEL_TARGETS)
    if which in ("all", "wheel_velocity"):
        cases.extend(Case("wheel_velocity", value) for value in DIRECT_WHEEL_TARGETS)
    return cases


def main() -> int:
    args = parse_args()
    if args.duration <= 0.0 or args.smoke_duration <= 0.0 or args.frame_period <= 0.0:
        raise ValueError("duration and frame period must be positive")
    if not MJCF_PATH.is_file():
        raise FileNotFoundError(f"official MuJoCo XML not found: {MJCF_PATH}")
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "logs" / "mujoco" / f"s10_speed_baseline_{timestamp}"
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    model.opt.timestep = DT
    ids = describe_model(model)
    metadata: dict[str, Any] = {
        "purpose": "MuJoCo-only S10 speed, wheel-contact, and slip baseline",
        "official_mjcf": str(MJCF_PATH),
        "official_policy": str(args.policy.resolve()),
        "dt_s": DT,
        "policy_hz": POLICY_HZ,
        "wheel_radius_m": ids["wheel_radius_m"],
        "control": {
            "leg_kp": [float(value) for value in KP[0:3]],
            "leg_kd": [float(value) for value in KD[0:3]],
            "wheel_kp": float(KP[3]),
            "wheel_kd": float(KD[3]),
        },
        "changes_to_official_assets": False,
    }
    print(f"[baseline] official XML: {MJCF_PATH}")
    print(f"[baseline] wheel radius read from MJCF: {ids['wheel_radius_m']:.3f} m")
    print(f"[baseline] output: {output_dir}")

    if args.gui_smoke:
        case = Case("wheel_velocity", -1.75)
        print(f"[baseline] GUI smoke: {case.name} for {args.smoke_duration:.1f} s", flush=True)
        summaries = [
            run_case(
                model,
                case,
                output_dir,
                args.smoke_duration,
                args.policy,
                not args.no_capture,
                args.frame_period,
                gui=True,
            )
        ]
        metadata["run_type"] = "gui_smoke"
    else:
        summaries = []
        for case in requested_cases(args.only):
            print(f"[baseline] running {case.name} for {args.duration:.1f} s", flush=True)
            summaries.append(
                run_case(
                    model,
                    case,
                    output_dir,
                    args.duration,
                    args.policy,
                    not args.no_capture,
                    args.frame_period,
                    gui=False,
                )
            )
        metadata["run_type"] = "full" if args.only == "all" else args.only
    metadata["cases"] = summaries
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[baseline] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
