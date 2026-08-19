#!/usr/bin/env python3
"""Probe the official S10 MuJoCo model for a wheel-assisted pit-exit hook.

This is an isolated diagnostic, not PPO training.  It loads the shipped
``S10_track.xml`` unchanged and controls the robot with the same raw MuJoCo
coordinate convention and PD/velocity gains used by the official deployment
stack.  It starts on the bottom of the official deep pit, approaches the exit
wall, then evaluates scripted front-wheel/rear-wheel/leg-posture combinations.

The probe does not use map coordinates to decide when to drive the robot.  The
approach changes phase only after a front-wheel terrain contact has been
observed.  Pit geometry values are retained exclusively for diagnostics and
post-run classification.

Examples:

    # One wall-roll combination across the four approach modes, useful to
    # verify the reset and contact labels.
    /usr/bin/python3 scripts/mujoco/s10_fronthook_scripted_probe.py --smoke

    # Curated visible scan (four approach modes, 14 wall-roll candidates).
    /usr/bin/python3 scripts/mujoco/s10_fronthook_scripted_probe.py

    # Full 4 x 3 x 4 x 7 matrix.  This is intentionally not the default.
    /usr/bin/python3 scripts/mujoco/s10_fronthook_scripted_probe.py --scan full

Results are written below ``logs/mujoco/s10_fronthook_scripted_<timestamp>/``.
Every candidate receives samples.csv, an MP4 and frames.  summary.csv and
summary.json provide the evaluated combination table and ranked candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
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
BASE_BODY_NAME = "base_link"
WHEEL_NAMES = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")
WHEEL_JOINT_NAMES = tuple(f"{name}_joint" for name in WHEEL_NAMES)
FRONT_INDICES = np.array([0, 1], dtype=np.int32)
REAR_INDICES = np.array([2, 3], dtype=np.int32)

# Measured against the unmodified official S10_track.xml.  These values label
# diagnostics only; no phase transition uses them as a control input.
PIT_BOTTOM_WP = np.array([11.1225, 33.0075, 0.1000], dtype=np.float64)
PIT_EXIT_WP = np.array([16.2750, 31.2900, 0.4750], dtype=np.float64)
EXIT_DIRECTION_XY = (PIT_EXIT_WP[:2] - PIT_BOTTOM_WP[:2])
EXIT_DIRECTION_XY /= np.linalg.norm(EXIT_DIRECTION_XY)
EXIT_YAW = math.atan2(float(EXIT_DIRECTION_XY[1]), float(EXIT_DIRECTION_XY[0]))
PIT_BOTTOM_Z = 0.1018
PLATFORM_Z = 0.4787
EXIT_WALL_ALONG_M = 1.65

THIGH_LENGTH_M = 0.25
SHANK_LENGTH_M = 0.25
STAND_HEIGHT_M = 0.48
SETTLE_SECONDS = 0.8
APPROACH_TIMEOUT_S = 6.0
ROLL_SECONDS = 3.5
LEG_KP = KP.astype(np.float64)
LEG_KD = KD.astype(np.float64)
WHEEL_ADDR = np.array([3, 7, 11, 15], dtype=np.int32)


@dataclass(frozen=True)
class Approach:
    mode: str  # cmd_vel or wheel_velocity
    value: float

    @property
    def label(self) -> str:
        return f"{self.mode}_{self.value:+.2f}".replace("-", "m").replace("+", "p").replace(".", "p")


@dataclass(frozen=True)
class RollCase:
    approach: Approach
    front_wheel_speed: float
    rear_wheel_speed: float
    leg_mode: str

    @property
    def label(self) -> str:
        return (
            f"{self.approach.label}_front_{self.front_wheel_speed:+.2f}"
            f"_rear_{self.rear_wheel_speed:+.2f}_{self.leg_mode}"
        ).replace("-", "m").replace("+", "p").replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", choices=("smoke", "curated", "full"), default="curated")
    parser.add_argument("--smoke", action="store_true", help="alias for --scan smoke")
    parser.add_argument("--case", type=str, default=None, help="only run a case whose label contains this text")
    parser.add_argument("--case-offset", type=int, default=0, help="skip this many cases after --case filtering")
    parser.add_argument("--max-cases", type=int, default=None, help="cap the selected cases after filtering")
    parser.add_argument("--roll-seconds", type=float, default=ROLL_SECONDS, help="wall-roll duration per candidate")
    parser.add_argument("--approach-timeout", type=float, default=APPROACH_TIMEOUT_S)
    parser.add_argument("--headless", action="store_true", help="disable the visible MuJoCo GUI")
    parser.add_argument("--no-capture", action="store_true", help="skip MP4 and PNG output")
    parser.add_argument("--frame-period", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def id_for(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise KeyError(f"missing MuJoCo {obj_type.name}: {name}")
    return int(obj_id)


def euler_from_xmat(xmat: np.ndarray) -> tuple[float, float, float]:
    rotation = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    pitch = math.atan2(-float(rotation[2, 0]), math.hypot(float(rotation[0, 0]), float(rotation[1, 0])))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return roll, pitch, yaw


def yaw_quaternion(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def pose_for_height(height_m: float) -> np.ndarray:
    hipy = -math.acos(
        (THIGH_LENGTH_M * THIGH_LENGTH_M + height_m * height_m - SHANK_LENGTH_M * SHANK_LENGTH_M)
        / (2.0 * height_m * THIGH_LENGTH_M)
    )
    knee = math.pi - math.acos(
        (THIGH_LENGTH_M * THIGH_LENGTH_M + SHANK_LENGTH_M * SHANK_LENGTH_M - height_m * height_m)
        / (2.0 * THIGH_LENGTH_M * SHANK_LENGTH_M)
    )
    return np.array(
        [0.0, hipy, knee, 0.0, 0.0, hipy, knee, 0.0, 0.0, -hipy, -knee, 0.0, 0.0, -hipy, -knee, 0.0],
        dtype=np.float64,
    )


STANDING_POSE = pose_for_height(STAND_HEIGHT_M)


def model_ids(model: mujoco.MjModel) -> dict[str, Any]:
    base_body_id = id_for(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME)
    wheel_body_ids = [id_for(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in WHEEL_NAMES]
    wheel_joint_ids = [id_for(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in WHEEL_JOINT_NAMES]
    wheel_geom_ids = [
        [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id]
        for body_id in wheel_body_ids
    ]
    robot_bodies: set[int] = set()
    for body_id in range(model.nbody):
        parent = body_id
        while parent > 0:
            if parent == base_body_id:
                robot_bodies.add(body_id)
                break
            parent = int(model.body_parentid[parent])
    robot_bodies.add(base_body_id)
    robot_geoms = {geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) in robot_bodies}
    base_geom_ids = {geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == base_body_id}
    return {
        "base_body_id": base_body_id,
        "wheel_body_ids": wheel_body_ids,
        "wheel_dof_addresses": [int(model.jnt_dofadr[joint_id]) for joint_id in wheel_joint_ids],
        "wheel_geom_ids": wheel_geom_ids,
        "robot_geoms": robot_geoms,
        "base_geom_ids": base_geom_ids,
    }


def reset_in_pit(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Place a normally standing S10 on the measured official pit bottom."""
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = np.array([PIT_BOTTOM_WP[0], PIT_BOTTOM_WP[1], PIT_BOTTOM_Z + STAND_HEIGHT_M], dtype=np.float64)
    data.qpos[3:7] = yaw_quaternion(EXIT_YAW)
    data.qpos[7:23] = STANDING_POSE
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def body_velocity(model: mujoco.MjModel, data: mujoco.MjData, base_body_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local = np.zeros(6, dtype=np.float64)
    world = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, local, 1)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, world, 0)
    return local[:3].copy(), local[3:].copy(), world[3:].copy()


def along(position_xy: np.ndarray) -> float:
    return float(np.dot(np.asarray(position_xy, dtype=np.float64) - PIT_BOTTOM_WP[:2], EXIT_DIRECTION_XY))


def wheel_centers(data: mujoco.MjData, ids: dict[str, Any]) -> np.ndarray:
    return np.asarray([data.xpos[body_id] for body_id in ids["wheel_body_ids"]], dtype=np.float64)


def is_robot_geom(geom_id: int, ids: dict[str, Any]) -> bool:
    return int(geom_id) in ids["robot_geoms"]


def contact_metrics(model: mujoco.MjModel, data: mujoco.MjData, ids: dict[str, Any]) -> dict[str, Any]:
    """Return terrain-only contact metrics and conservative wall/body labels."""
    geom_to_wheel = {
        geom_id: index for index, geom_ids in enumerate(ids["wheel_geom_ids"]) for geom_id in geom_ids
    }
    wheel_force = np.zeros(4, dtype=np.float64)
    wheel_contact = np.zeros(4, dtype=np.int32)
    front_wall_force = np.zeros(2, dtype=np.float64)
    front_wall_contact = np.zeros(2, dtype=np.int32)
    body_wall_force = 0.0
    max_force = 0.0
    contact_force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        wheel_index = geom_to_wheel.get(geom1, geom_to_wheel.get(geom2))
        robot_1, robot_2 = is_robot_geom(geom1, ids), is_robot_geom(geom2, ids)
        if robot_1 == robot_2:
            continue
        mujoco.mj_contactForce(model, data, contact_id, contact_force)
        magnitude = float(np.linalg.norm(contact_force[:3]))
        max_force = max(max_force, magnitude)
        # A horizontal contact normal identifies a vertical obstacle from
        # contact sensing alone.  The phase controller deliberately does not
        # consume exit coordinates or platform-height ground truth.
        wall_like_contact = abs(float(contact.frame[2])) < 0.35
        if wheel_index is not None:
            wheel_force[wheel_index] += magnitude
            wheel_contact[wheel_index] += 1
            if wheel_index < 2 and wall_like_contact:
                front_wall_force[wheel_index] += magnitude
                front_wall_contact[wheel_index] += 1
        elif wall_like_contact and (geom1 in ids["base_geom_ids"] or geom2 in ids["base_geom_ids"]):
            body_wall_force += magnitude
    return {
        "wheel_force": wheel_force,
        "wheel_contact": wheel_contact,
        "front_wall_force": front_wall_force,
        "front_wall_contact": front_wall_contact,
        "front_wall_contact_any": bool(np.any(front_wall_contact > 0)),
        "body_wall_contact": bool(body_wall_force > 1.0),
        "body_wall_force": body_wall_force,
        "max_contact_force": max_force,
    }


def leg_target(mode: str) -> np.ndarray:
    """Small symmetric postures relative to the official standing pose."""
    target = STANDING_POSE.copy()
    if mode in ("front_fold", "front_fold_rear_push"):
        target[[1, 5]] -= 0.22
        target[[2, 6]] += 0.32
    elif mode in ("front_extend", "front_extend_rear_push"):
        target[[1, 5]] += 0.10
        target[[2, 6]] -= 0.16
    elif mode in ("rear_extend", "rear_extend_all_drive"):
        target[[9, 13]] -= 0.10
        target[[10, 14]] += 0.16
    elif mode != "default":
        raise ValueError(f"unknown leg mode: {mode}")
    return target


class OfficialApproachPolicy:
    def __init__(self, command_mps: float, session: ort.InferenceSession | None = None):
        self.command_mps = float(command_mps)
        self.last_action = np.zeros(16, dtype=np.float32)
        self.position_target = STANDING_POSE.copy()
        self.velocity_target = np.zeros(16, dtype=np.float64)
        self.session = session or ort.InferenceSession(str(POLICY_PATH), providers=["CPUExecutionProvider"])

    def update(self, step: int, model: mujoco.MjModel, data: mujoco.MjData, ids: dict[str, Any]) -> None:
        if step % POLICY_STEPS != 0:
            return
        angular_local, _, _ = body_velocity(model, data, ids["base_body_id"])
        observation = build_observation(
            angular_local,
            data.qpos[3:7].copy(),
            np.array([self.command_mps, 0.0, 0.0], dtype=np.float32),
            data.qpos[7:23].copy(),
            data.qvel[6:22].copy(),
            self.last_action,
        )
        action = np.asarray(self.session.run(["actions"], {"obs": observation[None]})[0][0], dtype=np.float32)
        if not np.isfinite(action).all():
            raise RuntimeError("official policy emitted non-finite action")
        self.last_action = action
        position, velocity = decode_action(action)
        self.position_target = position.astype(np.float64)
        self.velocity_target = velocity.astype(np.float64)

    def apply(self, data: mujoco.MjData) -> None:
        data.ctrl[:] = LEG_KP * (self.position_target - data.qpos[7:23]) + LEG_KD * (
            self.velocity_target - data.qvel[6:22]
        )


def apply_direct_control(data: mujoco.MjData, position_target: np.ndarray, wheel_targets: np.ndarray) -> None:
    velocity_target = np.zeros(16, dtype=np.float64)
    velocity_target[WHEEL_ADDR] = wheel_targets
    data.ctrl[:] = LEG_KP * (position_target - data.qpos[7:23]) + LEG_KD * (velocity_target - data.qvel[6:22])


def diverged(data: mujoco.MjData) -> str | None:
    if not all(np.isfinite(array).all() for array in (data.qpos, data.qvel, data.qacc, data.ctrl)):
        return "non_finite_state"
    if float(np.max(np.abs(data.qvel))) > 1.0e4 or float(np.max(np.abs(data.qacc))) > 1.0e7:
        return "physics_divergence"
    return None


class Capture:
    def __init__(
        self,
        model: mujoco.MjModel,
        directory: Path,
        enabled: bool,
        frame_period: float,
        renderer: mujoco.Renderer | None = None,
        camera: mujoco.MjvCamera | None = None,
    ):
        self.enabled = enabled
        self.directory = directory
        self.frame_period = frame_period
        self.next_frame = 0.0
        self.index = 0
        self.renderer: mujoco.Renderer | None = None
        self.camera: mujoco.MjvCamera | None = None
        self.video: cv2.VideoWriter | None = None
        self.events: set[str] = set()
        self.owns_renderer = False
        if enabled:
            directory.mkdir(parents=True, exist_ok=True)
            self.renderer = renderer
            self.camera = camera
            if self.renderer is None or self.camera is None:
                self.renderer = mujoco.Renderer(model, height=480, width=640)
                self.camera = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(self.camera)
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.camera.distance = 3.2
                self.camera.azimuth = 130.0
                self.camera.elevation = -18.0
                self.owns_renderer = True
            self.video = cv2.VideoWriter(
                str(directory / "rollout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, 1.0 / frame_period), (640, 480)
            )

    def capture(self, data: mujoco.MjData, ids: dict[str, Any], relative_time: float, event: str | None = None) -> None:
        if not self.enabled:
            return
        periodic = relative_time + 1.0e-12 >= self.next_frame
        is_new_event = event is not None and event not in self.events
        if not periodic and not is_new_event:
            return
        if periodic:
            # The first sample after settling can be several frame periods
            # later than reset.  Advance to the next future deadline instead
            # of emitting one frame per subsequent physics step.
            self.next_frame = relative_time + self.frame_period
        if event is not None:
            self.events.add(event)
        assert self.renderer is not None and self.camera is not None
        base_pos = data.xpos[ids["base_body_id"]]
        self.camera.lookat[:] = base_pos
        self.camera.lookat[2] += 0.12
        self.renderer.update_scene(data, camera=self.camera)
        bgr = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        suffix = f"_{event}" if is_new_event else ""
        cv2.imwrite(str(self.directory / f"frame_{self.index:04d}_t{relative_time:05.2f}s{suffix}.png"), bgr)
        if self.video is not None and self.video.isOpened():
            self.video.write(bgr)
        self.index += 1

    def close(self) -> None:
        if self.video is not None:
            self.video.release()
        if self.owns_renderer and self.renderer is not None:
            self.renderer.close()


def configure_gui(viewer: Any, data: mujoco.MjData, ids: dict[str, Any]) -> None:
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = ids["base_body_id"]
        viewer.cam.distance = 3.2
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -18.0
        viewer.cam.lookat[:] = data.xpos[ids["base_body_id"]]


SAMPLE_FIELDS = (
    "step", "time_s", "phase", "approach_mode", "approach_target", "front_wheel_target_rad_s", "rear_wheel_target_rad_s",
    "leg_mode", "base_x_m", "base_y_m", "base_z_m", "base_along_m", "base_x_delta_m", "base_vx_body_mps",
    "roll_rad", "pitch_rad", "roll_rate_rad_s", "pitch_rate_rad_s", "fl_wheel_height_m", "fr_wheel_height_m",
    "front_wheel_height_m", "front_wheel_height_delta_m", "front_wheel_max_height_m", "rear_wheel_height_m",
    "fl_wheel_along_m", "fr_wheel_along_m", "fl_actual_wheel_rad_s", "fr_actual_wheel_rad_s", "hl_actual_wheel_rad_s",
    "hr_actual_wheel_rad_s", "front_actual_wheel_rad_s", "rear_actual_wheel_rad_s", "front_contact_force_N",
    "rear_contact_force_N", "front_lr_contact_diff_N", "front_wall_contact_force_N", "front_wall_contact_duration_s",
    "max_contact_force_N", "body_wall_contact", "body_wall_force_N", "front_top_contact", "hook_hold_steps",
    "high_wheel_speed_no_x_progress", "high_wheel_speed_no_height_progress", "sim_diverged", "failure_reason",
)


def make_row(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict[str, Any],
    case: RollCase,
    phase: str,
    step: int,
    start_time: float,
    previous_along: float,
    initial_front_height: float,
    best_front_height: float,
    wall_duration: float,
    hook_steps: int,
    reason: str | None,
) -> tuple[dict[str, Any], float, float]:
    _, angular, local_linear = body_velocity(model, data, ids["base_body_id"])
    base_pos = data.xpos[ids["base_body_id"]]
    centers = wheel_centers(data, ids)
    forces = contact_metrics(model, data, ids)
    roll, pitch, _ = euler_from_xmat(data.xmat[ids["base_body_id"]])
    current_along = along(base_pos[:2])
    front_height = float(np.mean(centers[FRONT_INDICES, 2]))
    rear_height = float(np.mean(centers[REAR_INDICES, 2]))
    wheel_velocity = data.qvel[ids["wheel_dof_addresses"]]
    front_actual = float(np.mean(wheel_velocity[FRONT_INDICES]))
    rear_actual = float(np.mean(wheel_velocity[REAR_INDICES]))
    front_force = float(np.sum(forces["wheel_force"][FRONT_INDICES]))
    rear_force = float(np.sum(forces["wheel_force"][REAR_INDICES]))
    front_wall_force = float(np.sum(forces["front_wall_force"]))
    front_top_contact = bool(
        np.all(centers[FRONT_INDICES, 2] >= PLATFORM_Z - 0.02)
        and np.all([along(center[:2]) >= EXIT_WALL_ALONG_M - 0.04 for center in centers[FRONT_INDICES]])
        and front_force > 20.0
        and abs(float(forces["wheel_force"][0] - forces["wheel_force"][1])) < max(80.0, front_force * 0.75)
    )
    high_speed = float(np.mean(np.abs(wheel_velocity))) > 2.0
    x_delta = current_along - previous_along
    height_delta = front_height - initial_front_height
    return {
        "step": step,
        "time_s": float(data.time - start_time),
        "phase": phase,
        "approach_mode": case.approach.mode,
        "approach_target": case.approach.value,
        "front_wheel_target_rad_s": case.front_wheel_speed,
        "rear_wheel_target_rad_s": case.rear_wheel_speed,
        "leg_mode": case.leg_mode,
        "base_x_m": float(base_pos[0]),
        "base_y_m": float(base_pos[1]),
        "base_z_m": float(base_pos[2]),
        "base_along_m": current_along,
        "base_x_delta_m": x_delta,
        "base_vx_body_mps": float(local_linear[0]),
        "roll_rad": roll,
        "pitch_rad": pitch,
        "roll_rate_rad_s": float(angular[0]),
        "pitch_rate_rad_s": float(angular[1]),
        "fl_wheel_height_m": float(centers[0, 2]),
        "fr_wheel_height_m": float(centers[1, 2]),
        "front_wheel_height_m": front_height,
        "front_wheel_height_delta_m": height_delta,
        "front_wheel_max_height_m": best_front_height,
        "rear_wheel_height_m": rear_height,
        "fl_wheel_along_m": along(centers[0, :2]),
        "fr_wheel_along_m": along(centers[1, :2]),
        "fl_actual_wheel_rad_s": float(wheel_velocity[0]),
        "fr_actual_wheel_rad_s": float(wheel_velocity[1]),
        "hl_actual_wheel_rad_s": float(wheel_velocity[2]),
        "hr_actual_wheel_rad_s": float(wheel_velocity[3]),
        "front_actual_wheel_rad_s": front_actual,
        "rear_actual_wheel_rad_s": rear_actual,
        "front_contact_force_N": front_force,
        "rear_contact_force_N": rear_force,
        "front_lr_contact_diff_N": abs(float(forces["wheel_force"][0] - forces["wheel_force"][1])),
        "front_wall_contact_force_N": front_wall_force,
        "front_wall_contact_duration_s": wall_duration,
        "max_contact_force_N": float(forces["max_contact_force"]),
        "body_wall_contact": int(forces["body_wall_contact"]),
        "body_wall_force_N": float(forces["body_wall_force"]),
        "front_top_contact": int(front_top_contact),
        "hook_hold_steps": hook_steps,
        "high_wheel_speed_no_x_progress": int(high_speed and x_delta <= 1.0e-5),
        "high_wheel_speed_no_height_progress": int(high_speed and height_delta <= 0.002),
        "sim_diverged": int(reason == "physics_divergence" or reason == "non_finite_state"),
        "failure_reason": reason or "",
    }, current_along, front_height


def safe_file_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_")


def case_failure(row: dict[str, Any]) -> str | None:
    if row["sim_diverged"]:
        return "physics_divergence"
    if row["body_wall_contact"]:
        return "body_wall_contact"
    if row["max_contact_force_N"] > 1200.0:
        return "contact_force_over_1200N"
    if abs(row["roll_rad"]) > 0.70:
        return "excessive_roll"
    if abs(row["pitch_rad"]) > 1.35:
        return "excessive_pitch"
    return None


def settle(model: mujoco.MjModel, data: mujoco.MjData, ids: dict[str, Any]) -> tuple[bool, str | None]:
    for _ in range(int(round(SETTLE_SECONDS / DT))):
        apply_direct_control(data, STANDING_POSE, np.zeros(4, dtype=np.float64))
        mujoco.mj_step(model, data)
        reason = diverged(data)
        if reason is not None:
            return False, reason
    forces = contact_metrics(model, data, ids)
    wheel_count = int(np.count_nonzero(forces["wheel_contact"] > 0))
    roll, pitch, _ = euler_from_xmat(data.xmat[ids["base_body_id"]])
    if wheel_count < 3:
        return False, f"unstable_reset_only_{wheel_count}_wheel_contacts"
    if abs(roll) > 0.18 or abs(pitch) > 0.30:
        return False, f"unstable_reset_roll_{roll:.3f}_pitch_{pitch:.3f}"
    return True, None


def run_case(
    model: mujoco.MjModel,
    ids: dict[str, Any],
    case: RollCase,
    output_dir: Path,
    gui: bool,
    capture_enabled: bool,
    frame_period: float,
    roll_seconds: float,
    approach_timeout_s: float,
    shared_renderer: mujoco.Renderer | None,
    shared_camera: mujoco.MjvCamera | None,
    policy_session: ort.InferenceSession | None,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    reset_in_pit(model, data)
    case_dir = output_dir / safe_file_label(case.label)
    case_dir.mkdir(parents=True, exist_ok=True)
    capture = Capture(model, case_dir, capture_enabled, frame_period, shared_renderer, shared_camera)
    viewer = mujoco.viewer.launch_passive(model, data) if gui else None
    if viewer is not None:
        configure_gui(viewer, data, ids)

    rows: list[dict[str, Any]] = []
    phase = "settle"
    reason: str | None = None
    start_time = float(data.time)
    previous_along = along(data.xpos[ids["base_body_id"], :2])
    best_front_height = -float("inf")
    initial_front_height: float | None = None
    wall_duration = 0.0
    hook_steps = 0
    contact_seen = False
    contact_time: float | None = None
    approach_controller = (
        OfficialApproachPolicy(case.approach.value, policy_session) if case.approach.mode == "cmd_vel" else None
    )
    wall_start_time: float | None = None
    wall_force_peak = 0.0
    started_at = time.perf_counter()
    try:
        stable, settle_reason = settle(model, data, ids)
        if not stable:
            reason = settle_reason
        capture.capture(data, ids, 0.0, "reset")
        if stable:
            centers = wheel_centers(data, ids)
            initial_front_height = float(np.mean(centers[FRONT_INDICES, 2]))
            best_front_height = initial_front_height
            phase = "approach"
        max_steps = int(round((SETTLE_SECONDS + approach_timeout_s + roll_seconds) / DT))
        for step in range(max_steps):
            if reason is not None or (viewer is not None and not viewer.is_running()):
                break
            elapsed = float(data.time - start_time)
            if phase == "approach":
                if case.approach.mode == "cmd_vel":
                    assert approach_controller is not None
                    approach_controller.update(step, model, data, ids)
                    approach_controller.apply(data)
                else:
                    apply_direct_control(data, STANDING_POSE, np.full(4, case.approach.value, dtype=np.float64))
            else:
                wheels = np.array(
                    [case.front_wheel_speed, case.front_wheel_speed, case.rear_wheel_speed, case.rear_wheel_speed],
                    dtype=np.float64,
                )
                apply_direct_control(data, leg_target(case.leg_mode), wheels)

            mujoco.mj_step(model, data)
            reason = diverged(data)
            metrics = contact_metrics(model, data, ids)
            if phase == "approach" and metrics["front_wall_contact_any"]:
                phase = "wall_roll"
                contact_seen = True
                contact_time = float(data.time)
                wall_start_time = contact_time
                capture.capture(data, ids, elapsed, "first_front_wall_contact")
            if phase == "approach" and elapsed >= SETTLE_SECONDS + approach_timeout_s:
                reason = "approach_timeout_no_front_wall_contact"
            if phase == "wall_roll" and wall_start_time is not None and float(data.time - wall_start_time) >= roll_seconds:
                reason = "roll_window_complete"
            row, previous_along, current_front_height = make_row(
                model, data, ids, case, phase, step + 1, start_time, previous_along,
                initial_front_height if initial_front_height is not None else 0.0,
                best_front_height if np.isfinite(best_front_height) else current_front_height,
                wall_duration, hook_steps, reason,
            )
            best_front_height = max(best_front_height, current_front_height)
            if metrics["front_wall_contact_any"]:
                wall_duration += DT
                wall_force_peak = max(wall_force_peak, float(np.sum(metrics["front_wall_force"])))
            if row["front_top_contact"]:
                hook_steps += 1
            elif phase == "wall_roll":
                hook_steps = 0
            if row["front_wheel_height_delta_m"] > 0.02:
                capture.capture(data, ids, row["time_s"], "front_height_delta_002")
            if row["front_wheel_height_delta_m"] > 0.05:
                capture.capture(data, ids, row["time_s"], "front_height_delta_005")
            if row["front_wheel_height_delta_m"] > 0.10:
                capture.capture(data, ids, row["time_s"], "front_height_delta_010")
            if row["front_top_contact"]:
                capture.capture(data, ids, row["time_s"], "front_top_contact")
            if row["max_contact_force_N"] > 500.0:
                capture.capture(data, ids, row["time_s"], "contact_force_over_500N")
            if row["body_wall_contact"]:
                capture.capture(data, ids, row["time_s"], "body_wall_contact")
            hard_failure = case_failure(row)
            if hard_failure is not None:
                reason = hard_failure
                row["failure_reason"] = reason
            rows.append(row)
            capture.capture(data, ids, row["time_s"])
            if viewer is not None and step % 10 == 0:
                viewer.sync()
                requested_wall = (step + 1) * DT
                remaining = requested_wall - (time.perf_counter() - started_at)
                if remaining > 0.0:
                    time.sleep(remaining)
            if reason is not None:
                break
    finally:
        if viewer is not None:
            viewer.close()
        capture.close()

    summary = summarize_case(case, rows, reason, contact_seen, contact_time, wall_force_peak, case_dir)
    save_csv(case_dir / "samples.csv", rows, SAMPLE_FIELDS)
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"[{case.label}] contact={summary['front_wall_contact_seen']} "
        f"dH={summary['front_wheel_height_delta_m']:.3f} m "
        f"maxH={summary['front_wheel_max_height_m']:.3f} m "
        f"top={summary['front_top_contact_any']} reason={summary['failure_reason']}",
        flush=True,
    )
    return summary


def save_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_case(
    case: RollCase,
    rows: list[dict[str, Any]],
    reason: str | None,
    contact_seen: bool,
    contact_time: float | None,
    wall_force_peak: float,
    case_dir: Path,
) -> dict[str, Any]:
    if not rows:
        return {
            "case": case.label, "approach_mode": case.approach.mode, "failure_reason": reason or "no_samples",
            "paths": {"directory": str(case_dir), "csv": str(case_dir / "samples.csv"), "video": str(case_dir / "rollout.mp4")},
        }

    values = {field: np.asarray([float(row[field]) for row in rows], dtype=np.float64) for field in SAMPLE_FIELDS if field not in {"phase", "approach_mode", "leg_mode", "failure_reason"}}
    height_delta = float(np.max(values["front_wheel_height_delta_m"]))
    max_height = float(np.max(values["front_wheel_height_m"]))
    front_top = bool(np.any(values["front_top_contact"] > 0.0))
    hook_hold_steps = int(np.max(values["hook_hold_steps"]))
    high_force_no_lift = bool(np.max(values["max_contact_force_N"]) > 500.0 and height_delta <= 0.02)
    spin_no_progress = bool(
        np.mean(values["high_wheel_speed_no_x_progress"]) > 0.25
        and np.mean(values["high_wheel_speed_no_height_progress"]) > 0.25
    )
    if front_top or hook_hold_steps > 0:
        outcome = "success_candidate"
    elif height_delta > 0.10:
        outcome = "strong_height_progress"
    elif height_delta > 0.05:
        outcome = "valid_height_progress"
    elif height_delta > 0.02:
        outcome = "weak_height_progress"
    elif high_force_no_lift:
        outcome = "hard_collision_no_lift"
    elif spin_no_progress:
        outcome = "wheel_spin_no_progress"
    elif reason == "approach_timeout_no_front_wall_contact":
        outcome = "approach_did_not_reach_wall"
    else:
        outcome = "no_effective_height_progress"

    return {
        "case": case.label,
        "approach_mode": case.approach.mode,
        "approach_target": case.approach.value,
        "front_wheel_target_rad_s": case.front_wheel_speed,
        "rear_wheel_target_rad_s": case.rear_wheel_speed,
        "leg_mode": case.leg_mode,
        "samples": len(rows),
        "duration_s": float(values["time_s"][-1]),
        "failure_reason": reason or "completed",
        "outcome": outcome,
        "front_wall_contact_seen": contact_seen,
        "front_wall_contact_time_s": contact_time,
        "front_wall_contact_duration_s": float(np.max(values["front_wall_contact_duration_s"])),
        "front_wall_contact_force_peak_N": wall_force_peak,
        "front_wheel_height_initial_m": float(values["front_wheel_height_m"][0]),
        "front_wheel_height_delta_m": height_delta,
        "front_wheel_max_height_m": max_height,
        "front_top_contact_any": front_top,
        "hook_hold_steps": hook_hold_steps,
        "base_along_final_m": float(values["base_along_m"][-1]),
        "base_along_max_m": float(np.max(values["base_along_m"])),
        "base_vx_body_mean_mps": float(np.mean(values["base_vx_body_mps"])),
        "front_actual_wheel_speed_mean_rad_s": float(np.mean(values["front_actual_wheel_rad_s"])),
        "rear_actual_wheel_speed_mean_rad_s": float(np.mean(values["rear_actual_wheel_rad_s"])),
        "front_contact_force_peak_N": float(np.max(values["front_contact_force_N"])),
        "rear_contact_force_peak_N": float(np.max(values["rear_contact_force_N"])),
        "front_lr_contact_diff_peak_N": float(np.max(values["front_lr_contact_diff_N"])),
        "max_contact_force_N": float(np.max(values["max_contact_force_N"])),
        "body_wall_contact_any": bool(np.any(values["body_wall_contact"] > 0.0)),
        "high_wheel_speed_no_x_progress_fraction": float(np.mean(values["high_wheel_speed_no_x_progress"])),
        "high_wheel_speed_no_height_progress_fraction": float(np.mean(values["high_wheel_speed_no_height_progress"])),
        "roll_abs_max_rad": float(np.max(np.abs(values["roll_rad"]))),
        "pitch_abs_max_rad": float(np.max(np.abs(values["pitch_rad"]))),
        "nan_or_inf": False,
        "paths": {
            "directory": str(case_dir),
            "csv": str(case_dir / "samples.csv"),
            "video": str(case_dir / "rollout.mp4"),
        },
    }


def approaches() -> list[Approach]:
    return [Approach("cmd_vel", 0.4), Approach("cmd_vel", 0.6), Approach("wheel_velocity", -5.0), Approach("wheel_velocity", -6.75)]


def wall_roll_cases(scan: str) -> list[tuple[float, float, str]]:
    if scan == "smoke":
        return [(-5.0, -5.0, "front_fold")]
    if scan == "full":
        return [
            (front, rear, posture)
            for front in (-3.0, -5.0, -6.75)
            for rear in (0.0, -3.0, -5.0, -6.75)
            for posture in (
                "default", "front_fold", "front_extend", "rear_extend", "front_fold_rear_push",
                "front_extend_rear_push", "rear_extend_all_drive",
            )
        ]
    # 14 representative candidates cover each wheel speed, each rear-push
    # level, and each posture once before committing to the 336-case sweep.
    return [
        (-3.0, 0.0, "default"), (-5.0, 0.0, "default"), (-6.75, 0.0, "default"),
        (-3.0, -3.0, "default"), (-5.0, -3.0, "default"), (-6.75, -3.0, "default"),
        (-5.0, -5.0, "default"), (-5.0, -6.75, "default"),
        (-5.0, -5.0, "front_fold"), (-5.0, -5.0, "front_extend"), (-5.0, -5.0, "rear_extend"),
        (-5.0, -5.0, "front_fold_rear_push"), (-5.0, -5.0, "front_extend_rear_push"),
        (-5.0, -5.0, "rear_extend_all_drive"),
    ]


def requested_cases(scan: str, query: str | None, offset: int, max_cases: int | None) -> list[RollCase]:
    selected_scan = "smoke" if scan == "smoke" else scan
    all_cases = [
        RollCase(approach, front, rear, posture)
        for approach in approaches()
        for front, rear, posture in wall_roll_cases(selected_scan)
    ]
    if query:
        all_cases = [case for case in all_cases if query in case.label]
    if offset < 0:
        raise ValueError("--case-offset must be non-negative")
    all_cases = all_cases[offset:]
    if max_cases is not None:
        all_cases = all_cases[:max_cases]
    if not all_cases:
        raise ValueError("no cases selected; adjust --case or --max-cases")
    return all_cases


def compact_summary(summaries: list[dict[str, Any]], path: Path) -> None:
    keys = (
        "case", "approach_mode", "approach_target", "front_wheel_target_rad_s", "rear_wheel_target_rad_s", "leg_mode",
        "outcome", "failure_reason", "front_wall_contact_seen", "front_wheel_height_delta_m", "front_wheel_max_height_m",
        "front_top_contact_any", "hook_hold_steps", "base_along_max_m", "front_contact_force_peak_N", "max_contact_force_N",
        "body_wall_contact_any", "high_wheel_speed_no_x_progress_fraction", "high_wheel_speed_no_height_progress_fraction",
        "roll_abs_max_rad", "pitch_abs_max_rad",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows([{key: summary.get(key, "") for key in keys} for summary in summaries])


def rank_summary(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    invalid = float(summary.get("body_wall_contact_any", False) or summary.get("failure_reason") not in {"completed", "roll_window_complete"})
    return (
        -invalid,
        float(summary.get("front_top_contact_any", False)),
        float(summary.get("front_wheel_height_delta_m", -1.0)),
        -float(summary.get("max_contact_force_N", 1.0e9)),
    )


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.scan = "smoke"
    if args.roll_seconds <= 0.0 or args.approach_timeout <= 0.0 or args.frame_period <= 0.0:
        raise ValueError("durations and frame period must be positive")
    if not MJCF_PATH.is_file() or not POLICY_PATH.is_file():
        raise FileNotFoundError(f"official model/policy missing: {MJCF_PATH} / {POLICY_PATH}")
    output_dir = args.output_dir.resolve() if args.output_dir else PROJECT_ROOT / "logs" / "mujoco" / (
        f"s10_fronthook_scripted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
    model.opt.timestep = DT
    ids = model_ids(model)
    cases = requested_cases(args.scan, args.case, args.case_offset, args.max_cases)
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"[fronthook] official XML: {MJCF_PATH}")
    print(f"[fronthook] output: {output_dir}")
    print(f"[fronthook] exit yaw={EXIT_YAW:.5f}, pit depth={PLATFORM_Z - PIT_BOTTOM_Z:.4f} m, cases={len(cases)}")
    print(f"[fronthook] GUI={'on' if not args.headless else 'off'}, no official assets are modified", flush=True)

    shared_renderer: mujoco.Renderer | None = None
    shared_camera: mujoco.MjvCamera | None = None
    shared_policy_session: ort.InferenceSession | None = None
    if any(case.approach.mode == "cmd_vel" for case in cases):
        shared_policy_session = ort.InferenceSession(str(POLICY_PATH), providers=["CPUExecutionProvider"])
    if not args.no_capture:
        shared_renderer = mujoco.Renderer(model, height=480, width=640)
        shared_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(shared_camera)
        shared_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        shared_camera.distance = 3.2
        shared_camera.azimuth = 130.0
        shared_camera.elevation = -18.0
    summaries = []
    try:
        for index, case in enumerate(cases, start=1):
            print(f"[fronthook] {index}/{len(cases)} {case.label}", flush=True)
            summaries.append(
                run_case(
                    model, ids, case, output_dir, not args.headless, not args.no_capture, args.frame_period,
                    args.roll_seconds, args.approach_timeout, shared_renderer, shared_camera, shared_policy_session,
                )
            )
    finally:
        if shared_renderer is not None:
            shared_renderer.close()
    ranked = sorted(summaries, key=rank_summary, reverse=True)
    compact_summary(summaries, output_dir / "summary.csv")
    report = {
        "purpose": "MuJoCo-only scripted FrontHook feasibility probe",
        "official_mjcf": str(MJCF_PATH),
        "official_assets_modified": False,
        "dt_s": DT,
        "pit_debug_geometry": {
            "bottom_waypoint_xyz": PIT_BOTTOM_WP.tolist(), "exit_waypoint_xyz": PIT_EXIT_WP.tolist(),
            "exit_direction_xy": EXIT_DIRECTION_XY.tolist(), "exit_yaw_rad": EXIT_YAW,
            "bottom_z_m": PIT_BOTTOM_Z, "platform_z_m": PLATFORM_Z, "exit_wall_along_m": EXIT_WALL_ALONG_M,
        },
        "scan": args.scan,
        "cases": summaries,
        "best_five": ranked[:5],
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_dir / "best_five.json").write_text(json.dumps(ranked[:5], indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print("[fronthook] best candidates:")
    for summary in ranked[:5]:
        print(
            f"  {summary['case']}: {summary.get('outcome')} dH={summary.get('front_wheel_height_delta_m', 0.0):.3f} "
            f"top={summary.get('front_top_contact_any')} failure={summary.get('failure_reason')}",
            flush=True,
        )
    print(f"[fronthook] wrote {output_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
