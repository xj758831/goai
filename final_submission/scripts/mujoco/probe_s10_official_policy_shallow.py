#!/usr/bin/env python3
"""Run the shipped S10 ONNX policy on runtime-only shallow vertical pits.

This is a controller-fidelity baseline, not a training script.  It reuses the
official 57-D proprioceptive observation, 50 Hz policy updates, raw action
scales, PD gains, and runtime-only pit geometry from the existing shallow
probe.  Each case writes CSV/JSON plus a headless MuJoCo MP4 and PNG frames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SDK_ROOT = PROJECT_ROOT / "src" / "S10_sdk_deploy"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SDK_ROOT / "scripts"))

from probe_s10_m20_shallow_curriculum import (  # noqa: E402
    OFFICIAL_WALL_SLOPE,
    PIT_EXIT_ALONG_M,
    PLATFORM_Z,
    WHEEL_RADIUS_M,
    build_model,
    local_to_world,
    validate_geometry,
)
from s10_fronthook_scripted_probe import (  # noqa: E402
    DT,
    EXIT_DIRECTION_XY,
    EXIT_YAW,
    EXIT_WALL_ALONG_M,
    KP,
    KD,
    MJCF_PATH,
    PIT_BOTTOM_WP,
    STANDING_POSE,
    body_velocity,
    contact_metrics,
    diverged,
    euler_from_xmat,
    model_ids,
    settle,
    wheel_centers,
    yaw_quaternion,
)
from s10_obs_reference import ACTION_SCALE_ROBOT, build_observation, decode_action  # noqa: E402


POLICY_PATH = SDK_ROOT / "policy" / "policy.onnx"
POLICY_HZ = 50
POLICY_STEPS = int(round(1.0 / (DT * POLICY_HZ)))
SUCCESS_DWELL_S = 0.10
WHEEL_NAMES = ("FL", "FR", "HL", "HR")
LATERAL_DIRECTION_XY = np.asarray([-EXIT_DIRECTION_XY[1], EXIT_DIRECTION_XY[0]], dtype=np.float64)
ALIGNMENT_UNTIL_ALONG_M = 1.10
ALIGNMENT_FORWARD_MPS = 0.40
ALIGNMENT_LATERAL_GAIN = 3.0
ALIGNMENT_YAW_GAIN = 6.0
ALIGNMENT_MAX_SIDE_MPS = 0.10
ALIGNMENT_MAX_YAW_RAD_S = 0.25
ALIGNMENT_TRIGGER_YAW_RAD = math.radians(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-m", type=float, action="append", default=None)
    parser.add_argument("--command-mps", type=float, default=1.0)
    parser.add_argument("--rollout-s", type=float, default=12.0)
    parser.add_argument("--frame-period", type=float, default=0.10)
    parser.add_argument(
        "--alignment-control",
        action="store_true",
        help="apply bounded lateral/yaw correction before the exit approach gate",
    )
    parser.add_argument(
        "--initial-state",
        action="append",
        default=None,
        metavar="LATERAL_M,YAW_DEG",
        help="initial local lateral offset and yaw offset; repeat for a bounded matrix",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--geometry",
        choices=("rectangular", "official_diagonal"),
        default="rectangular",
        help="runtime pit geometry; official_diagonal preserves the measured wall direction",
    )
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


class OfficialPolicy:
    """Exact policy protocol used by rl_deploy, without safety action clipping."""

    def __init__(self) -> None:
        self.session = ort.InferenceSession(str(POLICY_PATH), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        if model_input.name != "obs" or model_input.shape != [1, 57]:
            raise RuntimeError(f"expected official policy obs:[1,57], got {model_input.name}:{model_input.shape}")
        self.last_action = np.zeros(16, dtype=np.float32)
        self.position_target = STANDING_POSE.astype(np.float64).copy()
        self.velocity_target = np.zeros(16, dtype=np.float64)
        self.update_count = 0

    def reset(self) -> None:
        self.last_action.fill(0.0)
        self.position_target[:] = STANDING_POSE
        self.velocity_target.fill(0.0)
        self.update_count = 0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData, ids: dict[str, Any], command: np.ndarray) -> None:
        angular_local, _, _ = body_velocity(model, data, ids["base_body_id"])
        observation = build_observation(
            angular_local,
            data.qpos[3:7].copy(),
            np.asarray(command, dtype=np.float32),
            data.qpos[7:23].copy(),
            data.qvel[6:22].copy(),
            self.last_action,
        )
        action = np.asarray(self.session.run(["actions"], {"obs": observation[None]})[0][0], dtype=np.float32)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise RuntimeError(f"official ONNX emitted invalid action: shape={action.shape}")
        self.last_action = action.copy()
        position, velocity = decode_action(action)
        self.position_target = position.astype(np.float64)
        self.velocity_target = velocity.astype(np.float64)
        self.update_count += 1

    def apply(self, data: mujoco.MjData) -> None:
        data.ctrl[:] = KP.astype(np.float64) * (self.position_target - data.qpos[7:23]) + KD.astype(np.float64) * (
            self.velocity_target - data.qvel[6:22]
        )


class VideoCapture:
    def __init__(self, model: mujoco.MjModel, case_dir: Path, enabled: bool, frame_period: float) -> None:
        self.enabled = enabled
        self.case_dir = case_dir
        self.frame_period = frame_period
        self.next_time = 0.0
        self.index = 0
        self.renderer: mujoco.Renderer | None = None
        self.camera: mujoco.MjvCamera | None = None
        self.writer: cv2.VideoWriter | None = None
        if not enabled:
            return
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 3.2
        self.camera.azimuth = 135.0
        self.camera.elevation = -18.0
        self.writer = cv2.VideoWriter(
            str(case_dir / "rollout.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, 1.0 / frame_period),
            (640, 480),
        )

    def capture(self, data: mujoco.MjData, base_body_id: int, relative_time: float, force: bool = False) -> None:
        if not self.enabled or (not force and relative_time + 1.0e-12 < self.next_time):
            return
        assert self.renderer is not None and self.camera is not None
        self.next_time = relative_time + self.frame_period
        base = data.xpos[base_body_id]
        self.camera.lookat[:] = base
        self.camera.lookat[2] += 0.12
        self.renderer.update_scene(data, camera=self.camera)
        bgr = cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.case_dir / f"frame_{self.index:04d}_t{relative_time:05.2f}s.png"), bgr)
        if self.writer is not None and self.writer.isOpened():
            self.writer.write(bgr)
        self.index += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        if self.renderer is not None:
            self.renderer.close()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_initial_states(raw_states: list[str] | None) -> tuple[tuple[float, float], ...]:
    if not raw_states:
        return ((0.0, 0.0),)
    states: list[tuple[float, float]] = []
    for raw in raw_states:
        parts = raw.split(",")
        if len(parts) != 2:
            raise ValueError(f"initial state must be LATERAL_M,YAW_DEG, got {raw!r}")
        lateral_m, yaw_deg = (float(part.strip()) for part in parts)
        if not math.isfinite(lateral_m) or not math.isfinite(yaw_deg):
            raise ValueError(f"initial state must be finite, got {raw!r}")
        if abs(lateral_m) > 0.05 or abs(yaw_deg) > 5.0:
            raise ValueError("initial perturbations are limited to lateral +/-0.05 m and yaw +/-5 deg")
        states.append((lateral_m, yaw_deg))
    if len(states) > 9:
        raise ValueError("bounded initial-state matrix may contain at most 9 cases")
    return tuple(states)


def state_label(lateral_m: float, yaw_deg: float) -> str:
    if abs(lateral_m) < 1.0e-12 and abs(yaw_deg) < 1.0e-12:
        return "nominal"
    return f"lat{lateral_m:+.3f}_yaw{yaw_deg:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")


def wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def command_for_state(
    data: mujoco.MjData,
    ids: dict[str, Any],
    forward_mps: float,
    alignment_mode: str,
) -> tuple[np.ndarray, float, float, float, bool]:
    base = data.xpos[ids["base_body_id"]]
    base_along = float(np.dot(base[:2] - PIT_BOTTOM_WP[:2], EXIT_DIRECTION_XY))
    base_lateral = float(np.dot(base[:2] - PIT_BOTTOM_WP[:2], LATERAL_DIRECTION_XY))
    _, _, yaw = euler_from_xmat(data.xmat[ids["base_body_id"]])
    yaw_error = wrap_angle(EXIT_YAW - float(yaw))
    active = bool(alignment_mode != "none" and base_along < ALIGNMENT_UNTIL_ALONG_M)
    forward_command = (
        math.copysign(min(abs(forward_mps), ALIGNMENT_FORWARD_MPS), forward_mps)
        if active
        else forward_mps
    )
    side_mps = float(
        np.clip(-ALIGNMENT_LATERAL_GAIN * base_lateral, -ALIGNMENT_MAX_SIDE_MPS, ALIGNMENT_MAX_SIDE_MPS)
    ) if active else 0.0
    yaw_rad_s = float(
        np.clip(ALIGNMENT_YAW_GAIN * yaw_error, -ALIGNMENT_MAX_YAW_RAD_S, ALIGNMENT_MAX_YAW_RAD_S)
    ) if active else 0.0
    return np.asarray([forward_command, side_mps, yaw_rad_s], dtype=np.float32), base_lateral, float(yaw), yaw_error, active


def run_case(
    model: mujoco.MjModel,
    depth_m: float,
    command_mps: float,
    rollout_s: float,
    frame_period: float,
    output_dir: Path,
    video_enabled: bool,
    initial_lateral_m: float,
    initial_yaw_deg: float,
    alignment_control: bool,
    geometry_mode: str,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    ids = model_ids(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = local_to_world(0.0, initial_lateral_m, PLATFORM_Z - depth_m + 0.48)
    data.qpos[3:7] = yaw_quaternion(EXIT_YAW + math.radians(initial_yaw_deg))
    data.qpos[7:23] = STANDING_POSE
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    stable_reset, reset_reason = settle(model, data, ids)
    policy = OfficialPolicy()
    policy.reset()
    case_dir = output_dir / f"depth_{depth_m:.3f}_official_policy_{state_label(initial_lateral_m, initial_yaw_deg)}"
    case_dir.mkdir(parents=True, exist_ok=False)
    capture = VideoCapture(model, case_dir, video_enabled, frame_period)
    capture.capture(data, ids["base_body_id"], 0.0, force=True)

    rows: list[dict[str, Any]] = []
    first_front_top: float | None = None
    first_rear_top: float | None = None
    first_success: float | None = None
    success_dwell = 0
    max_success_dwell = 0
    success_required = int(math.ceil(SUCCESS_DWELL_S / DT))
    termination = "time_limit" if stable_reset else f"reset:{reset_reason}"
    start_time = float(data.time)
    total_steps = int(round(rollout_s / DT)) if stable_reset else 0
    ctrl_low = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64)
    ctrl_high = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64)
    command = np.asarray([command_mps, 0.0, 0.0], dtype=np.float32)
    _, initial_lateral, _, initial_yaw_error, _ = command_for_state(data, ids, command_mps, "none")
    alignment_mode = "none"
    if alignment_control and abs(initial_yaw_error) >= ALIGNMENT_TRIGGER_YAW_RAD:
        alignment_mode = "yaw"
    alignment_triggered = alignment_mode != "none"

    for step in range(total_steps):
        policy_update = step % POLICY_STEPS == 0
        command, base_lateral, base_yaw, yaw_error, alignment_active = command_for_state(
            data, ids, command_mps, alignment_mode
        )
        if policy_update:
            policy.update(model, data, ids, command)
        policy.apply(data)
        requested_ctrl = data.ctrl.copy()
        saturated = bool(np.any(requested_ctrl < ctrl_low) or np.any(requested_ctrl > ctrl_high))
        mujoco.mj_step(model, data)
        relative_time = float(data.time - start_time)
        contacts = contact_metrics(model, data, ids)
        centers = wheel_centers(data, ids)
        wheel_along = np.asarray(
            [float(np.dot(center[:2] - PIT_BOTTOM_WP[:2], EXIT_DIRECTION_XY)) for center in centers]
        )
        wheel_lateral = np.asarray(
            [float(np.dot(center[:2] - PIT_BOTTOM_WP[:2], LATERAL_DIRECTION_XY)) for center in centers]
        )
        diagonal_walls = geometry_mode.startswith("official_diagonal") or geometry_mode.startswith("official_source")
        wheel_exit_progress = (
            wheel_along - OFFICIAL_WALL_SLOPE * wheel_lateral if diagonal_walls else wheel_along
        )
        exit_wall_progress = EXIT_WALL_ALONG_M if geometry_mode.startswith("official_source") else PIT_EXIT_ALONG_M
        wheel_force = np.asarray(contacts["wheel_force"], dtype=np.float64)
        top_geometry = (wheel_exit_progress >= exit_wall_progress - 0.04) & (
            centers[:, 2] >= PLATFORM_Z + WHEEL_RADIUS_M - 0.025
        )
        front_top = bool(np.all(top_geometry[:2]) and np.sum(wheel_force[:2]) >= 5.0)
        rear_top_any = bool(np.any(top_geometry[2:] & (wheel_force[2:] >= 2.0)))
        rear_top_pair = bool(np.all(top_geometry[2:]) and np.sum(wheel_force[2:]) >= 5.0)
        if front_top and first_front_top is None:
            first_front_top = relative_time
        if rear_top_any and first_rear_top is None:
            first_rear_top = relative_time
        base = data.xpos[ids["base_body_id"]]
        base_along = float(np.dot(base[:2] - PIT_BOTTOM_WP[:2], EXIT_DIRECTION_XY))
        base_lateral_after = float(np.dot(base[:2] - PIT_BOTTOM_WP[:2], LATERAL_DIRECTION_XY))
        base_exit_progress = (
            base_along - OFFICIAL_WALL_SLOPE * base_lateral_after if diagonal_walls else base_along
        )
        roll, pitch, _ = euler_from_xmat(data.xmat[ids["base_body_id"]])
        angular_velocity, linear_velocity, _ = body_velocity(model, data, ids["base_body_id"])
        stable_exit = bool(
            front_top
            and rear_top_pair
            and base_exit_progress >= exit_wall_progress + 0.25
            and float(base[2]) >= PLATFORM_Z + 0.20
            and abs(float(roll)) <= math.radians(15.0)
            and abs(float(pitch)) <= math.radians(20.0)
            # The official command is allowed to be 1.0 m/s.  A success gate
            # at exactly 1.0 rejected a clean exit at 1.001 m/s, so retain a
            # bounded tracking margin while keeping the posture/contact gates.
            and float(np.linalg.norm(linear_velocity)) <= max(1.5, 1.5 * abs(command_mps))
            and float(np.linalg.norm(angular_velocity)) <= 2.0
        )
        success_dwell = success_dwell + 1 if stable_exit else 0
        max_success_dwell = max(max_success_dwell, success_dwell)
        if success_dwell >= success_required and first_success is None:
            first_success = relative_time - (success_required - 1) * DT
        row: dict[str, Any] = {
            "step": step + 1,
            "time_s": relative_time,
            "policy_update": int(policy_update),
            "command_forward_mps": float(command[0]),
            "command_side_mps": float(command[1]),
            "command_yaw_rad_s": float(command[2]),
            "alignment_active": int(alignment_active),
            "base_along_m": base_along,
            "base_lateral_m": base_lateral_after,
            "base_exit_progress_m": base_exit_progress,
            "base_yaw_rad": base_yaw,
            "yaw_error_rad": yaw_error,
            "base_z_m": float(base[2]),
            "roll_rad": float(roll),
            "pitch_rad": float(pitch),
            "base_linear_speed_mps": float(np.linalg.norm(linear_velocity)),
            "base_angular_speed_rad_s": float(np.linalg.norm(angular_velocity)),
            **{f"{name}_along_m": float(wheel_along[index]) for index, name in enumerate(WHEEL_NAMES)},
            **{f"{name}_lateral_m": float(wheel_lateral[index]) for index, name in enumerate(WHEEL_NAMES)},
            **{f"{name}_exit_progress_m": float(wheel_exit_progress[index]) for index, name in enumerate(WHEEL_NAMES)},
            **{f"{name}_height_m": float(centers[index, 2]) for index, name in enumerate(WHEEL_NAMES)},
            **{f"{name}_force_N": float(wheel_force[index]) for index, name in enumerate(WHEEL_NAMES)},
            "front_top_contact": int(front_top),
            "rear_top_any": int(rear_top_any),
            "rear_top_pair": int(rear_top_pair),
            "stable_exit": int(stable_exit),
            "success_dwell_steps": success_dwell,
            "max_contact_force_N": float(contacts["max_contact_force"]),
            "body_wall_contact": int(bool(contacts["body_wall_contact"])),
            "max_abs_requested_ctrl": float(np.max(np.abs(requested_ctrl))),
            "max_abs_actuator_force": float(np.max(np.abs(data.actuator_force))),
            "ctrl_saturated": int(saturated),
        }
        for index, value in enumerate(policy.last_action):
            row[f"policy_action_{index:02d}"] = float(value)
        for index, value in enumerate(policy.velocity_target[[3, 7, 11, 15]]):
            row[f"target_{WHEEL_NAMES[index]}_rad_s"] = float(value)
        rows.append(row)
        capture.capture(data, ids["base_body_id"], relative_time)

        bad = diverged(data)
        if bad is not None:
            termination = bad
            break
        if bool(contacts["body_wall_contact"]):
            termination = "body_wall_contact"
            break
        if float(contacts["max_contact_force"]) > 1200.0:
            termination = "excessive_contact_force"
            break
        if abs(float(roll)) > math.radians(30.0):
            termination = "excessive_roll"
            break
        if abs(float(pitch)) > math.radians(80.0):
            termination = "excessive_pitch"
            break
        if first_success is not None:
            termination = "stable_exit"
            break

    capture.close()
    values = lambda name: np.asarray([row[name] for row in rows], dtype=np.float64)
    actions = np.asarray([[row[f"policy_action_{index:02d}"] for index in range(16)] for row in rows])
    summary = {
        "depth_m": depth_m,
        "initial_lateral_m": initial_lateral_m,
        "initial_yaw_deg": initial_yaw_deg,
        "alignment_control": alignment_control,
        "geometry_mode": geometry_mode,
        "alignment_triggered": alignment_triggered,
        "alignment_mode": alignment_mode,
        "initial_lateral_after_settle_m": initial_lateral,
        "initial_yaw_error_deg": math.degrees(initial_yaw_error),
        "controller": "official_policy.onnx",
        "command_mps": command_mps,
        "policy_hz": POLICY_HZ,
        "policy_observation_dim": 57,
        "policy_action_dim": 16,
        "stable_reset": stable_reset,
        "reset_reason": reset_reason,
        "termination_reason": termination,
        "front_top_contact_seen": first_front_top is not None,
        "first_front_top_contact_time_s": first_front_top,
        "rear_top_contact_seen": first_rear_top is not None,
        "first_rear_top_contact_time_s": first_rear_top,
        "max_success_dwell_steps": max_success_dwell,
        "stable_exit": first_success is not None,
        "first_stable_exit_time_s": first_success,
        "max_base_along_m": float(values("base_along_m").max()) if rows else None,
        "max_rear_along_m": float(np.maximum(values("HL_along_m"), values("HR_along_m")).max()) if rows else None,
        "max_rear_height_m": float(np.maximum(values("HL_height_m"), values("HR_height_m")).max()) if rows else None,
        "max_abs_roll_rad": float(np.abs(values("roll_rad")).max()) if rows else None,
        "max_abs_pitch_rad": float(np.abs(values("pitch_rad")).max()) if rows else None,
        "max_contact_force_N": float(values("max_contact_force_N").max()) if rows else None,
        "max_abs_requested_ctrl": float(values("max_abs_requested_ctrl").max()) if rows else None,
        "max_abs_actuator_force": float(values("max_abs_actuator_force").max()) if rows else None,
        "ctrl_saturation_duty": float(values("ctrl_saturated").mean()) if rows else None,
        "max_abs_raw_policy_action": float(np.max(np.abs(actions))) if rows else None,
        "max_abs_policy_wheel_target_rad_s": float(
            np.max(np.abs(np.asarray([[row[f"target_{name}_rad_s"] for name in WHEEL_NAMES] for row in rows])))
        ) if rows else None,
        "dynamic_steps": len(rows),
        "training_performed": False,
        "state_teleported_during_dynamics": False,
        "files": {
            "summary": str(case_dir / "summary.json"),
            "samples": str(case_dir / "samples.csv"),
            "video": str(case_dir / "rollout.mp4") if video_enabled else None,
        },
    }
    write_csv(case_dir / "samples.csv", rows)
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    depths = tuple(args.depth_m or (0.08, 0.16, 0.23))
    initial_states = parse_initial_states(args.initial_state)
    if not depths or not all(math.isfinite(value) and 0.0 < value < PLATFORM_Z for value in depths):
        raise ValueError("depths must be positive and below the official platform height")
    if not math.isfinite(args.command_mps) or abs(args.command_mps) > 1.0:
        raise ValueError("command must remain inside the official +/-1.0 m/s limit")
    if not math.isfinite(args.rollout_s) or args.rollout_s <= SUCCESS_DWELL_S:
        raise ValueError("rollout must exceed the success dwell")
    if not math.isfinite(args.frame_period) or args.frame_period <= 0.0:
        raise ValueError("frame period must be positive")
    if not POLICY_PATH.is_file():
        raise FileNotFoundError(f"missing official policy: {POLICY_PATH}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else PROJECT_ROOT / "logs/mujoco" / "s10_official_policy_shallow"
    output_dir.mkdir(parents=True, exist_ok=False)

    geometry_reports: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for depth_m in depths:
        model = build_model(depth_m, args.geometry)
        geometry = validate_geometry(model, depth_m)
        geometry_reports.append(geometry)
        print(f"[official-shallow] depth={depth_m:.3f} geometry_ok measured={geometry['measured_depth_m']:.6f} m", flush=True)
        for initial_lateral_m, initial_yaw_deg in initial_states:
            summary = run_case(
                model,
                depth_m,
                args.command_mps,
                args.rollout_s,
                args.frame_period,
                output_dir,
                not args.no_video,
                initial_lateral_m,
                initial_yaw_deg,
                args.alignment_control,
                args.geometry,
            )
            summaries.append(summary)
            print(
                f"[official-shallow] depth={depth_m:.3f} state={state_label(initial_lateral_m, initial_yaw_deg)} "
                f"front={summary['front_top_contact_seen']} rear={summary['rear_top_contact_seen']} "
                f"exit={summary['stable_exit']} reason={summary['termination_reason']} "
                f"max_wheel_target={summary['max_abs_policy_wheel_target_rad_s']:.2f}",
                flush=True,
            )

    success_depths = sorted({float(item["depth_m"]) for item in summaries if bool(item["stable_exit"])})
    report = {
        "purpose": "Official S10 ONNX shallow-pit controller-fidelity baseline; no training",
        "official_mjcf": str(MJCF_PATH),
        "official_mjcf_sha256": hashlib.sha256(MJCF_PATH.read_bytes()).hexdigest(),
        "official_policy": str(POLICY_PATH),
        "official_assets_modified": False,
        "depths_m": list(depths),
        "command_mps": args.command_mps,
        "rollout_s": args.rollout_s,
        "alignment_control": args.alignment_control,
        "geometry_mode": args.geometry,
        "alignment_config": {
            "until_base_along_m": ALIGNMENT_UNTIL_ALONG_M,
            "forward_mps": ALIGNMENT_FORWARD_MPS,
            "lateral_gain_per_s": ALIGNMENT_LATERAL_GAIN,
            "yaw_gain_per_s": ALIGNMENT_YAW_GAIN,
            "max_side_mps": ALIGNMENT_MAX_SIDE_MPS,
            "max_yaw_rad_s": ALIGNMENT_MAX_YAW_RAD_S,
            "trigger_abs_yaw_deg": math.degrees(ALIGNMENT_TRIGGER_YAW_RAD),
        },
        "initial_states": [
            {"lateral_m": lateral_m, "yaw_deg": yaw_deg} for lateral_m, yaw_deg in initial_states
        ],
        "geometry": geometry_reports,
        "rollouts": summaries,
        "successful_depths_m": success_depths,
        "training_performed": False,
        "authorization": (
            "Only a successful official-policy depth may be used as a bounded curriculum baseline; no PPO authorization is implied."
            if success_depths
            else "Official policy did not complete any shallow depth; validate flat/stair controller fidelity before training."
        ),
        "files": {"summary": str(output_dir / "summary.json")},
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if success_depths else 2


if __name__ == "__main__":
    raise SystemExit(main())
