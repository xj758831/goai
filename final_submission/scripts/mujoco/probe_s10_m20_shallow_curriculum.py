#!/usr/bin/env python3
"""Run bounded S10 shallow-pit feasibility probes without training.

The official S10 robot, actuator configuration, control gains, and world
coordinates are loaded from the competition MJCF.  The 2185-geometries track
body is replaced only in an in-memory ``MjSpec`` by a rectangular vertical
pit whose top height, floor length, direction, and corridor width match the
measured exit segment.  Source XML, meshes, ONNX, and hardware files are never
written.

This probe is deliberately one-sided: a successful rollout can authorize a
later curriculum stage, while failure only rejects these three bounded
primitives and must not be reported as proof that no policy can succeed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from s10_fronthook_scripted_probe import (  # noqa: E402
    DT,
    EXIT_DIRECTION_XY,
    EXIT_YAW,
    MJCF_PATH,
    PIT_BOTTOM_WP,
    PLATFORM_Z,
    STANDING_POSE,
    apply_direct_control,
    body_velocity,
    contact_metrics,
    diverged,
    euler_from_xmat,
    model_ids,
    settle,
    wheel_centers,
    yaw_quaternion,
)


PIT_ENTRY_ALONG_M = -0.75
PIT_EXIT_ALONG_M = 1.75
PIT_HALF_WIDTH_M = 2.0
DECK_HALF_WIDTH_M = 6.0
PATCH_ALONG_MIN_M = -4.0
PATCH_ALONG_MAX_M = 5.0
WHEEL_RADIUS_M = 0.081
FRICTION = np.asarray([1.0, 0.005, 0.0001], dtype=np.float64)
OFFICIAL_WALL_SLOPE = -1.0 / 3.0
GEOMETRY_TOLERANCE_M = 1.0e-5
SUCCESS_DWELL_S = 0.10
WHEEL_NAMES = ("FL", "FR", "HL", "HR")


@dataclass(frozen=True)
class Primitive:
    name: str
    front_hipy_delta_rad: float = 0.0
    front_knee_delta_rad: float = 0.0
    rear_hipy_delta_rad: float = 0.0
    rear_knee_delta_rad: float = 0.0


PRIMITIVES = {
    "default": Primitive("default"),
    # Every offset stays inside the M20/S10 hipy/knee action scale of 0.25 rad.
    "front_fold": Primitive(
        "front_fold", front_hipy_delta_rad=-0.12, front_knee_delta_rad=0.20
    ),
    "rear_extend": Primitive(
        "rear_extend", rear_hipy_delta_rad=-0.10, rear_knee_delta_rad=0.16
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-m", type=float, action="append", default=None)
    parser.add_argument("--case", choices=tuple(PRIMITIVES), action="append", default=None)
    parser.add_argument("--wheel-speed-rad-s", type=float, default=-5.0)
    parser.add_argument("--rollout-s", type=float, default=7.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[str, ...]]:
    depths = tuple(args.depth_m or (0.08, 0.16, 0.23))
    cases = tuple(args.case or tuple(PRIMITIVES))
    if not depths or not all(math.isfinite(value) and 0.0 < value < PLATFORM_Z for value in depths):
        raise ValueError("depths must be finite, positive, and below the official platform height")
    if not math.isfinite(args.wheel_speed_rad_s) or abs(args.wheel_speed_rad_s) > 5.0:
        raise ValueError("wheel speed must remain inside the M20/S10 +/-5 rad/s action scale")
    if not math.isfinite(args.rollout_s) or args.rollout_s <= SUCCESS_DWELL_S:
        raise ValueError("rollout duration must exceed the success dwell")
    return depths, cases


def add_box(
    body: Any,
    *,
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    rgba: tuple[float, float, float, float],
    yaw_rad: float = 0.0,
) -> None:
    geom = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX)
    geom.pos = np.asarray(pos, dtype=np.float64)
    geom.size = np.asarray(size, dtype=np.float64)
    geom.quat = yaw_quaternion(yaw_rad)
    geom.rgba = np.asarray(rgba, dtype=np.float32)
    geom.friction = FRICTION.copy()
    geom.contype = 1
    geom.conaffinity = 1


def add_diagonal_pit(
    body: Any,
    *,
    depth_m: float,
) -> None:
    """Approximate the official pit's diagonal entry/exit walls with boxes."""
    top_thickness = 0.05
    wall_thickness = 0.02
    floor_thickness = 0.05
    normal = np.asarray([1.0, -OFFICIAL_WALL_SLOPE], dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    normal /= normal_norm
    wall_yaw = math.atan2(float(normal[1]), float(normal[0]))
    deck_normal_half = 8.0
    deck_tangent_half = 10.0

    for name, wall_x, direction in (
        ("entry", PIT_ENTRY_ALONG_M, -1.0),
        ("exit", PIT_EXIT_ALONG_M, 1.0),
    ):
        boundary_normal = wall_x / normal_norm
        center_normal = boundary_normal + direction * deck_normal_half
        add_box(
            body,
            name=f"official_diagonal_{name}_deck",
            pos=(float(normal[0] * center_normal), float(normal[1] * center_normal), -top_thickness),
            size=(deck_normal_half, deck_tangent_half, top_thickness),
            rgba=(0.32, 0.34, 0.37, 1.0),
            yaw_rad=wall_yaw,
        )
        add_box(
            body,
            name=f"official_diagonal_{name}_wall",
            pos=(wall_x, 0.0, -0.5 * depth_m),
            size=(wall_thickness, normal_norm * PIT_HALF_WIDTH_M, 0.5 * depth_m),
            rgba=(0.43, 0.45, 0.48, 1.0),
            yaw_rad=wall_yaw,
        )

    pit_center = 0.5 * (PIT_ENTRY_ALONG_M + PIT_EXIT_ALONG_M)
    pit_half_length = 0.5 * (PIT_EXIT_ALONG_M - PIT_ENTRY_ALONG_M)
    floor_half_length = pit_half_length + abs(OFFICIAL_WALL_SLOPE) * PIT_HALF_WIDTH_M
    add_box(
        body,
        name="official_diagonal_pit_floor",
        pos=(pit_center, 0.0, -depth_m - floor_thickness),
        size=(floor_half_length, PIT_HALF_WIDTH_M, floor_thickness),
        rgba=(0.20, 0.22, 0.25, 1.0),
    )
    for side in (-1.0, 1.0):
        side_center_x = pit_center + OFFICIAL_WALL_SLOPE * side * PIT_HALF_WIDTH_M
        add_box(
            body,
            name=f"official_diagonal_side_wall_{'left' if side > 0 else 'right'}",
            pos=(side_center_x, side * PIT_HALF_WIDTH_M, -0.5 * depth_m),
            size=(pit_half_length, wall_thickness, 0.5 * depth_m),
            rgba=(0.43, 0.45, 0.48, 1.0),
        )
        add_box(
            body,
            name=f"official_diagonal_side_deck_{'left' if side > 0 else 'right'}",
            pos=(
                0.5 * (PATCH_ALONG_MIN_M + PATCH_ALONG_MAX_M),
                side * 0.5 * (PIT_HALF_WIDTH_M + DECK_HALF_WIDTH_M),
                -top_thickness,
            ),
            size=(
                0.5 * (PATCH_ALONG_MAX_M - PATCH_ALONG_MIN_M),
                0.5 * (DECK_HALF_WIDTH_M - PIT_HALF_WIDTH_M),
                top_thickness,
            ),
            rgba=(0.32, 0.34, 0.37, 1.0),
        )


def build_model(depth_m: float, wall_mode: str = "rectangular") -> mujoco.MjModel:
    """Compile a runtime-only shallow pit around the official S10 robot."""
    if wall_mode not in {"rectangular", "official_diagonal"}:
        raise ValueError(f"unknown wall mode: {wall_mode}")
    spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
    for body in list(spec.bodies):
        if body.name in {"main_body", "track_overlay"}:
            spec.delete(body)

    terrain = spec.worldbody.add_body(name=f"s10_shallow_pit_{int(round(depth_m * 1000)):03d}mm")
    terrain.pos = np.asarray([PIT_BOTTOM_WP[0], PIT_BOTTOM_WP[1], PLATFORM_Z], dtype=np.float64)
    terrain.quat = yaw_quaternion(EXIT_YAW)
    top_thickness = 0.05
    wall_thickness = 0.02
    floor_thickness = 0.05
    pit_center = 0.5 * (PIT_ENTRY_ALONG_M + PIT_EXIT_ALONG_M)
    pit_half_length = 0.5 * (PIT_EXIT_ALONG_M - PIT_ENTRY_ALONG_M)

    if wall_mode == "official_diagonal":
        add_diagonal_pit(terrain, depth_m=depth_m)
    else:
        add_box(
            terrain,
            name="curriculum_entry_deck",
            pos=(0.5 * (PATCH_ALONG_MIN_M + PIT_ENTRY_ALONG_M), 0.0, -top_thickness),
            size=(0.5 * (PIT_ENTRY_ALONG_M - PATCH_ALONG_MIN_M), DECK_HALF_WIDTH_M, top_thickness),
            rgba=(0.32, 0.34, 0.37, 1.0),
        )
        add_box(
            terrain,
            name="curriculum_exit_deck",
            pos=(0.5 * (PIT_EXIT_ALONG_M + PATCH_ALONG_MAX_M), 0.0, -top_thickness),
            size=(0.5 * (PATCH_ALONG_MAX_M - PIT_EXIT_ALONG_M), DECK_HALF_WIDTH_M, top_thickness),
            rgba=(0.32, 0.34, 0.37, 1.0),
        )
        for side in (-1.0, 1.0):
            add_box(
                terrain,
                name=f"curriculum_side_deck_{'left' if side > 0 else 'right'}",
                pos=(pit_center, side * 0.5 * (PIT_HALF_WIDTH_M + DECK_HALF_WIDTH_M), -top_thickness),
                size=(pit_half_length, 0.5 * (DECK_HALF_WIDTH_M - PIT_HALF_WIDTH_M), top_thickness),
                rgba=(0.32, 0.34, 0.37, 1.0),
            )
        add_box(
            terrain,
            name="curriculum_pit_floor",
            pos=(pit_center, 0.0, -depth_m - floor_thickness),
            size=(pit_half_length, PIT_HALF_WIDTH_M, floor_thickness),
            rgba=(0.20, 0.22, 0.25, 1.0),
        )
        for name, along_position in (
            ("entry", PIT_ENTRY_ALONG_M),
            ("exit", PIT_EXIT_ALONG_M),
        ):
            add_box(
                terrain,
                name=f"curriculum_{name}_wall",
                pos=(along_position, 0.0, -0.5 * depth_m),
                size=(wall_thickness, PIT_HALF_WIDTH_M, 0.5 * depth_m),
                rgba=(0.43, 0.45, 0.48, 1.0),
            )
        for side in (-1.0, 1.0):
            add_box(
                terrain,
                name=f"curriculum_side_wall_{'left' if side > 0 else 'right'}",
                pos=(pit_center, side * PIT_HALF_WIDTH_M, -0.5 * depth_m),
                size=(pit_half_length, wall_thickness, 0.5 * depth_m),
                rgba=(0.43, 0.45, 0.48, 1.0),
            )

    model = spec.compile()
    model.opt.timestep = DT
    return model


def local_to_world(along_m: float, lateral_m: float, z_m: float) -> np.ndarray:
    lateral_direction = np.asarray([-EXIT_DIRECTION_XY[1], EXIT_DIRECTION_XY[0]], dtype=np.float64)
    xy = PIT_BOTTOM_WP[:2] + EXIT_DIRECTION_XY * along_m + lateral_direction * lateral_m
    return np.asarray([xy[0], xy[1], z_m], dtype=np.float64)


def ray_surface_height(model: mujoco.MjModel, along_m: float, lateral_m: float = 0.0) -> float:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    origin = local_to_world(along_m, lateral_m, PLATFORM_Z + 1.0)
    geom_id = np.asarray([-1], dtype=np.int32)
    collision_groups = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    distance = float(
        mujoco.mj_ray(
            model,
            data,
            origin,
            np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
            collision_groups,
            1,
            -1,
            geom_id,
        )
    )
    if distance < 0.0:
        raise RuntimeError(f"downward ray missed curriculum terrain at along={along_m}")
    return float(origin[2] - distance)


def validate_geometry(model: mujoco.MjModel, depth_m: float) -> dict[str, Any]:
    samples = {
        "entry_top": (PIT_ENTRY_ALONG_M - 0.10, PLATFORM_Z),
        "pit_floor": (0.0, PLATFORM_Z - depth_m),
        "exit_top": (PIT_EXIT_ALONG_M + 0.10, PLATFORM_Z),
    }
    measured = {name: ray_surface_height(model, along_m) for name, (along_m, _) in samples.items()}
    errors = {
        name: abs(measured[name] - expected)
        for name, (_, expected) in samples.items()
    }
    valid = max(errors.values()) <= GEOMETRY_TOLERANCE_M
    if not valid:
        raise RuntimeError(f"curriculum geometry validation failed: measured={measured}, errors={errors}")
    return {
        "requested_depth_m": depth_m,
        "measured_depth_m": measured["exit_top"] - measured["pit_floor"],
        "surface_height_m": measured,
        "max_surface_error_m": max(errors.values()),
        "valid": valid,
    }


def position_target(primitive: Primitive, active: bool) -> np.ndarray:
    target = STANDING_POSE.copy()
    if not active:
        return target
    for hipy_index, knee_index in ((1, 2), (5, 6)):
        target[hipy_index] += primitive.front_hipy_delta_rad
        target[knee_index] += primitive.front_knee_delta_rad
    for hipy_index, knee_index in ((9, 10), (13, 14)):
        target[hipy_index] += primitive.rear_hipy_delta_rad
        target[knee_index] += primitive.rear_knee_delta_rad
    return target


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_case(
    model: mujoco.MjModel,
    depth_m: float,
    primitive: Primitive,
    wheel_speed_rad_s: float,
    rollout_s: float,
    output_dir: Path,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    ids = model_ids(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = local_to_world(0.0, 0.0, PLATFORM_Z - depth_m + 0.48)
    data.qpos[3:7] = yaw_quaternion(EXIT_YAW)
    data.qpos[7:23] = STANDING_POSE
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    stable_reset, reset_reason = settle(model, data, ids)

    ctrl_low = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float64)
    ctrl_high = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    primitive_active = primitive.name == "default"
    success_dwell = 0
    success_required = int(math.ceil(SUCCESS_DWELL_S / DT))
    max_success_dwell = 0
    first_front_top_time: float | None = None
    first_rear_top_time: float | None = None
    first_success_time: float | None = None
    termination = "time_limit" if stable_reset else f"reset:{reset_reason}"
    total_steps = int(round(rollout_s / DT)) if stable_reset else 0

    for step in range(1, total_steps + 1):
        centers_before = wheel_centers(data, ids)
        front_along_before = np.asarray(
            [float(np.dot(center[:2] - PIT_BOTTOM_WP[:2], EXIT_DIRECTION_XY)) for center in centers_before[:2]]
        )
        if primitive.name != "default" and np.max(front_along_before) >= PIT_EXIT_ALONG_M - 0.12:
            primitive_active = True
        target = position_target(primitive, primitive_active)
        apply_direct_control(
            data,
            target,
            np.full(4, wheel_speed_rad_s, dtype=np.float64),
        )
        requested_ctrl = data.ctrl.copy()
        saturated = bool(np.any(requested_ctrl < ctrl_low) or np.any(requested_ctrl > ctrl_high))
        mujoco.mj_step(model, data)

        contacts = contact_metrics(model, data, ids)
        centers = wheel_centers(data, ids)
        wheel_along = np.asarray(
            [float(np.dot(center[:2] - PIT_BOTTOM_WP[:2], EXIT_DIRECTION_XY)) for center in centers]
        )
        wheel_force = np.asarray(contacts["wheel_force"], dtype=np.float64)
        top_geometry = (
            (wheel_along >= PIT_EXIT_ALONG_M - 0.04)
            & (centers[:, 2] >= PLATFORM_Z + WHEEL_RADIUS_M - 0.025)
        )
        front_top = bool(np.all(top_geometry[:2]) and np.sum(wheel_force[:2]) >= 5.0)
        rear_top_any = bool(np.any(top_geometry[2:] & (wheel_force[2:] >= 2.0)))
        rear_top_pair = bool(np.all(top_geometry[2:]) and np.sum(wheel_force[2:]) >= 5.0)
        if front_top and first_front_top_time is None:
            first_front_top_time = float(data.time)
        if rear_top_any and first_rear_top_time is None:
            first_rear_top_time = float(data.time)

        base = data.xpos[ids["base_body_id"]]
        base_along = float(np.dot(base[:2] - PIT_BOTTOM_WP[:2], EXIT_DIRECTION_XY))
        roll, pitch, _ = euler_from_xmat(data.xmat[ids["base_body_id"]])
        angular_velocity, linear_velocity, _ = body_velocity(model, data, ids["base_body_id"])
        stable_exit = bool(
            front_top
            and rear_top_pair
            and base_along >= PIT_EXIT_ALONG_M + 0.25
            and float(base[2]) >= PLATFORM_Z + 0.20
            and abs(float(roll)) <= math.radians(15.0)
            and abs(float(pitch)) <= math.radians(20.0)
            and float(np.linalg.norm(linear_velocity)) <= 1.0
            and float(np.linalg.norm(angular_velocity)) <= 1.5
        )
        success_dwell = success_dwell + 1 if stable_exit else 0
        max_success_dwell = max(max_success_dwell, success_dwell)
        if success_dwell >= success_required and first_success_time is None:
            first_success_time = float(data.time - (success_required - 1) * DT)

        rows.append(
            {
                "step": step,
                "time_s": float(data.time),
                "primitive_active": int(primitive_active),
                "base_along_m": base_along,
                "base_z_m": float(base[2]),
                "roll_rad": float(roll),
                "pitch_rad": float(pitch),
                "base_linear_speed_mps": float(np.linalg.norm(linear_velocity)),
                "base_angular_speed_rad_s": float(np.linalg.norm(angular_velocity)),
                "FL_along_m": float(wheel_along[0]),
                "FR_along_m": float(wheel_along[1]),
                "HL_along_m": float(wheel_along[2]),
                "HR_along_m": float(wheel_along[3]),
                "FL_height_m": float(centers[0, 2]),
                "FR_height_m": float(centers[1, 2]),
                "HL_height_m": float(centers[2, 2]),
                "HR_height_m": float(centers[3, 2]),
                "FL_force_N": float(wheel_force[0]),
                "FR_force_N": float(wheel_force[1]),
                "HL_force_N": float(wheel_force[2]),
                "HR_force_N": float(wheel_force[3]),
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
        )

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
        if first_success_time is not None:
            termination = "stable_exit"
            break

    case_dir = output_dir / f"depth_{depth_m:.3f}_{primitive.name}"
    case_dir.mkdir(parents=True, exist_ok=False)
    if rows:
        write_csv(case_dir / "samples.csv", rows)
    values = lambda name: np.asarray([row[name] for row in rows], dtype=np.float64)
    success = first_success_time is not None
    summary = {
        "depth_m": depth_m,
        "case": primitive.name,
        "wheel_speed_rad_s": wheel_speed_rad_s,
        "stable_reset": stable_reset,
        "reset_reason": reset_reason,
        "termination_reason": termination,
        "front_top_contact_seen": first_front_top_time is not None,
        "first_front_top_contact_time_s": first_front_top_time,
        "rear_top_contact_seen": first_rear_top_time is not None,
        "first_rear_top_contact_time_s": first_rear_top_time,
        "max_success_dwell_steps": max_success_dwell,
        "stable_exit": success,
        "first_stable_exit_time_s": first_success_time,
        "max_base_along_m": float(values("base_along_m").max()) if rows else None,
        "max_rear_along_m": float(np.maximum(values("HL_along_m"), values("HR_along_m")).max()) if rows else None,
        "max_rear_height_m": float(np.maximum(values("HL_height_m"), values("HR_height_m")).max()) if rows else None,
        "max_abs_roll_rad": float(np.abs(values("roll_rad")).max()) if rows else None,
        "max_abs_pitch_rad": float(np.abs(values("pitch_rad")).max()) if rows else None,
        "max_contact_force_N": float(values("max_contact_force_N").max()) if rows else None,
        "max_abs_requested_ctrl": float(values("max_abs_requested_ctrl").max()) if rows else None,
        "max_abs_actuator_force": float(values("max_abs_actuator_force").max()) if rows else None,
        "ctrl_saturation_duty": float(values("ctrl_saturated").mean()) if rows else None,
        "dynamic_steps": len(rows),
        "training_performed": False,
        "state_teleported_during_dynamics": False,
        "files": {
            "summary": str(case_dir / "summary.json"),
            "samples": str(case_dir / "samples.csv") if rows else None,
        },
    }
    (case_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    depths, case_names = validate_args(args)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT
        / "logs/mujoco"
        / f"s10_m20_shallow_curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    geometry_reports: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for depth_m in depths:
        model = build_model(depth_m)
        geometry_report = validate_geometry(model, depth_m)
        geometry_reports.append(geometry_report)
        print(
            f"[shallow] depth={depth_m:.3f} m geometry_ok measured={geometry_report['measured_depth_m']:.6f} m",
            flush=True,
        )
        for case_name in case_names:
            print(f"[shallow] depth={depth_m:.3f} case={case_name}", flush=True)
            summary = run_case(
                model,
                depth_m,
                PRIMITIVES[case_name],
                args.wheel_speed_rad_s,
                args.rollout_s,
                output_dir,
            )
            summaries.append(summary)
            print(
                f"[shallow] result case={case_name} front={summary['front_top_contact_seen']} "
                f"rear={summary['rear_top_contact_seen']} exit={summary['stable_exit']} "
                f"termination={summary['termination_reason']}",
                flush=True,
            )

    success_depths = sorted(
        {
            float(summary["depth_m"])
            for summary in summaries
            if bool(summary["stable_exit"])
        }
    )
    report = {
        "purpose": "S10 M20-style shallow vertical-pit feasibility gate; no training",
        "official_mjcf": str(MJCF_PATH),
        "official_mjcf_sha256": hashlib.sha256(MJCF_PATH.read_bytes()).hexdigest(),
        "official_assets_modified": False,
        "terrain_construction": "runtime-only MjSpec boxes; official top height/direction/2.50 m floor/4.00 m corridor",
        "depths_m": list(depths),
        "cases": list(case_names),
        "wheel_speed_rad_s": args.wheel_speed_rad_s,
        "geometry": geometry_reports,
        "rollouts": summaries,
        "successful_depths_m": success_depths,
        "training_performed": False,
        "authorization": (
            "Only successful depths may proceed to a bounded full-body curriculum pilot; failures reject these primitives only."
            if success_depths
            else "No shallow primitive succeeded; do not start PPO from this evidence."
        ),
        "files": {"summary": str(output_dir / "summary.json")},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if success_depths else 2


if __name__ == "__main__":
    raise SystemExit(main())
