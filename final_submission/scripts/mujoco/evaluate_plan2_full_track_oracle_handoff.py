#!/usr/bin/env python3
"""Run the official track with an oracle handoff to the locked plan2 pit expert.

The official 57-D ONNX policy and a local copy of the waypoint navigator drive
from the official start.  At scoring waypoint 15 the natural MuJoCo state is
saved, a short set of continuous-state handoff timings is audited, and the
locked v4 expert is invoked without reset or teleport.  On expert success the
official policy resumes toward waypoint 16 and the rest of the route.

This remains a simulation/oracle integration audit.  It does not claim that
lidar phase recognition or the competition short-peak rule is complete.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

import search_s10_m20_early_steering_rescue as rescue_runner
from evaluate_plan2_official_walk_expert_handoff import (
    COMMAND,
    CONTROL_STEPS,
    DT,
    HandoffExactPeakEnv,
    DynamicState,
    PhaseRecorder,
    RobotInsetRenderer,
    expert_kwargs,
    restore,
    run_expert,
    save_gif,
    snapshot,
)
from probe_s10_official_policy_shallow import OfficialPolicy
from s10_fronthook_scripted_probe import MJCF_PATH, contact_metrics, diverged, model_ids
from s10_speed_baseline import reset_official_start, run_official_standup
from search_s10_m20_approach_two_stage_rescue import asset_hashes, load_reference, sha256
from search_s10_m20_front_hook_pull import ORIGINAL_ROOT, local_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_YAML = PROJECT_ROOT / "config/path.yaml"
LOCKED_V4 = PROJECT_ROOT / (
    "logs/mujoco/plan2_hooked_tuck_closed_loop_exact_official_track_peak_20260812_v4/summary.json"
)
BASE_BODY_NAME = "base_link"
WHEEL_CONTINUOUS_NM = 14.0
LEG_CONTINUOUS_NM = 50.0
RECORD_CAMERA_DISTANCE = 5.0
RECORD_CAMERA_AZIMUTH = 130.0
RECORD_CAMERA_ELEVATION = -55.0
PIT_SCORE_INDEX = 15
EXIT_SCORE_INDEX = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sim-seconds", type=float, default=420.0)
    parser.add_argument("--max-forward", type=float, default=1.0)
    parser.add_argument("--frame-period", type=float, default=1.5)
    parser.add_argument("--pit-frame-period", type=float, default=0.08)
    parser.add_argument("--post-pit-seconds", type=float, default=30.0)
    parser.add_argument(
        "--handoff-prep-seconds",
        type=float,
        action="append",
        default=None,
        help="official-policy preparation after naturally reaching waypoint 15",
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


@dataclass(frozen=True)
class RoutePoint:
    xyz: np.ndarray
    kind: str
    score_index: int | None


def load_route(path: Path | None = None) -> list[RoutePoint]:
    route_path = local_path(path) if path is not None else local_path(PATH_YAML)
    payload = yaml.safe_load(route_path.read_text(encoding="utf-8"))
    points = []
    for item in payload["path"]:
        points.append(
            RoutePoint(
                xyz=np.asarray(item["pos"], dtype=np.float64),
                kind=str(item.get("kind", "waypoint")),
                score_index=None if item.get("index") is None else int(item["index"]),
            )
        )
    if [point.score_index for point in points if point.score_index is not None] != list(range(33)):
        raise RuntimeError("path.yaml does not contain the expected ordered 33 scoring waypoints")
    return points


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_xmat(xmat: np.ndarray) -> float:
    rotation = np.asarray(xmat, dtype=np.float64).reshape(3, 3)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def configure_continuous_limits(model: mujoco.MjModel) -> None:
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if not name:
            raise RuntimeError(f"actuator {actuator_id} has no name")
        limit = WHEEL_CONTINUOUS_NM if name.endswith("_wheel_joint") else LEG_CONTINUOUS_NM
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        model.actuator_ctrllimited[actuator_id] = True
        model.actuator_ctrlrange[actuator_id] = (-limit, limit)
        model.jnt_actfrclimited[joint_id] = True
        model.jnt_actfrcrange[joint_id] = (-limit, limit)


class TrackRecorder:
    default_frame_sink: Callable[[Image.Image], None] | None = None
    default_retain_frames = True
    default_robot_inset_enabled = False

    def __init__(self, model: mujoco.MjModel, enabled: bool, period_s: float) -> None:
        self.enabled = bool(enabled)
        self.period_s = float(period_s)
        self.next_time = -math.inf
        self.frames: list[Image.Image] = []
        self.frame_sink = type(self).default_frame_sink
        self.retain_frames = bool(type(self).default_retain_frames)
        self.frame_count = 0
        self.robot_inset = (
            RobotInsetRenderer(model)
            if self.enabled and type(self).default_robot_inset_enabled
            else None
        )
        self.renderer: mujoco.Renderer | None = None
        self.camera: mujoco.MjvCamera | None = None
        if self.enabled:
            self.renderer = mujoco.Renderer(model, height=480, width=640)
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.camera)
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.distance = RECORD_CAMERA_DISTANCE
            self.camera.azimuth = RECORD_CAMERA_AZIMUTH
            self.camera.elevation = RECORD_CAMERA_ELEVATION

    def capture(
        self,
        data: mujoco.MjData,
        base_body_id: int,
        *,
        phase: str,
        detail: str,
        force: bool = False,
    ) -> None:
        if not self.enabled or (not force and data.time + 1.0e-12 < self.next_time):
            return
        self.next_time = float(data.time) + self.period_s
        assert self.renderer is not None and self.camera is not None
        self.camera.lookat[:] = data.xpos[base_body_id]
        self.camera.lookat[2] += 0.20
        self.renderer.update_scene(data, camera=self.camera)
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image)
        lines = (phase, detail, f"SIM TIME {data.time:.2f}s")
        widths = [draw.textbbox((0, 0), line)[2] for line in lines]
        draw.rectangle((8, 8, max(widths) + 22, 66), fill=(0, 0, 0))
        draw.text((14, 12), phase, fill=(255, 255, 255))
        draw.text((14, 29), detail, fill=(255, 220, 70))
        draw.text((14, 46), lines[2], fill=(120, 235, 255))
        if self.robot_inset is not None:
            image = self.robot_inset.composite(image, data, base_body_id)
        self.frame_count += 1
        if self.frame_sink is not None:
            self.frame_sink(image)
        if self.retain_frames:
            self.frames.append(image)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
        if self.robot_inset is not None:
            self.robot_inset.close()


class Navigator:
    def __init__(self, route: list[RoutePoint], start_xy: np.ndarray, max_forward: float) -> None:
        self.route = route
        self.index = 0
        self.segment_start = np.asarray(start_xy, dtype=np.float64).copy()
        self.max_forward = float(max_forward)
        self.reached_scores: list[int] = []
        self.best_distance = math.inf
        self.last_improvement_time = 0.0
        self.recovery_attempt = 0
        self.recovery_phase: str | None = None
        self.recovery_started = 0.0
        self.strafe_sign = 1.0
        self.descent_start_z: float | None = None
        self.descent_heading: float | None = None
        self.descent_floor = False

    def current(self) -> RoutePoint | None:
        return None if self.index >= len(self.route) else self.route[self.index]

    def _radius(self, point: RoutePoint) -> float:
        return 0.2 if point.kind == "waypoint" else 0.5

    def update_reached(self, position: np.ndarray, sim_time: float) -> list[int]:
        newly_reached: list[int] = []
        while self.index < len(self.route):
            point = self.route[self.index]
            distance = float(np.linalg.norm(point.xyz[:2] - position[:2]))
            if distance > self._radius(point):
                break
            if point.score_index is not None:
                self.reached_scores.append(point.score_index)
                newly_reached.append(point.score_index)
            self.segment_start = point.xyz[:2].copy()
            self.index += 1
            self.best_distance = math.inf
            self.last_improvement_time = sim_time
            self.recovery_attempt = 0
            self.recovery_phase = None
            self.descent_start_z = None
            self.descent_heading = None
            self.descent_floor = False
        return newly_reached

    def command(self, position: np.ndarray, yaw: float, sim_time: float) -> np.ndarray:
        point = self.current()
        if point is None:
            return np.zeros(3, dtype=np.float32)
        delta = point.xyz[:2] - position[:2]
        distance = float(np.linalg.norm(delta))

        # The only deep downward scoring target is waypoint 15.  Match the
        # production navigator: align to the segment, cross the rim without
        # reverse recovery, then translate toward the bottom marker.
        if point.score_index == PIT_SCORE_INDEX:
            if self.descent_heading is None:
                segment = point.xyz[:2] - self.segment_start
                self.descent_heading = math.atan2(float(segment[1]), float(segment[0]))
                self.descent_start_z = float(position[2])
            heading_error = wrap(self.descent_heading - yaw)
            if abs(heading_error) > 0.18 and not self.descent_floor:
                return np.asarray([0.0, 0.0, np.clip(1.2 * heading_error, -0.30, 0.30)], dtype=np.float32)
            if self.descent_start_z is not None and position[2] <= self.descent_start_z - 0.32:
                self.descent_floor = True
            if not self.descent_floor:
                return np.asarray(
                    [min(self.max_forward, 1.0), 0.0, np.clip(0.8 * heading_error, -0.25, 0.25)],
                    dtype=np.float32,
                )
            if distance <= 1.0e-6:
                return np.zeros(3, dtype=np.float32)
            forward_error = math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1]
            side_error = -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1]
            return np.asarray(
                [
                    np.clip(0.45 * forward_error / distance, -0.45, 0.45),
                    np.clip(0.45 * side_error / distance, -0.30, 0.30),
                    np.clip(0.8 * heading_error, -0.25, 0.25),
                ],
                dtype=np.float32,
            )

        # Bounded recovery mirrors the production waypoint controller.
        if distance < self.best_distance - 0.05:
            self.best_distance = distance
            self.last_improvement_time = sim_time
        if self.recovery_phase is None and sim_time - self.last_improvement_time > 10.0:
            self.recovery_attempt += 1
            self.recovery_phase = "reverse"
            self.recovery_started = sim_time
        if self.recovery_phase == "reverse":
            if sim_time - self.recovery_started < 2.5:
                return np.asarray([-0.45, 0.0, 0.0], dtype=np.float32)
            self.recovery_phase = "strafe" if self.recovery_attempt >= 2 else "realign"
            self.recovery_started = sim_time
        if self.recovery_phase == "strafe":
            if sim_time - self.recovery_started < 2.0:
                return np.asarray([0.0, 0.4 * self.strafe_sign, 0.0], dtype=np.float32)
            self.strafe_sign *= -1.0
            self.recovery_phase = "realign"
            self.recovery_started = sim_time

        segment = point.xyz[:2] - self.segment_start
        length = float(np.linalg.norm(segment))
        if length <= 1.0e-6:
            carrot = point.xyz[:2]
            cross_track = 0.0
        else:
            unit = segment / length
            projection = float(np.clip(np.dot(position[:2] - self.segment_start, unit), 0.0, length))
            cross_track = float(np.cross(np.r_[unit, 0.0], np.r_[position[:2] - self.segment_start, 0.0])[2])
            carrot = self.segment_start + unit * min(length, projection + 1.2)
        heading_error = wrap(math.atan2(float(carrot[1] - position[1]), float(carrot[0] - position[0])) - yaw)
        if self.recovery_phase == "realign":
            if abs(heading_error) > 0.2 and sim_time - self.recovery_started < 6.0:
                return np.asarray(
                    [0.0, 0.0, np.clip(1.2 * heading_error, -0.8, 0.8)],
                    dtype=np.float32,
                )
            self.recovery_phase = None
            self.best_distance = distance
            self.last_improvement_time = sim_time
            self.segment_start = position[:2].copy()

        yaw_command = float(np.clip(1.2 * heading_error, -0.8, 0.8))
        if abs(heading_error) > 0.9:
            forward = 0.0
        else:
            forward = self.max_forward * math.cos(heading_error) * min(1.0, distance / 1.5)
            if abs(cross_track) > 0.4:
                forward *= 0.4 / abs(cross_track)
            forward = float(np.clip(forward, 0.0, self.max_forward))
        return np.asarray([forward, 0.0, yaw_command], dtype=np.float32)


def run_policy_segment(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict[str, Any],
    navigator: Navigator,
    recorder: TrackRecorder,
    *,
    max_sim_seconds: float,
    stop_after_score: int | None,
) -> dict[str, Any]:
    policy = OfficialPolicy()
    start_time = float(data.time)
    next_control_time = float(data.time)
    command = np.zeros(3, dtype=np.float32)
    max_force = 0.0
    max_roll = 0.0
    reason = "time_limit"
    while float(data.time - start_time) < max_sim_seconds:
        base = data.xpos[ids["base_body_id"]].copy()
        yaw = yaw_from_xmat(data.xmat[ids["base_body_id"]])
        reached = navigator.update_reached(base, float(data.time))
        for score in reached:
            point = navigator.current()
            detail = f"reached score {score}; next={None if point is None else point.score_index}"
            recorder.capture(data, ids["base_body_id"], phase="WALK POLICY", detail=detail, force=True)
            print(f"[full-track] reached score={score} t={data.time:.2f}s", flush=True)
        if stop_after_score is not None and stop_after_score in reached:
            reason = f"reached_score_{stop_after_score}"
            break
        if navigator.current() is None:
            reason = "route_complete"
            break
        if data.time + 1.0e-12 >= next_control_time:
            next_control_time = float(data.time) + 0.02
            command = navigator.command(base, yaw, float(data.time))
            policy.update(model, data, ids, command)
        policy.apply(data)
        mujoco.mj_step(model, data)
        contacts = contact_metrics(model, data, ids)
        max_force = max(max_force, float(contacts["max_contact_force"]))
        rotation = np.asarray(data.xmat[ids["base_body_id"]]).reshape(3, 3)
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        max_roll = max(max_roll, abs(roll))
        point = navigator.current()
        detail = (
            f"score={len(navigator.reached_scores)}/33 target="
            f"{None if point is None else point.score_index} t={data.time:.1f}s"
        )
        recorder.capture(data, ids["base_body_id"], phase="WALK POLICY", detail=detail)
        bad = diverged(data)
        if bad is not None:
            reason = bad
            break
    return {
        "reason": reason,
        "elapsed_seconds": float(data.time - start_time),
        "reached_scores": navigator.reached_scores.copy(),
        "route_index": navigator.index,
        "max_contact_force_n": max_force,
        "max_abs_roll_deg": math.degrees(max_roll),
        "command": command.tolist(),
    }


def prepare_state(
    source: DynamicState,
    seconds: float,
    *,
    video: bool,
    period_s: float,
) -> tuple[DynamicState, list[Image.Image]]:
    model = mujoco.MjModel.from_xml_path(str(local_path(MJCF_PATH)))
    model.opt.timestep = DT
    configure_continuous_limits(model)
    data = mujoco.MjData(model)
    ids = model_ids(model)
    restore(model, data, source)
    policy = OfficialPolicy()
    recorder = TrackRecorder(model, video, period_s)
    try:
        for step in range(int(round(seconds / DT))):
            if step % CONTROL_STEPS == 0:
                base = data.xpos[ids["base_body_id"]]
                target_yaw = math.atan2(31.2900 - float(base[1]), 16.2750 - float(base[0]))
                yaw_error = wrap(target_yaw - yaw_from_xmat(data.xmat[ids["base_body_id"]]))
                command = np.asarray([0.65, 0.0, np.clip(1.2 * yaw_error, -0.8, 0.8)], dtype=np.float32)
                policy.update(model, data, ids, command)
            policy.apply(data)
            mujoco.mj_step(model, data)
            recorder.capture(
                data,
                ids["base_body_id"],
                phase="HANDOFF PREP",
                detail=f"official ONNX toward score 16 ({seconds:.2f}s)",
            )
        return snapshot(data), recorder.frames.copy()
    finally:
        recorder.close()


def rank_report(report: dict[str, Any]) -> tuple[int, float]:
    return int(report["milestone_rank"]), float(report.get("tie_break_score", -math.inf))


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def main() -> int:
    args = parse_args()
    output = local_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    prep_seconds = args.handoff_prep_seconds or [0.0, 0.04, 0.10, 0.20, 0.40, 0.60]
    if any(not math.isfinite(value) or value < 0.0 for value in prep_seconds):
        raise ValueError("handoff preparation durations must be finite and non-negative")
    if args.max_sim_seconds <= 0.0 or args.max_forward <= 0.0:
        raise ValueError("simulation duration and maximum forward speed must be positive")

    locked_path = local_path(LOCKED_V4)
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    locked_report = locked["report"]
    reference_path = local_path(Path(locked["inputs"]["reference"]["path"]))
    phase_path = local_path(Path(locked["inputs"]["phase"]["path"]))
    before_assets = asset_hashes()
    locked_hash = sha256(locked_path)

    route = load_route()
    model = mujoco.MjModel.from_xml_path(str(local_path(MJCF_PATH)))
    model.opt.timestep = DT
    configure_continuous_limits(model)
    data = mujoco.MjData(model)
    ids = model_ids(model)
    reset_official_start(model, data)
    diverged_stand, stand_reason = run_official_standup(model, data)
    if diverged_stand:
        raise RuntimeError(f"official stand-up failed: {stand_reason}")

    recorder = TrackRecorder(model, bool(args.video), float(args.frame_period))
    recorder.capture(
        data,
        ids["base_body_id"],
        phase="WALK POLICY",
        detail="official start; 57-D ONNX + waypoint navigation",
        force=True,
    )
    navigator = Navigator(route, data.xpos[ids["base_body_id"]][:2], float(args.max_forward))
    pre_report = run_policy_segment(
        model,
        data,
        ids,
        navigator,
        recorder,
        max_sim_seconds=float(args.max_sim_seconds),
        stop_after_score=PIT_SCORE_INDEX,
    )
    natural_pit_state = snapshot(data)
    recorder.capture(
        data,
        ids["base_body_id"],
        phase="PIT HANDOFF POINT",
        detail=f"natural score 15 state; t={data.time:.2f}s",
        force=True,
    )
    pre_frames = recorder.frames.copy()
    recorder.close()

    if PIT_SCORE_INDEX not in pre_report["reached_scores"]:
        after_assets = asset_hashes()
        summary = {
            "purpose": "full-track oracle pit-expert integration audit",
            "deployable": False,
            "pit_reached": False,
            "pre_track": pre_report,
            "official_assets_unchanged": before_assets == after_assets,
            "locked_v4_unchanged": locked_hash == sha256(locked_path),
        }
        if args.video and pre_frames:
            save_gif(output / "full_track_rollout.gif", pre_frames, float(args.frame_period))
        (output / "summary.json").write_text(
            json.dumps(plain(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        print(f"[full-track] did not reach pit: {pre_report['reason']}", flush=True)
        return 2

    torch.set_num_threads(1)
    agent = load_reference(reference_path)
    original_env = rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv
    original_gif_recorder = rescue_runner.GifRecorder
    rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv = HandoffExactPeakEnv
    cases: list[dict[str, Any]] = []
    try:
        for seconds in prep_seconds:
            prepared, _ = prepare_state(
                natural_pit_state,
                float(seconds),
                video=False,
                period_s=float(args.pit_frame_period),
            )
            report = run_expert(
                agent,
                phase_path,
                expert_kwargs(locked_report),
                prepared,
            )
            cases.append({"prep_seconds": float(seconds), "report": report})
            print(
                f"[full-track] prep={seconds:.2f}s expert_success={report['success']} "
                f"milestone={report['milestone_name']} reason={report['termination_reason']}",
                flush=True,
            )
        successful = [case for case in cases if case["report"]["success"]]
        selected = min(successful, key=lambda case: case["prep_seconds"]) if successful else max(
            cases, key=lambda case: rank_report(case["report"])
        )
        selected_state, prep_frames = prepare_state(
            natural_pit_state,
            float(selected["prep_seconds"]),
            video=bool(args.video),
            period_s=float(args.pit_frame_period),
        )
        PhaseRecorder.last_frames = []
        if args.video:
            rescue_runner.GifRecorder = PhaseRecorder
        replay = run_expert(
            agent,
            phase_path,
            expert_kwargs(locked_report),
            selected_state,
            gif_path=output / "expert.gif" if args.video else None,
            trace_path=output / "expert_trace.npz",
            frame_period=float(args.pit_frame_period),
        )
        expert_instance = HandoffExactPeakEnv.last_instance
        if expert_instance is None:
            raise RuntimeError("expert environment instance was not retained")

        post_report: dict[str, Any] | None = None
        post_frames: list[Image.Image] = []
        if replay["success"]:
            configure_continuous_limits(expert_instance.model)
            post_recorder = TrackRecorder(
                expert_instance.model, bool(args.video), float(args.frame_period)
            )
            post_recorder.capture(
                expert_instance.data,
                expert_instance.ids["base_body_id"],
                phase="WALK POLICY RESUMED",
                detail="expert exit complete; target score 16",
                force=True,
            )
            post_report = run_policy_segment(
                expert_instance.model,
                expert_instance.data,
                expert_instance.ids,
                navigator,
                post_recorder,
                max_sim_seconds=float(args.post_pit_seconds),
                stop_after_score=None,
            )
            post_frames = post_recorder.frames.copy()
            post_recorder.close()

        combined_frames = pre_frames + prep_frames + PhaseRecorder.last_frames + post_frames
        if args.video and combined_frames:
            # GIF uses the pit period so the expert frames play at their true
            # rate; sparse route frames intentionally become a quick overview.
            save_gif(output / "full_track_rollout.gif", combined_frames, float(args.pit_frame_period))

        after_assets = asset_hashes()
        if before_assets != after_assets or locked_hash != sha256(locked_path):
            raise RuntimeError("official assets or locked v4 changed during full-track audit")
        summary = {
            "purpose": "full official track with oracle handoff to locked plan2 pit expert",
            "mechanical_integration_audit_only": True,
            "deployable": False,
            "oracle_phase_events_used": True,
            "lidar_takeover_used": False,
            "state_reset_or_teleport_at_handoff": False,
            "pre_track": pre_report,
            "natural_pit_state_reached": True,
            "handoff_cases": cases,
            "selected_prep_seconds": float(selected["prep_seconds"]),
            "selected_expert_replay": replay,
            "post_track": post_report,
            "official_assets_modified": False,
            "locked_v4_modified": False,
            "visualization": str(output / "full_track_rollout.gif") if args.video else None,
            "inputs": {
                "locked_v4": {"path": str(locked_path), "sha256": locked_hash},
                "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
                "phase": {"path": str(phase_path), "sha256": sha256(phase_path)},
                "official_track": {"path": str(local_path(MJCF_PATH)), "sha256": sha256(local_path(MJCF_PATH))},
                "path_yaml": {"path": str(local_path(PATH_YAML)), "sha256": sha256(local_path(PATH_YAML))},
            },
            "official_assets_before": before_assets,
            "official_assets_after": after_assets,
            "protected_original_root": str(ORIGINAL_ROOT),
        }
        (output / "summary.json").write_text(
            json.dumps(plain(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        print(
            f"[full-track] expert_success={replay['success']} "
            f"scores_before={len(pre_report['reached_scores'])} "
            f"scores_after={0 if post_report is None else len(post_report['reached_scores'])} "
            f"output={output}",
            flush=True,
        )
    finally:
        rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv = original_env
        rescue_runner.GifRecorder = original_gif_recorder
        HandoffExactPeakEnv.handoff_state = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
