#!/usr/bin/env python3
"""Evaluate a continuous official-ONNX -> pit expert -> official-ONNX handoff.

The locked v4 controller and official assets are read-only inputs.  A short
official-policy prefix is run in the exact official-track model, its complete
dynamic state is handed to the v4 controller without a reset or teleport, and
the official policy resumes from the expert's final state.  This is an oracle
integration audit, not a deployable perception result.
"""

from __future__ import annotations

import argparse
import hashlib
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
from PIL import Image, ImageDraw

import search_s10_m20_early_steering_rescue as rescue_runner
from evaluate_plan2_hooked_tuck_closed_loop_official_track import (
    ExactOfficialTrackPeakEnv,
)
from probe_s10_official_policy_shallow import OfficialPolicy
from s10_fronthook_scripted_probe import (
    DT,
    MJCF_PATH,
    model_ids,
    reset_in_pit,
    settle,
)
from search_s10_m20_approach_two_stage_rescue import (
    asset_hashes,
    load_reference,
    sha256,
)
from search_s10_m20_front_hook_pull import ORIGINAL_ROOT, local_path
from s10_m20_curriculum_env import exit_progress


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCKED_V4 = PROJECT_ROOT / (
    "logs/mujoco/plan2_hooked_tuck_closed_loop_exact_official_track_peak_20260812_v4/summary.json"
)
LEG_PEAK_NM = 76.4
WHEEL_PEAK_NM = 21.6
CONTROL_STEPS = 20
COMMAND = np.asarray([0.65, 0.0, 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prefix-seconds",
        type=float,
        action="append",
        default=None,
        help="official-ONNX duration before expert takeover; may be repeated",
    )
    parser.add_argument("--post-seconds", type=float, default=2.0)
    parser.add_argument("--frame-period", type=float, default=0.08)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class DynamicState:
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    qacc_warmstart: np.ndarray
    time: float


def snapshot(data: mujoco.MjData) -> DynamicState:
    return DynamicState(
        qpos=data.qpos.copy(),
        qvel=data.qvel.copy(),
        act=data.act.copy(),
        ctrl=data.ctrl.copy(),
        qacc_warmstart=data.qacc_warmstart.copy(),
        time=float(data.time),
    )


def restore(model: mujoco.MjModel, data: mujoco.MjData, state: DynamicState) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = state.qpos
    data.qvel[:] = state.qvel
    if data.act.size:
        data.act[:] = state.act
    data.ctrl[:] = state.ctrl
    data.qacc_warmstart[:] = state.qacc_warmstart
    data.time = state.time
    mujoco.mj_forward(model, data)


def configure_peak_limits(model: mujoco.MjModel) -> None:
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if not name:
            raise RuntimeError(f"actuator {actuator_id} has no name")
        peak = WHEEL_PEAK_NM if name.endswith("_wheel_joint") else LEG_PEAK_NM
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        model.actuator_ctrllimited[actuator_id] = True
        model.actuator_ctrlrange[actuator_id] = (-peak, peak)
        model.jnt_actfrclimited[joint_id] = True
        model.jnt_actfrcrange[joint_id] = (-peak, peak)


RECORD_CAMERA_DISTANCE = 5.0
RECORD_CAMERA_AZIMUTH = 130.0
RECORD_CAMERA_ELEVATION = -55.0
ROBOT_INSET_WIDTH = 240
ROBOT_INSET_HEIGHT = 180
ROBOT_INSET_MARGIN = 10


class RobotInsetRenderer:
    """Render an unobstructed, display-only top view of the simulated robot."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.renderer = mujoco.Renderer(
            model,
            height=ROBOT_INSET_HEIGHT,
            width=ROBOT_INSET_WIDTH,
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 1.8
        self.camera.azimuth = 0.0
        self.camera.elevation = -89.0
        self.scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.scene_option)
        # Group 0 contains the static course. Groups 1 and 2 retain the S10
        # collision/visual bodies and route overlay in this display-only inset.
        self.scene_option.geomgroup[:] = 0
        self.scene_option.geomgroup[1] = 1
        self.scene_option.geomgroup[2] = 1

    def composite(
        self,
        image: Image.Image,
        data: mujoco.MjData,
        base_body_id: int,
    ) -> Image.Image:
        self.camera.lookat[:] = data.xpos[base_body_id]
        self.camera.lookat[2] += 0.12
        self.renderer.update_scene(
            data,
            camera=self.camera,
            scene_option=self.scene_option,
        )
        inset = Image.fromarray(self.renderer.render()).convert("RGB")
        inset_draw = ImageDraw.Draw(inset)
        inset_draw.rectangle((0, 0, ROBOT_INSET_WIDTH - 1, 20), fill=(0, 0, 0))
        inset_draw.text((7, 5), "ROBOT VIEW | DISPLAY ONLY", fill=(255, 255, 255))
        result = image.copy()
        left = result.width - ROBOT_INSET_WIDTH - ROBOT_INSET_MARGIN
        top = result.height - ROBOT_INSET_HEIGHT - ROBOT_INSET_MARGIN
        result.paste(inset, (left, top))
        ImageDraw.Draw(result).rectangle(
            (
                left - 1,
                top - 1,
                left + ROBOT_INSET_WIDTH,
                top + ROBOT_INSET_HEIGHT,
            ),
            outline=(255, 255, 255),
            width=2,
        )
        return result

    def close(self) -> None:
        self.renderer.close()


def annotate(
    frame: Image.Image,
    phase: str,
    detail: str,
    sim_time_s: float | None = None,
) -> Image.Image:
    image = frame.copy()
    draw = ImageDraw.Draw(image)
    time_line = "SIM TIME --" if sim_time_s is None else f"SIM TIME {sim_time_s:.2f}s"
    lines = (phase, detail, time_line)
    widths = [draw.textbbox((0, 0), line)[2] for line in lines]
    draw.rectangle((8, 8, max(widths) + 22, 66), fill=(0, 0, 0))
    draw.text((14, 12), phase, fill=(255, 255, 255))
    draw.text((14, 29), detail, fill=(255, 220, 70))
    draw.text((14, 46), time_line, fill=(120, 235, 255))
    return image


class PhaseRecorder:
    """Drop-in recorder used by the existing locked rollout function."""

    last_frames: list[Image.Image] = []
    default_frame_sink: Callable[[Image.Image], None] | None = None
    default_retain_frames = True
    default_robot_inset_enabled = False

    def __init__(self, model: mujoco.MjModel, period_s: float) -> None:
        self.period_s = float(period_s)
        self.next_time = 0.0
        self.frames: list[Image.Image] = []
        self.frame_sink = type(self).default_frame_sink
        self.retain_frames = bool(type(self).default_retain_frames)
        self.frame_count = 0
        self.robot_inset = (
            RobotInsetRenderer(model)
            if type(self).default_robot_inset_enabled
            else None
        )
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = RECORD_CAMERA_DISTANCE
        self.camera.azimuth = RECORD_CAMERA_AZIMUTH
        self.camera.elevation = RECORD_CAMERA_ELEVATION

    def capture(self, env: Any) -> None:
        if env.data.time + 1.0e-12 < self.next_time:
            return
        self.next_time = float(env.data.time) + self.period_s
        self.camera.lookat[:] = env.data.xpos[env.ids["base_body_id"]]
        self.camera.lookat[2] += 0.12
        self.renderer.update_scene(env.data, camera=self.camera)
        frame = Image.fromarray(self.renderer.render())
        annotated = annotate(
            frame,
            "PIT EXPERT",
            "oracle phase events; no lidar takeover yet",
            float(env.data.time),
        )
        if self.robot_inset is not None:
            annotated = self.robot_inset.composite(
                annotated,
                env.data,
                env.ids["base_body_id"],
            )
        self.frame_count += 1
        if self.frame_sink is not None:
            self.frame_sink(annotated)
        if self.retain_frames:
            self.frames.append(annotated)

    def close(self, path: Path) -> str:
        self.renderer.close()
        if self.robot_inset is not None:
            self.robot_inset.close()
        if self.frame_count == 0:
            raise RuntimeError("expert recorder captured no frames")
        type(self).last_frames = (
            [frame.copy() for frame in self.frames] if self.retain_frames else []
        )
        if self.retain_frames:
            save_gif(path, self.frames, self.period_s)
        return str(path)


class HandoffExactPeakEnv(ExactOfficialTrackPeakEnv):
    handoff_state: DynamicState | None = None
    last_instance: "HandoffExactPeakEnv | None" = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        HandoffExactPeakEnv.last_instance = self

    def _reset_physics(self) -> tuple[bool, str | None]:
        if HandoffExactPeakEnv.handoff_state is None:
            return super()._reset_physics()
        restore(self.model, self.data, HandoffExactPeakEnv.handoff_state)
        return True, None


def render_motion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict[str, Any],
    *,
    seconds: float,
    period_s: float,
    phase: str,
    detail: str,
) -> tuple[list[Image.Image], OfficialPolicy]:
    policy = OfficialPolicy()
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = RECORD_CAMERA_DISTANCE
    camera.azimuth = RECORD_CAMERA_AZIMUTH
    camera.elevation = RECORD_CAMERA_ELEVATION
    frames: list[Image.Image] = []
    next_frame = float(data.time)
    total_steps = int(round(seconds / DT))
    for step in range(total_steps):
        if step % CONTROL_STEPS == 0:
            policy.update(model, data, ids, COMMAND)
        policy.apply(data)
        mujoco.mj_step(model, data)
        if data.time + 1.0e-12 >= next_frame:
            next_frame = float(data.time) + period_s
            camera.lookat[:] = data.xpos[ids["base_body_id"]]
            camera.lookat[2] += 0.12
            renderer.update_scene(data, camera=camera)
            frames.append(
                annotate(
                    Image.fromarray(renderer.render()),
                    phase,
                    detail,
                    float(data.time),
                )
            )
    renderer.close()
    return frames, policy


def build_prefix(seconds: float, period_s: float, video: bool) -> tuple[DynamicState, list[Image.Image]]:
    model = mujoco.MjModel.from_xml_path(str(local_path(MJCF_PATH)))
    model.opt.timestep = DT
    configure_peak_limits(model)
    data = mujoco.MjData(model)
    ids = model_ids(model)
    reset_in_pit(model, data)
    stable, reason = settle(model, data, ids)
    if not stable:
        raise RuntimeError(f"official prefix reset failed: {reason}")
    if seconds <= 0.0:
        return snapshot(data), []
    if video:
        frames, _ = render_motion(
            model,
            data,
            ids,
            seconds=seconds,
            period_s=period_s,
            phase="WALK POLICY",
            detail=f"official 57-D ONNX prefix ({seconds:.2f}s)",
        )
    else:
        policy = OfficialPolicy()
        for step in range(int(round(seconds / DT))):
            if step % CONTROL_STEPS == 0:
                policy.update(model, data, ids, COMMAND)
            policy.apply(data)
            mujoco.mj_step(model, data)
        frames = []
    return snapshot(data), frames


def expert_kwargs(report: dict[str, Any]) -> dict[str, Any]:
    rescue_candidate = np.asarray(
        report["candidate_pre_vector"] + report["candidate_post_vector"], dtype=np.float64
    )
    return {
        "rescue_candidate": rescue_candidate,
        "steering": np.asarray(report["steering_left_right"], dtype=np.float64),
        "label": "plan2_official_onnx_to_hooked_tuck_handoff",
        "depth_m": float(report["depth_m"]),
        "command_mps": float(report["command_mps"]),
        "initial_state": (0.0, 0.0),
        "rollout_seed": int(report["seed"]),
        "early_threshold": 0.10,
        "approach_threshold": 0.03,
        "prelift_from_early": True,
        "episode_seconds": 10.0,
        "leg_rate_per_second": 2.5,
        "post_from_approach": True,
        "post_delay_seconds": 0.15,
        "late_post_boost_delay_seconds": float(report["late_post_boost_delay_seconds"]),
        "late_post_boost_hold_seconds": float(report["late_post_boost_hold_seconds"]),
        "late_post_boost_vector": np.asarray(report["late_post_boost_vector"], dtype=np.float64),
        "late_post_target_limit": float(report["late_post_target_limit"]),
        "late_post_second_vector": np.asarray(report["late_post_second_vector"], dtype=np.float64),
        "late_post_second_delay_seconds": float(report["late_post_second_delay_seconds"]),
        "late_post_second_hold_seconds": float(report["late_post_second_hold_seconds"]),
        "late_post_height_vector": np.asarray(report["late_post_height_vector"], dtype=np.float64),
        "late_post_height_trigger_m": float(report["late_post_height_trigger_m"]),
        "late_post_height_hold_seconds": float(report["late_post_height_hold_seconds"]),
        "late_post_top_vector": np.asarray(report["late_post_top_vector"], dtype=np.float64),
        "late_post_top_front_leg_vector": np.asarray(
            report["late_post_top_front_leg_vector"], dtype=np.float64
        ),
        "late_post_top_trigger_m": float(report["late_post_top_trigger_m"]),
        "late_post_top_hold_seconds": float(report["late_post_top_hold_seconds"]),
        "late_post_top_score_delay_seconds": float(report["late_post_top_score_delay_seconds"]),
        "late_post_top_leg_rate_per_second": float(report["late_post_top_leg_rate_per_second"]),
        "precontact_leg_vector": np.asarray(report["precontact_leg_vector"], dtype=np.float64),
        "precontact_leg_delay_after_approach_seconds": float(
            report["precontact_leg_delay_after_approach_seconds"]
        ),
        "precontact_leg_hold_seconds": float(report["precontact_leg_hold_seconds"]),
        "precontact_leg_rate_per_second": float(report["precontact_leg_rate_per_second"]),
        "threepoint_monitor_enabled": True,
        "threepoint_min_leading_rear_contact_force_n": float(
            report["threepoint_min_leading_rear_contact_force_n"]
        ),
        "threepoint_max_abs_roll_deg": float(report["threepoint_max_abs_roll_deg"]),
        "pullover_launch_leg_vector": np.asarray(report["pullover_launch_leg_vector"], dtype=np.float64),
        "pullover_launch_delay_after_front_seconds": float(
            report["pullover_launch_delay_after_front_seconds"]
        ),
        "pullover_launch_hold_seconds": float(report["pullover_launch_hold_seconds"]),
        "pullover_tuck_leg_vector": np.asarray(report["pullover_tuck_leg_vector"], dtype=np.float64),
        "pullover_tuck_delay_after_launch_seconds": float(
            report["pullover_tuck_delay_after_launch_seconds"]
        ),
        "pullover_tuck_hold_seconds": float(report["pullover_tuck_hold_seconds"]),
        "pullover_leg_rate_per_second": float(report["pullover_leg_rate_per_second"]),
        "allow_body_wall_contact": False,
        "late_momentum_leg_vector": np.asarray(report["late_momentum_leg_vector"], dtype=np.float64),
        "late_momentum_leg_delay_seconds": float(report["late_momentum_leg_delay_seconds"]),
        "late_momentum_leg_hold_seconds": float(report["late_momentum_leg_hold_seconds"]),
        "late_momentum_leg_rate_per_second": float(report["late_momentum_leg_rate_per_second"]),
        "late_prejump_leg_vector": np.asarray(report["late_prejump_leg_vector"], dtype=np.float64),
        "late_prejump_leg_delay_seconds": float(report["late_prejump_leg_delay_seconds"]),
        "late_prejump_leg_hold_seconds": float(report["late_prejump_leg_hold_seconds"]),
        "late_prejump_leg_rate_per_second": float(report["late_prejump_leg_rate_per_second"]),
        "oracle_front_wheel_action": float(report["oracle_front_wheel_action"]),
        "oracle_freeze_front_legs": bool(report["oracle_freeze_front_legs"]),
        "oracle_front_anchor_delay_seconds": float(report["oracle_front_anchor_delay_seconds"]),
        "leg_peak_nm": LEG_PEAK_NM,
        "wheel_peak_nm": WHEEL_PEAK_NM,
        "max_continuous_exceedance_s": 0.5,
    }


def run_expert(
    agent: Any,
    phase_path: Path,
    kwargs: dict[str, Any],
    state: DynamicState | None,
    *,
    gif_path: Path | None = None,
    trace_path: Path | None = None,
    frame_period: float = 0.08,
    prebuilt_env: Any | None = None,
) -> dict[str, Any]:
    HandoffExactPeakEnv.handoff_state = state
    HandoffExactPeakEnv.last_instance = (
        prebuilt_env if isinstance(prebuilt_env, HandoffExactPeakEnv) else None
    )
    return rescue_runner.rollout(
        agent,
        phase_path,
        kwargs.pop("rescue_candidate"),
        kwargs.pop("steering"),
        **kwargs,
        gif_path=gif_path,
        gif_period=frame_period,
        trace_path=trace_path,
        prebuilt_env=prebuilt_env,
    )


def save_gif(path: Path, frames: list[Image.Image], period_s: float) -> None:
    if not frames:
        raise RuntimeError("cannot save an empty GIF")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=max(1, int(round(period_s * 1000.0))),
        loop=0,
    )


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
    prefix_seconds = args.prefix_seconds or [0.10, 0.20, 0.30, 0.40, 0.60]
    if not prefix_seconds or any(not math.isfinite(value) or value <= 0.0 for value in prefix_seconds):
        raise ValueError("prefix durations must be finite and positive")
    if args.post_seconds < 0.0 or args.frame_period <= 0.0:
        raise ValueError("post seconds must be non-negative and frame period positive")

    locked_path = local_path(LOCKED_V4)
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    locked_report = locked["report"]
    reference_path = local_path(Path(locked["inputs"]["reference"]["path"]))
    phase_path = local_path(Path(locked["inputs"]["phase"]["path"]))
    before_assets = asset_hashes()
    locked_hash_before = file_sha256(locked_path)

    torch.set_num_threads(1)
    agent = load_reference(reference_path)
    original_env = rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv
    original_recorder = rescue_runner.GifRecorder
    rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv = HandoffExactPeakEnv

    cases: list[dict[str, Any]] = []
    try:
        baseline = run_expert(
            agent,
            phase_path,
            expert_kwargs(locked_report),
            None,
        )
        baseline_matches = all(
            baseline.get(key) == locked_report.get(key)
            for key in (
                "success",
                "termination_reason",
                "episode_steps",
                "front_phase_trigger_step",
                "true_front_latch_step",
                "height_stage_trigger_step",
                "top_stage_trigger_step",
                "threepoint_trigger_step",
            )
        )
        if not baseline_matches:
            raise RuntimeError("isolated handoff entry did not reproduce locked v4 milestones")

        for seconds in prefix_seconds:
            state, _ = build_prefix(float(seconds), args.frame_period, False)
            report = run_expert(
                agent,
                phase_path,
                expert_kwargs(locked_report),
                state,
            )
            cases.append({"prefix_seconds": float(seconds), "report": report})
            print(
                f"[handoff] prefix={seconds:.2f}s success={report['success']} "
                f"milestone={report['milestone_name']} reason={report['termination_reason']}",
                flush=True,
            )

        successful = [case for case in cases if case["report"]["success"]]
        if successful:
            selected = max(successful, key=lambda case: case["prefix_seconds"])
        else:
            selected = max(
                cases,
                key=lambda case: (
                    int(case["report"]["milestone_rank"]),
                    float(case["report"].get("peak_front_top_margin_m", -1.0)),
                    -float(case["prefix_seconds"]),
                ),
            )

        selected_seconds = float(selected["prefix_seconds"])
        selected_state, prefix_frames = build_prefix(
            selected_seconds, args.frame_period, bool(args.video)
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
            frame_period=args.frame_period,
        )
        instance = HandoffExactPeakEnv.last_instance
        if instance is None:
            raise RuntimeError("handoff environment instance was not retained")

        post_frames: list[Image.Image] = []
        post_progress_m = 0.0
        if replay["success"] and args.post_seconds > 0.0:
            progress_before = exit_progress(instance.data.xpos[instance.ids["base_body_id"]])
            if args.video:
                post_frames, _ = render_motion(
                    instance.model,
                    instance.data,
                    instance.ids,
                    seconds=float(args.post_seconds),
                    period_s=args.frame_period,
                    phase="WALK POLICY RESUMED",
                    detail="official 57-D ONNX after expert exit",
                )
            else:
                policy = OfficialPolicy()
                for step in range(int(round(args.post_seconds / DT))):
                    if step % CONTROL_STEPS == 0:
                        policy.update(instance.model, instance.data, instance.ids, COMMAND)
                    policy.apply(instance.data)
                    mujoco.mj_step(instance.model, instance.data)
            progress_after = exit_progress(instance.data.xpos[instance.ids["base_body_id"]])
            post_progress_m = float(progress_after - progress_before)

        combined_frames = prefix_frames + PhaseRecorder.last_frames + post_frames
        if args.video:
            save_gif(output / "handoff_rollout.gif", combined_frames, args.frame_period)

        after_assets = asset_hashes()
        locked_hash_after = file_sha256(locked_path)
        if before_assets != after_assets or locked_hash_before != locked_hash_after:
            raise RuntimeError("protected assets or locked v4 summary changed during handoff audit")
        summary = {
            "purpose": "official ONNX to oracle pit expert to official ONNX continuous-state handoff",
            "mechanical_integration_audit_only": True,
            "deployable": False,
            "oracle_phase_events_used": True,
            "lidar_takeover_used": False,
            "state_reset_or_teleport_at_handoff": False,
            "official_track_modified": False,
            "locked_v4_modified": False,
            "baseline_matches_locked_v4_milestones": baseline_matches,
            "prefix_cases": cases,
            "selected_prefix_seconds": selected_seconds,
            "selected_replay": replay,
            "selected_replay_matches_sweep": all(
                replay.get(key) == selected["report"].get(key)
                for key in ("success", "termination_reason", "episode_steps", "threepoint_trigger_step")
            ),
            "walk_policy_resumed": bool(replay["success"] and args.post_seconds > 0.0),
            "post_walk_seconds": float(args.post_seconds),
            "post_walk_exit_progress_gain_m": post_progress_m,
            "visualization": str(output / "handoff_rollout.gif") if args.video else None,
            "inputs": {
                "locked_v4": {"path": str(locked_path), "sha256": locked_hash_before},
                "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
                "phase": {"path": str(phase_path), "sha256": sha256(phase_path)},
                "official_track": {"path": str(local_path(MJCF_PATH)), "sha256": sha256(local_path(MJCF_PATH))},
            },
            "official_assets_before": before_assets,
            "official_assets_after": after_assets,
            "protected_original_root": str(ORIGINAL_ROOT),
        }
        (output / "summary.json").write_text(
            json.dumps(plain(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
        print(
            f"[handoff] selected={selected_seconds:.2f}s success={replay['success']} "
            f"post_progress={post_progress_m:.3f}m output={output}",
            flush=True,
        )
    finally:
        rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv = original_env
        rescue_runner.GifRecorder = original_recorder
        HandoffExactPeakEnv.handoff_state = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
