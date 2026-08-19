#!/usr/bin/env python3
"""Condition the natural pit-bottom state before handing off to plan2 v4.

The official blind-walk policy drives continuously from the official start to
score 15.  The resulting full MuJoCo state is saved once, then a small set of
bounded, physical braking/retreat controllers is evaluated before invoking the
locked v4 pit expert.  No state component is teleported or overwritten during
the handoff.  The best case is replayed with a GIF.

This remains an oracle integration diagnostic: the expert still uses the
locked simulation-truth phase events and the competition peak-torque rule has
not been confirmed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from PIL import Image

import search_s10_m20_early_steering_rescue as rescue_runner
from evaluate_plan2_full_track_oracle_handoff import (
    PIT_SCORE_INDEX,
    Navigator,
    TrackRecorder,
    configure_continuous_limits,
    load_route,
    plain,
    rank_report,
    run_policy_segment,
    wrap,
    yaw_from_xmat,
)
from evaluate_plan2_official_walk_expert_handoff import (
    CONTROL_STEPS,
    DT,
    DynamicState,
    HandoffExactPeakEnv,
    PhaseRecorder,
    expert_kwargs,
    restore,
    run_expert,
    save_gif,
    snapshot,
)
from s10_fronthook_scripted_probe import (
    MJCF_PATH,
    PIT_EXIT_WP,
    STANDING_POSE,
    WHEEL_ADDR,
    along,
    apply_direct_control,
    contact_metrics,
    diverged,
    model_ids,
    wheel_centers,
)
from probe_s10_official_policy_shallow import OfficialPolicy
from s10_speed_baseline import reset_official_start, run_official_standup
from search_s10_m20_approach_two_stage_rescue import asset_hashes, load_reference, sha256
from search_s10_m20_front_hook_pull import ORIGINAL_ROOT, local_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCKED_V4 = PROJECT_ROOT / (
    "logs/mujoco/plan2_hooked_tuck_closed_loop_exact_official_track_peak_20260812_v4/summary.json"
)
LOCKED_TRACE = PROJECT_ROOT / (
    "logs/mujoco/plan2_hooked_tuck_closed_loop_exact_official_track_peak_20260812_v4/trace.npz"
)
LEG_INDICES = np.asarray([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14], dtype=np.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sim-seconds", type=float, default=420.0)
    parser.add_argument("--max-forward", type=float, default=1.0)
    parser.add_argument("--frame-period", type=float, default=1.5)
    parser.add_argument("--pit-frame-period", type=float, default=0.08)
    parser.add_argument("--post-pit-seconds", type=float, default=20.0)
    parser.add_argument(
        "--natural-state",
        type=Path,
        default=None,
        help="reuse a previously saved natural_pit_state.npz instead of rerunning the route",
    )
    parser.add_argument(
        "--tuned-natural-handoff",
        action="store_true",
        help="use the isolated natural-state launch/tuck parameters found by the grid audit",
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


@dataclass(frozen=True)
class ConditionerCase:
    label: str
    policy_command_mps: float
    policy_seconds: float
    leg_target: str
    wheel_speed_rad_s: float
    drive_seconds: float
    brake_seconds: float


def hide_waypoint_overlay_geoms(
    model: mujoco.MjModel,
    scores: Iterable[int],
) -> None:
    """Apply the same runtime-only marker visibility to an isolated renderer."""

    requested = {int(score) for score in scores}
    if not requested:
        return
    matched: set[int] = set()
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None:
            continue
        for prefix in ("track_waypoint_", "track_height_post_"):
            if not name.startswith(prefix):
                continue
            text = name[len(prefix) : len(prefix) + 3]
            if len(text) == 3 and text.isdigit() and int(text) in requested:
                model.geom_rgba[geom_id, 3] = 0.0
                matched.add(int(text))
            break
    missing = requested - matched
    if missing:
        raise RuntimeError(f"no visual overlay geoms found for scores {sorted(missing)}")


def build_conditioner_model() -> mujoco.MjModel:
    """Build the isolated conditioner model before the visible route starts."""

    model = mujoco.MjModel.from_xml_path(str(local_path(MJCF_PATH)))
    model.opt.timestep = DT
    configure_continuous_limits(model)
    return model


def conditioner_cases(tuned_natural_handoff: bool = False) -> list[ConditionerCase]:
    if tuned_natural_handoff:
        return [
            ConditionerCase(
                "policy_0.55mps_0.14s",
                0.55,
                0.14,
                "current",
                0.0,
                0.0,
                0.0,
            )
        ]
    cases = [ConditionerCase("none", 0.0, 0.0, "current", 0.0, 0.0, 0.0)]
    for command in (0.45, 0.55, 0.65, 0.75):
        for duration in (0.06, 0.08, 0.10, 0.12, 0.14):
            cases.append(
                ConditionerCase(
                    f"policy_{command:.2f}mps_{duration:.2f}s",
                    command,
                    duration,
                    "current",
                    0.0,
                    0.0,
                    0.0,
                )
            )
    for policy_seconds in (0.08, 0.10, 0.12):
        for brake_seconds in (0.02, 0.04, 0.06):
            cases.append(
                ConditionerCase(
                    f"policy_0.65mps_{policy_seconds:.2f}s_brake_current_{brake_seconds:.2f}s",
                    0.65,
                    policy_seconds,
                    "current",
                    0.0,
                    0.0,
                    brake_seconds,
                )
            )
    for policy_seconds in (0.08, 0.10, 0.12):
        for brake_seconds in (0.02, 0.04):
            cases.append(
                ConditionerCase(
                    f"policy_0.65mps_{policy_seconds:.2f}s_blend_locked_{brake_seconds:.2f}s",
                    0.65,
                    policy_seconds,
                    "locked",
                    0.0,
                    0.0,
                    brake_seconds,
                )
            )
    return cases


def configured_expert_kwargs(
    report: dict[str, Any], tuned_natural_handoff: bool
) -> dict[str, Any]:
    kwargs = expert_kwargs(report)
    if tuned_natural_handoff:
        kwargs["pullover_launch_delay_after_front_seconds"] = 0.0
        kwargs["pullover_launch_hold_seconds"] = 0.24
        kwargs["pullover_tuck_delay_after_launch_seconds"] = 0.14
        kwargs["pullover_tuck_hold_seconds"] = 0.16
    return kwargs


def save_state(path: Path, state: DynamicState) -> None:
    np.savez_compressed(
        path,
        qpos=state.qpos,
        qvel=state.qvel,
        act=state.act,
        ctrl=state.ctrl,
        qacc_warmstart=state.qacc_warmstart,
        time=np.asarray(state.time, dtype=np.float64),
    )


def load_state(path: Path) -> DynamicState:
    payload = np.load(path)
    return DynamicState(
        qpos=np.asarray(payload["qpos"], dtype=np.float64),
        qvel=np.asarray(payload["qvel"], dtype=np.float64),
        act=np.asarray(payload["act"], dtype=np.float64),
        ctrl=np.asarray(payload["ctrl"], dtype=np.float64),
        qacc_warmstart=np.asarray(payload["qacc_warmstart"], dtype=np.float64),
        time=float(payload["time"]),
    )


def roll_pitch(data: mujoco.MjData, base_body_id: int) -> tuple[float, float]:
    rotation = np.asarray(data.xmat[base_body_id], dtype=np.float64).reshape(3, 3)
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    pitch = math.atan2(
        -float(rotation[2, 0]),
        math.hypot(float(rotation[2, 1]), float(rotation[2, 2])),
    )
    return roll, pitch


def state_metrics(
    data: mujoco.MjData,
    ids: dict[str, Any],
    locked_reference: dict[str, np.ndarray],
) -> dict[str, Any]:
    base_id = ids["base_body_id"]
    roll, pitch = roll_pitch(data, base_id)
    joint_position = np.asarray(data.qpos[7:23], dtype=np.float64)
    joint_velocity = np.asarray(data.qvel[6:22], dtype=np.float64)
    wheels = wheel_centers(data, ids)
    wheel_progress = np.asarray([along(wheel[:2]) for wheel in wheels], dtype=np.float64)
    return {
        "base_progress_m": along(data.xpos[base_id][:2]),
        "base_z_m": float(data.xpos[base_id][2]),
        "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch),
        "wheel_progress_m": wheel_progress.tolist(),
        "leg_position_l2_to_locked": float(
            np.linalg.norm(
                joint_position[LEG_INDICES]
                - locked_reference["joint_position_rad"][LEG_INDICES]
            )
        ),
        "leg_velocity_l2_to_locked": float(
            np.linalg.norm(
                joint_velocity[LEG_INDICES]
                - locked_reference["joint_velocity_rad_s"][LEG_INDICES]
            )
        ),
        "wheel_velocity_l2_to_locked": float(
            np.linalg.norm(
                joint_velocity[WHEEL_ADDR]
                - locked_reference["joint_velocity_rad_s"][WHEEL_ADDR]
            )
        ),
        "leg_speed_l2_rad_s": float(np.linalg.norm(joint_velocity[LEG_INDICES])),
        "wheel_speed_l2_rad_s": float(np.linalg.norm(joint_velocity[WHEEL_ADDR])),
        "base_progress_error_to_locked_m": float(
            along(data.xpos[base_id][:2]) - float(locked_reference["base_progress_m"])
        ),
        "wheel_progress_l2_to_locked_m": float(
            np.linalg.norm(wheel_progress - locked_reference["wheel_progress_m"])
        ),
    }


def condition_state(
    source: DynamicState,
    case: ConditionerCase,
    locked_reference: dict[str, np.ndarray],
    *,
    video: bool,
    period_s: float,
    state_observer: Callable[[mujoco.MjData], None] | None = None,
    observer_period_s: float = 0.02,
    hidden_scores: Iterable[int] = (),
    prebuilt_model: mujoco.MjModel | None = None,
) -> tuple[DynamicState, dict[str, Any], list[Image.Image]]:
    if state_observer is not None and observer_period_s <= 0.0:
        raise ValueError("observer period must be positive")
    model = (
        prebuilt_model
        if prebuilt_model is not None
        else build_conditioner_model()
    )
    model.opt.timestep = DT
    configure_continuous_limits(model)
    data = mujoco.MjData(model)
    ids = model_ids(model)
    hide_waypoint_overlay_geoms(model, hidden_scores)
    restore(model, data, source)
    recorder = TrackRecorder(model, video, period_s)
    start_metrics = state_metrics(data, ids, locked_reference)
    max_force = 0.0
    max_roll = 0.0
    max_pitch = 0.0
    reason = "complete"
    observer_samples = 0
    last_observed_time = -math.inf
    next_observer_time = float(source.time) + observer_period_s

    def observe_state(*, force: bool = False) -> None:
        nonlocal last_observed_time, next_observer_time, observer_samples
        if state_observer is None:
            return
        current_time = float(data.time)
        if not force and current_time + 1.0e-12 < next_observer_time:
            return
        if current_time <= last_observed_time + 1.0e-12:
            return
        state_observer(data)
        observer_samples += 1
        last_observed_time = current_time
        next_observer_time = current_time + observer_period_s

    recorder.capture(
        data,
        ids["base_body_id"],
        phase="HANDOFF CONDITIONER",
        detail=f"{case.label}; begin",
        force=True,
    )

    policy = OfficialPolicy()
    policy_steps = int(round(case.policy_seconds / DT))
    for step in range(policy_steps):
        if step % CONTROL_STEPS == 0:
            base = data.xpos[ids["base_body_id"]]
            target_yaw = math.atan2(
                float(PIT_EXIT_WP[1] - base[1]),
                float(PIT_EXIT_WP[0] - base[0]),
            )
            yaw_error = wrap(target_yaw - yaw_from_xmat(data.xmat[ids["base_body_id"]]))
            command = np.asarray(
                [case.policy_command_mps, 0.0, np.clip(1.2 * yaw_error, -0.8, 0.8)],
                dtype=np.float32,
            )
            policy.update(model, data, ids, command)
        policy.apply(data)
        mujoco.mj_step(model, data)
        observe_state()
        contacts = contact_metrics(model, data, ids)
        max_force = max(max_force, float(contacts["max_contact_force"]))
        roll, pitch = roll_pitch(data, ids["base_body_id"])
        max_roll = max(max_roll, abs(roll))
        max_pitch = max(max_pitch, abs(pitch))
        recorder.capture(
            data,
            ids["base_body_id"],
            phase="HANDOFF CONDITIONER",
            detail=(
                f"official ONNX {case.policy_command_mps:.2f}m/s; "
                f"t={step * DT:.2f}/{case.policy_seconds:.2f}s"
            ),
        )
        bad = diverged(data)
        if bad is not None:
            reason = bad
            break

    initial_target = np.asarray(data.qpos[7:23], dtype=np.float64).copy()
    if case.leg_target == "current":
        final_target = initial_target.copy()
    elif case.leg_target == "standing":
        final_target = STANDING_POSE.astype(np.float64).copy()
    elif case.leg_target == "locked":
        final_target = np.asarray(locked_reference["joint_position_rad"], dtype=np.float64).copy()
    else:
        raise ValueError(f"unknown leg target: {case.leg_target}")

    total_seconds = case.drive_seconds + case.brake_seconds
    total_steps = int(round(total_seconds / DT))
    transition_seconds = min(0.15, max(total_seconds, DT))
    try:
        for step in range(total_steps if reason == "complete" else 0):
            elapsed = step * DT
            blend = min(1.0, elapsed / transition_seconds)
            position_target = initial_target + blend * (final_target - initial_target)
            wheel_speed = case.wheel_speed_rad_s if elapsed < case.drive_seconds else 0.0
            apply_direct_control(
                data,
                position_target,
                np.full(4, wheel_speed, dtype=np.float64),
            )
            mujoco.mj_step(model, data)
            observe_state()
            contacts = contact_metrics(model, data, ids)
            max_force = max(max_force, float(contacts["max_contact_force"]))
            roll, pitch = roll_pitch(data, ids["base_body_id"])
            max_roll = max(max_roll, abs(roll))
            max_pitch = max(max_pitch, abs(pitch))
            recorder.capture(
                data,
                ids["base_body_id"],
                phase="HANDOFF CONDITIONER",
                detail=(
                    f"{case.label}; progress={along(data.xpos[ids['base_body_id']][:2]):+.3f}m "
                    f"wheel={wheel_speed:+.1f}rad/s"
                ),
            )
            bad = diverged(data)
            if bad is not None:
                reason = bad
                break
            if abs(roll) > math.radians(25.0):
                reason = "excessive_roll"
                break
        recorder.capture(
            data,
            ids["base_body_id"],
            phase="HANDOFF CONDITIONER",
            detail=f"{case.label}; expert takeover",
            force=True,
        )
        observe_state(force=True)
        end_metrics = state_metrics(data, ids, locked_reference)
        report = {
            "label": case.label,
            "policy_command_mps": case.policy_command_mps,
            "policy_seconds": case.policy_seconds,
            "leg_target": case.leg_target,
            "wheel_speed_rad_s": case.wheel_speed_rad_s,
            "drive_seconds": case.drive_seconds,
            "brake_seconds": case.brake_seconds,
            "elapsed_seconds": float(data.time - source.time),
            "visible_state_observer_samples": observer_samples,
            "reason": reason,
            "max_contact_force_n": max_force,
            "max_abs_roll_deg": math.degrees(max_roll),
            "max_abs_pitch_deg": math.degrees(max_pitch),
            "start": start_metrics,
            "end": end_metrics,
        }
        return snapshot(data), report, recorder.frames.copy()
    finally:
        recorder.close()


def main() -> int:
    args = parse_args()
    output = local_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if args.max_sim_seconds <= 0.0 or args.max_forward <= 0.0:
        raise ValueError("simulation duration and maximum forward speed must be positive")

    locked_path = local_path(LOCKED_V4)
    locked_trace_path = local_path(LOCKED_TRACE)
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    locked_report = locked["report"]
    locked_trace = np.load(locked_trace_path)
    locked_reference = {
        "base_progress_m": np.asarray(locked_trace["base_progress_m"][0]),
        "wheel_progress_m": np.asarray(locked_trace["wheel_progress_m"][0]),
        "joint_position_rad": np.asarray(locked_trace["joint_position_rad"][0]),
        "joint_velocity_rad_s": np.asarray(locked_trace["joint_velocity_rad_s"][0]),
    }
    reference_path = local_path(Path(locked["inputs"]["reference"]["path"]))
    phase_path = local_path(Path(locked["inputs"]["phase"]["path"]))
    before_assets = asset_hashes()
    locked_hashes = {
        "summary": sha256(locked_path),
        "trace": sha256(locked_trace_path),
    }

    model = mujoco.MjModel.from_xml_path(str(local_path(MJCF_PATH)))
    model.opt.timestep = DT
    configure_continuous_limits(model)
    data = mujoco.MjData(model)
    ids = model_ids(model)
    navigator: Navigator | None
    if args.natural_state is None:
        route = load_route()
        reset_official_start(model, data)
        diverged_stand, stand_reason = run_official_standup(model, data)
        if diverged_stand:
            raise RuntimeError(f"official stand-up failed: {stand_reason}")
        pre_recorder = TrackRecorder(model, bool(args.video), float(args.frame_period))
        pre_recorder.capture(
            data,
            ids["base_body_id"],
            phase="WALK POLICY",
            detail="official start; natural route to score 15",
            force=True,
        )
        navigator = Navigator(route, data.xpos[ids["base_body_id"]][:2], float(args.max_forward))
        pre_report = run_policy_segment(
            model,
            data,
            ids,
            navigator,
            pre_recorder,
            max_sim_seconds=float(args.max_sim_seconds),
            stop_after_score=PIT_SCORE_INDEX,
        )
        natural_state = snapshot(data)
        save_state(output / "natural_pit_state.npz", natural_state)
        pre_recorder.capture(
            data,
            ids["base_body_id"],
            phase="PIT HANDOFF POINT",
            detail=f"natural score 15 state saved; t={data.time:.2f}s",
            force=True,
        )
        pre_frames = pre_recorder.frames.copy()
        pre_recorder.close()
        if PIT_SCORE_INDEX not in pre_report["reached_scores"]:
            raise RuntimeError(f"official policy did not reach score 15: {pre_report['reason']}")
        natural_state_source = str(output / "natural_pit_state.npz")
    else:
        source_path = local_path(args.natural_state)
        natural_state = load_state(source_path)
        restore(model, data, natural_state)
        pre_recorder = TrackRecorder(model, bool(args.video), float(args.pit_frame_period))
        pre_recorder.capture(
            data,
            ids["base_body_id"],
            phase="PIT HANDOFF POINT",
            detail="reused natural score 15 state; no teleport within candidate",
            force=True,
        )
        pre_frames = pre_recorder.frames.copy()
        pre_recorder.close()
        pre_report = {
            "reason": "reused_saved_natural_score_15_state",
            "reached_scores": list(range(PIT_SCORE_INDEX + 1)),
        }
        navigator = None
        natural_state_source = str(source_path)
    natural_metrics = state_metrics(data, ids, locked_reference)

    torch.set_num_threads(1)
    agent = load_reference(reference_path)
    original_env = rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv
    original_gif_recorder = rescue_runner.GifRecorder
    rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv = HandoffExactPeakEnv
    cases: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    replay: dict[str, Any] | None = None
    post_report: dict[str, Any] | None = None
    try:
        for case in conditioner_cases(bool(args.tuned_natural_handoff)):
            conditioned_state, conditioner_report, _ = condition_state(
                natural_state,
                case,
                locked_reference,
                video=False,
                period_s=float(args.pit_frame_period),
            )
            expert_report = run_expert(
                agent,
                phase_path,
                configured_expert_kwargs(
                    locked_report, bool(args.tuned_natural_handoff)
                ),
                conditioned_state,
            )
            result = {
                "conditioner": conditioner_report,
                "expert": expert_report,
                "case": plain(case.__dict__),
            }
            cases.append(result)
            print(
                f"[conditioned-handoff] {case.label} "
                f"progress={conditioner_report['end']['base_progress_m']:+.3f}m "
                f"leg_speed={conditioner_report['end']['leg_speed_l2_rad_s']:.2f} "
                f"success={expert_report['success']} "
                f"milestone={expert_report['milestone_name']} "
                f"reason={expert_report['termination_reason']}",
                flush=True,
            )

        successful = [item for item in cases if item["expert"]["success"]]
        if successful:
            selected = min(
                successful,
                key=lambda item: (
                    item["expert"]["max_contact_force_n"],
                    item["conditioner"]["elapsed_seconds"],
                ),
            )
        else:
            selected = max(cases, key=lambda item: rank_report(item["expert"]))

        selected_case = ConditionerCase(**selected["case"])
        selected_state, selected_conditioner, conditioner_frames = condition_state(
            natural_state,
            selected_case,
            locked_reference,
            video=bool(args.video),
            period_s=float(args.pit_frame_period),
        )
        PhaseRecorder.last_frames = []
        if args.video:
            rescue_runner.GifRecorder = PhaseRecorder
        replay = run_expert(
            agent,
            phase_path,
            configured_expert_kwargs(
                locked_report, bool(args.tuned_natural_handoff)
            ),
            selected_state,
            gif_path=output / "expert.gif" if args.video else None,
            trace_path=output / "expert_trace.npz",
            frame_period=float(args.pit_frame_period),
        )
        expert_instance = HandoffExactPeakEnv.last_instance
        if expert_instance is None:
            raise RuntimeError("expert environment instance was not retained")

        post_frames: list[Image.Image] = []
        if replay["success"] and navigator is not None:
            configure_continuous_limits(expert_instance.model)
            post_recorder = TrackRecorder(
                expert_instance.model,
                bool(args.video),
                float(args.frame_period),
            )
            post_recorder.capture(
                expert_instance.data,
                expert_instance.ids["base_body_id"],
                phase="WALK POLICY RESUMED",
                detail="conditioned expert exit complete; continuing route",
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

        combined_frames = pre_frames + conditioner_frames + PhaseRecorder.last_frames + post_frames
        if args.video and combined_frames:
            save_gif(
                output / "full_track_conditioned_rollout.gif",
                combined_frames,
                float(args.pit_frame_period),
            )

        after_assets = asset_hashes()
        locked_after = {
            "summary": sha256(locked_path),
            "trace": sha256(locked_trace_path),
        }
        if before_assets != after_assets or locked_hashes != locked_after:
            raise RuntimeError("official assets or locked v4 changed during conditioned handoff")
        summary = {
            "purpose": "natural full-track state conditioning before locked plan2 pit expert",
            "mechanical_integration_audit_only": True,
            "deployable": False,
            "training_performed": False,
            "tuned_natural_handoff_used": bool(args.tuned_natural_handoff),
            "oracle_phase_events_used": True,
            "lidar_takeover_used": False,
            "state_reset_or_teleport_at_handoff": False,
            "natural_state_source": natural_state_source,
            "pre_track": pre_report,
            "natural_state_metrics": natural_metrics,
            "cases": cases,
            "selected_case": selected["case"],
            "selected_conditioner_replay": selected_conditioner,
            "selected_expert_replay": replay,
            "post_track": post_report,
            "visualization": (
                str(output / "full_track_conditioned_rollout.gif") if args.video else None
            ),
            "official_assets_modified": False,
            "locked_v4_modified": False,
            "official_assets_before": before_assets,
            "official_assets_after": after_assets,
            "locked_v4_hashes_before": locked_hashes,
            "locked_v4_hashes_after": locked_after,
            "inputs": {
                "locked_v4": {"path": str(locked_path), "sha256": locked_hashes["summary"]},
                "locked_trace": {
                    "path": str(locked_trace_path),
                    "sha256": locked_hashes["trace"],
                },
                "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
                "phase": {"path": str(phase_path), "sha256": sha256(phase_path)},
                "official_track": {
                    "path": str(local_path(MJCF_PATH)),
                    "sha256": sha256(local_path(MJCF_PATH)),
                },
            },
            "protected_original_root": str(ORIGINAL_ROOT),
        }
        (output / "summary.json").write_text(
            json.dumps(plain(summary), indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[conditioned-handoff] selected={selected_case.label} "
            f"expert_success={replay['success']} output={output}",
            flush=True,
        )
    finally:
        rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv = original_env
        rescue_runner.GifRecorder = original_gif_recorder
        HandoffExactPeakEnv.handoff_state = None
    return 0 if replay is not None and replay["success"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
