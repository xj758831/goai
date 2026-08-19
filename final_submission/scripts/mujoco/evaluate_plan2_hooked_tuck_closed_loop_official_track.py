#!/usr/bin/env python3
"""Run the plan2 staged hooked-tuck controller on the exact official track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

import search_s10_m20_early_steering_rescue as rescue_runner
from evaluate_s10_m20_early_steering_teacher_bank_grid import load_steering
from search_s10_m20_approach_two_stage_rescue import (
    PHASE_ESTIMATOR,
    REFERENCE_CHECKPOINT,
    asset_hashes,
    load_reference,
    sha256,
)
from search_s10_m20_early_steering_rescue import load_candidate
from search_s10_m20_front_hook_pull import ORIGINAL_ROOT, local_path
from search_s10_m20_rear_placement_primitive import (
    DEFAULT_FIRST_BOOST,
    DEFAULT_LIFT_BOOST,
    DEFAULT_TEACHER,
    LEG_ACTIVE,
    LEG_WHEEL_ACTIVE,
    expand,
    expand_front,
    load_boost,
    load_height_crossing_placement,
)
from search_s10_m20_threepoint_rear_swing import (
    HEIGHT_SUMMARY,
    JOINT_SUMMARY,
    OLD_SUCCESS_SUMMARY,
    PRECONTACT_SUMMARY,
    finite_vector,
)
from s10_fronthook_scripted_probe import DT, MJCF_PATH, model_ids, reset_in_pit, settle
from s10_m20_phase_gated_residual import S10M20PhaseGatedEnv
from s10_m20_phase_gated_torque_envelope_env import (
    S10M20PhaseGatedTorqueEnvelopeEnv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUMMARY = PROJECT_ROOT / (
    "logs/mujoco/plan2_hooked_rear_tuck_additive_grid72_20260812_v2/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--torque-mode",
        choices=("official_continuous", "research_peak"),
        default="research_peak",
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class ExactOfficialTrackMixin:
    """Replace only the runtime terrain and reset of a staged controller env."""

    def _replace_with_exact_track(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(local_path(MJCF_PATH)))
        self.model.opt.timestep = DT
        self.data = mujoco.MjData(self.model)
        self.ids = model_ids(self.model)
        self.geometry = {
            "scope": "exact unmodified official S10_track.xml",
            "official_track_sha256": sha256(local_path(MJCF_PATH)),
        }

    def _reset_physics(self) -> tuple[bool, str | None]:
        if self._current_state != (0.0, 0.0):
            raise ValueError("exact-track mechanical audit currently supports nominal reset only")
        reset_in_pit(self.model, self.data)
        return settle(self.model, self.data, self.ids)


class ExactOfficialTrackPhaseEnv(ExactOfficialTrackMixin, S10M20PhaseGatedEnv):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._replace_with_exact_track()


class ExactOfficialTrackPeakEnv(
    ExactOfficialTrackMixin, S10M20PhaseGatedTorqueEnvelopeEnv
):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._replace_with_exact_track()
        self._configure_in_memory_limits()


def main() -> int:
    args = parse_args()
    output = local_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "reference": local_path(REFERENCE_CHECKPOINT),
        "phase": local_path(PHASE_ESTIMATOR),
        "teacher": local_path(DEFAULT_TEACHER),
        "first": local_path(DEFAULT_FIRST_BOOST),
        "lift": local_path(DEFAULT_LIFT_BOOST),
        "height_summary": local_path(HEIGHT_SUMMARY),
        "joint_summary": local_path(JOINT_SUMMARY),
        "old_success": local_path(OLD_SUCCESS_SUMMARY),
        "precontact": local_path(PRECONTACT_SUMMARY),
        "source": local_path(SOURCE_SUMMARY),
        "official_track": local_path(MJCF_PATH),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing plan2 input {name}: {path}")

    before_assets = asset_hashes()
    torch.set_num_threads(1)
    agent = load_reference(paths["reference"])
    candidate = load_candidate(paths["teacher"])
    steering = load_steering(paths["teacher"])
    first_boost = load_boost(paths["first"])
    lift_boost = load_boost(paths["lift"])
    height_placement, _ = load_height_crossing_placement(paths["height_summary"])

    joint_payload = json.loads(paths["joint_summary"].read_text(encoding="utf-8"))
    joint_vector = finite_vector(joint_payload["best_vector"], (10,), "joint vector")
    rear_transfer = joint_vector[: LEG_WHEEL_ACTIVE.size]
    front_coordination = joint_vector[LEG_WHEEL_ACTIVE.size :]
    old_case = json.loads(paths["old_success"].read_text(encoding="utf-8"))["best"]
    old_preload = finite_vector(old_case["leg_vector"], (12,), "old preload")
    pulse_case = json.loads(paths["precontact"].read_text(encoding="utf-8"))["best_search"]
    pulse = finite_vector(pulse_case["vector"], (12,), "precontact pulse")

    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    best = source["best"]
    best_report = best["report"]
    late_tuck = finite_vector(best["vector"], (12,), "late tuck")
    launch = finite_vector(best_report["pullover_launch_leg_vector"], (12,), "launch")
    source_tuck = finite_vector(best_report["pullover_tuck_leg_vector"], (12,), "source tuck")
    front_hook = source["fixed_front_hook"]

    original_base_class = rescue_runner.S10M20PhaseGatedEnv
    original_envelope_class = rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv
    rescue_runner.S10M20PhaseGatedEnv = ExactOfficialTrackPhaseEnv
    rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv = ExactOfficialTrackPeakEnv
    use_peak = args.torque_mode == "research_peak"
    try:
        report = rescue_runner.rollout(
            agent,
            paths["phase"],
            candidate,
            steering,
            label="plan2_hooked_tuck_closed_loop_exact_official_track",
            depth_m=float(source["depth_m"]),
            command_mps=0.65,
            initial_state=(0.0, 0.0),
            rollout_seed=118_000,
            early_threshold=0.10,
            approach_threshold=0.03,
            prelift_from_early=True,
            episode_seconds=10.0,
            leg_rate_per_second=2.5,
            post_from_approach=True,
            post_delay_seconds=0.15,
            late_post_boost_delay_seconds=0.80,
            late_post_boost_hold_seconds=0.35,
            late_post_boost_vector=first_boost,
            late_post_target_limit=1.0,
            late_post_second_vector=lift_boost,
            late_post_second_delay_seconds=0.35,
            late_post_second_hold_seconds=0.40,
            late_post_height_vector=expand(height_placement, LEG_ACTIVE),
            late_post_height_trigger_m=0.40,
            late_post_height_hold_seconds=0.40,
            late_post_top_vector=expand(rear_transfer, LEG_WHEEL_ACTIVE),
            late_post_top_front_leg_vector=expand_front(front_coordination),
            late_post_top_trigger_m=0.45,
            late_post_top_hold_seconds=0.35,
            late_post_top_score_delay_seconds=0.12,
            late_post_top_leg_rate_per_second=4.0,
            precontact_leg_vector=pulse,
            precontact_leg_delay_after_approach_seconds=float(pulse_case["delay_seconds"]),
            precontact_leg_hold_seconds=float(pulse_case["hold_seconds"]),
            precontact_leg_rate_per_second=4.0,
            threepoint_monitor_enabled=True,
            threepoint_min_leading_rear_contact_force_n=50.0,
            threepoint_max_abs_roll_deg=10.0,
            pullover_launch_leg_vector=launch,
            pullover_launch_delay_after_front_seconds=float(
                best_report["pullover_launch_delay_after_front_seconds"]
            ),
            pullover_launch_hold_seconds=float(best_report["pullover_launch_hold_seconds"]),
            pullover_tuck_leg_vector=source_tuck,
            pullover_tuck_delay_after_launch_seconds=float(
                best_report["pullover_tuck_delay_after_launch_seconds"]
            ),
            pullover_tuck_hold_seconds=float(best_report["pullover_tuck_hold_seconds"]),
            pullover_leg_rate_per_second=5.0,
            allow_body_wall_contact=False,
            late_momentum_leg_vector=old_preload,
            late_momentum_leg_delay_seconds=float(old_case["delay_seconds"]),
            late_momentum_leg_hold_seconds=float(old_case["hold_seconds"]),
            late_momentum_leg_rate_per_second=4.0,
            late_prejump_leg_vector=late_tuck,
            late_prejump_leg_delay_seconds=float(best["delay_seconds"]),
            late_prejump_leg_hold_seconds=float(best["hold_seconds"]),
            late_prejump_leg_rate_per_second=5.0,
            oracle_front_wheel_action=float(front_hook["front_wheel_action"]),
            oracle_freeze_front_legs=bool(front_hook["freeze_front_legs"]),
            oracle_front_anchor_delay_seconds=float(front_hook["anchor_delay_seconds"]),
            leg_peak_nm=76.4 if use_peak else 50.0,
            wheel_peak_nm=21.6 if use_peak else 14.0,
            max_continuous_exceedance_s=0.5,
            trace_path=output / "trace.npz",
            gif_path=output / "rollout.gif" if args.video else None,
        )
    finally:
        rescue_runner.S10M20PhaseGatedEnv = original_base_class
        rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv = original_envelope_class

    after_assets = asset_hashes()
    if before_assets != after_assets:
        raise RuntimeError("official assets changed during exact-track closed-loop audit")
    summary = {
        "purpose": "closed-loop staged plan2 hooked-tuck controller on the exact official track",
        "mechanical_feasibility_only": True,
        "deployable": False,
        "training_performed": False,
        "oracle_front_event_used": True,
        "oracle_height_top_events_used": True,
        "torque_mode": args.torque_mode,
        "in_memory_peak_limits_applied": use_peak,
        "competition_peak_threshold_confirmed": False,
        "official_assets_modified": False,
        "geometry_scope": "exact unmodified official S10_track.xml",
        "official_track_sha256": sha256(paths["official_track"]),
        "report": report,
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "protected_original_root": str(ORIGINAL_ROOT),
        "all_inputs_inside_plan2": True,
        "official_assets_before": before_assets,
        "official_assets_after": after_assets,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[official-closed-loop] mode={args.torque_mode} success={report['success']} "
        f"front={report['front_top_latched']} "
        f"rear={report['rear_top_latched']} base={report['base_cross_latched']} "
        f"reason={report['termination_reason']} roll={report['max_abs_roll_deg']:.2f} "
        f"contact={report['max_contact_force_n']:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
