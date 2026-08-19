#!/usr/bin/env python3
"""Audit early sustained front-wheel pulling after the front pair hooks the lip."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_s10_m20_early_steering_teacher_bank_grid import load_steering
from search_s10_m20_approach_two_stage_rescue import (
    PHASE_ESTIMATOR,
    REFERENCE_CHECKPOINT,
    asset_hashes,
    load_reference,
    sha256,
)
from search_s10_m20_early_steering_rescue import load_candidate, rollout
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT = Path("/home/xj/Downloads/goai_embodied_future_material-main")
SOURCE_SUMMARY = PROJECT_ROOT / (
    "logs/mujoco/s10_m20_latched_pullover_target037691_asym_grid420_"
    "20260812_v1/summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def local_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else PROJECT_ROOT / path
    resolved = candidate.expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"input escapes the isolated plan2 tree: {resolved}") from exc
    try:
        resolved.relative_to(ORIGINAL_ROOT)
    except ValueError:
        pass
    else:
        raise RuntimeError(f"input resolves into the protected original tree: {resolved}")
    return resolved


def optional(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    return -math.inf if value is None else float(value)


def body_contact(report: dict[str, Any]) -> bool:
    return bool(report.get("body_wall_contact_audit", {}).get("observed", False))


def result_key(report: dict[str, Any]) -> tuple[float, ...]:
    safe = not bool(report["hard_failure"])
    return (
        float(report["success"] and safe and not body_contact(report)),
        float(report["success"] and safe),
        float(report["base_cross_latched"] and safe),
        float(report["rear_top_latched"] and safe),
        optional(report, "peak_rear_top_proximity_after_top_m"),
        optional(report, "peak_rear_top_proximity_after_second_m"),
        optional(report, "peak_rear_progress_after_top_m"),
        optional(report, "peak_rear_height_after_top_m"),
        optional(report, "peak_base_progress_after_pullover_m"),
        optional(report, "front_support_rate_after_top"),
        float(not body_contact(report)),
        -float(report["hard_failure"]),
        -float(report["max_abs_roll_deg"]),
        -float(report["max_contact_force_n"]),
    )


def replay_key(report: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(report["success"]),
        bool(report["hard_failure"]),
        bool(report["rear_top_latched"]),
        bool(report["base_cross_latched"]),
        str(report["termination_reason"]),
        int(report["episode_steps"]),
        report["true_front_latch_step"],
        report["height_stage_trigger_step"],
        report["top_stage_trigger_step"],
        report["threepoint_trigger_step"],
        float(report["peak_rear_height_after_second_m"]),
        float(report["peak_rear_progress_after_second_m"]),
        float(report["peak_rear_top_proximity_after_second_m"]),
        float(report["peak_base_progress_after_pullover_m"]),
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("workers must stay in [1, 4] to avoid interfering with other runs")
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
    old_payload = json.loads(paths["old_success"].read_text(encoding="utf-8"))
    old_case = old_payload["best"]
    old_preload = finite_vector(old_case["leg_vector"], (12,), "old preload")
    pulse_payload = json.loads(paths["precontact"].read_text(encoding="utf-8"))
    pulse_case = pulse_payload["best_search"]
    pulse = finite_vector(pulse_case["vector"], (12,), "precontact pulse")

    source = json.loads(paths["source"].read_text(encoding="utf-8"))
    depth_m = float(source["depth_m"])
    source_case = source["best"]
    launch_case = source_case["launch"]
    tuck_case = source_case["tuck"]
    launch = finite_vector(launch_case["vector"], (12,), "launch vector")
    tuck = finite_vector(tuck_case["vector"], (12,), "tuck vector")
    expected = source["best_contact_strict_replay"]

    def evaluate(
        anchor_delay_s: float,
        front_wheel_action: float,
        freeze_front_legs: bool,
        label: str,
        *,
        trace: Path | None = None,
        gif: Path | None = None,
    ) -> dict[str, Any]:
        return rollout(
            agent,
            paths["phase"],
            candidate,
            steering,
            label=label,
            depth_m=depth_m,
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
            pullover_launch_delay_after_front_seconds=float(launch_case["delay_seconds"]),
            pullover_launch_hold_seconds=float(launch_case["hold_seconds"]),
            pullover_tuck_leg_vector=tuck,
            pullover_tuck_delay_after_launch_seconds=float(tuck_case["delay_seconds"]),
            pullover_tuck_hold_seconds=float(tuck_case["hold_seconds"]),
            pullover_leg_rate_per_second=5.0,
            allow_body_wall_contact=False,
            late_momentum_leg_vector=old_preload,
            late_momentum_leg_delay_seconds=float(old_case["delay_seconds"]),
            late_momentum_leg_hold_seconds=float(old_case["hold_seconds"]),
            late_momentum_leg_rate_per_second=4.0,
            oracle_front_wheel_action=front_wheel_action,
            oracle_freeze_front_legs=freeze_front_legs,
            oracle_front_anchor_delay_seconds=anchor_delay_s,
            leg_peak_nm=76.4,
            wheel_peak_nm=21.6,
            max_continuous_exceedance_s=0.5,
            trace_path=trace,
            gif_path=gif,
        )

    baseline = evaluate(
        0.82,
        -0.50,
        False,
        "locked_asymmetric_baseline",
        trace=output / "baseline_trace.npz",
    )
    baseline_matches = replay_key(baseline) == replay_key(expected)
    if not baseline_matches:
        (output / "baseline_mismatch.json").write_text(
            json.dumps({"actual": baseline, "expected": expected}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("front-hook audit did not reproduce the locked plan2 baseline")

    configurations = [
        (delay_s, wheel_action, freeze)
        for freeze in (False, True)
        for wheel_action in (-0.30, -0.50, -0.80, -1.20)
        for delay_s in (0.00, 0.08, 0.16, 0.24, 0.36, 0.50, 0.66, 0.82)
        if not (
            math.isclose(delay_s, 0.82)
            and math.isclose(wheel_action, -0.50)
            and not freeze
        )
    ]
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("max-cases must be positive")
        configurations = configurations[: args.max_cases]

    def run(item: tuple[int, tuple[float, float, bool]]) -> tuple[int, dict[str, Any]]:
        index, (delay_s, wheel_action, freeze) = item
        report = evaluate(
            delay_s,
            wheel_action,
            freeze,
            f"case{index:03d}_delay{delay_s:.2f}_wheel{wheel_action:.2f}_freeze{int(freeze)}",
        )
        return index, report

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        indexed_reports = list(executor.map(run, enumerate(configurations)))
    cases = [
        {
            "index": index,
            "anchor_delay_seconds": configurations[index][0],
            "front_wheel_action": configurations[index][1],
            "freeze_front_legs": configurations[index][2],
            "report": report,
        }
        for index, report in indexed_reports
    ]
    cases.sort(key=lambda item: result_key(item["report"]), reverse=True)
    best = cases[0]
    best_replay = evaluate(
        float(best["anchor_delay_seconds"]),
        float(best["front_wheel_action"]),
        bool(best["freeze_front_legs"]),
        "best_replay",
        trace=output / "best_trace.npz",
        gif=output / "gifs" / "best.gif",
    )

    after_assets = asset_hashes()
    if before_assets != after_assets:
        raise RuntimeError("official assets changed during front-hook pull audit")
    summary = {
        "purpose": "video-inspired sustained front-hook pull audit at the competition proxy depth",
        "video_mechanism": "front pair hooks the lip, then continuously rolls/pulls while the rear pair rises",
        "mechanical_feasibility_only": True,
        "deployable": False,
        "training_performed": False,
        "oracle_front_event_used": True,
        "body_wall_contact_allowed": False,
        "depth_m": depth_m,
        "baseline": baseline,
        "baseline_matches_source": baseline_matches,
        "grid": {
            "anchor_delays_seconds": [0.00, 0.08, 0.16, 0.24, 0.36, 0.50, 0.66, 0.82],
            "front_wheel_actions": [-0.30, -0.50, -0.80, -1.20],
            "freeze_front_legs": [False, True],
            "case_count": len(configurations),
        },
        "cases_ranked": cases,
        "best": best,
        "best_replay": best_replay,
        "success_count": sum(bool(item["report"]["success"]) for item in cases),
        "rear_top_count": sum(bool(item["report"]["rear_top_latched"]) for item in cases),
        "threepoint_count": sum(item["report"]["threepoint_trigger_step"] is not None for item in cases),
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
        f"[front-hook] cases={len(cases)} success={summary['success_count']} "
        f"rear={summary['rear_top_count']} threepoint={summary['threepoint_count']} "
        f"best_proximity={optional(best_replay, 'peak_rear_top_proximity_after_top_m'):.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
