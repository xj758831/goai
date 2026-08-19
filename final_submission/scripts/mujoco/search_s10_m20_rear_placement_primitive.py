#!/usr/bin/env python3
"""Search a height-triggered rear-leg placement primitive at a proxy pit.

This is an oracle mechanical-feasibility diagnostic.  It replays the fixed
approach, lift, and front-anchor sequence, then uses MuJoCo rear-wheel height
truth to trigger a third phase.  The height-triggered phase can search both
front and rear sagittal hip/knee channels; official assets and learned
checkpoints are read-only.
"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER = PROJECT_ROOT / (
    "logs/mujoco/s10_m20_depth0345_center_peak764_216_lateboost08_s030_"
    "frompeak_bound4_rate25_wheel08_cem40x3_20260811_v1/best_teacher.npz"
)
DEFAULT_FIRST_BOOST = PROJECT_ROOT / (
    "logs/mujoco/s10_m20_depth0345_center_peak764_216_independent_lateboost08_"
    "hold035_cem48x4_20260811_v1/best_late_boost.npz"
)
DEFAULT_LIFT_BOOST = PROJECT_ROOT / (
    "logs/mujoco/s10_m20_depth0345_center_peak764_216_oracleanchor_m05_delay082_"
    "secondphase035_hold040_rearprox_cem40x3_20260811_v1/best_late_boost.npz"
)
LEG_ACTIVE = np.asarray([1, 2, 4, 5], dtype=np.int64)
LEG_WHEEL_ACTIVE = np.asarray([1, 2, 4, 5, 6, 7], dtype=np.int64)
FRONT_ACTIVE = np.asarray([1, 2, 4, 5], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, default=REFERENCE_CHECKPOINT)
    parser.add_argument("--phase-estimator", type=Path, default=PHASE_ESTIMATOR)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--fixed-first-boost", type=Path, default=DEFAULT_FIRST_BOOST)
    parser.add_argument("--fixed-lift-boost", type=Path, default=DEFAULT_LIFT_BOOST)
    parser.add_argument("--fixed-height-placement-summary", type=Path, default=None)
    parser.add_argument("--warm-start-transfer-summary", type=Path, default=None)
    parser.add_argument("--fixed-top-transfer-summary", type=Path, default=None)
    parser.add_argument("--warm-start-front-summary", type=Path, default=None)
    parser.add_argument("--warm-start-height-summary", type=Path, default=None)
    parser.add_argument("--fixed-preload-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-m", type=float, default=0.345)
    parser.add_argument("--height-trigger-m", type=float, default=0.50)
    parser.add_argument("--height-delay-after-front-seconds", type=float, default=None)
    parser.add_argument("--placement-hold-seconds", type=float, default=0.40)
    parser.add_argument("--top-transfer-hold-seconds", type=float, default=0.35)
    parser.add_argument("--top-transfer-trigger-m", type=float, default=0.50)
    parser.add_argument("--top-transfer-score-delay-seconds", type=float, default=0.12)
    parser.add_argument("--top-transfer-leg-rate-per-second", type=float, default=2.5)
    parser.add_argument("--placement-bound", type=float, default=0.50)
    parser.add_argument("--wheel-bound", type=float, default=0.80)
    parser.add_argument("--front-leg-bound", type=float, default=0.50)
    parser.add_argument(
        "--include-rear-wheels",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--include-height-front-legs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="jointly search front and rear sagittal legs at the height trigger",
    )
    parser.add_argument(
        "--include-frog-prejump",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="jointly search a rear preload before the height-triggered release",
    )
    parser.add_argument("--prejump-bound", type=float, default=0.50)
    parser.add_argument("--prejump-delay-seconds", type=float, default=0.34)
    parser.add_argument("--prejump-hold-seconds", type=float, default=0.10)
    parser.add_argument(
        "--include-frog-rear-wheels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="search a small paired rear-wheel assist during the frog release",
    )
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--cem-update-rate", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=2026081161)
    parser.add_argument("--rollout-seed", type=int, default=118_000)
    return parser.parse_args()


def load_boost(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if "boost" not in payload:
        raise ValueError(f"boost array missing from {path}")
    values = np.asarray(payload["boost"], dtype=np.float64)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError(f"invalid boost in {path}: {values.shape}")
    return values


def expand(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    result = np.zeros(8, dtype=np.float64)
    if values.shape != active.shape:
        raise ValueError(f"value/active shape mismatch: {values.shape} vs {active.shape}")
    result[active] = values
    return result


def expand_front(values: np.ndarray) -> np.ndarray:
    result = np.zeros(6, dtype=np.float64)
    if values.shape != FRONT_ACTIVE.shape:
        raise ValueError(
            f"front value shape mismatch: {values.shape} vs {FRONT_ACTIVE.shape}"
        )
    result[FRONT_ACTIVE] = values
    return result


def seed_primitives(bounds: np.ndarray, include_rear_wheels: bool) -> list[np.ndarray]:
    # Each row is [HL hipy, HL knee, HR hipy, HR knee] in raw policy units.
    rows = (
        (0.0, 0.0, 0.0, 0.0),
        (-0.20, 0.40, -0.20, 0.40),
        (0.20, -0.40, 0.20, -0.40),
        (0.40, -0.20, 0.40, -0.20),
        (0.40, 0.20, 0.40, 0.20),
        (0.40, -0.20, 0.0, 0.0),
        (0.0, 0.0, 0.40, -0.20),
    )
    if include_rear_wheels:
        rows = tuple((*row, 0.0, 0.0) for row in rows) + (
            (0.0, 0.0, 0.0, 0.0, 0.50, 0.0),
            (0.0, 0.0, 0.0, 0.0, -0.50, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.50),
            (0.0, 0.0, 0.0, 0.0, 0.0, -0.50),
            (0.0, 0.0, 0.0, 0.0, 0.40, 0.40),
            (0.0, 0.0, 0.0, 0.0, -0.40, -0.40),
            (0.0, 0.0, 0.0, 0.0, 0.40, -0.40),
            (0.0, 0.0, 0.0, 0.0, -0.40, 0.40),
        )
    return [
        np.clip(np.asarray(row, dtype=np.float64), -bounds, bounds)
        for row in rows
    ]


def optional_float(value: Any, fallback: float = -math.inf) -> float:
    return fallback if value is None else float(value)


def load_height_crossing_placement(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    choices: list[tuple[float, float, np.ndarray, dict[str, Any]]] = []
    for iteration in payload.get("iterations", []):
        report = iteration["best"]
        progress = report.get("peak_rear_progress_above_height_gate_m")
        vector = np.asarray(iteration["best_vector"], dtype=np.float64)
        if (
            progress is not None
            and not report["hard_failure"]
            and vector.shape == (4,)
            and np.isfinite(vector).all()
        ):
            choices.append(
                (
                    float(progress),
                    optional_float(report.get("peak_rear_top_proximity_after_height_m")),
                    vector,
                    report,
                )
            )
    if not choices:
        raise ValueError(f"no safe height-crossing placement in {path}")
    _, _, vector, report = max(choices, key=lambda item: item[:2])
    return vector, report


def main() -> int:
    args = parse_args()
    if args.population < 8 or args.iterations < 1 or args.workers < 1:
        raise ValueError("population must be >=8; iterations/workers must be positive")
    if not 0.05 <= args.elite_fraction <= 0.75:
        raise ValueError("elite fraction must be in [0.05, 0.75]")
    if not 0.0 < args.cem_update_rate <= 1.0:
        raise ValueError("CEM update rate must be in (0, 1]")
    if not 0.05 <= args.placement_bound <= 0.8:
        raise ValueError("placement bound must be in [0.05, 0.8]")
    if not 0.05 <= args.wheel_bound <= 1.5:
        raise ValueError("wheel bound must be in [0.05, 1.5]")
    if not 0.05 <= args.front_leg_bound <= 0.8:
        raise ValueError("front leg bound must be in [0.05, 0.8]")
    if not 0.05 <= args.prejump_bound <= 0.8:
        raise ValueError("prejump bound must be in [0.05, 0.8]")
    if not 0.0 <= args.prejump_delay_seconds <= 0.50:
        raise ValueError("prejump delay must be in [0, 0.50] s")
    if not 0.06 <= args.prejump_hold_seconds <= 0.30:
        raise ValueError("prejump hold must be in [0.06, 0.30] s")
    if not 0.30 <= args.depth_m <= 0.38:
        raise ValueError("depth must be in [0.30, 0.38] m")
    if not 0.28 <= args.height_trigger_m <= 0.54:
        raise ValueError("height trigger must be in [0.28, 0.54] m")
    if args.height_delay_after_front_seconds is not None and not (
        0.0 <= args.height_delay_after_front_seconds <= 0.60
    ):
        raise ValueError("height delay after front must be in [0, 0.60] s")
    if not 0.10 <= args.placement_hold_seconds <= 0.80:
        raise ValueError("placement hold must be in [0.10, 0.80] s")
    if not 0.10 <= args.top_transfer_hold_seconds <= 0.80:
        raise ValueError("top transfer hold must be in [0.10, 0.80] s")
    if not 0.40 <= args.top_transfer_trigger_m <= 0.54:
        raise ValueError("top transfer trigger must be in [0.40, 0.54] m")
    if not 0.0 <= args.top_transfer_score_delay_seconds <= 0.30:
        raise ValueError("top transfer score delay must be in [0, 0.30] s")
    if not 0.1 <= args.top_transfer_leg_rate_per_second <= 5.0:
        raise ValueError("top transfer leg rate must be in [0.1, 5.0]")

    paths = {
        "reference": args.reference_checkpoint.expanduser().resolve(),
        "phase": args.phase_estimator.expanduser().resolve(),
        "teacher": args.teacher.expanduser().resolve(),
        "first": args.fixed_first_boost.expanduser().resolve(),
        "lift": args.fixed_lift_boost.expanduser().resolve(),
    }
    if args.fixed_height_placement_summary is not None:
        paths["height_placement_summary"] = (
            args.fixed_height_placement_summary.expanduser().resolve()
        )
    if args.warm_start_transfer_summary is not None:
        paths["warm_start_transfer_summary"] = (
            args.warm_start_transfer_summary.expanduser().resolve()
        )
    if args.fixed_top_transfer_summary is not None:
        paths["fixed_top_transfer_summary"] = (
            args.fixed_top_transfer_summary.expanduser().resolve()
        )
    if args.warm_start_front_summary is not None:
        paths["warm_start_front_summary"] = (
            args.warm_start_front_summary.expanduser().resolve()
        )
    if args.fixed_preload_summary is not None:
        paths["fixed_preload_summary"] = (
            args.fixed_preload_summary.expanduser().resolve()
        )
    if args.warm_start_height_summary is not None:
        paths["warm_start_height_summary"] = (
            args.warm_start_height_summary.expanduser().resolve()
        )
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    before_assets = asset_hashes()
    torch.set_num_threads(1)
    agent = load_reference(paths["reference"])
    candidate = load_candidate(paths["teacher"])
    steering = load_steering(paths["teacher"])
    first_boost = load_boost(paths["first"])
    lift_boost = load_boost(paths["lift"])
    top_transfer_mode = "height_placement_summary" in paths
    height_joint_mode = bool(args.include_height_front_legs)
    frog_joint_mode = bool(args.include_frog_prejump)
    frog_wheel_mode = bool(args.include_frog_rear_wheels)
    if frog_joint_mode and not height_joint_mode:
        raise ValueError("frog prejump requires height-stage front/rear joint search")
    if frog_wheel_mode and not frog_joint_mode:
        raise ValueError("frog rear-wheel assist requires frog prejump mode")
    if frog_wheel_mode and args.include_rear_wheels:
        raise ValueError("frog rear-wheel assist replaces the legacy rear-wheel mode")
    if height_joint_mode and top_transfer_mode:
        raise ValueError("height-stage joint search cannot use top-transfer mode")
    if height_joint_mode and args.include_rear_wheels:
        raise ValueError(
            "height-stage joint search isolates leg coordination; rear-wheel search is disabled"
        )
    if top_transfer_mode:
        fixed_height_placement, fixed_height_report = load_height_crossing_placement(
            paths["height_placement_summary"]
        )
    else:
        fixed_height_placement = np.zeros(4, dtype=np.float64)
        fixed_height_report = None
    front_coordination_mode = "fixed_top_transfer_summary" in paths
    joint_coordination_mode = "warm_start_front_summary" in paths
    if front_coordination_mode and joint_coordination_mode:
        raise ValueError("fixed-front and joint coordination modes are exclusive")
    if front_coordination_mode:
        if not top_transfer_mode:
            raise ValueError("front coordination requires fixed height placement")
        if args.include_rear_wheels:
            raise ValueError("front coordination fixes the rear-wheel transfer")
        top_payload = json.loads(
            paths["fixed_top_transfer_summary"].read_text(encoding="utf-8")
        )
        fixed_top_transfer = np.asarray(top_payload["best_vector"], dtype=np.float64)
        if (
            fixed_top_transfer.shape != LEG_WHEEL_ACTIVE.shape
            or not np.isfinite(fixed_top_transfer).all()
        ):
            raise ValueError("fixed top transfer must contain a finite 6-vector")
        fixed_top_report = top_payload["best_replay"]
    else:
        fixed_top_transfer = np.zeros(LEG_WHEEL_ACTIVE.size, dtype=np.float64)
        fixed_top_report = None
    if joint_coordination_mode:
        if not top_transfer_mode:
            raise ValueError("joint coordination requires fixed height placement")
        if not args.include_rear_wheels:
            raise ValueError("joint coordination requires rear-wheel search")
        if "warm_start_transfer_summary" not in paths:
            raise ValueError("joint coordination requires a warm-start transfer")
        front_payload = json.loads(
            paths["warm_start_front_summary"].read_text(encoding="utf-8")
        )
        warm_front = np.asarray(front_payload["best_vector"], dtype=np.float64)
        if warm_front.shape != FRONT_ACTIVE.shape or not np.isfinite(warm_front).all():
            raise ValueError("warm-start front summary must contain a finite 4-vector")
        warm_front_report = front_payload["best_replay"]
    else:
        warm_front = np.zeros(FRONT_ACTIVE.size, dtype=np.float64)
        warm_front_report = None
    search_active = (
        np.arange(LEG_ACTIVE.size * 2 + 2 + FRONT_ACTIVE.size, dtype=np.int64)
        if frog_wheel_mode
        else (
            np.arange(LEG_ACTIVE.size * 2 + FRONT_ACTIVE.size, dtype=np.int64)
            if frog_joint_mode
            else (
                np.arange(LEG_ACTIVE.size + FRONT_ACTIVE.size, dtype=np.int64)
                if height_joint_mode
                else (
                    np.arange(LEG_WHEEL_ACTIVE.size + FRONT_ACTIVE.size, dtype=np.int64)
                    if joint_coordination_mode
                    else (
                        FRONT_ACTIVE
                        if front_coordination_mode
                        else (LEG_WHEEL_ACTIVE if args.include_rear_wheels else LEG_ACTIVE)
                    )
                )
            )
        )
    )
    warm_start = np.zeros(search_active.size, dtype=np.float64)
    height_warm_report: dict[str, Any] | None = None
    height_warm_constraints: dict[str, Any] = {}
    if "warm_start_height_summary" in paths:
        if not height_joint_mode:
            raise ValueError(
                "warm-start height summary requires height-stage joint search"
            )
        height_payload = json.loads(
            paths["warm_start_height_summary"].read_text(encoding="utf-8")
        )
        height_warm_report = height_payload["best_replay"]
        height_warm_constraints = height_payload.get("constraints", {})
        height_values = np.asarray(height_payload["best_vector"], dtype=np.float64)
        if frog_wheel_mode and height_values.shape == (
            LEG_ACTIVE.size * 2 + FRONT_ACTIVE.size,
        ):
            warm_start[: LEG_ACTIVE.size * 2] = height_values[: LEG_ACTIVE.size * 2]
            warm_start[LEG_ACTIVE.size * 2 + 2 :] = height_values[
                LEG_ACTIVE.size * 2 :
            ]
        elif frog_joint_mode and height_values.shape == (
            LEG_ACTIVE.size + FRONT_ACTIVE.size,
        ):
            warm_start[LEG_ACTIVE.size :] = height_values
        elif frog_joint_mode and height_values.shape == LEG_ACTIVE.shape:
            warm_start[LEG_ACTIVE.size : LEG_ACTIVE.size * 2] = height_values
        elif height_values.shape == LEG_ACTIVE.shape:
            warm_start[: LEG_ACTIVE.size] = height_values
        elif height_values.shape == search_active.shape:
            warm_start[:] = height_values
        else:
            raise ValueError(
                f"height warm-start shape {height_values.shape} incompatible with {search_active.shape}"
            )
        if not np.isfinite(warm_start).all():
            raise ValueError("height warm-start contains non-finite values")
    if "warm_start_transfer_summary" in paths:
        if not top_transfer_mode or front_coordination_mode:
            raise ValueError(
                "warm-start transfer requires rear top-transfer search mode"
            )
        warm_payload = json.loads(
            paths["warm_start_transfer_summary"].read_text(encoding="utf-8")
        )
        warm_values = np.asarray(warm_payload["best_vector"], dtype=np.float64)
        if joint_coordination_mode and warm_values.shape == LEG_WHEEL_ACTIVE.shape:
            warm_start[: LEG_WHEEL_ACTIVE.size] = warm_values
        elif warm_values.shape == search_active.shape:
            warm_start[:] = warm_values
        elif args.include_rear_wheels and warm_values.shape == LEG_ACTIVE.shape:
            warm_start[: LEG_ACTIVE.size] = warm_values
        else:
            raise ValueError(
                f"warm-start shape {warm_values.shape} incompatible with {search_active.shape}"
            )
        if not np.isfinite(warm_start).all():
            raise ValueError("warm-start transfer contains non-finite values")
    if joint_coordination_mode:
        warm_start[LEG_WHEEL_ACTIVE.size :] = warm_front
    if "fixed_preload_summary" in paths:
        preload_payload = json.loads(
            paths["fixed_preload_summary"].read_text(encoding="utf-8")
        )
        preload_case = preload_payload["best"]
        preload_vector = np.asarray(preload_case["leg_vector"], dtype=np.float64)
        if preload_vector.shape != (12,) or not np.isfinite(preload_vector).all():
            raise ValueError("fixed preload summary must contain a finite 12-vector")
        preload_delay_seconds = float(preload_case["delay_seconds"])
        preload_hold_seconds = float(preload_case["hold_seconds"])
    else:
        preload_vector = np.zeros(12, dtype=np.float64)
        preload_delay_seconds = 0.0
        preload_hold_seconds = 0.20

    def height_rear(values: np.ndarray) -> np.ndarray:
        if frog_joint_mode:
            return values[LEG_ACTIVE.size : LEG_ACTIVE.size * 2]
        if height_joint_mode:
            return values[: LEG_ACTIVE.size]
        return values

    def height_front(values: np.ndarray) -> np.ndarray:
        offset = (
            LEG_ACTIVE.size * 2 + 2
            if frog_wheel_mode
            else (LEG_ACTIVE.size * 2 if frog_joint_mode else LEG_ACTIVE.size)
        )
        return values[offset:]

    def height_post(values: np.ndarray) -> np.ndarray:
        if frog_wheel_mode:
            result = expand(height_rear(values), LEG_ACTIVE)
            result[6:8] = values[LEG_ACTIVE.size * 2 : LEG_ACTIVE.size * 2 + 2]
            return result
        return expand(
            fixed_height_placement if top_transfer_mode else height_rear(values),
            LEG_ACTIVE if top_transfer_mode or height_joint_mode else search_active,
        )

    def frog_prejump(values: np.ndarray) -> np.ndarray:
        result = np.zeros(12, dtype=np.float64)
        if frog_joint_mode:
            result[6 + FRONT_ACTIVE] = values[: LEG_ACTIVE.size]
        return result

    def evaluate(values: np.ndarray, label: str, *, trace: Path | None = None, gif: Path | None = None) -> dict[str, Any]:
        return rollout(
            agent,
            paths["phase"],
            candidate,
            steering,
            label=label,
            depth_m=float(args.depth_m),
            command_mps=0.65,
            initial_state=(0.0, 0.0),
            rollout_seed=int(args.rollout_seed),
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
            late_post_height_vector=height_post(values),
            late_post_height_front_leg_vector=(
                expand_front(height_front(values))
                if height_joint_mode
                else None
            ),
            late_post_height_trigger_m=float(args.height_trigger_m),
            late_post_height_delay_after_front_seconds=(
                None
                if args.height_delay_after_front_seconds is None
                else float(args.height_delay_after_front_seconds)
            ),
            late_post_height_hold_seconds=float(args.placement_hold_seconds),
            late_post_top_vector=(
                expand(values[: LEG_WHEEL_ACTIVE.size], LEG_WHEEL_ACTIVE)
                if joint_coordination_mode
                else (
                    expand(fixed_top_transfer, LEG_WHEEL_ACTIVE)
                    if front_coordination_mode
                    else (expand(values, search_active) if top_transfer_mode else None)
                )
            ),
            late_post_top_front_leg_vector=(
                expand_front(values[LEG_WHEEL_ACTIVE.size :])
                if joint_coordination_mode
                else (expand_front(values) if front_coordination_mode else None)
            ),
            late_post_top_trigger_m=float(args.top_transfer_trigger_m),
            late_post_top_hold_seconds=float(args.top_transfer_hold_seconds),
            late_post_top_score_delay_seconds=float(
                args.top_transfer_score_delay_seconds
            ),
            late_post_top_leg_rate_per_second=float(
                args.top_transfer_leg_rate_per_second
            ),
            late_momentum_leg_vector=preload_vector,
            late_momentum_leg_delay_seconds=preload_delay_seconds,
            late_momentum_leg_hold_seconds=preload_hold_seconds,
            late_momentum_leg_rate_per_second=4.0,
            late_prejump_leg_vector=frog_prejump(values),
            late_prejump_leg_delay_seconds=float(args.prejump_delay_seconds),
            late_prejump_leg_hold_seconds=float(args.prejump_hold_seconds),
            late_prejump_leg_rate_per_second=5.0,
            oracle_front_wheel_action=-0.5,
            oracle_front_anchor_delay_seconds=0.82,
            leg_peak_nm=76.4,
            wheel_peak_nm=21.6,
            max_continuous_exceedance_s=0.5,
            trace_path=trace,
            gif_path=gif,
        )

    def key(report: dict[str, Any]) -> tuple[float, ...]:
        if top_transfer_mode:
            return (
                float(report["success"] and not report["hard_failure"]),
                -float(report["hard_failure"]),
                float(report["rear_top_latched"]),
                optional_float(report["peak_rear_top_proximity_after_top_m"]),
                optional_float(report["peak_rear_progress_after_top_m"]),
                optional_float(report["rear_progress_at_best_proximity_after_top_m"]),
                optional_float(report["rear_height_at_best_proximity_after_top_m"]),
                float(report["front_support_rate_after_top"]),
                -float(report["max_abs_roll_deg"]),
            )
        return (
            float(report["success"] and not report["hard_failure"]),
            -float(report["hard_failure"]),
            float(report["rear_top_latched"]),
            optional_float(report["peak_rear_top_proximity_after_height_m"]),
            optional_float(report["peak_rear_progress_above_height_gate_m"]),
            optional_float(report["rear_progress_at_best_proximity_after_height_m"]),
            optional_float(report["rear_height_at_best_proximity_after_height_m"]),
            float(report["front_support_rate_after_height"]),
            -float(report["max_abs_roll_deg"]),
        )

    baseline_values = (
        warm_start.copy()
        if joint_coordination_mode or height_joint_mode
        else np.zeros(search_active.size, dtype=np.float64)
    )
    baseline = evaluate(
        baseline_values,
        "baseline_locked_warm_start"
        if joint_coordination_mode or height_joint_mode
        else "baseline_zero_placement",
        trace=output / "baseline_trace.npz",
    )
    if frog_joint_mode:
        assert height_warm_report is not None
        source_prejump_delay = height_warm_constraints.get("prejump_delay_seconds")
        source_prejump_hold = height_warm_constraints.get("prejump_hold_seconds")
        source_height_delay = height_warm_constraints.get(
            "height_delay_after_front_seconds"
        )
        timing_matches = (
            source_prejump_delay is not None
            and source_prejump_hold is not None
            and source_height_delay is not None
            and math.isclose(
                float(source_prejump_delay),
                float(args.prejump_delay_seconds),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and math.isclose(
                float(source_prejump_hold),
                float(args.prejump_hold_seconds),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            and args.height_delay_after_front_seconds is not None
            and math.isclose(
                float(source_height_delay),
                float(args.height_delay_after_front_seconds),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        if timing_matches:
            expected = {
                "rear_top_latched": bool(height_warm_report["rear_top_latched"]),
                "height_stage_trigger_step": int(
                    height_warm_report["height_stage_trigger_step"]
                ),
                "peak_rear_top_proximity_after_height_m": float(
                    height_warm_report["peak_rear_top_proximity_after_height_m"]
                ),
                "rear_height_at_best_proximity_after_height_m": float(
                    height_warm_report["rear_height_at_best_proximity_after_height_m"]
                ),
            }
            checks = {
                "rear_top_matches": bool(baseline["rear_top_latched"])
                == expected["rear_top_latched"],
                "height_trigger_matches": int(baseline["height_stage_trigger_step"])
                == expected["height_stage_trigger_step"],
                "proximity_matches": math.isclose(
                    float(baseline["peak_rear_top_proximity_after_height_m"]),
                    expected["peak_rear_top_proximity_after_height_m"],
                    rel_tol=0.0,
                    abs_tol=1.0e-10,
                ),
                "height_matches": math.isclose(
                    float(baseline["rear_height_at_best_proximity_after_height_m"]),
                    expected["rear_height_at_best_proximity_after_height_m"],
                    rel_tol=0.0,
                    abs_tol=1.0e-10,
                ),
            }
        else:
            expected = {
                "source_summary": str(paths["warm_start_height_summary"]),
                "timing_changed": True,
                "evaluation_depth_m": float(args.depth_m),
            }
            checks = {
                "evaluation_depth_matches": math.isclose(
                    float(baseline["depth_m"]),
                    float(args.depth_m),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ),
                "height_trigger_observed": baseline["height_stage_trigger_step"]
                is not None,
                "baseline_executed": int(baseline["episode_steps"]) > 0,
            }
    elif height_joint_mode:
        expected = {
            "source_summary": str(paths.get("warm_start_height_summary", "none")),
            "evaluation_depth_m": float(args.depth_m),
        }
        checks = {
            "evaluation_depth_matches": math.isclose(
                float(baseline["depth_m"]),
                float(args.depth_m),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "height_trigger_observed": baseline["height_stage_trigger_step"] is not None,
            "baseline_executed": int(baseline["episode_steps"]) > 0,
        }
    elif joint_coordination_mode:
        assert warm_front_report is not None
        expected = {
            "rear_top_latched": bool(warm_front_report["rear_top_latched"]),
            "peak_rear_top_proximity_after_top_m": float(
                warm_front_report["peak_rear_top_proximity_after_top_m"]
            ),
            "peak_rear_progress_after_top_m": float(
                warm_front_report["peak_rear_progress_after_top_m"]
            ),
        }
        checks = {
            "rear_top_matches": bool(baseline["rear_top_latched"])
            == expected["rear_top_latched"],
            "height_trigger_observed": baseline["height_stage_trigger_step"] is not None,
            "top_trigger_observed": baseline["top_stage_trigger_step"] is not None,
            "top_proximity_matches": math.isclose(
                float(baseline["peak_rear_top_proximity_after_top_m"]),
                expected["peak_rear_top_proximity_after_top_m"],
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ),
            "top_progress_matches": math.isclose(
                float(baseline["peak_rear_progress_after_top_m"]),
                expected["peak_rear_progress_after_top_m"],
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ),
        }
    elif front_coordination_mode:
        assert fixed_top_report is not None
        expected = {
            "rear_top_latched": bool(fixed_top_report["rear_top_latched"]),
            "peak_rear_top_proximity_after_top_m": float(
                fixed_top_report["peak_rear_top_proximity_after_top_m"]
            ),
            "peak_rear_progress_after_top_m": float(
                fixed_top_report["peak_rear_progress_after_top_m"]
            ),
        }
        checks = {
            "rear_top_matches": bool(baseline["rear_top_latched"])
            == expected["rear_top_latched"],
            "height_trigger_observed": baseline["height_stage_trigger_step"] is not None,
            "top_trigger_observed": baseline["top_stage_trigger_step"] is not None,
            "top_proximity_matches": math.isclose(
                float(baseline["peak_rear_top_proximity_after_top_m"]),
                expected["peak_rear_top_proximity_after_top_m"],
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ),
            "top_progress_matches": math.isclose(
                float(baseline["peak_rear_progress_after_top_m"]),
                expected["peak_rear_progress_after_top_m"],
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ),
        }
    elif top_transfer_mode:
        assert fixed_height_report is not None
        expected = {
            "rear_top_latched": bool(fixed_height_report["rear_top_latched"]),
            "peak_rear_top_proximity_after_height_m": float(
                fixed_height_report["peak_rear_top_proximity_after_height_m"]
            ),
            "peak_rear_progress_above_height_gate_m": float(
                fixed_height_report["peak_rear_progress_above_height_gate_m"]
            ),
        }
        checks = {
            "rear_top_matches": bool(baseline["rear_top_latched"])
            == expected["rear_top_latched"],
            "height_trigger_observed": baseline["height_stage_trigger_step"] is not None,
            "top_trigger_observed": baseline["top_stage_trigger_step"] is not None,
            "proximity_matches": math.isclose(
                float(baseline["peak_rear_top_proximity_after_height_m"]),
                expected["peak_rear_top_proximity_after_height_m"],
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ),
            "height_crossing_progress_matches": math.isclose(
                float(baseline["peak_rear_progress_above_height_gate_m"]),
                expected["peak_rear_progress_above_height_gate_m"],
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ),
        }
    elif not math.isclose(args.depth_m, 0.345, rel_tol=0.0, abs_tol=1.0e-12):
        expected = {
            "source_depth_m": 0.345,
            "evaluation_depth_m": float(args.depth_m),
        }
        checks = {
            "evaluation_depth_matches": math.isclose(
                float(baseline["depth_m"]),
                float(args.depth_m),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "baseline_executed": int(baseline["episode_steps"]) > 0,
        }
    else:
        expected = {
            "rear_top_latched": False,
            "peak_rear_height_after_second_m": 0.5438989022580232,
            "peak_rear_progress_after_second_m": 1.6425254392203774,
        }
        checks = {
            "rear_top_matches": bool(baseline["rear_top_latched"])
            == expected["rear_top_latched"],
            "height_trigger_observed": baseline["height_stage_trigger_step"] is not None,
            "height_matches": math.isclose(
                float(baseline["peak_rear_height_after_second_m"]),
                expected["peak_rear_height_after_second_m"],
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ),
            "progress_matches": math.isclose(
                float(baseline["peak_rear_progress_after_second_m"]),
                expected["peak_rear_progress_after_second_m"],
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ),
        }
    if not all(checks.values()):
        (output / "baseline_mismatch.json").write_text(
            json.dumps({"checks": checks, "baseline": baseline}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"locked baseline did not reproduce: {checks}")

    if frog_wheel_mode:
        bounds = np.asarray(
            [float(args.prejump_bound)] * 4
            + [float(args.placement_bound)] * 4
            + [0.20] * 2
            + [float(args.front_leg_bound)] * 4,
            dtype=np.float64,
        )
    elif frog_joint_mode:
        bounds = np.asarray(
            [float(args.prejump_bound)] * 4
            + [float(args.placement_bound)] * 4
            + [float(args.front_leg_bound)] * 4,
            dtype=np.float64,
        )
    elif height_joint_mode:
        bounds = np.asarray(
            [float(args.placement_bound)] * 4
            + [float(args.front_leg_bound)] * 4,
            dtype=np.float64,
        )
    elif joint_coordination_mode:
        bounds = np.asarray(
            [float(args.placement_bound)] * 4
            + [float(args.wheel_bound)] * 2
            + [float(args.front_leg_bound)] * 4,
            dtype=np.float64,
        )
    elif front_coordination_mode:
        bounds = np.full(
            search_active.size, float(args.front_leg_bound), dtype=np.float64
        )
    else:
        bounds = np.asarray(
            [float(args.placement_bound)] * 4
            + ([float(args.wheel_bound)] * 2 if args.include_rear_wheels else []),
            dtype=np.float64,
        )
    warm_start = np.clip(warm_start, -bounds, bounds)
    mean = warm_start.copy()
    if frog_wheel_mode:
        std = np.asarray(
            [0.10] * 4 + [0.10] * 4 + [0.05] * 2 + [0.10] * 4,
            dtype=np.float64,
        )
    elif frog_joint_mode:
        std = np.asarray(
            [0.10] * 4 + [0.10] * 4 + [0.10] * 4,
            dtype=np.float64,
        )
    elif height_joint_mode:
        std = np.asarray([0.10] * 4 + [0.10] * 4, dtype=np.float64)
    elif joint_coordination_mode:
        std = np.asarray([0.10] * 4 + [0.16] * 2 + [0.10] * 4, dtype=np.float64)
    elif front_coordination_mode:
        std = np.full(
            search_active.size,
            min(0.18, 0.55 * float(args.front_leg_bound)),
            dtype=np.float64,
        )
    else:
        std = np.asarray(
            [min(0.22, 0.55 * float(args.placement_bound))] * 4
            + (
                [min(0.32, 0.55 * float(args.wheel_bound))] * 2
                if args.include_rear_wheels
                else []
            ),
            dtype=np.float64,
        )
    floor = np.full(search_active.size, 0.02, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    all_ranked: list[tuple[dict[str, Any], np.ndarray]] = []
    iterations: list[dict[str, Any]] = []

    for iteration in range(1, args.iterations + 1):
        if joint_coordination_mode or height_joint_mode:
            population = [warm_start.copy()] if iteration == 1 else [warm_start.copy(), mean.copy()]
        else:
            population = (
                seed_primitives(bounds, args.include_rear_wheels)
                if iteration == 1
                else [np.zeros(search_active.size), mean.copy()]
            )
        if not any(np.array_equal(values, warm_start) for values in population):
            population.append(warm_start.copy())
        while len(population) < args.population:
            population.append(
                np.clip(
                    mean + std * rng.normal(size=search_active.size),
                    -bounds,
                    bounds,
                )
            )
        population = population[: args.population]

        def run(item: tuple[int, np.ndarray]) -> tuple[int, dict[str, Any]]:
            index, values = item
            return index, evaluate(values, f"iter{iteration:02d}_candidate{index:03d}")

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            evaluated = list(executor.map(run, enumerate(population)))
        ranked = [(report, population[index]) for index, report in evaluated]
        ranked.sort(key=lambda item: key(item[0]), reverse=True)
        all_ranked.extend(ranked)
        eligible = [item for item in ranked if not item[0]["hard_failure"]]
        elite_count = max(2, int(math.ceil(args.population * args.elite_fraction)))
        elites = eligible[:elite_count]
        update = len(elites) >= 2
        if update:
            elite_values = np.stack([values for _, values in elites])
            rate = float(args.cem_update_rate)
            mean = np.clip(
                (1.0 - rate) * mean + rate * elite_values.mean(axis=0),
                -bounds,
                bounds,
            )
            std = np.maximum((1.0 - rate) * std + rate * elite_values.std(axis=0), floor)
        best = ranked[0][0]
        proximity_name = (
            "peak_rear_top_proximity_after_top_m"
            if top_transfer_mode
            else "peak_rear_top_proximity_after_height_m"
        )
        progress_name = (
            "peak_rear_progress_after_top_m"
            if top_transfer_mode
            else "peak_rear_progress_above_height_gate_m"
        )
        iterations.append(
            {
                "iteration": iteration,
                "eligible_count": len(eligible),
                "best_vector": ranked[0][1].tolist(),
                "best": best,
            }
        )
        np.savez_compressed(output / f"population_iter_{iteration:04d}.npz", values=np.stack(population))
        print(
            f"[{'frog-joint-coordination' if frog_joint_mode else ('height-joint-coordination' if height_joint_mode else ('joint-coordination' if joint_coordination_mode else ('front-coordination' if front_coordination_mode else ('rear-top-transfer' if top_transfer_mode else 'rear-placement'))))}] "
            f"iter={iteration}/{args.iterations} "
            f"success={int(best['success'])} rear={int(best['rear_top_latched'])} "
            f"prox={best[proximity_name]} progress={best[progress_name]} "
            f"roll={best['max_abs_roll_deg']:.2f} hard={int(best['hard_failure'])}",
            flush=True,
        )
        if best["success"] or not update:
            break

    best_report, best_vector = max(all_ranked, key=lambda item: key(item[0]))
    depth_tag = f"{args.depth_m:.6f}".replace(".", "")
    gif_path = output / "gifs" / (
        f"best_oracle_success_depth{depth_tag}.gif"
        if best_report["success"]
        else f"best_oracle_failure_depth{depth_tag}.gif"
    )
    replay = evaluate(
        best_vector,
        "best_replay",
        trace=output / "best_trace.npz",
        gif=gif_path,
    )
    result_name = (
        "best_frog_joint_coordination.npz"
        if frog_joint_mode
        else (
            "best_height_joint_coordination.npz"
            if height_joint_mode
            else (
                "best_joint_coordination.npz"
                if joint_coordination_mode
                else (
                    "best_front_coordination.npz"
                    if front_coordination_mode
                    else (
                        "best_rear_top_transfer.npz"
                        if top_transfer_mode
                        else "best_rear_placement.npz"
                    )
                )
            )
        )
    )
    result_key = (
        "frog_joint_coordination"
        if frog_joint_mode
        else (
            "height_joint_coordination"
            if height_joint_mode
            else (
                "joint_coordination"
                if joint_coordination_mode
                else (
                    "front_coordination"
                    if front_coordination_mode
                    else ("transfer" if top_transfer_mode else "placement")
                )
            )
        )
    )
    np.savez_compressed(output / result_name, **{result_key: best_vector})
    after_assets = asset_hashes()
    if before_assets != after_assets:
        raise RuntimeError("official assets changed during rear-placement search")
    summary = {
        "purpose": (
            f"two-phase frog preload/release front/rear coordination oracle at {args.depth_m:.6f} m"
            if frog_joint_mode
            else (
                f"height-triggered joint front/rear sagittal coordination oracle at {args.depth_m:.6f} m"
                if height_joint_mode
                else (
                    f"top-triggered joint front/rear sagittal coordination oracle at {args.depth_m:.6f} m"
                    if joint_coordination_mode
                    else (
                        f"top-triggered front sagittal coordination oracle at {args.depth_m:.6f} m"
                        if front_coordination_mode
                        else (
                            f"top-height-triggered rear sagittal transfer oracle at {args.depth_m:.6f} m"
                            if top_transfer_mode
                            else f"height-triggered rear sagittal placement oracle at {args.depth_m:.6f} m"
                        )
                    )
                )
            )
        ),
        "mechanical_feasibility_only": True,
        "deployable": False,
        "geometry_scope": "official-derived diagonal vertical-wall proxy pit, not full official track mesh",
        "depth_m": float(args.depth_m),
        "reference_checkpoint": str(paths["reference"]),
        "reference_checkpoint_sha256": sha256(paths["reference"]),
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items() if name != "reference"},
        "official_assets_before": before_assets,
        "official_assets_after": after_assets,
        "constraints": {
            "continuous_torque_limits_nm": {"leg": 50.0, "wheel": 14.0},
            "research_peak_torque_limits_nm": {"leg": 76.4, "wheel": 21.6},
            "competition_peak_threshold_confirmed": False,
            "max_continuous_exceedance_s": 0.5,
            "500_n": "report_only",
            "1200_n": "hard_failure",
            "roll_25_deg": "hard_failure",
            "body_wall_contact": "hard_failure",
            "active_post_indices": search_active.tolist(),
            "search_vector_layout": (
                "[pre HL hipy, pre HL knee, pre HR hipy, pre HR knee, release HL hipy, release HL knee, release HR hipy, release HR knee, HL wheel, HR wheel, FL hipy, FL knee, FR hipy, FR knee]"
                if frog_wheel_mode
                else (
                    "[pre HL hipy, pre HL knee, pre HR hipy, pre HR knee, release HL hipy, release HL knee, release HR hipy, release HR knee, FL hipy, FL knee, FR hipy, FR knee]"
                    if frog_joint_mode
                    else (
                        "[HL hipy, HL knee, HR hipy, HR knee, FL hipy, FL knee, FR hipy, FR knee]"
                        if height_joint_mode
                        else (
                            "[HL hipy, HL knee, HR hipy, HR knee, HL wheel, HR wheel, FL hipy, FL knee, FR hipy, FR knee]"
                            if joint_coordination_mode
                            else None
                        )
                    )
                )
            ),
            "joint_coordination_searched": bool(
                height_joint_mode or joint_coordination_mode
            ),
            "front_coordination_searched": bool(
                height_joint_mode or front_coordination_mode or joint_coordination_mode
            ),
            "height_trigger_m": float(args.height_trigger_m),
            "height_delay_after_front_seconds": args.height_delay_after_front_seconds,
            "placement_hold_seconds": float(args.placement_hold_seconds),
            "prejump_bound_raw": (
                float(args.prejump_bound) if frog_joint_mode else None
            ),
            "prejump_delay_seconds": (
                float(args.prejump_delay_seconds) if frog_joint_mode else None
            ),
            "prejump_hold_seconds": (
                float(args.prejump_hold_seconds) if frog_joint_mode else None
            ),
            "top_transfer_hold_seconds": float(args.top_transfer_hold_seconds),
            "top_transfer_trigger_m": float(args.top_transfer_trigger_m),
            "top_transfer_score_delay_seconds": float(
                args.top_transfer_score_delay_seconds
            ),
            "top_transfer_leg_rate_per_second": float(
                args.top_transfer_leg_rate_per_second
            ),
            "placement_bound_raw": float(args.placement_bound),
            "wheel_bound_raw": (
                0.20
                if frog_wheel_mode
                else (float(args.wheel_bound) if args.include_rear_wheels else None)
            ),
            "front_leg_bound_raw": (
                float(args.front_leg_bound)
                if height_joint_mode or front_coordination_mode or joint_coordination_mode
                else None
            ),
            "rear_wheels_searched": bool(
                args.include_rear_wheels or frog_wheel_mode
            ),
            "fixed_height_placement": (
                fixed_height_placement.tolist() if top_transfer_mode else None
            ),
            "warm_start_transfer": warm_start.tolist(),
            "warm_start_height_rear": (
                height_rear(warm_start).tolist() if height_joint_mode else None
            ),
            "warm_start_height_front": (
                height_front(warm_start).tolist() if height_joint_mode else None
            ),
            "warm_start_prejump_rear": (
                warm_start[: LEG_ACTIVE.size].tolist() if frog_joint_mode else None
            ),
            "fixed_top_transfer": (
                fixed_top_transfer.tolist() if front_coordination_mode else None
            ),
            "warm_start_rear_transfer": (
                warm_start[: LEG_WHEEL_ACTIVE.size].tolist()
                if joint_coordination_mode
                else None
            ),
            "warm_start_front_coordination": (
                warm_start[LEG_WHEEL_ACTIVE.size :].tolist()
                if joint_coordination_mode
                else None
            ),
            "fixed_preload_vector": preload_vector.tolist(),
            "fixed_preload_delay_seconds": preload_delay_seconds,
            "fixed_preload_hold_seconds": preload_hold_seconds,
        },
        "optimizer": {
            "population": int(args.population),
            "requested_iterations": int(args.iterations),
            "completed_iterations": len(iterations),
            "workers": int(args.workers),
            "seed": int(args.seed),
        },
        "baseline_expected": expected,
        "baseline_checks": checks,
        "baseline": baseline,
        "iterations": iterations,
        "best_vector": best_vector.tolist(),
        "best_search_report": best_report,
        "best_replay": replay,
        "best_replay_matches": key(replay) == key(best_report),
        "outputs": {
            result_key: str(output / result_name),
            "baseline_trace": str(output / "baseline_trace.npz"),
            "best_trace": str(output / "best_trace.npz"),
            "gif": str(gif_path),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[result] success={int(replay['success'])} rear={int(replay['rear_top_latched'])} "
        f"prox={replay[proximity_name]} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
