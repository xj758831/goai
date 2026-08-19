#!/usr/bin/env python3
"""Search an early bounded wheel-differential stage before two-stage rescue."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from search_s10_m20_approach_two_stage_rescue import (
    ACTION_DIM,
    CONTROL_DT,
    DWELL_STEPS,
    HARD_FAILURES,
    PHASE_ESTIMATOR,
    PHASE_THRESHOLD,
    POST_ACTIVE,
    PRE_ACTIVE,
    PROJECT_ROOT,
    REFERENCE_CHECKPOINT,
    S10M20PhaseGatedEnv,
    TERRAIN_OBS_DIM,
    GifRecorder,
    asset_hashes,
    candidate_key,
    limits,
    load_reference,
    rank_and_score,
    sha256,
    smoothstep,
)
from s10_m20_curriculum_env import (
    CONTACT_THRESHOLD_N,
    PIT_EXIT_ALONG_M,
    PLATFORM_Z,
    WHEEL_RADIUS_M,
)
from s10_m20_lidar_torque_envelope_env import (
    LEG_CONTINUOUS_LIMIT_NM,
    MAX_CONTINUOUS_EXCEEDANCE_S,
    WHEEL_CONTINUOUS_LIMIT_NM,
)
from s10_m20_phase_gated_torque_envelope_env import (
    S10M20PhaseGatedTorqueEnvelopeEnv,
)


DEFAULT_RESCUE_CANDIDATE = Path(
    "logs/mujoco/s10_m20_two_stage_selector_observability_19cases_20260810_v2/"
    "selector_gate.npz"
)
EARLY_WHEEL_INDICES = np.asarray([12, 13, 14, 15], dtype=np.int64)
WHEEL_RATE_PER_SECOND = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, default=REFERENCE_CHECKPOINT)
    parser.add_argument("--phase-estimator", type=Path, default=PHASE_ESTIMATOR)
    parser.add_argument("--rescue-candidate", type=Path, default=DEFAULT_RESCUE_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--depth-m", type=float, default=0.21)
    parser.add_argument("--command-mps", type=float, default=0.8)
    parser.add_argument("--lateral-m", type=float, default=0.03)
    parser.add_argument("--yaw-deg", type=float, default=-3.0)
    parser.add_argument("--rollout-seed", type=int, default=41_001)
    parser.add_argument("--early-terrain-threshold", type=float, default=0.05)
    parser.add_argument("--approach-terrain-threshold", type=float, default=0.03)
    parser.add_argument(
        "--prelift-from-early",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="start the six front-preparation residuals at the early terrain gate",
    )
    parser.add_argument("--wheel-bound", type=float, default=0.60)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--cem-update-rate", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--gif-frame-period", type=float, default=0.08)
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--leg-rate-per-second", type=float, default=0.65)
    parser.add_argument(
        "--post-from-approach", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--post-delay-seconds", type=float, default=0.25)
    parser.add_argument("--front-post-delay-seconds", type=float, default=0.0)
    parser.add_argument("--late-post-boost-scale", type=float, default=0.0)
    parser.add_argument("--late-post-boost-delay-seconds", type=float, default=1.0)
    parser.add_argument("--late-post-boost-hold-seconds", type=float, default=0.35)
    parser.add_argument("--late-post-target-limit", type=float, default=1.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.population < 8 or args.iterations < 1 or args.workers < 1:
        raise ValueError("population must be >=8; iterations and workers must be positive")
    if not 0.03 <= args.early_terrain_threshold <= 0.10:
        raise ValueError("early terrain threshold must be in [0.03, 0.10]")
    if not 0.0 < args.approach_terrain_threshold <= args.early_terrain_threshold:
        raise ValueError("approach threshold must be positive and no larger than early threshold")
    if not 0.05 <= args.wheel_bound <= 1.0:
        raise ValueError("wheel bound must be in [0.05, 1.0]")
    if args.episode_seconds < 3.0:
        raise ValueError("episode-seconds must be at least 3")
    if not 0.1 <= args.leg_rate_per_second <= 5.0:
        raise ValueError("leg-rate-per-second must be in [0.1, 5.0]")
    if args.post_delay_seconds < 0.0:
        raise ValueError("post-delay-seconds must be non-negative")
    if not 0.05 <= args.elite_fraction <= 0.75 or not 0.0 < args.cem_update_rate <= 1.0:
        raise ValueError("invalid CEM parameters")
    reference = args.reference_checkpoint.expanduser().resolve()
    phase = args.phase_estimator.expanduser().resolve()
    rescue_path = args.rescue_candidate.expanduser().resolve()
    for path in (reference, phase, rescue_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = PROJECT_ROOT / "logs" / "mujoco" / f"s10_m20_early_steering_{stamp}"
    else:
        output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return reference, phase, rescue_path, output


def load_candidate(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if "candidate" not in payload:
        raise ValueError(f"candidate array missing from {path}")
    candidate = np.asarray(payload["candidate"], dtype=np.float64)
    if candidate.shape != (14,) or not np.isfinite(candidate).all():
        raise ValueError(f"invalid rescue candidate shape/value: {candidate.shape}")
    return candidate


def structured_seeds(bound: float) -> list[tuple[str, np.ndarray]]:
    values = (
        ("zero", (0.0, 0.0)),
        ("left_faster", (0.30, -0.30)),
        ("right_faster", (-0.30, 0.30)),
        ("left_only_forward", (0.30, 0.0)),
        ("right_only_forward", (0.0, 0.30)),
        ("left_only_reverse", (-0.30, 0.0)),
        ("right_only_reverse", (0.0, -0.30)),
        ("common_forward", (0.20, 0.20)),
        ("common_reverse", (-0.20, -0.20)),
    )
    return [
        (name, np.clip(np.asarray(value, dtype=np.float64), -bound, bound))
        for name, value in values
    ]


def rollout(
    agent: Any,
    phase_path: Path,
    rescue_candidate: np.ndarray,
    steering: np.ndarray,
    *,
    label: str,
    depth_m: float,
    command_mps: float,
    initial_state: tuple[float, float],
    rollout_seed: int,
    early_threshold: float,
    approach_threshold: float,
    prelift_from_early: bool = False,
    episode_seconds: float = 6.0,
    leg_rate_per_second: float = 0.65,
    post_from_approach: bool = False,
    post_delay_seconds: float = 0.25,
    front_post_delay_seconds: float = 0.0,
    late_post_boost_scale: float = 0.0,
    late_post_boost_delay_seconds: float = 1.0,
    late_post_boost_hold_seconds: float = 0.35,
    late_post_boost_vector: np.ndarray | None = None,
    late_post_target_limit: float = 1.0,
    late_post_second_vector: np.ndarray | None = None,
    late_post_second_delay_seconds: float = 1.30,
    late_post_second_hold_seconds: float = 0.50,
    late_post_height_vector: np.ndarray | None = None,
    late_post_height_front_leg_vector: np.ndarray | None = None,
    late_post_height_trigger_m: float = 0.50,
    late_post_height_delay_after_front_seconds: float | None = None,
    late_post_height_hold_seconds: float = 0.40,
    late_post_height_dwell_steps: int = 1,
    late_post_top_vector: np.ndarray | None = None,
    late_post_top_front_leg_vector: np.ndarray | None = None,
    late_post_top_trigger_m: float | None = None,
    late_post_top_hold_seconds: float = 0.35,
    late_post_top_dwell_steps: int = 1,
    late_post_top_score_delay_seconds: float = 0.12,
    late_post_top_leg_rate_per_second: float | None = None,
    precontact_leg_vector: np.ndarray | None = None,
    precontact_leg_delay_after_approach_seconds: float = 1.20,
    precontact_leg_hold_seconds: float = 0.20,
    precontact_leg_rate_per_second: float | None = None,
    threepoint_swing_leg_vector: np.ndarray | None = None,
    threepoint_delay_seconds: float = 0.0,
    threepoint_hold_seconds: float = 0.20,
    threepoint_leg_rate_per_second: float | None = None,
    threepoint_monitor_enabled: bool = False,
    threepoint_freeze_leading_leg: bool = False,
    threepoint_min_leading_rear_contact_force_n: float = 50.0,
    threepoint_max_abs_roll_deg: float = 10.0,
    threepoint_trigger_dwell_steps: int = 1,
    pullover_launch_leg_vector: np.ndarray | None = None,
    pullover_launch_delay_after_front_seconds: float = 0.0,
    pullover_launch_hold_seconds: float = 0.18,
    pullover_tuck_leg_vector: np.ndarray | None = None,
    pullover_tuck_delay_after_launch_seconds: float = 0.16,
    pullover_tuck_hold_seconds: float = 0.20,
    pullover_leg_rate_per_second: float | None = None,
    pullover_top_rear_wheel_vector: np.ndarray | None = None,
    pullover_top_rear_wheel_delay_seconds: float = 0.0,
    pullover_top_rear_wheel_hold_seconds: float = 0.16,
    allow_body_wall_contact: bool = False,
    late_momentum_leg_vector: np.ndarray | None = None,
    late_momentum_leg_delay_seconds: float = 0.0,
    late_momentum_leg_hold_seconds: float = 0.20,
    late_momentum_leg_rate_per_second: float | None = None,
    late_prejump_leg_vector: np.ndarray | None = None,
    late_prejump_leg_delay_seconds: float = 0.30,
    late_prejump_leg_hold_seconds: float = 0.14,
    late_prejump_leg_rate_per_second: float | None = None,
    late_momentum_wheel_vector: np.ndarray | None = None,
    late_momentum_wheel_delay_seconds: float = 0.0,
    late_momentum_wheel_hold_seconds: float = 0.40,
    late_front_wheel_vector: np.ndarray | None = None,
    late_front_wheel_delay_seconds: float = 0.60,
    late_front_wheel_hold_seconds: float = 1.20,
    late_front_leg_vector: np.ndarray | None = None,
    oracle_front_wheel_action: float | None = None,
    oracle_freeze_front_legs: bool = False,
    oracle_front_anchor_delay_seconds: float = 0.0,
    pre_fade_start_seconds: float = 0.65,
    pre_fade_duration_seconds: float = 0.35,
    leg_peak_nm: float = LEG_CONTINUOUS_LIMIT_NM,
    wheel_peak_nm: float = WHEEL_CONTINUOUS_LIMIT_NM,
    max_continuous_exceedance_s: float = MAX_CONTINUOUS_EXCEEDANCE_S,
    expected_early_step: int | None = None,
    expected_prefix_sha256: str | None = None,
    gif_path: Path | None = None,
    gif_period: float = 0.08,
    trace_path: Path | None = None,
    prebuilt_env: Any | None = None,
) -> dict[str, Any]:
    steering = np.asarray(steering, dtype=np.float64)
    if steering.shape != (2,) or not np.isfinite(steering).all():
        raise ValueError("steering must be a finite left/right pair")
    if pre_fade_start_seconds < 0.0 or pre_fade_duration_seconds <= 0.0:
        raise ValueError("pre fade start must be non-negative and duration positive")
    if front_post_delay_seconds < 0.0:
        raise ValueError("front-post delay must be non-negative")
    if not 0.0 <= late_post_boost_scale <= 0.35:
        raise ValueError("late post boost scale must be in [0, 0.35]")
    if late_post_boost_delay_seconds < 0.0 or late_post_boost_hold_seconds <= 0.0:
        raise ValueError("late post boost delay must be non-negative and hold positive")
    if not 1.0 <= late_post_target_limit <= 1.5:
        raise ValueError("late post target limit must be in [1.0, 1.5]")
    if late_post_boost_vector is None:
        boost_vector = late_post_boost_scale * rescue_candidate[6:]
    else:
        boost_vector = np.asarray(late_post_boost_vector, dtype=np.float64)
        if boost_vector.shape != (8,) or not np.isfinite(boost_vector).all():
            raise ValueError("late post boost vector must be a finite 8-vector")
        if late_post_boost_scale != 0.0:
            raise ValueError("use either late post boost scale or vector, not both")
    second_stage_configured = late_post_second_vector is not None
    if late_post_second_vector is None:
        second_boost_vector = np.zeros(8, dtype=np.float64)
    else:
        second_boost_vector = np.asarray(late_post_second_vector, dtype=np.float64)
        if second_boost_vector.shape != (8,) or not np.isfinite(second_boost_vector).all():
            raise ValueError("late post second vector must be a finite 8-vector")
    if late_post_second_delay_seconds < 0.0 or late_post_second_hold_seconds <= 0.0:
        raise ValueError("late post second delay must be non-negative and hold positive")
    height_stage_configured = (
        late_post_height_vector is not None
        or late_post_height_front_leg_vector is not None
    )
    if late_post_height_vector is None:
        height_boost_vector = np.zeros(8, dtype=np.float64)
    else:
        height_boost_vector = np.asarray(late_post_height_vector, dtype=np.float64)
        if height_boost_vector.shape != (8,) or not np.isfinite(height_boost_vector).all():
            raise ValueError("late post height vector must be a finite 8-vector")
    if late_post_height_front_leg_vector is None:
        height_front_leg_vector = np.zeros(6, dtype=np.float64)
    else:
        height_front_leg_vector = np.asarray(
            late_post_height_front_leg_vector, dtype=np.float64
        )
        if (
            height_front_leg_vector.shape != (6,)
            or not np.isfinite(height_front_leg_vector).all()
        ):
            raise ValueError(
                "late post height front-leg vector must be a finite 6-vector"
            )
    if not math.isfinite(late_post_height_trigger_m) or late_post_height_trigger_m <= 0.0:
        raise ValueError("late post height trigger must be finite and positive")
    if late_post_height_delay_after_front_seconds is not None and (
        not math.isfinite(late_post_height_delay_after_front_seconds)
        or late_post_height_delay_after_front_seconds < 0.0
    ):
        raise ValueError("late post height delay after front must be finite and non-negative")
    if late_post_height_hold_seconds <= 0.0:
        raise ValueError("late post height hold must be positive")
    if late_post_height_dwell_steps < 1:
        raise ValueError("late post height dwell steps must be positive")
    top_stage_configured = late_post_top_vector is not None
    if late_post_top_vector is None:
        top_boost_vector = np.zeros(8, dtype=np.float64)
    else:
        top_boost_vector = np.asarray(late_post_top_vector, dtype=np.float64)
        if top_boost_vector.shape != (8,) or not np.isfinite(top_boost_vector).all():
            raise ValueError("late post top vector must be a finite 8-vector")
    if late_post_top_front_leg_vector is None:
        top_front_leg_vector = np.zeros(6, dtype=np.float64)
    else:
        top_front_leg_vector = np.asarray(
            late_post_top_front_leg_vector, dtype=np.float64
        )
        if (
            top_front_leg_vector.shape != (6,)
            or not np.isfinite(top_front_leg_vector).all()
        ):
            raise ValueError("late post top front-leg vector must be a finite 6-vector")
    top_trigger_m = (
        PLATFORM_Z + WHEEL_RADIUS_M - 0.025
        if late_post_top_trigger_m is None
        else float(late_post_top_trigger_m)
    )
    if not math.isfinite(top_trigger_m) or top_trigger_m <= 0.0:
        raise ValueError("late post top trigger must be finite and positive")
    if late_post_top_hold_seconds <= 0.0:
        raise ValueError("late post top hold must be positive")
    if late_post_top_dwell_steps < 1:
        raise ValueError("late post top dwell steps must be positive")
    if late_post_top_score_delay_seconds < 0.0:
        raise ValueError("late post top score delay must be non-negative")
    if late_post_top_leg_rate_per_second is not None and not (
        0.1 <= late_post_top_leg_rate_per_second <= 5.0
    ):
        raise ValueError("late post top leg rate must be in [0.1, 5.0]")
    if precontact_leg_vector is None:
        precontact_vector = np.zeros(12, dtype=np.float64)
    else:
        precontact_vector = np.asarray(precontact_leg_vector, dtype=np.float64)
        if precontact_vector.shape != (12,) or not np.isfinite(
            precontact_vector
        ).all():
            raise ValueError("precontact leg vector must be a finite 12-vector")
        if np.max(np.abs(precontact_vector)) > 0.30:
            raise ValueError("precontact leg residual exceeds the 0.30 raw-action bound")
    if (
        precontact_leg_delay_after_approach_seconds < 0.0
        or precontact_leg_hold_seconds <= 0.0
    ):
        raise ValueError("precontact leg delay must be non-negative and hold positive")
    if precontact_leg_rate_per_second is not None and not (
        0.1 <= precontact_leg_rate_per_second <= 5.0
    ):
        raise ValueError("precontact leg rate must be in [0.1, 5.0]")
    if threepoint_swing_leg_vector is None:
        threepoint_swing_vector = np.zeros(3, dtype=np.float64)
    else:
        threepoint_swing_vector = np.asarray(
            threepoint_swing_leg_vector, dtype=np.float64
        )
        if threepoint_swing_vector.shape != (3,) or not np.isfinite(
            threepoint_swing_vector
        ).all():
            raise ValueError("threepoint swing leg vector must be a finite 3-vector")
        if np.max(np.abs(threepoint_swing_vector)) > 0.30:
            raise ValueError("threepoint swing leg residual exceeds the 0.30 raw-action bound")
    if threepoint_delay_seconds < 0.0 or threepoint_hold_seconds <= 0.0:
        raise ValueError("threepoint delay must be non-negative and hold positive")
    if threepoint_leg_rate_per_second is not None and not (
        0.1 <= threepoint_leg_rate_per_second <= 5.0
    ):
        raise ValueError("threepoint leg rate must be in [0.1, 5.0]")
    if not math.isfinite(threepoint_min_leading_rear_contact_force_n) or not (
        0.0 <= threepoint_min_leading_rear_contact_force_n <= 1200.0
    ):
        raise ValueError("threepoint rear contact threshold must be in [0, 1200] N")
    if not math.isfinite(threepoint_max_abs_roll_deg) or not (
        0.0 < threepoint_max_abs_roll_deg <= 25.0
    ):
        raise ValueError("threepoint roll threshold must be in (0, 25] deg")
    if threepoint_trigger_dwell_steps < 1:
        raise ValueError("threepoint trigger dwell steps must be positive")
    if pullover_launch_leg_vector is None:
        pullover_launch_vector = np.zeros(12, dtype=np.float64)
    else:
        pullover_launch_vector = np.asarray(
            pullover_launch_leg_vector, dtype=np.float64
        )
        if pullover_launch_vector.shape != (12,) or not np.isfinite(
            pullover_launch_vector
        ).all():
            raise ValueError("pullover launch leg vector must be a finite 12-vector")
        if np.max(np.abs(pullover_launch_vector)) > 0.8:
            raise ValueError("pullover launch leg residual exceeds the 0.8 raw-action bound")
    if pullover_tuck_leg_vector is None:
        pullover_tuck_vector = np.zeros(12, dtype=np.float64)
    else:
        pullover_tuck_vector = np.asarray(pullover_tuck_leg_vector, dtype=np.float64)
        if pullover_tuck_vector.shape != (12,) or not np.isfinite(
            pullover_tuck_vector
        ).all():
            raise ValueError("pullover tuck leg vector must be a finite 12-vector")
        if np.max(np.abs(pullover_tuck_vector)) > 0.8:
            raise ValueError("pullover tuck leg residual exceeds the 0.8 raw-action bound")
    if (
        pullover_launch_delay_after_front_seconds < 0.0
        or pullover_launch_hold_seconds <= 0.0
        or pullover_tuck_delay_after_launch_seconds < 0.0
        or pullover_tuck_hold_seconds <= 0.0
    ):
        raise ValueError("pullover delays must be non-negative and holds positive")
    if pullover_leg_rate_per_second is not None and not (
        0.1 <= pullover_leg_rate_per_second <= 5.0
    ):
        raise ValueError("pullover leg rate must be in [0.1, 5.0]")
    if pullover_top_rear_wheel_vector is None:
        pullover_top_rear_wheel = np.zeros(2, dtype=np.float64)
    else:
        pullover_top_rear_wheel = np.asarray(
            pullover_top_rear_wheel_vector, dtype=np.float64
        )
        if pullover_top_rear_wheel.shape != (2,) or not np.isfinite(
            pullover_top_rear_wheel
        ).all():
            raise ValueError("pullover top rear-wheel vector must be a finite pair")
        if np.max(np.abs(pullover_top_rear_wheel)) > 3.0:
            raise ValueError("pullover top rear-wheel residual exceeds the 3.0 raw-action bound")
    if (
        pullover_top_rear_wheel_delay_seconds < 0.0
        or pullover_top_rear_wheel_hold_seconds <= 0.0
    ):
        raise ValueError("pullover top rear-wheel delay must be non-negative and hold positive")
    if late_momentum_leg_vector is None:
        momentum_leg_vector = np.zeros(12, dtype=np.float64)
    else:
        momentum_leg_vector = np.asarray(late_momentum_leg_vector, dtype=np.float64)
        if (
            momentum_leg_vector.shape != (12,)
            or not np.isfinite(momentum_leg_vector).all()
        ):
            raise ValueError("late momentum leg vector must be a finite 12-vector")
        if np.max(np.abs(momentum_leg_vector)) > 0.8:
            raise ValueError("late momentum leg residual exceeds the 0.8 raw-action bound")
    if (
        late_momentum_leg_delay_seconds < 0.0
        or late_momentum_leg_hold_seconds <= 0.0
    ):
        raise ValueError("late momentum leg delay must be non-negative and hold positive")
    if late_momentum_leg_rate_per_second is not None and not (
        0.1 <= late_momentum_leg_rate_per_second <= 5.0
    ):
        raise ValueError("late momentum leg rate must be in [0.1, 5.0]")
    if late_prejump_leg_vector is None:
        prejump_leg_vector = np.zeros(12, dtype=np.float64)
    else:
        prejump_leg_vector = np.asarray(late_prejump_leg_vector, dtype=np.float64)
        if (
            prejump_leg_vector.shape != (12,)
            or not np.isfinite(prejump_leg_vector).all()
        ):
            raise ValueError("late prejump leg vector must be a finite 12-vector")
        if np.max(np.abs(prejump_leg_vector)) > 0.8:
            raise ValueError("late prejump leg residual exceeds the 0.8 raw-action bound")
    if late_prejump_leg_delay_seconds < 0.0 or late_prejump_leg_hold_seconds <= 0.0:
        raise ValueError("late prejump leg delay must be non-negative and hold positive")
    if late_prejump_leg_rate_per_second is not None and not (
        0.1 <= late_prejump_leg_rate_per_second <= 5.0
    ):
        raise ValueError("late prejump leg rate must be in [0.1, 5.0]")
    if late_momentum_wheel_vector is None:
        momentum_wheel_vector = np.zeros(4, dtype=np.float64)
    else:
        momentum_wheel_vector = np.asarray(
            late_momentum_wheel_vector, dtype=np.float64
        )
        if (
            momentum_wheel_vector.shape != (4,)
            or not np.isfinite(momentum_wheel_vector).all()
        ):
            raise ValueError("late momentum wheel vector must be a finite 4-vector")
        if np.max(np.abs(momentum_wheel_vector)) > 3.0:
            raise ValueError("late momentum wheel residual exceeds the 3.0 raw-action bound")
    if (
        late_momentum_wheel_delay_seconds < 0.0
        or late_momentum_wheel_hold_seconds <= 0.0
    ):
        raise ValueError("late momentum wheel delay must be non-negative and hold positive")
    if late_front_wheel_vector is None:
        front_wheel_vector = np.zeros(2, dtype=np.float64)
    else:
        front_wheel_vector = np.asarray(late_front_wheel_vector, dtype=np.float64)
        if front_wheel_vector.shape != (2,) or not np.isfinite(front_wheel_vector).all():
            raise ValueError("late front wheel vector must be a finite 2-vector")
        if np.max(np.abs(front_wheel_vector)) > 3.0:
            raise ValueError("late front wheel residual exceeds the 3.0 raw-action bound")
    if late_front_wheel_delay_seconds < 0.0 or late_front_wheel_hold_seconds <= 0.0:
        raise ValueError("late front wheel delay must be non-negative and hold positive")
    if late_front_leg_vector is None:
        front_leg_vector = np.zeros(6, dtype=np.float64)
    else:
        front_leg_vector = np.asarray(late_front_leg_vector, dtype=np.float64)
        if front_leg_vector.shape != (6,) or not np.isfinite(front_leg_vector).all():
            raise ValueError("late front leg vector must be a finite 6-vector")
        if np.max(np.abs(front_leg_vector)) > 0.6:
            raise ValueError("late front leg residual exceeds the 0.6 raw-action bound")
    if oracle_front_wheel_action is not None:
        if not math.isfinite(oracle_front_wheel_action) or not -3.0 <= oracle_front_wheel_action <= 1.0:
            raise ValueError("oracle front wheel action must be in [-3.0, 1.0]")
    if oracle_front_anchor_delay_seconds < 0.0:
        raise ValueError("oracle front anchor delay must be non-negative")
    peak_requested = (
        leg_peak_nm > LEG_CONTINUOUS_LIMIT_NM
        or wheel_peak_nm > WHEEL_CONTINUOUS_LIMIT_NM
    )
    env_class = (
        S10M20PhaseGatedTorqueEnvelopeEnv
        if peak_requested
        else S10M20PhaseGatedEnv
    )
    env_class.phase_estimator_path = phase_path
    envelope_kwargs = (
        {
            "leg_peak_nm": leg_peak_nm,
            "wheel_peak_nm": wheel_peak_nm,
            "max_continuous_exceedance_s": max_continuous_exceedance_s,
            "allow_body_wall_contact": bool(allow_body_wall_contact),
        }
        if peak_requested
        else {}
    )
    if allow_body_wall_contact and not peak_requested:
        raise ValueError(
            "body-wall contact relaxation is available only in the audited torque-envelope environment"
        )
    owns_env = prebuilt_env is None
    if owns_env:
        env = env_class(
            depth_m=depth_m,
            command_mps=command_mps,
            training_states=(initial_state,),
            max_episode_steps=int(round(episode_seconds / CONTROL_DT)),
            **envelope_kwargs,
        )
    else:
        if not isinstance(prebuilt_env, env_class):
            raise TypeError(
                f"prebuilt environment must be an instance of {env_class.__name__}"
            )
        env = prebuilt_env
    recorder = GifRecorder(env.model, gif_period) if gif_path is not None else None
    early_streak = 0
    approach_streak = 0
    front_streak = 0
    early_step: int | None = None
    approach_step: int | None = None
    front_step: int | None = None
    true_front_step: int | None = None
    prefix_actions: list[np.ndarray] = []
    previous_residual = np.zeros(ACTION_DIM, dtype=np.float64)
    rate_per_second = np.asarray(
        [leg_rate_per_second] * 12 + [WHEEL_RATE_PER_SECOND] * 4
    )
    post_steps = 0
    post_front_steps = 0
    residual_l2_sum = 0.0
    alignment_l2_sum = 0.0
    alignment_steps = 0
    initial_rear = 0.0
    peak_rear_after_front = 0.0
    initial_rear_pair_height = 0.0
    peak_rear_pair_height_after_front = 0.0
    initial_front_pair_height = 0.0
    peak_front_pair_height = 0.0
    initial_front_pair_progress = 0.0
    peak_front_pair_progress = 0.0
    peak_front_top_margin = -math.inf
    initial_base = 0.0
    peak_base_after_approach = -math.inf
    peak_rear_height_after_second = -math.inf
    peak_rear_progress_after_second = -math.inf
    peak_rear_proximity_after_second = -math.inf
    front_support_steps_after_second = 0
    steps_after_second = 0
    height_stage_streak = 0
    height_stage_trigger_step: int | None = None
    peak_rear_proximity_after_height = -math.inf
    rear_height_at_best_proximity = -math.inf
    rear_progress_at_best_proximity = -math.inf
    peak_rear_progress_above_height_gate = -math.inf
    front_support_steps_after_height = 0
    steps_after_height = 0
    top_stage_streak = 0
    top_stage_trigger_step: int | None = None
    threepoint_streak = 0
    threepoint_trigger_step: int | None = None
    threepoint_leading_rear_index: int | None = None
    threepoint_lagging_rear_index: int | None = None
    threepoint_leading_action_hold = np.zeros(3, dtype=np.float32)
    threepoint_trigger_height_m: list[float] | None = None
    threepoint_trigger_progress_m: list[float] | None = None
    threepoint_trigger_force_n: list[float] | None = None
    peak_threepoint_lagging_height = -math.inf
    peak_threepoint_lagging_progress = -math.inf
    peak_threepoint_lagging_top_margin = -math.inf
    threepoint_front_support_steps = 0
    threepoint_leading_geometry_steps = 0
    threepoint_leading_contact_steps = 0
    threepoint_steps = 0
    peak_rear_proximity_after_top = -math.inf
    rear_height_at_best_proximity_after_top = -math.inf
    rear_progress_at_best_proximity_after_top = -math.inf
    peak_rear_progress_after_top = -math.inf
    front_support_steps_after_top = 0
    steps_after_top = 0
    peak_base_progress_after_pullover = -math.inf
    peak_base_z_after_pullover = -math.inf
    peak_pitch_after_pullover = -math.inf
    pullover_steps = 0
    final_info: dict[str, Any] | None = None
    oracle_front_anchor_latched = False
    oracle_front_seen_step: int | None = None
    oracle_front_leg_hold = np.zeros(6, dtype=np.float32)
    trace: dict[str, list[Any]] = {
        name: []
        for name in (
            "time_s",
            "terrain_min",
            "front_probability",
            "residual",
            "final_action",
            "roll_rad",
            "pitch_rad",
            "base_z_m",
            "base_progress_m",
            "wheel_height_m",
            "wheel_progress_m",
            "wheel_force_n",
            "actuator_force_nm",
            "joint_position_rad",
            "joint_velocity_rad_s",
            "front_top",
            "rear_top",
            "base_cross",
            "body_wall_contact",
            "threepoint_active",
            "threepoint_leading_rear_index",
            "threepoint_lagging_rear_index",
        )
    }
    try:
        actor_obs, _, info = env.reset(seed=rollout_seed, initial_state=initial_state)
        terrain_obs = actor_obs[:TERRAIN_OBS_DIM].copy()
        initial_metrics = info["metrics"]
        initial_rear = float(np.min(initial_metrics["wheel_progress_m"][2:4]))
        peak_rear_after_front = initial_rear
        initial_rear_pair_height = float(
            np.min(initial_metrics["wheel_height_m"][2:4])
        )
        peak_rear_pair_height_after_front = initial_rear_pair_height
        initial_front_pair_height = float(
            np.min(initial_metrics["wheel_height_m"][:2])
        )
        peak_front_pair_height = initial_front_pair_height
        initial_front_pair_progress = float(
            np.min(initial_metrics["wheel_progress_m"][:2])
        )
        peak_front_pair_progress = initial_front_pair_progress
        peak_front_top_margin = min(
            initial_front_pair_height - (PLATFORM_Z + WHEEL_RADIUS_M - 0.025),
            initial_front_pair_progress - (PIT_EXIT_ALONG_M - 0.04),
        )
        initial_base = float(initial_metrics["base_progress_m"])
        peak_base_after_approach = initial_base
        if recorder is not None:
            recorder.capture(env)
        for step in range(env.max_episode_steps):
            terrain_min = float(np.min(terrain_obs[129:]))
            early_streak = early_streak + 1 if terrain_min <= early_threshold else 0
            if early_step is None and early_streak >= DWELL_STEPS:
                early_step = step
            approach_streak = approach_streak + 1 if terrain_min <= approach_threshold else 0
            if approach_step is None and approach_streak >= DWELL_STEPS:
                approach_step = step
            front_probability = float(actor_obs[-1])
            front_streak = front_streak + 1 if front_probability >= PHASE_THRESHOLD else 0
            if front_step is None and front_streak >= DWELL_STEPS:
                front_step = step
            with torch.inference_mode():
                nominal = (
                    agent.deterministic(
                        torch.as_tensor(actor_obs[:129], dtype=torch.float32).unsqueeze(0)
                    )
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            target = np.zeros(ACTION_DIM, dtype=np.float64)
            if early_step is None:
                prefix_actions.append(nominal.copy())
            elif approach_step is None:
                early_age = (step - early_step) * CONTROL_DT
                envelope = smoothstep(early_age / 0.15)
                if prelift_from_early:
                    target[PRE_ACTIVE] = rescue_candidate[:6] * envelope
                target[EARLY_WHEEL_INDICES] = np.asarray(
                    [steering[0], steering[1], steering[0], steering[1]],
                    dtype=np.float64,
                ) * envelope
                alignment_l2_sum += float(np.linalg.norm(target[EARLY_WHEEL_INDICES]))
                alignment_steps += 1
            else:
                approach_age = (step - approach_step) * CONTROL_DT
                pre_age = (
                    (step - early_step) * CONTROL_DT
                    if prelift_from_early and early_step is not None
                    else approach_age
                )
                pre_envelope = smoothstep(pre_age / 0.15)
                if approach_age > pre_fade_start_seconds:
                    pre_envelope *= 1.0 - smoothstep(
                        (approach_age - pre_fade_start_seconds)
                        / pre_fade_duration_seconds
                    )
                target[PRE_ACTIVE] = rescue_candidate[:6] * pre_envelope
                post_start_step = (
                    None
                    if front_step is None
                    else front_step
                    + int(round(front_post_delay_seconds / CONTROL_DT))
                )
                if post_start_step is None and post_from_approach:
                    post_start_step = approach_step + int(
                        round(post_delay_seconds / CONTROL_DT)
                    )
                if post_start_step is not None:
                    post_age = (step - post_start_step) * CONTROL_DT
                    target[POST_ACTIVE] = rescue_candidate[6:] * smoothstep(post_age / 0.15)
                if np.any(precontact_vector != 0.0):
                    precontact_age = (
                        step
                        - approach_step
                        - int(
                            round(
                                precontact_leg_delay_after_approach_seconds
                                / CONTROL_DT
                            )
                        )
                    ) * CONTROL_DT
                    if precontact_age >= 0.0:
                        precontact_envelope = smoothstep(precontact_age / 0.06)
                        if precontact_age > precontact_leg_hold_seconds:
                            precontact_envelope *= 1.0 - smoothstep(
                                (precontact_age - precontact_leg_hold_seconds) / 0.10
                            )
                        target[:12] += precontact_vector * precontact_envelope
                        target[:12] = np.clip(
                            target[:12],
                            -late_post_target_limit,
                            late_post_target_limit,
                        )
                if (
                    threepoint_trigger_step is not None
                    and threepoint_lagging_rear_index is not None
                    and np.any(threepoint_swing_vector != 0.0)
                ):
                    threepoint_age = (
                        step
                        - threepoint_trigger_step
                        - int(round(threepoint_delay_seconds / CONTROL_DT))
                    ) * CONTROL_DT
                    if threepoint_age >= 0.0:
                        threepoint_envelope = smoothstep(threepoint_age / 0.04)
                        if threepoint_age > threepoint_hold_seconds:
                            threepoint_envelope *= 1.0 - smoothstep(
                                (threepoint_age - threepoint_hold_seconds) / 0.08
                            )
                        lagging_action_start = (
                            6 if threepoint_lagging_rear_index == 2 else 9
                        )
                        target[lagging_action_start : lagging_action_start + 3] += (
                            threepoint_swing_vector * threepoint_envelope
                        )
                        target[lagging_action_start : lagging_action_start + 3] = (
                            np.clip(
                                target[lagging_action_start : lagging_action_start + 3],
                                -late_post_target_limit,
                                late_post_target_limit,
                            )
                        )
                if true_front_step is not None and np.any(
                    pullover_launch_vector != 0.0
                ):
                    pullover_launch_age = (
                        step
                        - true_front_step
                        - int(
                            round(
                                pullover_launch_delay_after_front_seconds
                                / CONTROL_DT
                            )
                        )
                    ) * CONTROL_DT
                    if pullover_launch_age >= 0.0:
                        pullover_launch_envelope = smoothstep(
                            pullover_launch_age / 0.04
                        )
                        if pullover_launch_age > pullover_launch_hold_seconds:
                            pullover_launch_envelope *= 1.0 - smoothstep(
                                (
                                    pullover_launch_age
                                    - pullover_launch_hold_seconds
                                )
                                / 0.08
                            )
                        target[:12] += (
                            pullover_launch_vector * pullover_launch_envelope
                        )
                        target[:12] = np.clip(
                            target[:12],
                            -late_post_target_limit,
                            late_post_target_limit,
                        )
                if true_front_step is not None and np.any(
                    pullover_tuck_vector != 0.0
                ):
                    pullover_tuck_age = (
                        step
                        - true_front_step
                        - int(
                            round(
                                (
                                    pullover_launch_delay_after_front_seconds
                                    + pullover_tuck_delay_after_launch_seconds
                                )
                                / CONTROL_DT
                            )
                        )
                    ) * CONTROL_DT
                    if pullover_tuck_age >= 0.0:
                        pullover_tuck_envelope = smoothstep(
                            pullover_tuck_age / 0.04
                        )
                        if pullover_tuck_age > pullover_tuck_hold_seconds:
                            pullover_tuck_envelope *= 1.0 - smoothstep(
                                (pullover_tuck_age - pullover_tuck_hold_seconds)
                                / 0.08
                            )
                        target[:12] += pullover_tuck_vector * pullover_tuck_envelope
                        target[:12] = np.clip(
                            target[:12],
                            -late_post_target_limit,
                            late_post_target_limit,
                        )
                if front_step is not None and np.any(boost_vector != 0.0):
                    boost_age = (
                        step
                        - front_step
                        - int(round(late_post_boost_delay_seconds / CONTROL_DT))
                    ) * CONTROL_DT
                    if boost_age >= 0.0:
                        boost_envelope = smoothstep(boost_age / 0.08)
                        if boost_age > late_post_boost_hold_seconds:
                            boost_envelope *= 1.0 - smoothstep(
                                (boost_age - late_post_boost_hold_seconds) / 0.15
                            )
                        target[POST_ACTIVE] += (
                            boost_vector * boost_envelope
                        )
                        target[POST_ACTIVE] = np.clip(
                            target[POST_ACTIVE],
                            -late_post_target_limit,
                            late_post_target_limit,
                        )
                if front_step is not None and np.any(second_boost_vector != 0.0):
                    second_age = (
                        step
                        - front_step
                        - int(round(late_post_second_delay_seconds / CONTROL_DT))
                    ) * CONTROL_DT
                    if second_age >= 0.0:
                        second_envelope = smoothstep(second_age / 0.08)
                        if second_age > late_post_second_hold_seconds:
                            second_envelope *= 1.0 - smoothstep(
                                (second_age - late_post_second_hold_seconds) / 0.15
                            )
                        target[POST_ACTIVE] += second_boost_vector * second_envelope
                        target[POST_ACTIVE] = np.clip(
                            target[POST_ACTIVE],
                            -late_post_target_limit,
                            late_post_target_limit,
                        )
                if height_stage_trigger_step is not None and (
                    np.any(height_boost_vector != 0.0)
                    or np.any(height_front_leg_vector != 0.0)
                ):
                    height_stage_age = (step - height_stage_trigger_step) * CONTROL_DT
                    if height_stage_age >= 0.0:
                        height_stage_envelope = smoothstep(height_stage_age / 0.06)
                        if height_stage_age > late_post_height_hold_seconds:
                            height_stage_envelope *= 1.0 - smoothstep(
                                (height_stage_age - late_post_height_hold_seconds) / 0.12
                            )
                        if np.any(height_boost_vector != 0.0):
                            target[POST_ACTIVE] += (
                                height_boost_vector * height_stage_envelope
                            )
                            target[POST_ACTIVE] = np.clip(
                                target[POST_ACTIVE],
                                -late_post_target_limit,
                                late_post_target_limit,
                            )
                        if np.any(height_front_leg_vector != 0.0):
                            target[:6] += (
                                height_front_leg_vector * height_stage_envelope
                            )
                            target[:6] = np.clip(
                                target[:6],
                                -late_post_target_limit,
                                late_post_target_limit,
                            )
                if top_stage_trigger_step is not None and (
                    np.any(top_boost_vector != 0.0)
                    or np.any(top_front_leg_vector != 0.0)
                ):
                    top_stage_age = (step - top_stage_trigger_step) * CONTROL_DT
                    if top_stage_age >= 0.0:
                        top_stage_envelope = smoothstep(top_stage_age / 0.06)
                        if top_stage_age > late_post_top_hold_seconds:
                            top_stage_envelope *= 1.0 - smoothstep(
                                (top_stage_age - late_post_top_hold_seconds) / 0.12
                            )
                        if np.any(top_boost_vector != 0.0):
                            target[POST_ACTIVE] += (
                                top_boost_vector * top_stage_envelope
                            )
                            target[POST_ACTIVE] = np.clip(
                                target[POST_ACTIVE],
                                -late_post_target_limit,
                                late_post_target_limit,
                            )
                        if np.any(top_front_leg_vector != 0.0):
                            target[:6] += top_front_leg_vector * top_stage_envelope
                            target[:6] = np.clip(
                                target[:6],
                                -late_post_target_limit,
                                late_post_target_limit,
                            )
                if top_stage_trigger_step is not None and np.any(
                    pullover_top_rear_wheel != 0.0
                ):
                    pullover_wheel_age = (
                        step
                        - top_stage_trigger_step
                        - int(
                            round(
                                pullover_top_rear_wheel_delay_seconds
                                / CONTROL_DT
                            )
                        )
                    ) * CONTROL_DT
                    if pullover_wheel_age >= 0.0:
                        pullover_wheel_envelope = smoothstep(
                            pullover_wheel_age / 0.04
                        )
                        if (
                            pullover_wheel_age
                            > pullover_top_rear_wheel_hold_seconds
                        ):
                            pullover_wheel_envelope *= 1.0 - smoothstep(
                                (
                                    pullover_wheel_age
                                    - pullover_top_rear_wheel_hold_seconds
                                )
                                / 0.08
                            )
                        target[14:16] += (
                            pullover_top_rear_wheel
                            * pullover_wheel_envelope
                        )
                        target[14:16] = np.clip(target[14:16], -3.0, 3.0)
                if front_step is not None and np.any(momentum_leg_vector != 0.0):
                    momentum_leg_age = (
                        step
                        - front_step
                        - int(round(late_momentum_leg_delay_seconds / CONTROL_DT))
                    ) * CONTROL_DT
                    if momentum_leg_age >= 0.0:
                        momentum_leg_envelope = smoothstep(momentum_leg_age / 0.06)
                        if momentum_leg_age > late_momentum_leg_hold_seconds:
                            momentum_leg_envelope *= 1.0 - smoothstep(
                                (momentum_leg_age - late_momentum_leg_hold_seconds) / 0.10
                            )
                        target[:12] += momentum_leg_vector * momentum_leg_envelope
                        target[:12] = np.clip(
                            target[:12],
                            -late_post_target_limit,
                            late_post_target_limit,
                        )
                if front_step is not None and np.any(prejump_leg_vector != 0.0):
                    prejump_leg_age = (
                        step
                        - front_step
                        - int(round(late_prejump_leg_delay_seconds / CONTROL_DT))
                    ) * CONTROL_DT
                    if prejump_leg_age >= 0.0:
                        prejump_leg_envelope = smoothstep(prejump_leg_age / 0.04)
                        if prejump_leg_age > late_prejump_leg_hold_seconds:
                            prejump_leg_envelope *= 1.0 - smoothstep(
                                (prejump_leg_age - late_prejump_leg_hold_seconds)
                                / 0.06
                            )
                        target[:12] += prejump_leg_vector * prejump_leg_envelope
                        target[:12] = np.clip(
                            target[:12],
                            -late_post_target_limit,
                            late_post_target_limit,
                        )
                if front_step is not None and np.any(momentum_wheel_vector != 0.0):
                    momentum_age = (
                        step
                        - front_step
                        - int(round(late_momentum_wheel_delay_seconds / CONTROL_DT))
                    ) * CONTROL_DT
                    if momentum_age >= 0.0:
                        momentum_envelope = smoothstep(momentum_age / 0.08)
                        if momentum_age > late_momentum_wheel_hold_seconds:
                            momentum_envelope *= 1.0 - smoothstep(
                                (momentum_age - late_momentum_wheel_hold_seconds) / 0.12
                            )
                        target[12:16] += momentum_wheel_vector * momentum_envelope
                        target[12:16] = np.clip(target[12:16], -3.0, 3.0)
                if front_step is not None and np.any(front_wheel_vector != 0.0):
                    front_wheel_age = (
                        step
                        - front_step
                        - int(round(late_front_wheel_delay_seconds / CONTROL_DT))
                    ) * CONTROL_DT
                    if front_wheel_age >= 0.0:
                        front_wheel_envelope = smoothstep(front_wheel_age / 0.15)
                        if front_wheel_age > late_front_wheel_hold_seconds:
                            front_wheel_envelope *= 1.0 - smoothstep(
                                (front_wheel_age - late_front_wheel_hold_seconds) / 0.20
                            )
                        target[12:14] = front_wheel_vector * front_wheel_envelope
                if front_step is not None and np.any(front_leg_vector != 0.0):
                    front_leg_age = (
                        step
                        - front_step
                        - int(round(late_front_wheel_delay_seconds / CONTROL_DT))
                    ) * CONTROL_DT
                    if front_leg_age >= 0.0:
                        front_leg_envelope = smoothstep(front_leg_age / 0.15)
                        if front_leg_age > late_front_wheel_hold_seconds:
                            front_leg_envelope *= 1.0 - smoothstep(
                                (front_leg_age - late_front_wheel_hold_seconds) / 0.20
                            )
                        target[0:6] = front_leg_vector * front_leg_envelope
                post_steps += 1
            if early_step is not None:
                active_rate_per_second = rate_per_second
                if (
                    (
                        approach_step is not None
                        and np.any(precontact_vector != 0.0)
                        and precontact_leg_rate_per_second is not None
                    )
                    or (
                        threepoint_trigger_step is not None
                        and np.any(threepoint_swing_vector != 0.0)
                        and threepoint_leg_rate_per_second is not None
                    )
                    or (
                        true_front_step is not None
                        and (
                            np.any(pullover_launch_vector != 0.0)
                            or np.any(pullover_tuck_vector != 0.0)
                        )
                        and pullover_leg_rate_per_second is not None
                    )
                    or (
                        front_step is not None
                        and (
                        (
                            np.any(momentum_leg_vector != 0.0)
                            and late_momentum_leg_rate_per_second is not None
                        )
                        or (
                            np.any(prejump_leg_vector != 0.0)
                            and late_prejump_leg_rate_per_second is not None
                        )
                    )
                    )
                ):
                    active_rate_per_second = rate_per_second.copy()
                    active_rate_per_second[:12] = max(
                        rate
                        for vector, rate in (
                            (
                                precontact_vector,
                                precontact_leg_rate_per_second,
                            ),
                            (
                                threepoint_swing_vector,
                                threepoint_leg_rate_per_second,
                            ),
                            (
                                pullover_launch_vector,
                                pullover_leg_rate_per_second,
                            ),
                            (
                                pullover_tuck_vector,
                                pullover_leg_rate_per_second,
                            ),
                            (
                                momentum_leg_vector,
                                late_momentum_leg_rate_per_second,
                            ),
                            (
                                prejump_leg_vector,
                                late_prejump_leg_rate_per_second,
                            ),
                        )
                        if np.any(vector != 0.0) and rate is not None
                    )
                if (
                    top_stage_trigger_step is not None
                    and (
                        np.any(top_boost_vector != 0.0)
                        or np.any(top_front_leg_vector != 0.0)
                    )
                    and late_post_top_leg_rate_per_second is not None
                ):
                    active_rate_per_second = rate_per_second.copy()
                    active_rate_per_second[:6] = late_post_top_leg_rate_per_second
                    active_rate_per_second[6:12] = (
                        late_post_top_leg_rate_per_second
                    )
                maximum_delta = active_rate_per_second * CONTROL_DT
                residual = previous_residual + np.clip(
                    target - previous_residual, -maximum_delta, maximum_delta
                )
                previous_residual = residual
                residual_l2_sum += float(np.linalg.norm(residual))
            else:
                residual = target
            final_action = nominal + residual.astype(np.float32)
            if (
                oracle_front_seen_step is not None
                and step
                >= oracle_front_seen_step
                + int(round(oracle_front_anchor_delay_seconds / CONTROL_DT))
            ):
                oracle_front_anchor_latched = True
            if oracle_front_anchor_latched:
                if oracle_freeze_front_legs:
                    final_action[:6] = oracle_front_leg_hold
                if oracle_front_wheel_action is not None:
                    final_action[12:14] = float(oracle_front_wheel_action)
            if (
                threepoint_trigger_step is not None
                and threepoint_freeze_leading_leg
                and threepoint_leading_rear_index is not None
                and step >= threepoint_trigger_step
            ):
                leading_action_start = (
                    6 if threepoint_leading_rear_index == 2 else 9
                )
                final_action[leading_action_start : leading_action_start + 3] = (
                    threepoint_leading_action_hold
                )
            actor_obs, _, _, terminated, truncated, final_info = env.step(final_action)
            metrics = final_info["metrics"]
            if oracle_front_seen_step is None and bool(metrics["front_top"]):
                oracle_front_seen_step = step + 1
                oracle_front_leg_hold[:] = final_action[:6]
            terrain_obs = actor_obs[:TERRAIN_OBS_DIM].copy()
            if true_front_step is None and bool(metrics["front_top_latched"]):
                true_front_step = step + 1
            if true_front_step is not None and step + 1 >= true_front_step:
                peak_base_progress_after_pullover = max(
                    peak_base_progress_after_pullover,
                    float(metrics["base_progress_m"]),
                )
                peak_base_z_after_pullover = max(
                    peak_base_z_after_pullover,
                    float(metrics["base_z_m"]),
                )
                peak_pitch_after_pullover = max(
                    peak_pitch_after_pullover,
                    float(metrics["pitch_rad"]),
                )
                pullover_steps += 1
            peak_front_pair_height = max(
                peak_front_pair_height,
                float(np.min(metrics["wheel_height_m"][:2])),
            )
            peak_front_pair_progress = max(
                peak_front_pair_progress,
                float(np.min(metrics["wheel_progress_m"][:2])),
            )
            front_height_margin = float(np.min(metrics["wheel_height_m"][:2])) - (
                PLATFORM_Z + WHEEL_RADIUS_M - 0.025
            )
            front_progress_margin = float(np.min(metrics["wheel_progress_m"][:2])) - (
                PIT_EXIT_ALONG_M - 0.04
            )
            peak_front_top_margin = max(
                peak_front_top_margin,
                min(front_height_margin, front_progress_margin),
            )
            if approach_step is not None:
                post_front_steps += int(bool(metrics["front_top"]))
                peak_base_after_approach = max(
                    peak_base_after_approach, float(metrics["base_progress_m"])
                )
            if front_step is not None:
                peak_rear_after_front = max(
                    peak_rear_after_front,
                    float(np.min(metrics["wheel_progress_m"][2:4])),
                )
            if bool(metrics["front_top_latched"]):
                peak_rear_pair_height_after_front = max(
                    peak_rear_pair_height_after_front,
                    float(np.min(metrics["wheel_height_m"][2:4])),
                )
            rear_height_gate = PLATFORM_Z + WHEEL_RADIUS_M - 0.025
            rear_progress_gate = PIT_EXIT_ALONG_M - 0.04
            rear_top_geometry = (
                np.asarray(metrics["wheel_height_m"][2:4]) >= rear_height_gate
            ) & (
                np.asarray(metrics["wheel_progress_m"][2:4]) >= rear_progress_gate
            )
            if (
                threepoint_trigger_step is None
                and (
                    threepoint_monitor_enabled
                    or np.any(threepoint_swing_vector != 0.0)
                    or threepoint_freeze_leading_leg
                )
            ):
                leading_geometry_indices = np.flatnonzero(rear_top_geometry)
                threepoint_ready = bool(
                    metrics["front_top"]
                    and leading_geometry_indices.size == 1
                    and float(
                        metrics["wheel_force_n"][2 + leading_geometry_indices[0]]
                    )
                    >= threepoint_min_leading_rear_contact_force_n
                    and abs(math.degrees(float(metrics["roll_rad"])))
                    <= threepoint_max_abs_roll_deg
                )
                threepoint_streak = threepoint_streak + 1 if threepoint_ready else 0
                if threepoint_streak >= threepoint_trigger_dwell_steps:
                    leading_local = int(leading_geometry_indices[0])
                    threepoint_leading_rear_index = 2 + leading_local
                    threepoint_lagging_rear_index = 3 - leading_local
                    threepoint_trigger_step = step + 1
                    leading_action_start = 6 if leading_local == 0 else 9
                    threepoint_leading_action_hold[:] = final_action[
                        leading_action_start : leading_action_start + 3
                    ]
                    threepoint_trigger_height_m = np.asarray(
                        metrics["wheel_height_m"][2:4], dtype=np.float64
                    ).tolist()
                    threepoint_trigger_progress_m = np.asarray(
                        metrics["wheel_progress_m"][2:4], dtype=np.float64
                    ).tolist()
                    threepoint_trigger_force_n = np.asarray(
                        metrics["wheel_force_n"][2:4], dtype=np.float64
                    ).tolist()
            if (
                threepoint_trigger_step is not None
                and threepoint_leading_rear_index is not None
                and threepoint_lagging_rear_index is not None
                and step + 1 >= threepoint_trigger_step
            ):
                lagging_index = threepoint_lagging_rear_index
                leading_local = threepoint_leading_rear_index - 2
                lagging_height = float(metrics["wheel_height_m"][lagging_index])
                lagging_progress = float(metrics["wheel_progress_m"][lagging_index])
                lagging_margin = min(
                    lagging_height - rear_height_gate,
                    lagging_progress - rear_progress_gate,
                )
                peak_threepoint_lagging_height = max(
                    peak_threepoint_lagging_height, lagging_height
                )
                peak_threepoint_lagging_progress = max(
                    peak_threepoint_lagging_progress, lagging_progress
                )
                peak_threepoint_lagging_top_margin = max(
                    peak_threepoint_lagging_top_margin, lagging_margin
                )
                threepoint_front_support_steps += int(bool(metrics["front_top"]))
                threepoint_leading_geometry_steps += int(
                    bool(rear_top_geometry[leading_local])
                )
                threepoint_leading_contact_steps += int(
                    float(metrics["wheel_force_n"][threepoint_leading_rear_index])
                    >= CONTACT_THRESHOLD_N
                )
                threepoint_steps += 1
            if second_stage_configured and front_step is not None:
                second_start_step = front_step + int(
                    round(late_post_second_delay_seconds / CONTROL_DT)
                )
                if step >= second_start_step:
                    rear_height = float(np.min(metrics["wheel_height_m"][2:4]))
                    rear_progress = float(np.min(metrics["wheel_progress_m"][2:4]))
                    height_margin = rear_height - (
                        PLATFORM_Z + WHEEL_RADIUS_M - 0.025
                    )
                    progress_margin = rear_progress - (PIT_EXIT_ALONG_M - 0.04)
                    proximity = -math.hypot(
                        min(0.0, height_margin), min(0.0, progress_margin)
                    )
                    peak_rear_height_after_second = max(
                        peak_rear_height_after_second, rear_height
                    )
                    peak_rear_progress_after_second = max(
                        peak_rear_progress_after_second, rear_progress
                    )
                    peak_rear_proximity_after_second = max(
                        peak_rear_proximity_after_second, proximity
                    )
                    front_support_steps_after_second += int(bool(metrics["front_top"]))
                    steps_after_second += 1
            rear_height = float(np.min(metrics["wheel_height_m"][2:4]))
            if height_stage_configured and height_stage_trigger_step is None:
                if late_post_height_delay_after_front_seconds is None:
                    height_ready = (
                        bool(metrics["front_top"])
                        and rear_height >= late_post_height_trigger_m
                    )
                else:
                    height_ready = (
                        front_step is not None
                        and step
                        >= front_step
                        + int(
                            round(
                                late_post_height_delay_after_front_seconds
                                / CONTROL_DT
                            )
                        )
                    )
                height_stage_streak = height_stage_streak + 1 if height_ready else 0
                if height_stage_streak >= late_post_height_dwell_steps:
                    height_stage_trigger_step = step + 1
            if height_stage_trigger_step is not None and step >= height_stage_trigger_step:
                rear_progress = float(np.min(metrics["wheel_progress_m"][2:4]))
                height_margin = rear_height - (
                    PLATFORM_Z + WHEEL_RADIUS_M - 0.025
                )
                progress_margin = rear_progress - (PIT_EXIT_ALONG_M - 0.04)
                proximity = -math.hypot(
                    min(0.0, height_margin), min(0.0, progress_margin)
                )
                if proximity > peak_rear_proximity_after_height:
                    peak_rear_proximity_after_height = proximity
                    rear_height_at_best_proximity = rear_height
                    rear_progress_at_best_proximity = rear_progress
                if height_margin >= 0.0:
                    peak_rear_progress_above_height_gate = max(
                        peak_rear_progress_above_height_gate, rear_progress
                    )
                front_support_steps_after_height += int(bool(metrics["front_top"]))
                steps_after_height += 1
            if top_stage_configured and top_stage_trigger_step is None:
                top_ready = (
                    bool(metrics["front_top"])
                    and rear_height >= top_trigger_m
                )
                top_stage_streak = top_stage_streak + 1 if top_ready else 0
                if top_stage_streak >= late_post_top_dwell_steps:
                    top_stage_trigger_step = step + 1
            top_score_start_step = (
                None
                if top_stage_trigger_step is None
                else top_stage_trigger_step
                + int(round(late_post_top_score_delay_seconds / CONTROL_DT))
            )
            if top_score_start_step is not None and step >= top_score_start_step:
                rear_progress = float(np.min(metrics["wheel_progress_m"][2:4]))
                height_margin = rear_height - (
                    PLATFORM_Z + WHEEL_RADIUS_M - 0.025
                )
                progress_margin = rear_progress - (PIT_EXIT_ALONG_M - 0.04)
                proximity = -math.hypot(
                    min(0.0, height_margin), min(0.0, progress_margin)
                )
                if proximity > peak_rear_proximity_after_top:
                    peak_rear_proximity_after_top = proximity
                    rear_height_at_best_proximity_after_top = rear_height
                    rear_progress_at_best_proximity_after_top = rear_progress
                peak_rear_progress_after_top = max(
                    peak_rear_progress_after_top, rear_progress
                )
                front_support_steps_after_top += int(bool(metrics["front_top"]))
                steps_after_top += 1
            if trace_path is not None:
                values = {
                    "time_s": float(final_info["time_s"]),
                    "terrain_min": terrain_min,
                    "front_probability": front_probability,
                    "residual": residual.copy(),
                    "final_action": final_action.copy(),
                    "roll_rad": float(metrics["roll_rad"]),
                    "pitch_rad": float(metrics["pitch_rad"]),
                    "base_z_m": float(metrics["base_z_m"]),
                    "base_progress_m": float(metrics["base_progress_m"]),
                    "wheel_height_m": np.asarray(metrics["wheel_height_m"]).copy(),
                    "wheel_progress_m": np.asarray(metrics["wheel_progress_m"]).copy(),
                    "wheel_force_n": np.asarray(metrics["wheel_force_n"]).copy(),
                    "actuator_force_nm": np.asarray(env.data.actuator_force).copy(),
                    "joint_position_rad": np.asarray(env.data.qpos[7:23]).copy(),
                    "joint_velocity_rad_s": np.asarray(env.data.qvel[6:22]).copy(),
                    "front_top": bool(metrics["front_top"]),
                    "rear_top": bool(metrics["rear_top"]),
                    "base_cross": bool(metrics["base_cross"]),
                    "body_wall_contact": bool(metrics["body_wall_contact"]),
                    "threepoint_active": bool(
                        threepoint_trigger_step is not None
                        and step + 1 >= threepoint_trigger_step
                    ),
                    "threepoint_leading_rear_index": (
                        -1
                        if threepoint_leading_rear_index is None
                        else threepoint_leading_rear_index
                    ),
                    "threepoint_lagging_rear_index": (
                        -1
                        if threepoint_lagging_rear_index is None
                        else threepoint_lagging_rear_index
                    ),
                }
                for name, value in values.items():
                    trace[name].append(value)
            if recorder is not None:
                recorder.capture(env)
            if terminated or truncated:
                break
        if final_info is None:
            raise RuntimeError("early-steering rollout produced no transition")
        prefix = np.stack(prefix_actions).astype(np.float32).tobytes() if prefix_actions else b""
        prefix_sha = hashlib.sha256(prefix).hexdigest()
        if expected_early_step is not None and early_step != expected_early_step:
            raise RuntimeError(f"candidate {label} changed early trigger step")
        if expected_prefix_sha256 is not None and prefix_sha != expected_prefix_sha256:
            raise RuntimeError(f"candidate {label} changed pre-intervention prefix")
        metrics = final_info["metrics"]
        reason = str(final_info["termination_reason"])
        hard_failure = reason in HARD_FAILURES or reason not in {"success", "time_limit", ""}
        report: dict[str, Any] = {
            "label": label,
            "depth_m": depth_m,
            "command_mps": command_mps,
            "lateral_m": initial_state[0],
            "yaw_deg": initial_state[1],
            "seed": rollout_seed,
            "success": bool(final_info["success"]),
            "termination_reason": reason,
            "hard_failure": bool(hard_failure),
            "alignment_triggered": early_step is not None,
            "alignment_trigger_step": early_step,
            "approach_triggered": approach_step is not None,
            "approach_trigger_step": approach_step,
            "front_phase_trigger_step": front_step,
            "true_front_latch_step": true_front_step,
            "prefix_action_count": len(prefix_actions),
            "prefix_action_sha256": prefix_sha,
            "front_top_latched": bool(metrics["front_top_latched"]),
            "rear_top_latched": bool(metrics["rear_top_latched"]),
            "base_cross_latched": bool(metrics["base_cross_latched"]),
            "time_s": float(final_info["time_s"]),
            "episode_steps": int(final_info["episode_step"]),
            "max_abs_roll_deg": math.degrees(float(metrics["episode_max_roll_rad"])),
            "max_abs_pitch_deg": math.degrees(float(metrics["episode_max_pitch_rad"])),
            "max_contact_force_n": float(metrics["episode_max_force_n"]),
            "contact_force_above_500_n": bool(metrics["episode_max_force_n"] > 500.0),
            "front_support_rate_after_approach": post_front_steps / max(post_steps, 1),
            "rear_pair_progress_gain_after_front_m": peak_rear_after_front - initial_rear,
            "rear_pair_height_gain_after_front_m": (
                peak_rear_pair_height_after_front - initial_rear_pair_height
            ),
            "front_pair_height_gain_m": peak_front_pair_height - initial_front_pair_height,
            "front_pair_progress_gain_m": peak_front_pair_progress - initial_front_pair_progress,
            "peak_front_top_margin_m": peak_front_top_margin,
            "base_progress_gain_after_approach_m": peak_base_after_approach - initial_base,
            "mean_residual_l2": residual_l2_sum / max(alignment_steps + post_steps, 1),
            "mean_alignment_target_l2": alignment_l2_sum / max(alignment_steps, 1),
            "steering_left_right": steering.tolist(),
            "candidate_pre_vector": rescue_candidate[:6].tolist(),
            "candidate_post_vector": rescue_candidate[6:].tolist(),
            "prelift_from_early": bool(prelift_from_early),
            "pre_fade_start_seconds": float(pre_fade_start_seconds),
            "pre_fade_duration_seconds": float(pre_fade_duration_seconds),
            "front_post_delay_seconds": float(front_post_delay_seconds),
            "late_post_boost_scale": float(late_post_boost_scale),
            "late_post_boost_delay_seconds": float(late_post_boost_delay_seconds),
            "late_post_boost_hold_seconds": float(late_post_boost_hold_seconds),
            "late_post_boost_vector": boost_vector.tolist(),
            "late_post_target_limit": float(late_post_target_limit),
            "late_post_second_vector": second_boost_vector.tolist(),
            "late_post_second_delay_seconds": float(late_post_second_delay_seconds),
            "late_post_second_hold_seconds": float(late_post_second_hold_seconds),
            "second_stage_configured": bool(second_stage_configured),
            "late_post_height_vector": height_boost_vector.tolist(),
            "late_post_height_front_leg_vector": height_front_leg_vector.tolist(),
            "late_post_height_trigger_m": float(late_post_height_trigger_m),
            "late_post_height_delay_after_front_seconds": (
                None
                if late_post_height_delay_after_front_seconds is None
                else float(late_post_height_delay_after_front_seconds)
            ),
            "late_post_height_hold_seconds": float(late_post_height_hold_seconds),
            "late_post_height_dwell_steps": int(late_post_height_dwell_steps),
            "height_stage_configured": bool(height_stage_configured),
            "height_stage_trigger_step": height_stage_trigger_step,
            "late_post_top_vector": top_boost_vector.tolist(),
            "late_post_top_front_leg_vector": top_front_leg_vector.tolist(),
            "late_post_top_trigger_m": float(top_trigger_m),
            "late_post_top_hold_seconds": float(late_post_top_hold_seconds),
            "late_post_top_dwell_steps": int(late_post_top_dwell_steps),
            "late_post_top_score_delay_seconds": float(
                late_post_top_score_delay_seconds
            ),
            "late_post_top_leg_rate_per_second": (
                None
                if late_post_top_leg_rate_per_second is None
                else float(late_post_top_leg_rate_per_second)
            ),
            "top_stage_configured": bool(top_stage_configured),
            "top_stage_trigger_step": top_stage_trigger_step,
            "peak_rear_height_after_second_m": (
                None
                if not math.isfinite(peak_rear_height_after_second)
                else peak_rear_height_after_second
            ),
            "peak_rear_progress_after_second_m": (
                None
                if not math.isfinite(peak_rear_progress_after_second)
                else peak_rear_progress_after_second
            ),
            "peak_rear_top_proximity_after_second_m": (
                None
                if not math.isfinite(peak_rear_proximity_after_second)
                else peak_rear_proximity_after_second
            ),
            "front_support_rate_after_second": (
                front_support_steps_after_second / max(steps_after_second, 1)
            ),
            "peak_rear_top_proximity_after_height_m": (
                None
                if not math.isfinite(peak_rear_proximity_after_height)
                else peak_rear_proximity_after_height
            ),
            "rear_height_at_best_proximity_after_height_m": (
                None
                if not math.isfinite(rear_height_at_best_proximity)
                else rear_height_at_best_proximity
            ),
            "rear_progress_at_best_proximity_after_height_m": (
                None
                if not math.isfinite(rear_progress_at_best_proximity)
                else rear_progress_at_best_proximity
            ),
            "peak_rear_progress_above_height_gate_m": (
                None
                if not math.isfinite(peak_rear_progress_above_height_gate)
                else peak_rear_progress_above_height_gate
            ),
            "front_support_rate_after_height": (
                front_support_steps_after_height / max(steps_after_height, 1)
            ),
            "peak_rear_top_proximity_after_top_m": (
                None
                if not math.isfinite(peak_rear_proximity_after_top)
                else peak_rear_proximity_after_top
            ),
            "rear_height_at_best_proximity_after_top_m": (
                None
                if not math.isfinite(rear_height_at_best_proximity_after_top)
                else rear_height_at_best_proximity_after_top
            ),
            "rear_progress_at_best_proximity_after_top_m": (
                None
                if not math.isfinite(rear_progress_at_best_proximity_after_top)
                else rear_progress_at_best_proximity_after_top
            ),
            "peak_rear_progress_after_top_m": (
                None
                if not math.isfinite(peak_rear_progress_after_top)
                else peak_rear_progress_after_top
            ),
            "front_support_rate_after_top": (
                front_support_steps_after_top / max(steps_after_top, 1)
            ),
            "final_rear_pair_height_m": float(
                np.min(metrics["wheel_height_m"][2:4])
            ),
            "final_rear_pair_progress_m": float(
                np.min(metrics["wheel_progress_m"][2:4])
            ),
            "precontact_leg_vector": precontact_vector.tolist(),
            "precontact_leg_delay_after_approach_seconds": float(
                precontact_leg_delay_after_approach_seconds
            ),
            "precontact_leg_hold_seconds": float(precontact_leg_hold_seconds),
            "precontact_leg_rate_per_second": precontact_leg_rate_per_second,
            "pullover_launch_leg_vector": pullover_launch_vector.tolist(),
            "pullover_launch_delay_after_front_seconds": float(
                pullover_launch_delay_after_front_seconds
            ),
            "pullover_launch_hold_seconds": float(pullover_launch_hold_seconds),
            "pullover_tuck_leg_vector": pullover_tuck_vector.tolist(),
            "pullover_tuck_delay_after_launch_seconds": float(
                pullover_tuck_delay_after_launch_seconds
            ),
            "pullover_tuck_hold_seconds": float(pullover_tuck_hold_seconds),
            "pullover_leg_rate_per_second": pullover_leg_rate_per_second,
            "pullover_top_rear_wheel_vector": pullover_top_rear_wheel.tolist(),
            "pullover_top_rear_wheel_delay_seconds": float(
                pullover_top_rear_wheel_delay_seconds
            ),
            "pullover_top_rear_wheel_hold_seconds": float(
                pullover_top_rear_wheel_hold_seconds
            ),
            "pullover_observed_steps": int(pullover_steps),
            "peak_base_progress_after_pullover_m": (
                None
                if not math.isfinite(peak_base_progress_after_pullover)
                else peak_base_progress_after_pullover
            ),
            "peak_base_z_after_pullover_m": (
                None
                if not math.isfinite(peak_base_z_after_pullover)
                else peak_base_z_after_pullover
            ),
            "peak_pitch_after_pullover_deg": (
                None
                if not math.isfinite(peak_pitch_after_pullover)
                else math.degrees(peak_pitch_after_pullover)
            ),
            "allow_body_wall_contact": bool(allow_body_wall_contact),
            "threepoint_swing_leg_vector": threepoint_swing_vector.tolist(),
            "threepoint_delay_seconds": float(threepoint_delay_seconds),
            "threepoint_hold_seconds": float(threepoint_hold_seconds),
            "threepoint_leg_rate_per_second": threepoint_leg_rate_per_second,
            "threepoint_monitor_enabled": bool(threepoint_monitor_enabled),
            "threepoint_freeze_leading_leg": bool(threepoint_freeze_leading_leg),
            "threepoint_min_leading_rear_contact_force_n": float(
                threepoint_min_leading_rear_contact_force_n
            ),
            "threepoint_max_abs_roll_deg": float(threepoint_max_abs_roll_deg),
            "threepoint_trigger_dwell_steps": int(threepoint_trigger_dwell_steps),
            "threepoint_trigger_step": threepoint_trigger_step,
            "threepoint_leading_rear_index": threepoint_leading_rear_index,
            "threepoint_lagging_rear_index": threepoint_lagging_rear_index,
            "threepoint_trigger_rear_height_m": threepoint_trigger_height_m,
            "threepoint_trigger_rear_progress_m": threepoint_trigger_progress_m,
            "threepoint_trigger_rear_force_n": threepoint_trigger_force_n,
            "peak_threepoint_lagging_height_m": (
                None
                if not math.isfinite(peak_threepoint_lagging_height)
                else peak_threepoint_lagging_height
            ),
            "peak_threepoint_lagging_progress_m": (
                None
                if not math.isfinite(peak_threepoint_lagging_progress)
                else peak_threepoint_lagging_progress
            ),
            "peak_threepoint_lagging_top_margin_m": (
                None
                if not math.isfinite(peak_threepoint_lagging_top_margin)
                else peak_threepoint_lagging_top_margin
            ),
            "threepoint_front_support_rate": (
                threepoint_front_support_steps / max(threepoint_steps, 1)
            ),
            "threepoint_leading_geometry_rate": (
                threepoint_leading_geometry_steps / max(threepoint_steps, 1)
            ),
            "threepoint_leading_contact_rate": (
                threepoint_leading_contact_steps / max(threepoint_steps, 1)
            ),
            "late_momentum_leg_vector": momentum_leg_vector.tolist(),
            "late_momentum_leg_delay_seconds": float(
                late_momentum_leg_delay_seconds
            ),
            "late_momentum_leg_hold_seconds": float(
                late_momentum_leg_hold_seconds
            ),
            "late_momentum_leg_rate_per_second": late_momentum_leg_rate_per_second,
            "late_prejump_leg_vector": prejump_leg_vector.tolist(),
            "late_prejump_leg_delay_seconds": float(
                late_prejump_leg_delay_seconds
            ),
            "late_prejump_leg_hold_seconds": float(
                late_prejump_leg_hold_seconds
            ),
            "late_prejump_leg_rate_per_second": late_prejump_leg_rate_per_second,
            "late_momentum_wheel_vector": momentum_wheel_vector.tolist(),
            "late_momentum_wheel_delay_seconds": float(
                late_momentum_wheel_delay_seconds
            ),
            "late_momentum_wheel_hold_seconds": float(
                late_momentum_wheel_hold_seconds
            ),
            "late_front_wheel_vector": front_wheel_vector.tolist(),
            "late_front_wheel_delay_seconds": float(late_front_wheel_delay_seconds),
            "late_front_wheel_hold_seconds": float(late_front_wheel_hold_seconds),
            "late_front_leg_vector": front_leg_vector.tolist(),
            "oracle_front_anchor_latched": bool(oracle_front_anchor_latched),
            "oracle_front_wheel_action": oracle_front_wheel_action,
            "oracle_freeze_front_legs": bool(oracle_freeze_front_legs),
            "oracle_front_anchor_delay_seconds": float(
                oracle_front_anchor_delay_seconds
            ),
        }
        if "torque_envelope" in final_info:
            report["torque_envelope"] = final_info["torque_envelope"]
        if "body_wall_contact_audit" in final_info:
            report["body_wall_contact_audit"] = final_info[
                "body_wall_contact_audit"
            ]
        rank, score = rank_and_score(report)
        report["milestone_rank"] = rank
        report["milestone_name"] = {
            0: "hard_failure_or_no_approach",
            1: "front_support_retained",
            2: "rear_progress_with_front_support",
            3: "rear_top",
            4: "stable_exit",
        }[rank]
        report["tie_break_score"] = score - report["mean_alignment_target_l2"]
        if trace_path is not None:
            np.savez_compressed(
                trace_path,
                **{name: np.asarray(values) for name, values in trace.items()},
            )
        return report
    finally:
        if recorder is not None and gif_path is not None:
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            recorder.close(gif_path)
        if owns_env:
            env.close()


def main() -> int:
    args = parse_args()
    reference_path, phase_path, rescue_path, output = validate_args(args)
    before_assets = asset_hashes()
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    agent = load_reference(reference_path)
    rescue_candidate = load_candidate(rescue_path)
    case_kwargs = {
        "depth_m": float(args.depth_m),
        "command_mps": float(args.command_mps),
        "initial_state": (float(args.lateral_m), float(args.yaw_deg)),
        "rollout_seed": int(args.rollout_seed),
        "early_threshold": float(args.early_terrain_threshold),
        "approach_threshold": float(args.approach_terrain_threshold),
        "prelift_from_early": bool(args.prelift_from_early),
        "episode_seconds": float(args.episode_seconds),
        "leg_rate_per_second": float(args.leg_rate_per_second),
        "post_from_approach": bool(args.post_from_approach),
        "post_delay_seconds": float(args.post_delay_seconds),
        "front_post_delay_seconds": float(args.front_post_delay_seconds),
        "late_post_boost_scale": float(args.late_post_boost_scale),
        "late_post_boost_delay_seconds": float(args.late_post_boost_delay_seconds),
        "late_post_boost_hold_seconds": float(args.late_post_boost_hold_seconds),
        "late_post_target_limit": float(args.late_post_target_limit),
    }
    zero = np.zeros(2, dtype=np.float64)
    mean = zero.copy()
    std = np.full(2, min(0.25, args.wheel_bound * 0.6), dtype=np.float64)
    floor = np.full(2, 0.04, dtype=np.float64)
    baseline = rollout(
        agent,
        phase_path,
        rescue_candidate,
        zero,
        label="zero_early_steering",
        trace_path=output / "baseline_trace.npz",
        **case_kwargs,
    )
    print(
        f"[baseline] reason={baseline['termination_reason']} "
        f"early={baseline['alignment_trigger_step']} approach={baseline['approach_trigger_step']} "
        f"front={baseline['front_phase_trigger_step']} roll={baseline['max_abs_roll_deg']:.2f}deg",
        flush=True,
    )
    seeds = structured_seeds(float(args.wheel_bound))
    reports: list[dict[str, Any]] = []
    all_candidates: list[tuple[dict[str, Any], np.ndarray]] = []
    for iteration in range(1, args.iterations + 1):
        population: list[tuple[str, np.ndarray]] = []
        if iteration == 1:
            population.extend((name, value.copy()) for name, value in seeds)
        else:
            population.extend([("zero", zero.copy()), ("cem_mean", mean.copy())])
        while len(population) < args.population:
            sample = np.clip(
                mean + std * rng.normal(size=2), -args.wheel_bound, args.wheel_bound
            )
            population.append((f"sample_{len(population):03d}", sample))
        population = population[: args.population]

        def evaluate(item: tuple[int, tuple[str, np.ndarray]]) -> tuple[dict[str, Any], np.ndarray]:
            index, (name, value) = item
            report = rollout(
                agent,
                phase_path,
                rescue_candidate,
                value,
                label=f"iter{iteration:02d}_{index:03d}_{name}",
                expected_early_step=baseline["alignment_trigger_step"],
                expected_prefix_sha256=baseline["prefix_action_sha256"],
                **case_kwargs,
            )
            return report, value

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            evaluated = list(executor.map(evaluate, enumerate(population)))
        all_candidates.extend(evaluated)
        ordered = sorted(evaluated, key=candidate_key, reverse=True)
        eligible = [item for item in ordered if not item[0]["hard_failure"]]
        elite_count = max(2, int(math.ceil(args.population * args.elite_fraction)))
        elites = eligible[:elite_count]
        update = len(elites) >= 2
        if update:
            values = np.stack([item[1] for item in elites])
            rate = args.cem_update_rate
            mean = np.clip(
                (1.0 - rate) * mean + rate * values.mean(axis=0),
                -args.wheel_bound,
                args.wheel_bound,
            )
            std = np.maximum((1.0 - rate) * std + rate * values.std(axis=0), floor)
        best = ordered[0][0]
        iteration_report = {
            "iteration": iteration,
            "population": len(evaluated),
            "eligible_count": len(eligible),
            "elite_count": len(elites) if update else 0,
            "cem_update_applied": update,
            "rear_top_count": sum(bool(item[0]["rear_top_latched"]) for item in evaluated),
            "safe_success_count": sum(bool(item[0]["success"]) for item in evaluated),
            "hard_failure_count": sum(bool(item[0]["hard_failure"]) for item in evaluated),
            "best": best,
            "candidates": [item[0] for item in evaluated],
        }
        reports.append(iteration_report)
        np.savez_compressed(
            output / f"population_iter_{iteration:04d}.npz",
            labels=np.asarray([item[0]["label"] for item in evaluated]),
            steering=np.stack([item[1] for item in evaluated]),
        )
        print(
            f"[cem] iter={iteration}/{args.iterations} rank={best['milestone_rank']} "
            f"eligible={len(eligible)}/{len(evaluated)} "
            f"rear_top={iteration_report['rear_top_count']} "
            f"success={iteration_report['safe_success_count']} update={int(update)}",
            flush=True,
        )
        if not update:
            break
    best_summary, best_steering = max(all_candidates, key=candidate_key)
    np.savez_compressed(output / "best_steering.npz", steering=best_steering)
    gif_dir = output / "gifs"
    baseline_replay = rollout(
        agent,
        phase_path,
        rescue_candidate,
        zero,
        label="zero_early_steering_replay",
        gif_path=gif_dir / "zero_early_steering.gif",
        gif_period=float(args.gif_frame_period),
        expected_early_step=baseline["alignment_trigger_step"],
        expected_prefix_sha256=baseline["prefix_action_sha256"],
        **case_kwargs,
    )
    best_replay = rollout(
        agent,
        phase_path,
        rescue_candidate,
        best_steering,
        label="best_early_steering_replay",
        gif_path=gif_dir / "best_early_steering.gif",
        gif_period=float(args.gif_frame_period),
        trace_path=output / "best_trace.npz",
        expected_early_step=baseline["alignment_trigger_step"],
        expected_prefix_sha256=baseline["prefix_action_sha256"],
        **case_kwargs,
    )
    replay_match = all(
        best_replay[key] == best_summary[key]
        for key in (
            "success",
            "termination_reason",
            "episode_steps",
            "alignment_trigger_step",
            "approach_trigger_step",
            "front_phase_trigger_step",
            "front_top_latched",
            "rear_top_latched",
            "base_cross_latched",
        )
    )
    after_assets = asset_hashes()
    if before_assets != after_assets:
        raise RuntimeError("official assets changed during early-steering search")
    summary = {
        "purpose": "early bounded wheel-differential alignment before two-stage rescue",
        "geometry_scope": "official-derived diagonal vertical-wall proxy pit, not full official track mesh",
        "reference_checkpoint": str(reference_path),
        "reference_checkpoint_sha256": sha256(reference_path),
        "phase_estimator": str(phase_path),
        "phase_estimator_sha256": sha256(phase_path),
        "rescue_candidate": str(rescue_path),
        "rescue_candidate_sha256": sha256(rescue_path),
        "case": {
            "depth_m": case_kwargs["depth_m"],
            "command_mps": case_kwargs["command_mps"],
            "lateral_m": case_kwargs["initial_state"][0],
            "yaw_deg": case_kwargs["initial_state"][1],
            "seed": case_kwargs["rollout_seed"],
        },
        "constraints": {
            "continuous_torque_limits_nm": {"leg": 50.0, "wheel": 14.0},
            "500_n": "report_only",
            "1200_n": "hard_failure",
            "roll_25_deg": "hard_failure",
            "body_wall_contact": "hard_failure",
            "early_wheel_indices": EARLY_WHEEL_INDICES.tolist(),
            "wheel_residual_bound": float(args.wheel_bound),
            "wheel_residual_rate_per_second": WHEEL_RATE_PER_SECOND,
            "early_terrain_threshold": float(args.early_terrain_threshold),
            "approach_terrain_threshold": float(args.approach_terrain_threshold),
            "prelift_from_early": bool(args.prelift_from_early),
            "episode_seconds": float(args.episode_seconds),
            "leg_rate_per_second": float(args.leg_rate_per_second),
            "post_from_approach": bool(args.post_from_approach),
            "post_delay_seconds": float(args.post_delay_seconds),
        },
        "optimizer": {
            "kind": "ranked diagonal CEM over left/right early wheel residuals",
            "population": args.population,
            "requested_iterations": args.iterations,
            "completed_iterations": len(reports),
            "workers": args.workers,
            "seed": args.seed,
        },
        "baseline": baseline,
        "baseline_replay": baseline_replay,
        "iterations": reports,
        "best_search_candidate": best_summary,
        "best_replay": best_replay,
        "best_replay_match": replay_match,
        "official_assets_modified": False,
        "official_asset_hashes_before": before_assets,
        "official_asset_hashes_after": after_assets,
        "files": {
            "best_steering": str(output / "best_steering.npz"),
            "baseline_gif": str(gif_dir / "zero_early_steering.gif"),
            "best_gif": str(gif_dir / "best_early_steering.gif"),
            "summary": str(output / "summary.json"),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(
        f"[result] best={best_replay['milestone_name']} "
        f"steering={best_steering.tolist()} success={int(best_replay['success'])} "
        f"replay={int(replay_match)} summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
