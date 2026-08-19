#!/usr/bin/env python3
"""Search a bounded single-rear-leg swing after verified three-point support."""

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
    FRONT_ACTIVE,
    LEG_ACTIVE,
    LEG_WHEEL_ACTIVE,
    expand,
    expand_front,
    load_boost,
    load_height_crossing_placement,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRECONTACT_SUMMARY = PROJECT_ROOT / (
    "logs/mujoco/s10_m20_precontact_teacher_depth03456_local60_"
    "20260812_v1/summary.json"
)
SOURCE_0350_SUMMARY = PROJECT_ROOT / (
    "logs/mujoco/s10_m20_threepoint_probe_depth0350_from03456_"
    "20260812_v1/summary.json"
)
OLD_SUCCESS_SUMMARY = PROJECT_ROOT / (
    "logs/mujoco/s10_m20_depth0345_oracle_frontpreload_magaxis7_"
    "20260811_v1/summary.json"
)
HEIGHT_SUMMARY = PROJECT_ROOT / (
    "logs/mujoco/s10_m20_depth0345_oracle_rearplacement_trigger040_hold040_"
    "cem48x4_20260811_v1/summary.json"
)
JOINT_SUMMARY = PROJECT_ROOT / (
    "logs/mujoco/s10_m20_depth0345_oracle_jointcoord_warmfrontrear_rate40_"
    "cem48x4_20260811_v1/summary.json"
)
DEPTH_M = 0.350


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--top-k-no-freeze", type=int, default=6)
    parser.add_argument("--contact-threshold-n", type=float, default=50.0)
    parser.add_argument("--roll-threshold-deg", type=float, default=10.0)
    return parser.parse_args()


def finite_vector(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite {shape} vector")
    return result


def optional_float(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    return -math.inf if value is None else float(value)


def result_key(report: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(report["success"] and not report["hard_failure"]),
        float(report["base_cross_latched"] and not report["hard_failure"]),
        float(report["rear_top_latched"] and not report["hard_failure"]),
        -float(report["hard_failure"]),
        optional_float(report, "peak_threepoint_lagging_top_margin_m"),
        optional_float(report, "peak_threepoint_lagging_progress_m"),
        float(report["threepoint_leading_geometry_rate"]),
        float(report["threepoint_leading_contact_rate"]),
        float(report["threepoint_front_support_rate"]),
        float(report["base_progress_gain_after_approach_m"]),
        -float(report["max_abs_roll_deg"]),
        -float(report["max_contact_force_n"]),
    )


def regression_key(report: dict[str, Any]) -> tuple[Any, ...]:
    return (
        bool(report["success"]),
        bool(report["hard_failure"]),
        bool(report["rear_top_latched"]),
        bool(report["base_cross_latched"]),
        str(report["termination_reason"]),
        int(report["episode_steps"]),
        float(report["peak_rear_height_after_second_m"]),
        float(report["peak_rear_progress_after_second_m"]),
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.top_k_no_freeze < 1:
        raise ValueError("workers and top-k-no-freeze must be positive")
    if not 0.0 <= args.contact_threshold_n <= 300.0:
        raise ValueError("contact threshold must be in [0, 300] N")
    if not 1.0 <= args.roll_threshold_deg <= 25.0:
        raise ValueError("roll threshold must be in [1, 25] deg")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    paths = {
        "reference": REFERENCE_CHECKPOINT.expanduser().resolve(),
        "phase": PHASE_ESTIMATOR.expanduser().resolve(),
        "teacher": DEFAULT_TEACHER.expanduser().resolve(),
        "first": DEFAULT_FIRST_BOOST.expanduser().resolve(),
        "lift": DEFAULT_LIFT_BOOST.expanduser().resolve(),
        "height_summary": HEIGHT_SUMMARY.expanduser().resolve(),
        "joint_summary": JOINT_SUMMARY.expanduser().resolve(),
        "old_success": OLD_SUCCESS_SUMMARY.expanduser().resolve(),
        "precontact": PRECONTACT_SUMMARY.expanduser().resolve(),
        "source_0350": SOURCE_0350_SUMMARY.expanduser().resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

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
    pulse_delay = float(pulse_case["delay_seconds"])
    pulse_hold = float(pulse_case["hold_seconds"])

    expected_0350 = json.loads(paths["source_0350"].read_text(encoding="utf-8"))[
        "best_search_replay"
    ]

    def evaluate(
        vector: np.ndarray,
        hold_s: float,
        freeze: bool,
        label: str,
        *,
        monitor: bool = True,
        trace: Path | None = None,
        gif: Path | None = None,
    ) -> dict[str, Any]:
        return rollout(
            agent,
            paths["phase"],
            candidate,
            steering,
            label=label,
            depth_m=DEPTH_M,
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
            precontact_leg_delay_after_approach_seconds=pulse_delay,
            precontact_leg_hold_seconds=pulse_hold,
            precontact_leg_rate_per_second=4.0,
            threepoint_swing_leg_vector=vector,
            threepoint_hold_seconds=hold_s,
            threepoint_leg_rate_per_second=4.0,
            threepoint_monitor_enabled=monitor,
            threepoint_freeze_leading_leg=freeze,
            threepoint_min_leading_rear_contact_force_n=float(
                args.contact_threshold_n
            ),
            threepoint_max_abs_roll_deg=float(args.roll_threshold_deg),
            late_momentum_leg_vector=old_preload,
            late_momentum_leg_delay_seconds=float(old_case["delay_seconds"]),
            late_momentum_leg_hold_seconds=float(old_case["hold_seconds"]),
            late_momentum_leg_rate_per_second=4.0,
            oracle_front_wheel_action=-0.5,
            oracle_front_anchor_delay_seconds=0.82,
            leg_peak_nm=76.4,
            wheel_peak_nm=21.6,
            max_continuous_exceedance_s=0.5,
            trace_path=trace,
            gif_path=gif,
        )

    zero = np.zeros(3, dtype=np.float64)
    baseline = evaluate(
        zero,
        0.20,
        False,
        "baseline_monitor_only",
        trace=output / "baseline_trace.npz",
    )
    baseline_match = regression_key(baseline) == regression_key(expected_0350)
    if not baseline_match:
        (output / "baseline_mismatch.json").write_text(
            json.dumps(
                {"actual": baseline, "expected": expected_0350},
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("threepoint monitor changed the locked 0.350 m baseline")

    patterns = {
        "hipy_pos": np.asarray([0.0, 1.0, 0.0]),
        "hipy_neg": np.asarray([0.0, -1.0, 0.0]),
        "knee_pos": np.asarray([0.0, 0.0, 1.0]),
        "knee_neg": np.asarray([0.0, 0.0, -1.0]),
        "hipy_pos_knee_pos": np.asarray([0.0, 1.0, 1.0]),
        "hipy_pos_knee_neg": np.asarray([0.0, 1.0, -1.0]),
        "hipy_neg_knee_pos": np.asarray([0.0, -1.0, 1.0]),
        "hipy_neg_knee_neg": np.asarray([0.0, -1.0, -1.0]),
    }
    frozen_configs = [
        (name, scale, hold_s, np.clip(direction * scale, -0.30, 0.30))
        for name, direction in patterns.items()
        for scale in (0.15, 0.30)
        for hold_s in (0.12, 0.20, 0.28)
    ]

    def run_frozen(
        item: tuple[int, tuple[str, float, float, np.ndarray]],
    ) -> tuple[int, dict[str, Any]]:
        index, (name, scale, hold_s, vector) = item
        return index, evaluate(
            vector,
            hold_s,
            True,
            f"frozen{index:03d}_{name}_s{scale:.2f}_h{hold_s:.2f}",
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        frozen_reports = list(executor.map(run_frozen, enumerate(frozen_configs)))
    ranked_frozen = sorted(
        (
            (report, frozen_configs[index], index)
            for index, report in frozen_reports
        ),
        key=lambda item: result_key(item[0]),
        reverse=True,
    )

    no_freeze_configs = [item[1] for item in ranked_frozen[: args.top_k_no_freeze]]

    def run_no_freeze(
        item: tuple[int, tuple[str, float, float, np.ndarray]],
    ) -> tuple[int, dict[str, Any]]:
        index, (name, scale, hold_s, vector) = item
        return index, evaluate(
            vector,
            hold_s,
            False,
            f"free{index:03d}_{name}_s{scale:.2f}_h{hold_s:.2f}",
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        no_freeze_reports = list(
            executor.map(run_no_freeze, enumerate(no_freeze_configs))
        )

    all_cases: list[dict[str, Any]] = [
        {
            "stage": "freeze_leading",
            "index": index,
            "name": config[0],
            "scale": config[1],
            "hold_seconds": config[2],
            "vector": config[3].tolist(),
            "report": report,
        }
        for report, config, index in ranked_frozen
    ]
    all_cases.extend(
        {
            "stage": "no_freeze",
            "index": index,
            "name": no_freeze_configs[index][0],
            "scale": no_freeze_configs[index][1],
            "hold_seconds": no_freeze_configs[index][2],
            "vector": no_freeze_configs[index][3].tolist(),
            "report": report,
        }
        for index, report in no_freeze_reports
    )
    all_cases.append(
        {
            "stage": "freeze_only",
            "index": -1,
            "name": "zero",
            "scale": 0.0,
            "hold_seconds": 0.20,
            "vector": zero.tolist(),
            "report": evaluate(zero, 0.20, True, "freeze_only"),
        }
    )
    all_cases.sort(key=lambda item: result_key(item["report"]), reverse=True)
    best = all_cases[0]
    best_replay = evaluate(
        finite_vector(best["vector"], (3,), "best vector"),
        float(best["hold_seconds"]),
        best["stage"] != "no_freeze",
        "best_replay",
        trace=output / "best_trace.npz",
        gif=output / "gifs" / "best.gif",
    )

    after_assets = asset_hashes()
    if before_assets != after_assets:
        raise RuntimeError("official assets changed during threepoint search")
    summary = {
        "purpose": "verified three-point support and bounded lagging-rear-leg swing audit",
        "mechanical_feasibility_only": True,
        "deployable": False,
        "training_performed": False,
        "depth_m": DEPTH_M,
        "baseline_monitor_only": baseline,
        "baseline_matches_locked_0350": baseline_match,
        "event_definition": {
            "front_pair_top_contact": True,
            "exactly_one_rear_wheel_above_top_geometry_gate": True,
            "leading_rear_contact_force_n": float(args.contact_threshold_n),
            "max_abs_roll_deg": float(args.roll_threshold_deg),
        },
        "grid": {
            "patterns": list(patterns),
            "scales": [0.15, 0.30],
            "hold_seconds": [0.12, 0.20, 0.28],
            "frozen_case_count": len(frozen_configs),
            "no_freeze_case_count": len(no_freeze_configs),
        },
        "cases_ranked": all_cases,
        "best": best,
        "best_replay": best_replay,
        "best_replay_matches": result_key(best_replay) == result_key(best["report"]),
        "rear_top_observed": bool(best_replay["rear_top_latched"]),
        "base_cross_observed": bool(best_replay["base_cross_latched"]),
        "success_observed": bool(best_replay["success"]),
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "official_assets_before": before_assets,
        "official_assets_after": after_assets,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[threepoint] cases={len(all_cases)} trigger={best_replay['threepoint_trigger_step']} "
        f"best={best['stage']}/{best['name']} scale={best['scale']:.2f} "
        f"hold={best['hold_seconds']:.2f} rear={int(best_replay['rear_top_latched'])} "
        f"base={int(best_replay['base_cross_latched'])} success={int(best_replay['success'])} "
        f"lag_x={optional_float(best_replay, 'peak_threepoint_lagging_progress_m'):.6f} "
        f"lead_geom={best_replay['threepoint_leading_geometry_rate']:.3f} "
        f"lead_contact={best_replay['threepoint_leading_contact_rate']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
