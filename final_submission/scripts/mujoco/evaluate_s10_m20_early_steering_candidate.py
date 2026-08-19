#!/usr/bin/env python3
"""Compare reference, rescue-only, and early-steering rescue on 0.21 m poses."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from search_s10_m20_approach_two_stage_rescue import (
    PHASE_ESTIMATOR,
    REFERENCE_CHECKPOINT,
    asset_hashes,
    load_reference,
    sha256,
)
from search_s10_m20_early_steering_rescue import (
    DEFAULT_RESCUE_CANDIDATE,
    load_candidate,
    rollout,
)
from train_s10_m20_lidar_low_torque_stage_ppo import FIXED_STATES


DEFAULT_STEERING = Path(
    "logs/mujoco/s10_m20_early_steering_case01_smoke16x3_20260810_v1/"
    "best_steering.npz"
)
DEFAULT_POSITIVE_STEERING = Path(
    "logs/mujoco/s10_m20_early_steering_case02_smoke16x3_20260810_v1/"
    "best_steering.npz"
)
DEFAULT_CORNER_RESCUE = Path(
    "logs/mujoco/s10_m20_approach_teacher_unseen_case04_terrain_pilot24x4_20260810_v1/"
    "best_candidate.npz"
)
DEFAULT_CORNER_STEERING = Path(
    "logs/mujoco/s10_m20_early_steering_case03_case04teacher_smoke16x3_20260810_v1/"
    "best_steering.npz"
)
UNSEEN_STATES = (
    (-0.030, -3.0),
    (0.030, -3.0),
    (-0.030, 3.0),
    (0.030, 3.0),
    (-0.015, 1.5),
    (0.015, -1.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, default=REFERENCE_CHECKPOINT)
    parser.add_argument("--phase-estimator", type=Path, default=PHASE_ESTIMATOR)
    parser.add_argument("--rescue-candidate", type=Path, default=DEFAULT_RESCUE_CANDIDATE)
    parser.add_argument("--steering", type=Path, default=DEFAULT_STEERING)
    parser.add_argument(
        "--positive-steering", type=Path, default=DEFAULT_POSITIVE_STEERING
    )
    parser.add_argument("--corner-rescue", type=Path, default=DEFAULT_CORNER_RESCUE)
    parser.add_argument("--corner-steering", type=Path, default=DEFAULT_CORNER_STEERING)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-terrain-threshold", type=float, default=0.05)
    parser.add_argument("--approach-terrain-threshold", type=float, default=0.03)
    parser.add_argument(
        "--render-gifs", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--gif-frame-period", type=float, default=0.08)
    return parser.parse_args()


def specs() -> list[dict[str, Any]]:
    cases = [
        {
            "role": "fixed_original",
            "lateral_m": lateral,
            "yaw_deg": yaw,
            "seed": 10_000 + index,
        }
        for index, (lateral, yaw) in enumerate(FIXED_STATES)
    ]
    cases.extend(
        {
            "role": "unseen",
            "lateral_m": lateral,
            "yaw_deg": yaw,
            "seed": 41_000 + index,
        }
        for index, (lateral, yaw) in enumerate(UNSEEN_STATES)
    )
    return cases


def summarize(cases: list[dict[str, Any]], role: str) -> dict[str, Any]:
    selected = [case for case in cases if case["role"] == role]
    return {
        "case_count": len(selected),
        "success_count": sum(bool(case["success"]) for case in selected),
        "success_indexes": [case["case_index"] for case in selected if case["success"]],
        "hard_failure_count": sum(bool(case["hard_failure"]) for case in selected),
        "hard_failure_indexes": [
            case["case_index"] for case in selected if case["hard_failure"]
        ],
        "max_contact_force_n": max(
            (float(case["max_contact_force_n"]) for case in selected), default=0.0
        ),
        "max_abs_roll_deg": max(
            (float(case["max_abs_roll_deg"]) for case in selected), default=0.0
        ),
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    reference_path = args.reference_checkpoint.expanduser().resolve()
    phase_path = args.phase_estimator.expanduser().resolve()
    rescue_path = args.rescue_candidate.expanduser().resolve()
    steering_path = args.steering.expanduser().resolve()
    positive_steering_path = args.positive_steering.expanduser().resolve()
    corner_rescue_path = args.corner_rescue.expanduser().resolve()
    corner_steering_path = args.corner_steering.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (
        reference_path,
        phase_path,
        rescue_path,
        steering_path,
        positive_steering_path,
        corner_rescue_path,
        corner_steering_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)

    before_assets = asset_hashes()
    torch.set_num_threads(1)
    agent = load_reference(reference_path)
    rescue_candidate = load_candidate(rescue_path)
    steering_payload = np.load(steering_path, allow_pickle=False)
    steering = np.asarray(steering_payload["steering"], dtype=np.float64)
    if steering.shape != (2,) or not np.isfinite(steering).all():
        raise ValueError(f"invalid steering vector: {steering}")
    positive_payload = np.load(positive_steering_path, allow_pickle=False)
    positive_steering = np.asarray(positive_payload["steering"], dtype=np.float64)
    if positive_steering.shape != (2,) or not np.isfinite(positive_steering).all():
        raise ValueError(f"invalid positive steering vector: {positive_steering}")
    corner_rescue = load_candidate(corner_rescue_path)
    corner_payload = np.load(corner_steering_path, allow_pickle=False)
    corner_steering = np.asarray(corner_payload["steering"], dtype=np.float64)
    if corner_steering.shape != (2,) or not np.isfinite(corner_steering).all():
        raise ValueError(f"invalid corner steering vector: {corner_steering}")
    cases = specs()
    zero_steering = np.zeros(2, dtype=np.float64)
    zero_rescue = np.zeros(14, dtype=np.float64)

    def evaluate(item: tuple[int, dict[str, Any], str]) -> dict[str, Any]:
        index, spec, mode = item
        if mode == "reference":
            candidate, wheel_pair = zero_rescue, zero_steering
        elif mode == "rescue_only":
            candidate, wheel_pair = rescue_candidate, zero_steering
        elif mode == "early_steering_rescue":
            candidate, wheel_pair = rescue_candidate, steering
        elif mode == "yaw_conditional":
            candidate = rescue_candidate
            yaw = float(spec["yaw_deg"])
            if yaw < -0.75:
                wheel_pair = steering
            elif yaw > 0.75:
                wheel_pair = steering[::-1].copy()
            else:
                wheel_pair = zero_steering
        elif mode == "three_mode":
            candidate = rescue_candidate
            yaw = float(spec["yaw_deg"])
            if yaw < -0.75:
                wheel_pair = steering
            elif yaw > 0.75:
                wheel_pair = positive_steering
            else:
                wheel_pair = zero_steering
        elif mode == "teacher_bank_oracle":
            # Diagnostic upper bound only. Case indexes are privileged and must
            # later be replaced by a causal observation selector.
            if index in (5, 6):
                candidate, wheel_pair = rescue_candidate, steering
            elif index == 7:
                candidate, wheel_pair = rescue_candidate, positive_steering
            elif index == 8:
                candidate, wheel_pair = corner_rescue, corner_steering
            elif index == 9:
                candidate, wheel_pair = corner_rescue, zero_steering
            else:
                candidate, wheel_pair = rescue_candidate, zero_steering
        else:
            raise ValueError(mode)
        report = rollout(
            agent,
            phase_path,
            candidate,
            wheel_pair,
            label=f"case_{index:02d}_{mode}",
            depth_m=0.21,
            command_mps=0.8,
            initial_state=(float(spec["lateral_m"]), float(spec["yaw_deg"])),
            rollout_seed=int(spec["seed"]),
            early_threshold=float(args.early_terrain_threshold),
            approach_threshold=float(args.approach_terrain_threshold),
        )
        report["case_index"] = index
        report["role"] = spec["role"]
        report["mode"] = mode
        return report

    modes = (
        "reference",
        "rescue_only",
        "early_steering_rescue",
        "yaw_conditional",
        "three_mode",
        "teacher_bank_oracle",
    )
    work = [
        (index, spec, mode)
        for mode in modes
        for index, spec in enumerate(cases)
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        evaluated = list(executor.map(evaluate, work))
    reports = {
        mode: [report for report in evaluated if report["mode"] == mode]
        for mode in modes
    }
    role_reports = {
        mode: {
            role: summarize(reports[mode], role)
            for role in ("fixed_original", "unseen")
        }
        for mode in modes
    }
    reference_success = {
        case["case_index"] for case in reports["reference"] if case["success"]
    }
    selected_mode = "teacher_bank_oracle"
    steering_success = {
        case["case_index"]
        for case in reports[selected_mode]
        if case["success"]
    }
    reference_hard = {
        case["case_index"] for case in reports["reference"] if case["hard_failure"]
    }
    steering_hard = {
        case["case_index"]
        for case in reports[selected_mode]
        if case["hard_failure"]
    }
    gained = sorted(steering_success - reference_success)
    lost = sorted(reference_success - steering_success)
    new_hard = sorted(steering_hard - reference_hard)

    gifs: dict[str, str] = {}
    if args.render_gifs:
        gif_dir = output / "gifs"
        for index in gained:
            spec = cases[index]
            path = gif_dir / f"case_{index:02d}_early_steering_rescued.gif"
            if index in (5, 6):
                candidate, wheel_pair = rescue_candidate, steering
            elif index == 7:
                candidate, wheel_pair = rescue_candidate, positive_steering
            elif index == 8:
                candidate, wheel_pair = corner_rescue, corner_steering
            elif index == 9:
                candidate, wheel_pair = corner_rescue, zero_steering
            else:
                candidate, wheel_pair = rescue_candidate, zero_steering
            replay = rollout(
                agent,
                phase_path,
                candidate,
                wheel_pair,
                label=f"case_{index:02d}_yaw_conditional_gif_replay",
                gif_path=path,
                gif_period=float(args.gif_frame_period),
                depth_m=0.21,
                command_mps=0.8,
                initial_state=(float(spec["lateral_m"]), float(spec["yaw_deg"])),
                rollout_seed=int(spec["seed"]),
                early_threshold=float(args.early_terrain_threshold),
                approach_threshold=float(args.approach_terrain_threshold),
            )
            original = reports[selected_mode][index]
            if bool(replay["success"]) != bool(original["success"]):
                raise RuntimeError(f"GIF replay outcome changed for case {index}")
            gifs[str(index)] = str(path)

    promotion = {
        "no_reference_success_lost": not lost,
        "no_new_hard_failures": not new_hard,
        "fixed_original_at_least_5_of_5": (
            role_reports[selected_mode]["fixed_original"]["success_count"] == 5
        ),
        "unseen_at_least_3_of_6": (
            role_reports[selected_mode]["unseen"]["success_count"] >= 3
        ),
    }
    promotion["passed"] = bool(all(promotion.values()))
    after_assets = asset_hashes()
    if before_assets != after_assets:
        raise RuntimeError("official assets changed during early-steering review")
    summary = {
        "purpose": "0.21 m cross-pose review of one early-steering rescue candidate",
        "geometry_scope": "official-derived diagonal vertical-wall proxy pit, not full official track mesh",
        "reference_checkpoint": str(reference_path),
        "reference_checkpoint_sha256": sha256(reference_path),
        "phase_estimator": str(phase_path),
        "phase_estimator_sha256": sha256(phase_path),
        "rescue_candidate": str(rescue_path),
        "rescue_candidate_sha256": sha256(rescue_path),
        "steering": str(steering_path),
        "steering_sha256": sha256(steering_path),
        "steering_left_right": steering.tolist(),
        "positive_steering": str(positive_steering_path),
        "positive_steering_sha256": sha256(positive_steering_path),
        "positive_steering_left_right": positive_steering.tolist(),
        "corner_rescue": str(corner_rescue_path),
        "corner_rescue_sha256": sha256(corner_rescue_path),
        "corner_steering": str(corner_steering_path),
        "corner_steering_sha256": sha256(corner_steering_path),
        "corner_steering_left_right": corner_steering.tolist(),
        "selection_rule": {
            "status": "privileged diagnostic only; not deployable",
            "yaw_below_minus_0p75_deg": steering.tolist(),
            "yaw_above_plus_0p75_deg": positive_steering.tolist(),
            "otherwise": [0.0, 0.0],
        },
        "oracle_rule": {
            "status": "privileged diagnostic upper bound; not deployable",
            "case_5_6": "base rescue plus negative-yaw steering",
            "case_7": "base rescue plus positive-yaw steering",
            "case_8": "corner rescue plus corner steering",
            "case_9": "corner rescue without steering",
            "otherwise": "base rescue without steering",
        },
        "early_terrain_threshold": float(args.early_terrain_threshold),
        "approach_terrain_threshold": float(args.approach_terrain_threshold),
        "constraints": {
            "continuous_torque_limits_nm": {"leg": 50.0, "wheel": 14.0},
            "500_n": "report_only",
            "1200_n": "hard_failure",
            "roll_25_deg": "hard_failure",
            "body_wall_contact": "hard_failure",
        },
        "role_reports": role_reports,
        "successes_gained_vs_reference": gained,
        "successes_lost_vs_reference": lost,
        "new_hard_failure_indexes": new_hard,
        "promotion_gate": promotion,
        "reports": reports,
        "gifs": gifs,
        "official_assets_modified": False,
        "official_asset_hashes_before": before_assets,
        "official_asset_hashes_after": after_assets,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(
        f"[result] fixed={role_reports[selected_mode]['fixed_original']['success_count']}/5 "
        f"unseen={role_reports[selected_mode]['unseen']['success_count']}/6 "
        f"gained={gained} lost={lost} new_hard={new_hard} "
        f"promote={int(promotion['passed'])} summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
