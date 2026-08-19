#!/usr/bin/env python3
"""Train 0.22 m stable-exit capability while retaining 0.18 m fixed five."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import train_s10_m20_curriculum_ppo as base_runner
import train_s10_m20_lidar_curriculum_ppo as lidar_runner
from s10_m20_lidar_capability_env import (
    S10M20LidarCapabilityEnv,
    S10M20LidarCapabilityVecEnv,
)


TARGET_DEPTH_M = 0.22
RETENTION_DEPTH_M = 0.18
FIXED_STATES = (
    (0.0, 0.0),
    (0.02, 0.0),
    (-0.02, 0.0),
    (0.0, 2.0),
    (0.0, -2.0),
)


def configure_runner() -> None:
    lidar_runner.configure_runner()
    base_runner.S10M20CurriculumEnv = S10M20LidarCapabilityEnv
    base_runner.S10M20CurriculumVecEnv = S10M20LidarCapabilityVecEnv
    evaluate_single_depth = base_runner.evaluate_policy
    collect_single_depth_anchors = base_runner.collect_successful_baseline_anchors

    def evaluate_policy(
        agent: Any,
        *,
        depth_m: float,
        command_mps: float,
        device: Any,
        validation_states: tuple[tuple[float, float], ...] = FIXED_STATES,
        gif_dir: Path | None = None,
        gif_frame_period: float = 0.10,
    ) -> dict[str, Any]:
        if abs(depth_m - TARGET_DEPTH_M) > 1.0e-12:
            raise ValueError(f"this wrapper requires --depth-m {TARGET_DEPTH_M}")
        target = evaluate_single_depth(
            agent,
            depth_m=TARGET_DEPTH_M,
            command_mps=command_mps,
            device=device,
            validation_states=FIXED_STATES,
            gif_dir=gif_dir / "target_depth_0p22" if gif_dir is not None else None,
            gif_frame_period=gif_frame_period,
        )
        retention = evaluate_single_depth(
            agent,
            depth_m=RETENTION_DEPTH_M,
            command_mps=command_mps,
            device=device,
            validation_states=FIXED_STATES,
            gif_dir=gif_dir / "retention_depth_0p18" if gif_dir is not None else None,
            gif_frame_period=gif_frame_period,
        )
        cases: list[dict[str, Any]] = []
        for role, case_depth, report in (
            ("target", TARGET_DEPTH_M, target),
            ("retention", RETENTION_DEPTH_M, retention),
        ):
            for case in report["cases"]:
                annotated = dict(case)
                annotated["validation_role"] = role
                annotated["depth_m"] = case_depth
                cases.append(annotated)
        successful = [index for index, case in enumerate(cases) if case["safe_success"]]
        physical_target = sum(
            bool(case["front_top_latched"])
            and bool(case["rear_top_latched"])
            and bool(case["base_cross_latched"])
            for case in target["cases"]
        )
        target_front_top = sum(bool(case["front_top_latched"]) for case in target["cases"])
        target_rear_top = sum(bool(case["rear_top_latched"]) for case in target["cases"])
        target_base_cross = sum(bool(case["base_cross_latched"]) for case in target["cases"])
        return {
            "depth_m": TARGET_DEPTH_M,
            "target_depth_m": TARGET_DEPTH_M,
            "retention_depth_m": RETENTION_DEPTH_M,
            "success_gate": "stable physical exit; 500 N report-only",
            "cases": cases,
            "safe_success_count": len(successful),
            "safe_success_rate": len(successful) / len(cases),
            "safe_success_indexes": successful,
            "mean_return": base_runner.mean(
                [float(case["episode_return"]) for case in cases]
            ),
            "max_force_n": max(float(case["max_force_n"]) for case in cases),
            "target_capability_success_count": target["safe_success_count"],
            "target_physical_completion_count": physical_target,
            "target_front_top_count": target_front_top,
            "target_rear_top_count": target_rear_top,
            "target_base_cross_count": target_base_cross,
            "retention_capability_success_count": retention["safe_success_count"],
            "target_evaluation": target,
            "retention_evaluation": retention,
        }

    def validation_score(
        report: dict[str, Any],
    ) -> tuple[int, int, int, int, int, float, float]:
        target = report["target_evaluation"]
        return (
            int(report["target_capability_success_count"]),
            int(report["target_base_cross_count"]),
            int(report["target_rear_top_count"]),
            int(report["target_front_top_count"]),
            int(report["retention_capability_success_count"]),
            float(target["mean_return"]),
            -float(target["max_force_n"]),
        )

    def collect_retention_anchors(
        agent: Any, *, depth_m: float, command_mps: float, device: Any
    ) -> tuple[Any, Any, list[int]]:
        if abs(depth_m - TARGET_DEPTH_M) > 1.0e-12:
            raise ValueError("unexpected target depth while collecting anchors")
        observations, actions, indexes = collect_single_depth_anchors(
            agent,
            depth_m=RETENTION_DEPTH_M,
            command_mps=command_mps,
            device=device,
        )
        return observations, actions, [len(FIXED_STATES) + index for index in indexes]

    base_runner.evaluate_policy = evaluate_policy
    base_runner.validation_score = validation_score
    base_runner.collect_successful_baseline_anchors = collect_retention_anchors
    original_parse_args = base_runner.parse_args

    def parse_args() -> Any:
        args = original_parse_args()
        if abs(args.depth_m - TARGET_DEPTH_M) > 1.0e-12:
            raise ValueError(f"use --depth-m {TARGET_DEPTH_M} with this wrapper")
        args.validation_gate = "0p22_capability_fixed5_plus_0p18_retention_fixed5"
        args.validation_case_count = 10
        args.force_500n_role = "report_only"
        args.hard_contact_force_limit_n = 1200.0
        return args

    base_runner.parse_args = parse_args


def main() -> int:
    configure_runner()
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
