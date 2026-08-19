#!/usr/bin/env python3
"""Train one low-torque S10 depth stage with multi-depth retention gates."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

import train_s10_m20_curriculum_ppo as base_runner
import train_s10_m20_lidar_curriculum_ppo as lidar_runner
from s10_m20_lidar_capability_env import S10M20LidarCapabilityEnv
from train_s10_m20_lidar_021_capability_downstream_actor_ppo import (
    LidarDownstreamActorCritic,
)


FIXED_STATES = (
    (0.0, 0.0),
    (0.02, 0.0),
    (-0.02, 0.0),
    (0.0, 2.0),
    (0.0, -2.0),
)
LEG_LIMIT_NM = 50.0
WHEEL_LIMIT_NM = 14.0


def parse_stage_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target-depth-m", type=float, required=True)
    parser.add_argument("--retention-depth-m", type=float, action="append", default=[])
    return parser.parse_known_args(argv)


def unique_depths(values: list[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("stage depths must be finite and positive")
        if not any(math.isclose(value, old, rel_tol=0.0, abs_tol=1.0e-12) for old in result):
            result.append(float(value))
    return tuple(result)


class S10M20LidarLowTorqueCapabilityEnv(S10M20LidarCapabilityEnv):
    """Capability environment that asserts the official local torque limits."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.model.nu)
        ]
        expected = np.asarray(
            [WHEEL_LIMIT_NM if name and name.endswith("_wheel_joint") else LEG_LIMIT_NM for name in actuator_names],
            dtype=np.float64,
        )
        joint_ids = np.asarray(self.model.actuator_trnid[:, 0], dtype=np.int32)
        actual_ctrl = np.max(np.abs(self.model.actuator_ctrlrange), axis=1)
        actual_joint = np.asarray(
            [np.max(np.abs(self.model.jnt_actfrcrange[joint_id])) for joint_id in joint_ids],
            dtype=np.float64,
        )
        if self.model.nu != 16 or np.count_nonzero(expected == WHEEL_LIMIT_NM) != 4:
            raise RuntimeError("unexpected S10 actuator mapping")
        if not np.allclose(actual_ctrl, expected) or not np.allclose(actual_joint, expected):
            raise RuntimeError(
                f"low-torque training requires official 50/14 Nm limits, got {actual_ctrl.tolist()}"
            )


def depth_schedule(
    *, target_depth_m: float, retention_depths_m: tuple[float, ...], num_envs: int
) -> tuple[float, ...]:
    target_count = max(1, (num_envs + 1) // 2)
    support_candidates = [
        target_depth_m - 0.005,
        target_depth_m - 0.010,
        target_depth_m - 0.020,
        *reversed(retention_depths_m),
    ]
    support = [
        value
        for value in unique_depths(support_candidates)
        if value > 0.0 and value < target_depth_m - 1.0e-12
    ]
    if not support:
        support = [target_depth_m]
    schedule = [target_depth_m] * target_count
    while len(schedule) < num_envs:
        schedule.append(support[(len(schedule) - target_count) % len(support)])
    return tuple(schedule)


def configure_runner(
    *, target_depth_m: float, retention_depths_m: tuple[float, ...]
) -> tuple[float, ...]:
    lidar_runner.configure_runner()
    evaluate_single_depth = base_runner.evaluate_policy

    class MixedDepthCapabilityVecEnv:
        def __init__(self, num_envs: int, **kwargs: Any) -> None:
            if num_envs <= 0:
                raise ValueError("num_envs must be positive")
            requested_depth = float(kwargs.pop("depth_m"))
            if not math.isclose(requested_depth, target_depth_m, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("vector environment received the wrong target depth")
            self.depths_m = depth_schedule(
                target_depth_m=target_depth_m,
                retention_depths_m=retention_depths_m,
                num_envs=num_envs,
            )
            self.envs = [
                S10M20LidarLowTorqueCapabilityEnv(depth_m=depth, **kwargs)
                for depth in self.depths_m
            ]
            self.num_envs = int(num_envs)
            self.action_dim = self.envs[0].action_dim
            self.actor_observation_dim = self.envs[0].actor_observation_dim
            self.critic_observation_dim = self.envs[0].critic_observation_dim

        def reset(self) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
            transitions = [env.reset(seed=index) for index, env in enumerate(self.envs)]
            return (
                np.stack([transition[0] for transition in transitions]),
                np.stack([transition[1] for transition in transitions]),
                [transition[2] for transition in transitions],
            )

        def reset_at(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
            return self.envs[index].reset()

        def close(self) -> None:
            for env in self.envs:
                env.close()

    def evaluate_policy(
        agent: Any,
        *,
        depth_m: float,
        command_mps: float,
        device: torch.device,
        validation_states: tuple[tuple[float, float], ...] = FIXED_STATES,
        gif_dir: Path | None = None,
        gif_frame_period: float = 0.10,
    ) -> dict[str, Any]:
        if not math.isclose(depth_m, target_depth_m, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"this stage requires --depth-m {target_depth_m}")
        reports: list[tuple[str, float, dict[str, Any]]] = []
        for role, case_depth in (
            ("target", target_depth_m),
            *(("retention", value) for value in retention_depths_m),
        ):
            depth_label = f"{case_depth:.10f}".rstrip("0").rstrip(".").replace(".", "p")
            report = evaluate_single_depth(
                agent,
                depth_m=case_depth,
                command_mps=command_mps,
                device=device,
                validation_states=FIXED_STATES,
                gif_dir=gif_dir / f"{role}_{depth_label}" if gif_dir is not None else None,
                gif_frame_period=gif_frame_period,
            )
            reports.append((role, case_depth, report))

        cases: list[dict[str, Any]] = []
        for role, case_depth, report in reports:
            for case in report["cases"]:
                annotated = dict(case)
                annotated["validation_role"] = role
                annotated["depth_m"] = case_depth
                cases.append(annotated)
        successful = [index for index, case in enumerate(cases) if case["safe_success"]]
        target = reports[0][2]
        retention_reports = {f"{depth:.10f}": report for _, depth, report in reports[1:]}
        return {
            "depth_m": target_depth_m,
            "target_depth_m": target_depth_m,
            "retention_depths_m": list(retention_depths_m),
            "success_gate": "stable physical exit; 500 N report-only; official 50/14 Nm",
            "cases": cases,
            "safe_success_count": len(successful),
            "safe_success_rate": len(successful) / len(cases),
            "safe_success_indexes": successful,
            "mean_return": base_runner.mean([float(case["episode_return"]) for case in cases]),
            "max_force_n": max(float(case["max_force_n"]) for case in cases),
            "target_capability_success_count": int(target["safe_success_count"]),
            "target_front_top_count": sum(bool(case["front_top_latched"]) for case in target["cases"]),
            "target_rear_top_count": sum(bool(case["rear_top_latched"]) for case in target["cases"]),
            "target_base_cross_count": sum(bool(case["base_cross_latched"]) for case in target["cases"]),
            "retention_capability_success_counts": {
                key: int(report["safe_success_count"]) for key, report in retention_reports.items()
            },
            "target_evaluation": target,
            "retention_evaluations": retention_reports,
        }

    def validation_score(report: dict[str, Any]) -> tuple[int, int, int, int, int, float, float]:
        target = report["target_evaluation"]
        return (
            int(report["target_capability_success_count"]),
            int(report["target_base_cross_count"]),
            int(report["target_rear_top_count"]),
            int(report["target_front_top_count"]),
            sum(int(value) for value in report["retention_capability_success_counts"].values()),
            float(target["mean_return"]),
            -float(target["max_force_n"]),
        )

    def collect_all_successful_anchors(
        agent: Any, *, depth_m: float, command_mps: float, device: torch.device
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        if not math.isclose(depth_m, target_depth_m, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("unexpected target depth while collecting anchors")
        was_training = agent.training
        agent.eval()
        accepted_observations: list[np.ndarray] = []
        accepted_actions: list[np.ndarray] = []
        accepted_cases: list[int] = []
        all_depths = (target_depth_m, *retention_depths_m)
        try:
            for depth_index, anchor_depth in enumerate(all_depths):
                for state_index, state in enumerate(FIXED_STATES):
                    env = S10M20LidarLowTorqueCapabilityEnv(
                        depth_m=anchor_depth,
                        command_mps=command_mps,
                        training_states=(state,),
                    )
                    observations: list[np.ndarray] = []
                    actions: list[np.ndarray] = []
                    try:
                        actor_obs, _, _ = env.reset(
                            seed=20_000 + depth_index * 100 + state_index,
                            initial_state=state,
                        )
                        final_info: dict[str, Any] | None = None
                        for _ in range(env.max_episode_steps):
                            with torch.inference_mode():
                                action = (
                                    agent.deterministic(
                                        torch.as_tensor(
                                            actor_obs, dtype=torch.float32, device=device
                                        ).unsqueeze(0)
                                    )
                                    .squeeze(0)
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32)
                                )
                            observations.append(actor_obs.copy())
                            actions.append(action.copy())
                            actor_obs, _, _, terminated, truncated, final_info = env.step(action)
                            if terminated or truncated:
                                break
                        if final_info is not None and final_info["success"]:
                            accepted_observations.extend(observations)
                            accepted_actions.extend(actions)
                            accepted_cases.append(depth_index * len(FIXED_STATES) + state_index)
                    finally:
                        env.close()
        finally:
            agent.train(was_training)
        if not accepted_observations:
            raise RuntimeError("baseline policy produced no successful anchor trajectories")
        return (
            np.stack(accepted_observations).astype(np.float32),
            np.stack(accepted_actions).astype(np.float32),
            accepted_cases,
        )

    schedule = depth_schedule(
        target_depth_m=target_depth_m,
        retention_depths_m=retention_depths_m,
        num_envs=16,
    )
    base_runner.S10M20CurriculumEnv = S10M20LidarLowTorqueCapabilityEnv
    base_runner.S10M20CurriculumVecEnv = MixedDepthCapabilityVecEnv
    base_runner.M20InitializedActorCritic = LidarDownstreamActorCritic
    base_runner.evaluate_policy = evaluate_policy
    base_runner.validation_score = validation_score
    base_runner.collect_successful_baseline_anchors = collect_all_successful_anchors
    original_parse_args = base_runner.parse_args

    def parse_args() -> Any:
        args = original_parse_args()
        if not math.isclose(args.depth_m, target_depth_m, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"use --depth-m {target_depth_m} for this stage")
        args.target_depth_m = target_depth_m
        args.retention_depths_m = list(retention_depths_m)
        args.torque_limits_nm = {"leg": LEG_LIMIT_NM, "wheel": WHEEL_LIMIT_NM}
        args.actor_trainable_scope = (
            "first_layer_lidar_columns_plus_downstream_actor_layers; legacy input columns frozen"
        )
        args.validation_gate = "target_fixed5_plus_all_retention_fixed5"
        args.validation_case_count = len(FIXED_STATES) * (1 + len(retention_depths_m))
        args.force_500n_role = "report_only"
        args.training_depth_schedule_example_16_envs = list(schedule)
        return args

    base_runner.parse_args = parse_args
    return schedule


def main() -> int:
    stage_args, remaining = parse_stage_args(sys.argv[1:])
    target_depth_m = float(stage_args.target_depth_m)
    retention_depths_m = unique_depths(list(stage_args.retention_depth_m))
    if any(depth >= target_depth_m for depth in retention_depths_m):
        raise ValueError("every retention depth must be shallower than the target")
    sys.argv = [sys.argv[0], *remaining]
    configure_runner(
        target_depth_m=target_depth_m,
        retention_depths_m=retention_depths_m,
    )
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
