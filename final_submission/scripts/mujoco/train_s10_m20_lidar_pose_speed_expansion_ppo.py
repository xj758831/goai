#!/usr/bin/env python3
"""Expand the learned S10 policy from sparse deep-pit successes.

This process is intentionally isolated from the earlier balanced-reward pilot.
It trains on the pose/speed/depth combinations that the current checkpoint
actually completes, while retaining the fixed 0.21 m and 0.18 m validation
gates supplied by the audited low-torque stage runner.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import torch

import train_s10_m20_lidar_low_torque_stage_ppo as stage_runner
import train_s10_m20_lidar_curriculum_ppo as lidar_runner
import train_s10_m20_curriculum_ppo as base_runner
from s10_m20_lidar_balanced_stage_env import S10M20LidarBalancedStageEnv
from train_s10_m20_curriculum_ppo import VALIDATION_STATES


FIXED_STATES = tuple(VALIDATION_STATES)

# These are successful or immediately neighboring entry states found by the
# read-only search.  Neighboring states are included to widen the basin rather
# than memorize one exact initial pose.
POSE_SPEED_SCHEDULE: tuple[tuple[float, float, tuple[tuple[float, float], ...]], ...] = (
    (
        0.225,
        0.70,
        ((0.02, 0.0), (-0.03, -2.0), (-0.04, -3.0), (-0.02, -4.0), (0.0, -4.0)),
    ),
    (
        0.23,
        0.70,
        ((-0.04, 0.0), (-0.03, 0.0), (-0.01, -3.0), (0.02, -4.0), (0.04, 0.0)),
    ),
    (
        0.235,
        0.70,
        ((-0.05, -4.0), (-0.04, 0.0), (-0.02, -1.0), (-0.01, -5.0), (0.04, 0.0)),
    ),
    (
        0.24,
        0.80,
        ((-0.04, -5.0), (-0.03, -5.0), (-0.03, -4.0), (-0.02, -5.0)),
    ),
    (
        0.25,
        0.70,
        ((-0.04, -5.0), (-0.03, -5.0), (-0.02, -4.5), (-0.01, -4.0)),
    ),
    (
        0.26,
        0.70,
        ((-0.04, -4.0), (-0.03, -3.5), (-0.02, -3.0), (-0.01, -3.0)),
    ),
    (
        0.27,
        0.65,
        ((-0.04, -4.5), (-0.035, -4.5), (-0.03, -4.0), (-0.02, -4.0)),
    ),
    (0.18, 0.80, FIXED_STATES),
)

# Exact successful cases provide the anchor trajectories.  The full fixed
# 0.21/0.18 validation set is added as well by collect_anchors below.
SEARCH_ANCHOR_SPECS = (
    (0.225, 0.70, 0.02, 0.0),
    (0.225, 0.70, -0.03, -2.0),
    (0.23, 0.70, -0.04, 0.0),
    (0.23, 0.70, -0.03, 0.0),
    (0.235, 0.70, -0.02, -1.0),
    (0.24, 0.80, -0.03, -5.0),
    (0.25, 0.70, -0.03, -5.0),
    (0.26, 0.70, -0.03, -3.5),
    (0.27, 0.65, -0.035, -4.5),
)


class PoseSpeedExpansionVecEnv:
    """Serial MuJoCo env API with a fixed, auditable depth/speed schedule."""

    def __init__(self, num_envs: int, **kwargs: Any) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if num_envs != len(POSE_SPEED_SCHEDULE):
            raise ValueError(
                f"targeted expansion requires exactly {len(POSE_SPEED_SCHEDULE)} environments"
            )
        self.envs = [
            S10M20LidarBalancedStageEnv(
                depth_m=depth_m,
                command_mps=command_mps,
                training_states=states,
            )
            for depth_m, command_mps, states in POSE_SPEED_SCHEDULE
        ]
        self.num_envs = num_envs
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
        return self.envs[index].reset(seed=10_000 + index)

    def close(self) -> None:
        for env in self.envs:
            env.close()


def collect_anchors(
    agent: Any,
    *,
    depth_m: float,
    command_mps: float,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Replay known successes and retain only trajectories that truly exit."""

    del depth_m, command_mps
    was_training = agent.training
    agent.eval()
    observation_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    accepted: list[int] = []
    specs = list(SEARCH_ANCHOR_SPECS)
    specs.extend((0.21, 0.80, lateral, yaw) for lateral, yaw in FIXED_STATES)
    specs.extend((0.18, 0.80, lateral, yaw) for lateral, yaw in FIXED_STATES)
    try:
        for index, (anchor_depth, anchor_speed, lateral, yaw) in enumerate(specs):
            env = S10M20LidarBalancedStageEnv(
                depth_m=anchor_depth,
                command_mps=anchor_speed,
                training_states=((lateral, yaw),),
            )
            observations: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            try:
                actor_obs, _, _ = env.reset(
                    seed=30_000 + index,
                    initial_state=(lateral, yaw),
                )
                final_info: dict[str, Any] | None = None
                for _ in range(env.max_episode_steps):
                    with torch.inference_mode():
                        action = (
                            agent.deterministic(
                                torch.as_tensor(actor_obs, dtype=torch.float32, device=device).unsqueeze(0)
                            )
                            .squeeze(0)
                            .detach()
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
                    observation_rows.extend(observations)
                    action_rows.extend(actions)
                    accepted.append(index)
            finally:
                env.close()
    finally:
        agent.train(was_training)
    if not observation_rows:
        raise RuntimeError("targeted expansion produced no successful anchor trajectory")
    return (
        np.stack(observation_rows).astype(np.float32),
        np.stack(action_rows).astype(np.float32),
        accepted,
    )


def configure_runner() -> None:
    # Stage runner supplies the audited target/retention fixed-case evaluator;
    # only its training vector environment and anchor collector are replaced.
    stage_runner.S10M20LidarLowTorqueCapabilityEnv = S10M20LidarBalancedStageEnv
    stage_args, _ = stage_runner.parse_stage_args(sys.argv[1:])
    target_depth_m = float(stage_args.target_depth_m)
    retention_depths_m = stage_runner.unique_depths(list(stage_args.retention_depth_m))
    if not math.isclose(target_depth_m, 0.21, abs_tol=1.0e-12):
        raise ValueError("targeted expansion validation is fixed to --target-depth-m 0.21")
    if any(depth >= target_depth_m for depth in retention_depths_m):
        raise ValueError("retention depths must be shallower than target")
    stage_runner.configure_runner(
        target_depth_m=target_depth_m,
        retention_depths_m=retention_depths_m,
    )
    base_runner.S10M20CurriculumVecEnv = PoseSpeedExpansionVecEnv
    base_runner.collect_successful_baseline_anchors = collect_anchors
    original_parse_args = base_runner.parse_args

    def parse_args() -> Any:
        args = original_parse_args()
        args.actor_trainable_scope = (
            "first_layer_lidar_columns_plus_downstream_actor_layers; targeted pose-speed expansion"
        )
        args.training_depth_schedule_example_8_envs = [
            {"depth_m": depth, "command_mps": speed, "states": list(states)}
            for depth, speed, states in POSE_SPEED_SCHEDULE
        ]
        args.anchor_source = "read-only pose/speed search successes plus fixed 0.21/0.18 validation successes"
        return args

    base_runner.parse_args = parse_args


def main() -> int:
    stage_args, remaining = stage_runner.parse_stage_args(sys.argv[1:])
    if not math.isclose(float(stage_args.target_depth_m), 0.21, abs_tol=1.0e-12):
        raise ValueError("targeted expansion validation is fixed to --target-depth-m 0.21")
    retention_depths_m = stage_runner.unique_depths(list(stage_args.retention_depth_m))
    if any(depth >= 0.21 for depth in retention_depths_m):
        raise ValueError("retention depths must be shallower than target")
    configure_runner()
    sys.argv = [sys.argv[0], *remaining]
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
