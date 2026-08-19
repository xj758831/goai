#!/usr/bin/env python3
"""Competition-capability S10 lidar environment with force recorded, not gated."""

from __future__ import annotations

import math
from typing import Any

from s10_m20_curriculum_env import PIT_EXIT_ALONG_M, SAFE_CONTACT_FORCE_N
from s10_m20_lidar_curriculum_env import (
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    S10M20LidarCurriculumEnv,
)


class S10M20LidarCapabilityEnv(S10M20LidarCurriculumEnv):
    """Prioritize stable physical exit while retaining hard collision limits."""

    def _reward(self, metrics: dict[str, Any], action: Any) -> dict[str, float]:
        rewards = super()._reward(metrics, action)
        rewards["contact_force_penalty"] = 0.0
        return rewards

    def _stable_exit(self, metrics: dict[str, Any]) -> bool:
        return bool(
            metrics["front_top"]
            and metrics["rear_top"]
            and metrics["base_progress_m"] >= PIT_EXIT_ALONG_M + 0.25
            and abs(metrics["roll_rad"]) <= math.radians(15.0)
            and abs(metrics["pitch_rad"]) <= math.radians(20.0)
        )

    def _info(
        self, metrics: dict[str, Any], rewards: dict[str, float], reason: str
    ) -> dict[str, Any]:
        info = super()._info(metrics, rewards, reason)
        info.update(
            {
                "validation_gate": "stable_physical_exit_force_report_only",
                "force_quality_threshold_n": SAFE_CONTACT_FORCE_N,
                "force_quality_pass": bool(
                    metrics["episode_max_force_n"] <= SAFE_CONTACT_FORCE_N
                ),
            }
        )
        return info


class S10M20LidarCapabilityVecEnv:
    def __init__(self, num_envs: int, **kwargs: Any) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.envs = [S10M20LidarCapabilityEnv(**kwargs) for _ in range(num_envs)]
        self.num_envs = int(num_envs)
        self.action_dim = self.envs[0].action_dim
        self.actor_observation_dim = ACTOR_OBS_DIM
        self.critic_observation_dim = CRITIC_OBS_DIM

    def reset(self) -> tuple[Any, Any, list[dict[str, Any]]]:
        import numpy as np

        transitions = [env.reset(seed=index) for index, env in enumerate(self.envs)]
        return (
            np.stack([transition[0] for transition in transitions]),
            np.stack([transition[1] for transition in transitions]),
            [transition[2] for transition in transitions],
        )

    def reset_at(self, index: int) -> tuple[Any, Any, dict[str, Any]]:
        return self.envs[index].reset()

    def close(self) -> None:
        for env in self.envs:
            env.close()


if __name__ == "__main__":
    print("Environment library; use the capability evaluator or PPO wrapper")
