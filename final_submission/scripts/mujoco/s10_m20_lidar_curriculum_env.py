#!/usr/bin/env python3
"""72-beam deployable sensor extension for the S10 M20 curriculum.

The first 57 actor observations and the 16-D action stay unchanged.  A 72-beam
scan is appended using the same mount, angles, range normalization, collision
mask, 25 Hz update rate, and 50 Hz zero-order hold as the current S10 MuJoCo
ROS/deployment path.  Official assets are not modified.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from s10_m20_curriculum_env import (
    ACTOR_OBS_DIM as BASE_ACTOR_OBS_DIM,
    PIT_EXIT_ALONG_M,
    PITCH_LIMIT_RAD,
    PLATFORM_Z,
    PRIVILEGED_OBS_DIM,
    ROLL_LIMIT_RAD,
    SAFE_CONTACT_FORCE_N,
    S10M20CurriculumEnv,
)


LIDAR_BEAMS = 72
LIDAR_RANGE_MAX_M = 10.0
LIDAR_UPDATE_STEPS = 2
LIDAR_MOUNT_OFFSET = np.array([0.22341, 0.0, -0.0001], dtype=np.float64)
LIDAR_GEOMGROUP = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
LIDAR_ANGLES = np.linspace(-np.pi, np.pi, LIDAR_BEAMS, endpoint=False, dtype=np.float64)
ACTOR_OBS_DIM = BASE_ACTOR_OBS_DIM + LIDAR_BEAMS
CRITIC_OBS_DIM = ACTOR_OBS_DIM + PRIVILEGED_OBS_DIM


class S10M20LidarCurriculumEnv(S10M20CurriculumEnv):
    actor_observation_dim = ACTOR_OBS_DIM
    critic_observation_dim = CRITIC_OBS_DIM

    def __init__(self, **kwargs: Any) -> None:
        self._lidar_obs = np.ones(LIDAR_BEAMS, dtype=np.float32)
        self._lidar_last_update_step = -LIDAR_UPDATE_STEPS
        super().__init__(**kwargs)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self._lidar_obs.fill(1.0)
        self._lidar_last_update_step = -LIDAR_UPDATE_STEPS
        return super().reset(**kwargs)

    def _scan_lidar(self) -> np.ndarray:
        rotation = self.data.xmat[self.ids["base_body_id"]].reshape(3, 3)
        origin = self.data.xpos[self.ids["base_body_id"]] + rotation @ LIDAR_MOUNT_OFFSET
        ranges = np.full(LIDAR_BEAMS, LIDAR_RANGE_MAX_M, dtype=np.float32)
        for index, angle in enumerate(LIDAR_ANGLES):
            direction = rotation @ np.array(
                [math.cos(angle), math.sin(angle), 0.0], dtype=np.float64
            )
            geom_id = np.array([-1], dtype=np.int32)
            distance = float(
                mujoco.mj_ray(
                    self.model,
                    self.data,
                    origin,
                    direction,
                    LIDAR_GEOMGROUP,
                    1,
                    self.ids["base_body_id"],
                    geom_id,
                )
            )
            if distance >= 0.0:
                ranges[index] = min(LIDAR_RANGE_MAX_M, max(0.0, distance))
        return np.clip(ranges / LIDAR_RANGE_MAX_M, 0.0, 1.0)

    def _actor_observation(self) -> np.ndarray:
        official_obs = super()._actor_observation()
        if self._episode_step - self._lidar_last_update_step >= LIDAR_UPDATE_STEPS:
            self._lidar_obs = self._scan_lidar()
            self._lidar_last_update_step = self._episode_step
        actor_obs = np.concatenate([official_obs, self._lidar_obs]).astype(np.float32)
        if actor_obs.shape != (ACTOR_OBS_DIM,) or not np.isfinite(actor_obs).all():
            raise FloatingPointError("invalid 129-D deployable S10 actor observation")
        return actor_obs

    def _observations(self, metrics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        actor_obs = self._actor_observation()
        privileged = np.concatenate(
            [
                np.tanh(metrics["wheel_force_n"] / 250.0).astype(np.float32),
                np.clip(
                    (metrics["wheel_height_m"] - PLATFORM_Z) / 0.35, -2.0, 2.0
                ).astype(np.float32),
                np.clip(
                    (metrics["wheel_progress_m"] - PIT_EXIT_ALONG_M) / 1.0,
                    -3.0,
                    3.0,
                ).astype(np.float32),
                np.asarray(
                    [
                        np.clip(
                            (metrics["base_progress_m"] - PIT_EXIT_ALONG_M) / 1.0,
                            -3.0,
                            3.0,
                        ),
                        np.clip(metrics["base_lateral_m"] / 0.25, -2.0, 2.0),
                        np.clip((metrics["base_z_m"] - PLATFORM_Z) / 0.5, -2.0, 2.0),
                        np.clip(metrics["roll_rad"] / ROLL_LIMIT_RAD, -2.0, 2.0),
                        np.clip(metrics["pitch_rad"] / PITCH_LIMIT_RAD, -2.0, 2.0),
                        np.clip(metrics["base_forward_velocity_mps"] / 2.0, -2.0, 2.0),
                        float(metrics["front_top"]),
                        float(metrics["rear_top"]),
                        float(metrics["front_top_latched"]),
                        float(metrics["rear_top_latched"]),
                        np.clip(
                            metrics["episode_max_force_n"] / SAFE_CONTACT_FORCE_N,
                            0.0,
                            4.0,
                        ),
                        np.clip(
                            (metrics["wheel_force_n"][2] - metrics["wheel_force_n"][3])
                            / 250.0,
                            -4.0,
                            4.0,
                        ),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        critic_obs = np.concatenate([actor_obs, privileged]).astype(np.float32)
        if privileged.shape != (PRIVILEGED_OBS_DIM,) or critic_obs.shape != (CRITIC_OBS_DIM,):
            raise AssertionError(
                f"bad lidar curriculum observation shapes "
                f"{actor_obs.shape}/{privileged.shape}/{critic_obs.shape}"
            )
        return actor_obs, critic_obs

    def _info(self, metrics: dict[str, Any], rewards: dict[str, float], reason: str) -> dict[str, Any]:
        info = super()._info(metrics, rewards, reason)
        info.update(
            {
                "actor_observation_dim": ACTOR_OBS_DIM,
                "critic_observation_dim": CRITIC_OBS_DIM,
                "lidar_beams": LIDAR_BEAMS,
                "lidar_range_max_m": LIDAR_RANGE_MAX_M,
                "lidar_update_hz": 25.0,
            }
        )
        return info


class S10M20LidarCurriculumVecEnv:
    def __init__(self, num_envs: int, **kwargs: Any) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.envs = [S10M20LidarCurriculumEnv(**kwargs) for _ in range(num_envs)]
        self.num_envs = int(num_envs)
        self.action_dim = self.envs[0].action_dim
        self.actor_observation_dim = ACTOR_OBS_DIM
        self.critic_observation_dim = CRITIC_OBS_DIM

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


if __name__ == "__main__":
    print("Environment library; use train_s10_m20_lidar_curriculum_ppo.py")
