#!/usr/bin/env python3
"""MuJoCo terrain-lidar extension matching the repository Isaac ray layout."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from s10_m20_curriculum_env import (
    PITCH_LIMIT_RAD,
    PIT_EXIT_ALONG_M,
    PLATFORM_Z,
    PRIVILEGED_OBS_DIM,
    ROLL_LIMIT_RAD,
    SAFE_CONTACT_FORCE_N,
)
from s10_m20_lidar_balanced_stage_env import S10M20LidarBalancedStageEnv
from s10_m20_lidar_curriculum_env import (
    LIDAR_GEOMGROUP,
    LIDAR_MOUNT_OFFSET,
    LIDAR_RANGE_MAX_M,
)


TERRAIN_LIDAR_CHANNELS = 5
TERRAIN_LIDAR_HORIZONTAL_BEAMS = 13
TERRAIN_LIDAR_BEAMS = TERRAIN_LIDAR_CHANNELS * TERRAIN_LIDAR_HORIZONTAL_BEAMS
TERRAIN_LIDAR_VERTICAL_FOV_DEG = (-45.0, -10.0)
TERRAIN_LIDAR_HORIZONTAL_FOV_DEG = (-60.0, 60.0)
TERRAIN_LIDAR_VERTICAL_ANGLES = np.linspace(
    math.radians(TERRAIN_LIDAR_VERTICAL_FOV_DEG[0]),
    math.radians(TERRAIN_LIDAR_VERTICAL_FOV_DEG[1]),
    TERRAIN_LIDAR_CHANNELS,
    dtype=np.float64,
)
TERRAIN_LIDAR_HORIZONTAL_ANGLES = np.linspace(
    math.radians(TERRAIN_LIDAR_HORIZONTAL_FOV_DEG[0]),
    math.radians(TERRAIN_LIDAR_HORIZONTAL_FOV_DEG[1]),
    TERRAIN_LIDAR_HORIZONTAL_BEAMS,
    dtype=np.float64,
)
BASE_LIDAR_ACTOR_OBS_DIM = 129
ACTOR_OBS_DIM = BASE_LIDAR_ACTOR_OBS_DIM + TERRAIN_LIDAR_BEAMS
CRITIC_OBS_DIM = ACTOR_OBS_DIM + PRIVILEGED_OBS_DIM
TERRAIN_LIDAR_UPDATE_STEPS = 2


class S10M20TerrainLidarBalancedStageEnv(S10M20LidarBalancedStageEnv):
    """Balanced capability environment with deployable downward terrain rays."""

    actor_observation_dim = ACTOR_OBS_DIM
    critic_observation_dim = CRITIC_OBS_DIM

    def __init__(self, **kwargs: Any) -> None:
        self._terrain_lidar_obs = np.ones(TERRAIN_LIDAR_BEAMS, dtype=np.float32)
        self._terrain_lidar_last_update_step = -TERRAIN_LIDAR_UPDATE_STEPS
        super().__init__(**kwargs)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self._terrain_lidar_obs.fill(1.0)
        self._terrain_lidar_last_update_step = -TERRAIN_LIDAR_UPDATE_STEPS
        return super().reset(**kwargs)

    def _scan_terrain_lidar(self) -> np.ndarray:
        rotation = self.data.xmat[self.ids["base_body_id"]].reshape(3, 3)
        origin = self.data.xpos[self.ids["base_body_id"]] + rotation @ LIDAR_MOUNT_OFFSET
        ranges = np.full(TERRAIN_LIDAR_BEAMS, LIDAR_RANGE_MAX_M, dtype=np.float32)
        index = 0
        for vertical in TERRAIN_LIDAR_VERTICAL_ANGLES:
            cos_vertical = math.cos(float(vertical))
            for horizontal in TERRAIN_LIDAR_HORIZONTAL_ANGLES:
                direction_local = np.asarray(
                    [
                        cos_vertical * math.cos(float(horizontal)),
                        cos_vertical * math.sin(float(horizontal)),
                        math.sin(float(vertical)),
                    ],
                    dtype=np.float64,
                )
                direction = rotation @ direction_local
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
                index += 1
        return np.clip(ranges / LIDAR_RANGE_MAX_M, 0.0, 1.0)

    def _actor_observation(self) -> np.ndarray:
        horizontal = super()._actor_observation()
        if self._episode_step - self._terrain_lidar_last_update_step >= TERRAIN_LIDAR_UPDATE_STEPS:
            self._terrain_lidar_obs = self._scan_terrain_lidar()
            self._terrain_lidar_last_update_step = self._episode_step
        actor_obs = np.concatenate([horizontal, self._terrain_lidar_obs]).astype(np.float32)
        if actor_obs.shape != (ACTOR_OBS_DIM,) or not np.isfinite(actor_obs).all():
            raise FloatingPointError("invalid 194-D terrain-lidar actor observation")
        return actor_obs

    def _observations(self, metrics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        actor_obs = self._actor_observation()
        privileged = np.concatenate(
            [
                np.tanh(metrics["wheel_force_n"] / 250.0).astype(np.float32),
                np.clip((metrics["wheel_height_m"] - PLATFORM_Z) / 0.35, -2.0, 2.0).astype(np.float32),
                np.clip((metrics["wheel_progress_m"] - PIT_EXIT_ALONG_M) / 1.0, -3.0, 3.0).astype(np.float32),
                np.asarray(
                    [
                        np.clip((metrics["base_progress_m"] - PIT_EXIT_ALONG_M) / 1.0, -3.0, 3.0),
                        np.clip(metrics["base_lateral_m"] / 0.25, -2.0, 2.0),
                        np.clip((metrics["base_z_m"] - PLATFORM_Z) / 0.5, -2.0, 2.0),
                        np.clip(metrics["roll_rad"] / ROLL_LIMIT_RAD, -2.0, 2.0),
                        np.clip(metrics["pitch_rad"] / PITCH_LIMIT_RAD, -2.0, 2.0),
                        np.clip(metrics["base_forward_velocity_mps"] / 2.0, -2.0, 2.0),
                        float(metrics["front_top"]),
                        float(metrics["rear_top"]),
                        float(metrics["front_top_latched"]),
                        float(metrics["rear_top_latched"]),
                        np.clip(metrics["episode_max_force_n"] / SAFE_CONTACT_FORCE_N, 0.0, 4.0),
                        np.clip((metrics["wheel_force_n"][2] - metrics["wheel_force_n"][3]) / 250.0, -4.0, 4.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        critic_obs = np.concatenate([actor_obs, privileged]).astype(np.float32)
        if privileged.shape != (PRIVILEGED_OBS_DIM,) or critic_obs.shape != (CRITIC_OBS_DIM,):
            raise AssertionError(
                f"bad terrain-lidar observation shapes {actor_obs.shape}/{privileged.shape}/{critic_obs.shape}"
            )
        return actor_obs, critic_obs

    def _info(self, metrics: dict[str, Any], rewards: dict[str, float], reason: str) -> dict[str, Any]:
        info = super()._info(metrics, rewards, reason)
        info.update(
            {
                "actor_observation_dim": ACTOR_OBS_DIM,
                "critic_observation_dim": CRITIC_OBS_DIM,
                "terrain_lidar_beams": TERRAIN_LIDAR_BEAMS,
                "terrain_lidar_channels": TERRAIN_LIDAR_CHANNELS,
                "terrain_lidar_horizontal_beams": TERRAIN_LIDAR_HORIZONTAL_BEAMS,
                "terrain_lidar_vertical_fov_deg": list(TERRAIN_LIDAR_VERTICAL_FOV_DEG),
                "terrain_lidar_horizontal_fov_deg": list(TERRAIN_LIDAR_HORIZONTAL_FOV_DEG),
                "terrain_lidar_update_hz": 25.0,
            }
        )
        return info


if __name__ == "__main__":
    print("Environment library; use the terrain-lidar trainer")
