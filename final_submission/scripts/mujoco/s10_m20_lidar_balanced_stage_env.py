#!/usr/bin/env python3
"""Balanced-pair reward for the official-derived S10 MuJoCo pit.

This is an isolated research environment.  It keeps the capability
termination and the official local 50/14 N*m actuator limits, but replaces the
old one-wheel ``max`` progress reward with pairwise minimum progress and
high-water potentials.  The latter prevents repeatedly lifting and lowering a
single wheel from producing dense reward.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from train_s10_m20_lidar_low_torque_stage_ppo import (
    FIXED_STATES,
    LEG_LIMIT_NM,
    WHEEL_LIMIT_NM,
    S10M20LidarLowTorqueCapabilityEnv,
)


class S10M20LidarBalancedStageEnv(S10M20LidarLowTorqueCapabilityEnv):
    """Capability environment with balanced front/rear pair shaping."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reset_pair_highwaters()

    def _reset_pair_highwaters(self) -> None:
        self._front_progress_highwater = -math.inf
        self._rear_progress_highwater = -math.inf
        self._front_height_highwater = -math.inf
        self._rear_height_highwater = -math.inf
        self._base_progress_highwater = -math.inf

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        transition = super().reset(**kwargs)
        metrics = transition[2]["metrics"]
        self._front_progress_highwater = float(np.min(metrics["wheel_progress_m"][0:2]))
        self._rear_progress_highwater = float(np.min(metrics["wheel_progress_m"][2:4]))
        self._front_height_highwater = float(np.min(metrics["wheel_height_m"][0:2]))
        self._rear_height_highwater = float(np.min(metrics["wheel_height_m"][2:4]))
        self._base_progress_highwater = float(metrics["base_progress_m"])
        return transition

    @staticmethod
    def _new_highwater(value: float, old: float) -> tuple[float, float]:
        new = max(old, value)
        return new, max(0.0, new - old)

    def _reward(self, metrics: dict[str, Any], action: np.ndarray) -> dict[str, float]:
        front_progress = float(np.min(metrics["wheel_progress_m"][0:2]))
        rear_progress = float(np.min(metrics["wheel_progress_m"][2:4]))
        front_height = float(np.min(metrics["wheel_height_m"][0:2]))
        rear_height = float(np.min(metrics["wheel_height_m"][2:4]))
        base_progress = float(metrics["base_progress_m"])

        self._front_progress_highwater, front_progress_gain = self._new_highwater(
            front_progress, self._front_progress_highwater
        )
        self._rear_progress_highwater, rear_progress_gain = self._new_highwater(
            rear_progress, self._rear_progress_highwater
        )
        self._front_height_highwater, front_height_gain = self._new_highwater(
            front_height, self._front_height_highwater
        )
        self._rear_height_highwater, rear_height_gain = self._new_highwater(
            rear_height, self._rear_height_highwater
        )
        self._base_progress_highwater, base_progress_gain = self._new_highwater(
            base_progress, self._base_progress_highwater
        )

        previous = self._previous_metrics
        front_new = metrics["front_top"] and not previous["front_top_latched"]
        rear_new = metrics["rear_top"] and not previous["rear_top_latched"]
        base_cross_new = metrics["base_cross"] and not previous["base_cross_latched"]
        action_rate = float(np.mean(np.square(action - self._last_action)))
        action_acceleration = float(
            np.mean(np.square(action - 2.0 * self._last_action + self._previous_action))
        )
        roll_abs = abs(float(metrics["roll_rad"]))
        roll_penalty = max(0.0, roll_abs - math.radians(10.0)) / math.radians(15.0)

        # Stage potentials use the less-progressed wheel in each pair.  Before
        # front-top only the front pair is shaped; after the latch the rear
        # pair takes over, while base progress remains useful throughout.
        terms = {
            "velocity_tracking_reward": 0.50
            * math.exp(-((metrics["base_forward_velocity_mps"] - self.command_mps) / 0.6) ** 2),
            "base_progress_reward": 2.0 * float(np.clip(base_progress_gain / 0.02, 0.0, 1.0)),
            "front_progress_reward": 1.25
            * float(not metrics["front_top_latched"])
            * float(np.clip(front_progress_gain / 0.02, 0.0, 1.0)),
            "front_height_reward": 1.25
            * float(not metrics["front_top_latched"])
            * float(np.clip(front_height_gain / 0.01, 0.0, 1.0)),
            "front_top_reward": 6.0 * float(front_new),
            "front_support_reward": 0.40 * float(metrics["front_top"]),
            "rear_progress_reward": 2.5
            * float(metrics["front_top_latched"])
            * float(np.clip(rear_progress_gain / 0.02, 0.0, 1.0)),
            "rear_height_reward": 2.5
            * float(metrics["front_top_latched"])
            * float(np.clip(rear_height_gain / 0.01, 0.0, 1.0)),
            "rear_top_reward": 8.0 * float(rear_new),
            "base_cross_reward": 4.0 * float(base_cross_new),
            "roll_stability_reward": 0.20 * math.exp(-((roll_abs / 0.20) ** 2)),
            "roll_proximity_penalty": -0.35 * float(np.clip(roll_penalty, 0.0, 1.0)),
            "lateral_penalty": -0.20
            * float(np.clip(abs(metrics["base_lateral_m"]) / 0.25, 0.0, 2.0)),
            "contact_force_penalty": 0.0,
            "body_wall_penalty": -5.0 * float(metrics["body_wall_contact"]),
            "actuator_force_penalty": -2.0e-5 * metrics["mean_abs_actuator_force"] ** 2,
            "action_rate_penalty": -0.01 * action_rate,
            "action_smoothness_penalty": -0.01 * action_acceleration,
        }
        return {name: 0.10 * value for name, value in terms.items()}


if __name__ == "__main__":
    env = S10M20LidarBalancedStageEnv(depth_m=0.21, training_states=FIXED_STATES)
    try:
        actor, critic, info = env.reset(seed=0, initial_state=FIXED_STATES[0])
        assert actor.shape == (129,)
        assert critic.shape == (153,)
        assert info["official_assets_modified"] is False
        print("balanced-stage smoke: OK")
    finally:
        env.close()
