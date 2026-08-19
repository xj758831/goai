#!/usr/bin/env python3
"""Full-body S10 MuJoCo curriculum environment initialized from M20 policy work.

The actor-facing interface stays deployable: the official 57-D observation and
the official 16-D action convention.  Contact forces and runtime pit geometry
are critic-only privileged features and reward/termination diagnostics.
Official robot, ONNX, and track files are never modified.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SDK_ROOT = PROJECT_ROOT / "src" / "S10_sdk_deploy"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SDK_ROOT / "scripts"))

from probe_s10_m20_shallow_curriculum import (  # noqa: E402
    OFFICIAL_WALL_SLOPE,
    PIT_EXIT_ALONG_M,
    WHEEL_RADIUS_M,
    build_model,
    local_to_world,
    validate_geometry,
)
from s10_fronthook_scripted_probe import (  # noqa: E402
    DT,
    EXIT_DIRECTION_XY,
    EXIT_YAW,
    PLATFORM_Z,
    STANDING_POSE,
    body_velocity,
    contact_metrics,
    diverged,
    euler_from_xmat,
    model_ids,
    settle,
    wheel_centers,
    yaw_quaternion,
)
from s10_obs_reference import ACTION_DIM, BASE_OBS_DIM, KD, KP, build_observation, decode_action  # noqa: E402


CONTROL_DECIMATION = 20
CONTROL_DT = DT * CONTROL_DECIMATION
MAX_EPISODE_SECONDS = 6.0
MAX_EPISODE_STEPS = int(round(MAX_EPISODE_SECONDS / CONTROL_DT))
SUCCESS_DWELL_STEPS = int(round(0.10 / CONTROL_DT))
ROLL_LIMIT_RAD = math.radians(25.0)
PITCH_LIMIT_RAD = math.radians(75.0)
SAFE_CONTACT_FORCE_N = 500.0
CONTACT_FORCE_LIMIT_N = 1200.0
CONTACT_THRESHOLD_N = 5.0
FRONT_INDICES = np.array([0, 1], dtype=np.int32)
REAR_INDICES = np.array([2, 3], dtype=np.int32)
LATERAL_DIRECTION_XY = np.asarray([-EXIT_DIRECTION_XY[1], EXIT_DIRECTION_XY[0]], dtype=np.float64)
PIT_ORIGIN_XY = local_to_world(0.0, 0.0, 0.0)[:2]

ACTOR_OBS_DIM = BASE_OBS_DIM
PRIVILEGED_OBS_DIM = 24
CRITIC_OBS_DIM = ACTOR_OBS_DIM + PRIVILEGED_OBS_DIM

DEFAULT_TRAINING_STATES = (
    (0.02, 0.0),
    (0.02, 0.0),
    (0.02, 0.0),
    (0.0, 0.0),
    (-0.02, 0.0),
    (0.0, 2.0),
    (0.0, -2.0),
)
VALIDATION_STATES = (
    (0.0, 0.0),
    (0.02, 0.0),
    (-0.02, 0.0),
    (0.0, 2.0),
    (0.0, -2.0),
)


def exit_progress(position_xyz: np.ndarray) -> float:
    offset = np.asarray(position_xyz[:2], dtype=np.float64) - PIT_ORIGIN_XY
    along = float(np.dot(offset, EXIT_DIRECTION_XY))
    lateral = float(np.dot(offset, LATERAL_DIRECTION_XY))
    return along - OFFICIAL_WALL_SLOPE * lateral


def lateral_position(position_xyz: np.ndarray) -> float:
    return float(np.dot(np.asarray(position_xyz[:2], dtype=np.float64) - PIT_ORIGIN_XY, LATERAL_DIRECTION_XY))


class S10M20CurriculumEnv:
    """One serial full-action S10 environment on an official-derived pit."""

    action_dim = ACTION_DIM
    actor_observation_dim = ACTOR_OBS_DIM
    critic_observation_dim = CRITIC_OBS_DIM
    control_dt = CONTROL_DT

    def __init__(
        self,
        *,
        depth_m: float = 0.08,
        command_mps: float = 0.8,
        training_states: tuple[tuple[float, float], ...] = DEFAULT_TRAINING_STATES,
        max_episode_steps: int = MAX_EPISODE_STEPS,
    ) -> None:
        if not 0.0 < depth_m < PLATFORM_Z:
            raise ValueError("depth_m must be positive and below the official platform height")
        if not math.isfinite(command_mps) or abs(command_mps) > 1.0:
            raise ValueError("command_mps must stay inside the official +/-1.0 m/s range")
        if not training_states:
            raise ValueError("training_states cannot be empty")
        self.depth_m = float(depth_m)
        self.command_mps = float(command_mps)
        self.training_states = tuple((float(lateral), float(yaw)) for lateral, yaw in training_states)
        self.max_episode_steps = int(max_episode_steps)
        self.model = build_model(self.depth_m, wall_mode="official_diagonal")
        geometry = validate_geometry(self.model, self.depth_m)
        if not geometry["valid"]:
            raise RuntimeError(f"invalid official-derived pit geometry: {geometry}")
        self.geometry = geometry
        self.data = mujoco.MjData(self.model)
        self.ids = model_ids(self.model)
        self._rng = np.random.default_rng(0)
        self._current_state = (0.0, 0.0)
        self._command = np.asarray([self.command_mps, 0.0, 0.0], dtype=np.float32)
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._episode_step = 0
        self._episode_time_s = 0.0
        self._success_hold_steps = 0
        self._front_top_latched = False
        self._rear_top_latched = False
        self._base_cross_latched = False
        self._episode_max_force_n = 0.0
        self._episode_max_roll_rad = 0.0
        self._episode_max_pitch_rad = 0.0
        self._initial_metrics: dict[str, Any] = {}
        self._previous_metrics: dict[str, Any] = {}
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def _select_state(self, state: tuple[float, float] | None) -> tuple[float, float]:
        if state is None:
            state = self.training_states[int(self._rng.integers(0, len(self.training_states)))]
        lateral, yaw = (float(value) for value in state)
        if abs(lateral) > 0.05 or abs(yaw) > 5.0:
            raise ValueError("initial state exceeds bounded lateral/yaw perturbations")
        return lateral, yaw

    def _reset_physics(self) -> tuple[bool, str | None]:
        lateral_m, yaw_deg = self._current_state
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = local_to_world(0.0, lateral_m, PLATFORM_Z - self.depth_m + 0.48)
        self.data.qpos[3:7] = yaw_quaternion(EXIT_YAW + math.radians(yaw_deg))
        self.data.qpos[7:23] = STANDING_POSE
        self.data.qvel.fill(0.0)
        self.data.ctrl.fill(0.0)
        mujoco.mj_forward(self.model, self.data)
        return settle(self.model, self.data, self.ids)

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_state: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("environment is closed")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current_state = self._select_state(initial_state)
        stable, reason = self._reset_physics()
        if not stable:
            raise RuntimeError(f"S10 curriculum reset failed: {reason}")
        self._last_action.fill(0.0)
        self._previous_action.fill(0.0)
        self._episode_step = 0
        self._episode_time_s = 0.0
        self._success_hold_steps = 0
        self._front_top_latched = False
        self._rear_top_latched = False
        self._base_cross_latched = False
        self._episode_max_force_n = 0.0
        self._episode_max_roll_rad = 0.0
        self._episode_max_pitch_rad = 0.0
        metrics = self._metrics(control_peak_force_n=0.0, body_wall_seen=False)
        self._initial_metrics = self._copy_metrics(metrics)
        self._previous_metrics = self._copy_metrics(metrics)
        actor_obs, critic_obs = self._observations(metrics)
        return actor_obs, critic_obs, self._info(metrics, {}, "")

    @staticmethod
    def _copy_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        return {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in metrics.items()}

    def _apply_action(self, action: np.ndarray) -> None:
        position_target, velocity_target = decode_action(action)
        self.data.ctrl[:] = KP.astype(np.float64) * (position_target.astype(np.float64) - self.data.qpos[7:23]) + KD.astype(
            np.float64
        ) * (velocity_target.astype(np.float64) - self.data.qvel[6:22])

    def _metrics(self, *, control_peak_force_n: float, body_wall_seen: bool) -> dict[str, Any]:
        centers = wheel_centers(self.data, self.ids)
        contacts = contact_metrics(self.model, self.data, self.ids)
        forces = np.asarray(contacts["wheel_force"], dtype=np.float64)
        wheel_progress = np.asarray([exit_progress(center) for center in centers], dtype=np.float64)
        top_geometry = (wheel_progress >= PIT_EXIT_ALONG_M - 0.04) & (
            centers[:, 2] >= PLATFORM_Z + WHEEL_RADIUS_M - 0.025
        )
        front_top = bool(np.all(top_geometry[FRONT_INDICES]) and forces[FRONT_INDICES].sum() >= CONTACT_THRESHOLD_N)
        rear_top = bool(np.all(top_geometry[REAR_INDICES]) and forces[REAR_INDICES].sum() >= CONTACT_THRESHOLD_N)
        self._front_top_latched |= front_top
        self._rear_top_latched |= rear_top
        base = self.data.xpos[self.ids["base_body_id"]]
        base_progress = exit_progress(base)
        base_cross = bool(base_progress >= PIT_EXIT_ALONG_M + 0.25)
        self._base_cross_latched |= base_cross
        angular_body, linear_body, _ = body_velocity(self.model, self.data, self.ids["base_body_id"])
        roll, pitch, _ = euler_from_xmat(self.data.xmat[self.ids["base_body_id"]])
        force_peak = max(float(contacts["max_contact_force"]), float(control_peak_force_n))
        self._episode_max_force_n = max(self._episode_max_force_n, force_peak)
        self._episode_max_roll_rad = max(self._episode_max_roll_rad, abs(float(roll)))
        self._episode_max_pitch_rad = max(self._episode_max_pitch_rad, abs(float(pitch)))
        return {
            "base_progress_m": base_progress,
            "base_lateral_m": lateral_position(base),
            "base_z_m": float(base[2]),
            "base_forward_velocity_mps": float(linear_body[0]),
            "base_angular_speed_rad_s": float(np.linalg.norm(angular_body)),
            "roll_rad": float(roll),
            "pitch_rad": float(pitch),
            "wheel_force_n": forces,
            "wheel_height_m": centers[:, 2].copy(),
            "wheel_progress_m": wheel_progress,
            "front_top": front_top,
            "rear_top": rear_top,
            "front_top_latched": self._front_top_latched,
            "rear_top_latched": self._rear_top_latched,
            "base_cross": base_cross,
            "base_cross_latched": self._base_cross_latched,
            "body_wall_contact": bool(contacts["body_wall_contact"] or body_wall_seen),
            "control_peak_force_n": force_peak,
            "episode_max_force_n": self._episode_max_force_n,
            "episode_max_roll_rad": self._episode_max_roll_rad,
            "episode_max_pitch_rad": self._episode_max_pitch_rad,
            "mean_abs_actuator_force": float(np.mean(np.abs(self.data.actuator_force))),
            "finite": diverged(self.data) is None,
        }

    def _actor_observation(self) -> np.ndarray:
        angular_body, _, _ = body_velocity(self.model, self.data, self.ids["base_body_id"])
        observation = build_observation(
            angular_body,
            self.data.qpos[3:7].copy(),
            self._command,
            self.data.qpos[7:23].copy(),
            self.data.qvel[6:22].copy(),
            self._last_action,
        )
        if observation.shape != (ACTOR_OBS_DIM,) or not np.isfinite(observation).all():
            raise FloatingPointError("invalid deployable S10 actor observation")
        return observation

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
            raise AssertionError(f"bad curriculum observation shapes {actor_obs.shape}/{privileged.shape}/{critic_obs.shape}")
        return actor_obs, critic_obs

    def _reward(self, metrics: dict[str, Any], action: np.ndarray) -> dict[str, float]:
        previous = self._previous_metrics
        base_delta = float(metrics["base_progress_m"] - previous["base_progress_m"])
        front_delta = float(np.max(metrics["wheel_progress_m"][FRONT_INDICES]) - np.max(previous["wheel_progress_m"][FRONT_INDICES]))
        rear_delta = float(np.max(metrics["wheel_progress_m"][REAR_INDICES]) - np.max(previous["wheel_progress_m"][REAR_INDICES]))
        front_height_delta = float(np.max(metrics["wheel_height_m"][FRONT_INDICES]) - np.max(previous["wheel_height_m"][FRONT_INDICES]))
        rear_height_delta = float(np.max(metrics["wheel_height_m"][REAR_INDICES]) - np.max(previous["wheel_height_m"][REAR_INDICES]))
        front_new = metrics["front_top"] and not previous["front_top_latched"]
        rear_new = metrics["rear_top"] and not previous["rear_top_latched"]
        base_cross_new = metrics["base_cross"] and not previous["base_cross_latched"]
        force_excess = max(0.0, metrics["episode_max_force_n"] - SAFE_CONTACT_FORCE_N) / SAFE_CONTACT_FORCE_N
        action_rate = float(np.mean(np.square(action - self._last_action)))
        action_acceleration = float(np.mean(np.square(action - 2.0 * self._last_action + self._previous_action)))
        terms = {
            "velocity_tracking_reward": 0.50 * math.exp(-((metrics["base_forward_velocity_mps"] - self.command_mps) / 0.6) ** 2),
            "base_progress_reward": 2.0 * float(np.clip(base_delta / 0.02, -1.0, 1.0)),
            "front_progress_reward": 0.75 * float(not metrics["front_top_latched"]) * float(np.clip(front_delta / 0.02, -1.0, 1.0)),
            "front_height_reward": 0.75 * float(not metrics["front_top_latched"]) * float(np.clip(front_height_delta / 0.01, 0.0, 1.0)),
            "front_top_reward": 6.0 * float(front_new),
            "front_support_reward": 0.40 * float(metrics["front_top"]),
            "rear_progress_reward": 2.0 * float(metrics["front_top_latched"]) * float(np.clip(rear_delta / 0.02, -1.0, 1.0)),
            "rear_height_reward": 2.0 * float(metrics["front_top_latched"]) * float(np.clip(rear_height_delta / 0.01, 0.0, 1.0)),
            "rear_top_reward": 8.0 * float(rear_new),
            "base_cross_reward": 4.0 * float(base_cross_new),
            "roll_stability_reward": 0.20 * math.exp(-((metrics["roll_rad"] / 0.20) ** 2)),
            "lateral_penalty": -0.20 * float(np.clip(abs(metrics["base_lateral_m"]) / 0.25, 0.0, 2.0)),
            "contact_force_penalty": -1.0 * float(np.clip(force_excess, 0.0, 2.0)),
            "body_wall_penalty": -5.0 * float(metrics["body_wall_contact"]),
            "actuator_force_penalty": -2.0e-5 * metrics["mean_abs_actuator_force"] ** 2,
            "action_rate_penalty": -0.01 * action_rate,
            "action_smoothness_penalty": -0.01 * action_acceleration,
        }
        # Milestones and terminal outcomes must dominate episode length.  The
        # earlier shallow reward allowed a long unsafe rollout to outscore a
        # quick safe exit simply by collecting more dense tracking reward.
        return {name: 0.10 * value for name, value in terms.items()}

    def _stable_exit(self, metrics: dict[str, Any]) -> bool:
        return bool(
            metrics["front_top"]
            and metrics["rear_top"]
            and metrics["base_progress_m"] >= PIT_EXIT_ALONG_M + 0.25
            and abs(metrics["roll_rad"]) <= math.radians(15.0)
            and abs(metrics["pitch_rad"]) <= math.radians(20.0)
            and metrics["episode_max_force_n"] <= SAFE_CONTACT_FORCE_N
        )

    def _termination_reason(self, metrics: dict[str, Any]) -> str:
        bad = diverged(self.data)
        if bad is not None:
            return bad
        if metrics["body_wall_contact"]:
            return "body_wall_contact"
        if metrics["control_peak_force_n"] > CONTACT_FORCE_LIMIT_N:
            return "excessive_contact_force"
        if abs(metrics["roll_rad"]) > ROLL_LIMIT_RAD:
            return "excessive_roll"
        if abs(metrics["pitch_rad"]) > PITCH_LIMIT_RAD:
            return "excessive_pitch"
        return ""

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("environment is closed")
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (ACTION_DIM,) or not np.isfinite(action_array).all():
            raise ValueError(f"action must be finite shape {(ACTION_DIM,)}, got {action_array.shape}")
        action_array = np.clip(action_array, -100.0, 100.0)
        control_peak_force = 0.0
        body_wall_seen = False
        for _ in range(CONTROL_DECIMATION):
            self._apply_action(action_array)
            mujoco.mj_step(self.model, self.data)
            contacts = contact_metrics(self.model, self.data, self.ids)
            control_peak_force = max(control_peak_force, float(contacts["max_contact_force"]))
            body_wall_seen |= bool(contacts["body_wall_contact"])
        self._episode_step += 1
        self._episode_time_s += CONTROL_DT
        metrics = self._metrics(control_peak_force_n=control_peak_force, body_wall_seen=body_wall_seen)
        rewards = self._reward(metrics, action_array)
        reason = self._termination_reason(metrics)
        self._success_hold_steps = self._success_hold_steps + 1 if self._stable_exit(metrics) else 0
        if self._success_hold_steps >= SUCCESS_DWELL_STEPS:
            reason = "success"
            rewards["success_reward"] = 100.0
        elif self._episode_step >= self.max_episode_steps and not reason:
            reason = "time_limit"
        if reason and reason != "success":
            rewards["failure_penalty"] = -100.0
        self._previous_metrics = self._copy_metrics(metrics)
        self._previous_action[:] = self._last_action
        self._last_action[:] = action_array
        actor_obs, critic_obs = self._observations(metrics)
        terminated = bool(reason and reason != "time_limit")
        truncated = reason == "time_limit"
        return actor_obs, critic_obs, float(sum(rewards.values())), terminated, truncated, self._info(metrics, rewards, reason)

    def _info(self, metrics: dict[str, Any], rewards: dict[str, float], reason: str) -> dict[str, Any]:
        return {
            "depth_m": self.depth_m,
            "geometry": dict(self.geometry),
            "initial_state": {"lateral_m": self._current_state[0], "yaw_deg": self._current_state[1]},
            "episode_step": self._episode_step,
            "time_s": self._episode_time_s,
            "metrics": metrics,
            "reward_terms": rewards,
            "termination_reason": reason,
            "success": reason == "success",
            "last_action": self._last_action.copy(),
            "official_assets_modified": False,
            "actor_observation_dim": ACTOR_OBS_DIM,
            "critic_observation_dim": CRITIC_OBS_DIM,
        }


class S10M20CurriculumVecEnv:
    """Serial vector wrapper compatible with the local PPO runners."""

    def __init__(self, num_envs: int, **kwargs: Any) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.envs = [S10M20CurriculumEnv(**kwargs) for _ in range(num_envs)]
        self.num_envs = int(num_envs)
        self.action_dim = ACTION_DIM
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
    print("Environment library; use train_s10_m20_curriculum_ppo.py")
