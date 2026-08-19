#!/usr/bin/env python3
"""Phase-gated S10 environment with a research-only short-peak torque envelope.

The official MJCF stays untouched.  Peak limits are applied only to the
environment-owned in-memory model, and every actuator's continuous time above
the 50/14 Nm limits is measured at the 1 kHz MuJoCo physics rate.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

from s10_fronthook_scripted_probe import contact_metrics
from s10_m20_curriculum_env import ACTION_DIM, CONTROL_DECIMATION, SUCCESS_DWELL_STEPS
from s10_m20_lidar_torque_envelope_env import (
    EXPECTED_ACTUATORS,
    LEG_CONTINUOUS_LIMIT_NM,
    MAX_CONTINUOUS_EXCEEDANCE_S,
    WHEEL_CONTINUOUS_LIMIT_NM,
)
from s10_m20_phase_gated_residual import S10M20PhaseGatedEnv


class S10M20PhaseGatedTorqueEnvelopeEnv(S10M20PhaseGatedEnv):
    """Phase-gated environment with bounded in-memory actuator peak limits."""

    def __init__(
        self,
        *,
        leg_peak_nm: float = LEG_CONTINUOUS_LIMIT_NM,
        wheel_peak_nm: float = WHEEL_CONTINUOUS_LIMIT_NM,
        max_continuous_exceedance_s: float = MAX_CONTINUOUS_EXCEEDANCE_S,
        allow_body_wall_contact: bool = False,
        **kwargs: Any,
    ) -> None:
        values = (leg_peak_nm, wheel_peak_nm, max_continuous_exceedance_s)
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise ValueError("torque envelope values must be finite and positive")
        if leg_peak_nm < LEG_CONTINUOUS_LIMIT_NM or wheel_peak_nm < WHEEL_CONTINUOUS_LIMIT_NM:
            raise ValueError("peak limits cannot be below the continuous 50/14 Nm limits")

        self.leg_peak_nm = float(leg_peak_nm)
        self.wheel_peak_nm = float(wheel_peak_nm)
        self.max_continuous_exceedance_s = float(max_continuous_exceedance_s)
        self.allow_body_wall_contact = bool(allow_body_wall_contact)
        super().__init__(**kwargs)

        if self.model.nu != EXPECTED_ACTUATORS:
            raise RuntimeError(f"expected {EXPECTED_ACTUATORS} S10 actuators, found {self.model.nu}")
        names: list[str] = []
        joint_ids: list[int] = []
        wheel_mask: list[bool] = []
        for actuator_id in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            if not name:
                raise RuntimeError(f"S10 actuator {actuator_id} has no name")
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                raise RuntimeError(f"S10 actuator {name} is not attached to a joint")
            names.append(name)
            joint_ids.append(joint_id)
            wheel_mask.append(name.endswith("_wheel_joint"))

        self.actuator_names = tuple(names)
        self._joint_ids = np.asarray(joint_ids, dtype=np.int32)
        self._wheel_mask = np.asarray(wheel_mask, dtype=bool)
        if int(np.count_nonzero(self._wheel_mask)) != 4:
            raise RuntimeError(
                f"expected four S10 wheel actuators, found {np.count_nonzero(self._wheel_mask)}"
            )
        if len(set(joint_ids)) != self.model.nu:
            raise RuntimeError("S10 torque-envelope evaluation requires one actuator per joint")

        self._continuous_limits_nm = np.where(
            self._wheel_mask, WHEEL_CONTINUOUS_LIMIT_NM, LEG_CONTINUOUS_LIMIT_NM
        ).astype(np.float64)
        self._peak_limits_nm = np.where(
            self._wheel_mask, self.wheel_peak_nm, self.leg_peak_nm
        ).astype(np.float64)
        self._configure_in_memory_limits()
        self._current_exceedance_s = np.zeros(self.model.nu, dtype=np.float64)
        self._longest_exceedance_s = np.zeros(self.model.nu, dtype=np.float64)
        self._peak_abs_torque_nm = np.zeros(self.model.nu, dtype=np.float64)
        self._duration_violation = False
        self._body_wall_contact_duration_s = 0.0
        self._body_wall_current_duration_s = 0.0
        self._body_wall_longest_duration_s = 0.0
        self._body_wall_peak_force_n = 0.0

    def _configure_in_memory_limits(self) -> None:
        for actuator_id, joint_id in enumerate(self._joint_ids):
            peak = float(self._peak_limits_nm[actuator_id])
            self.model.actuator_ctrllimited[actuator_id] = True
            self.model.actuator_ctrlrange[actuator_id] = (-peak, peak)
            self.model.jnt_actfrclimited[joint_id] = True
            self.model.jnt_actfrcrange[joint_id] = (-peak, peak)
        actual_ctrl = np.max(np.abs(self.model.actuator_ctrlrange), axis=1)
        actual_joint = np.asarray(
            [
                np.max(np.abs(self.model.jnt_actfrcrange[joint_id]))
                for joint_id in self._joint_ids
            ],
            dtype=np.float64,
        )
        if not np.allclose(actual_ctrl, self._peak_limits_nm) or not np.allclose(
            actual_joint, self._peak_limits_nm
        ):
            raise RuntimeError("failed to apply the in-memory S10 torque envelope")

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self._current_exceedance_s.fill(0.0)
        self._longest_exceedance_s.fill(0.0)
        self._peak_abs_torque_nm.fill(0.0)
        self._duration_violation = False
        self._body_wall_contact_duration_s = 0.0
        self._body_wall_current_duration_s = 0.0
        self._body_wall_longest_duration_s = 0.0
        self._body_wall_peak_force_n = 0.0
        return super().reset(**kwargs)

    def _sample_body_wall_contact(self, contacts: dict[str, Any]) -> None:
        timestep = float(self.model.opt.timestep)
        if bool(contacts["body_wall_contact"]):
            self._body_wall_contact_duration_s += timestep
            self._body_wall_current_duration_s += timestep
            self._body_wall_longest_duration_s = max(
                self._body_wall_longest_duration_s,
                self._body_wall_current_duration_s,
            )
            self._body_wall_peak_force_n = max(
                self._body_wall_peak_force_n,
                float(contacts["body_wall_force"]),
            )
        else:
            self._body_wall_current_duration_s = 0.0

    def _sample_torque_envelope(self) -> None:
        torque = np.abs(np.asarray(self.data.actuator_force, dtype=np.float64))
        self._peak_abs_torque_nm = np.maximum(self._peak_abs_torque_nm, torque)
        exceeded = torque > self._continuous_limits_nm
        self._current_exceedance_s[exceeded] += float(self.model.opt.timestep)
        self._current_exceedance_s[~exceeded] = 0.0
        self._longest_exceedance_s = np.maximum(
            self._longest_exceedance_s, self._current_exceedance_s
        )
        self._duration_violation |= bool(
            np.any(
                self._current_exceedance_s
                > self.max_continuous_exceedance_s + 1.0e-12
            )
        )

    def _torque_envelope_metrics(self) -> dict[str, Any]:
        leg_mask = ~self._wheel_mask
        return {
            "leg_peak_limit_nm": self.leg_peak_nm,
            "wheel_peak_limit_nm": self.wheel_peak_nm,
            "leg_continuous_limit_nm": LEG_CONTINUOUS_LIMIT_NM,
            "wheel_continuous_limit_nm": WHEEL_CONTINUOUS_LIMIT_NM,
            "max_continuous_exceedance_s": self.max_continuous_exceedance_s,
            "max_leg_abs_torque_nm": float(np.max(self._peak_abs_torque_nm[leg_mask])),
            "max_wheel_abs_torque_nm": float(np.max(self._peak_abs_torque_nm[self._wheel_mask])),
            "max_leg_continuous_exceedance_s": float(
                np.max(self._longest_exceedance_s[leg_mask])
            ),
            "max_wheel_continuous_exceedance_s": float(
                np.max(self._longest_exceedance_s[self._wheel_mask])
            ),
            "duration_violation": bool(self._duration_violation),
        }

    def _termination_reason(self, metrics: dict[str, Any]) -> str:
        checked_metrics = metrics
        if self.allow_body_wall_contact and bool(metrics["body_wall_contact"]):
            checked_metrics = dict(metrics)
            checked_metrics["body_wall_contact"] = False
        reason = super()._termination_reason(checked_metrics)
        if reason:
            return reason
        if self._duration_violation:
            return "torque_duration_violation"
        return ""

    def _info(
        self, metrics: dict[str, Any], rewards: dict[str, float], reason: str
    ) -> dict[str, Any]:
        info = super()._info(metrics, rewards, reason)
        info["torque_envelope"] = self._torque_envelope_metrics()
        info["torque_envelope_in_memory_only"] = True
        info["competition_peak_threshold_confirmed"] = False
        info["body_wall_contact_audit"] = {
            "allowed_for_this_rollout": self.allow_body_wall_contact,
            "observed": self._body_wall_contact_duration_s > 0.0,
            "total_duration_s": self._body_wall_contact_duration_s,
            "longest_continuous_duration_s": self._body_wall_longest_duration_s,
            "peak_force_n": self._body_wall_peak_force_n,
        }
        return info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("environment is closed")
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (ACTION_DIM,) or not np.isfinite(action_array).all():
            raise ValueError(
                f"action must be finite shape {(ACTION_DIM,)}, got {action_array.shape}"
            )
        action_array = np.clip(action_array, -100.0, 100.0)

        control_peak_force = 0.0
        body_wall_seen = False
        physics_steps = 0
        for _ in range(CONTROL_DECIMATION):
            self._apply_action(action_array)
            mujoco.mj_step(self.model, self.data)
            physics_steps += 1
            self._sample_torque_envelope()
            contacts = contact_metrics(self.model, self.data, self.ids)
            self._sample_body_wall_contact(contacts)
            control_peak_force = max(
                control_peak_force, float(contacts["max_contact_force"])
            )
            body_wall_seen |= bool(contacts["body_wall_contact"])
            if self._duration_violation:
                break

        self._episode_step += 1
        self._episode_time_s += physics_steps * float(self.model.opt.timestep)
        metrics = self._metrics(
            control_peak_force_n=control_peak_force, body_wall_seen=body_wall_seen
        )
        rewards = self._reward(metrics, action_array)
        reason = self._termination_reason(metrics)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._stable_exit(metrics) else 0
        )
        if not reason and self._success_hold_steps >= SUCCESS_DWELL_STEPS:
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
        return (
            actor_obs,
            critic_obs,
            float(sum(rewards.values())),
            terminated,
            truncated,
            self._info(metrics, rewards, reason),
        )


if __name__ == "__main__":
    print("Environment library; use the robust-teacher search with explicit peak limits")
