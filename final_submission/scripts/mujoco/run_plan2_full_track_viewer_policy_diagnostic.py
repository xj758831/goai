#!/usr/bin/env python3
"""Run the unchanged full-track viewer while recording invalid ONNX outputs.

This wrapper replaces only the policy object's validation method.  Finite
actions, observation construction, decoding, and PD targets are identical to
``run_plan2_full_track_viewer.py``.  On a non-finite action it writes the
current state plus a short pre-failure history, then raises the same safety
failure instead of silently continuing.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

import probe_s10_official_policy_shallow as official
import run_plan2_full_track_viewer as runner
from s10_fronthook_scripted_probe import body_velocity
from s10_obs_reference import build_observation, decode_action


def _output_dir() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args.output_dir.resolve()


DIAGNOSTIC_DIR = _output_dir()
WP29_XY = np.asarray([30.9350, 14.7150], dtype=np.float64)
WP29_CAPTURE_RADIUS_M = 0.20
POLICY_ACTION_CLIP = float(os.environ.get("PLAN2_POLICY_ACTION_CLIP", "0"))
if POLICY_ACTION_CLIP < 0.0:
    raise ValueError("PLAN2_POLICY_ACTION_CLIP must be zero (disabled) or positive")
POLICY_ACTION_GUARD = float(os.environ.get("PLAN2_POLICY_ACTION_GUARD", "0"))
if POLICY_ACTION_GUARD < 0.0:
    raise ValueError("PLAN2_POLICY_ACTION_GUARD must be zero (disabled) or positive")
if POLICY_ACTION_CLIP > 0.0 and POLICY_ACTION_GUARD > 0.0:
    raise ValueError("action clip and action guard cannot be enabled together")


class DiagnosticOfficialPolicy(official.OfficialPolicy):
    """Official policy with a state snapshot at the first invalid action."""

    last_instance: "DiagnosticOfficialPolicy | None" = None

    def __init__(self) -> None:
        super().__init__()
        self._history: deque[dict[str, np.ndarray | float]] = deque(maxlen=8)
        self._wp29_saved = False
        self.raw_action_abs_max = 0.0
        self.clipped_value_count = 0
        self.guard_trigger_count = 0
        self.guard_reinfer_success_count = 0
        self.guard_zero_fallback_count = 0
        self.guard_events: list[dict[str, Any]] = []
        DiagnosticOfficialPolicy.last_instance = self

    def write_run_report(self) -> None:
        DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
        report = {
            "policy_action_clip": None if POLICY_ACTION_CLIP == 0.0 else POLICY_ACTION_CLIP,
            "policy_updates": self.update_count,
            "raw_action_abs_max": self.raw_action_abs_max,
            "clipped_value_count": self.clipped_value_count,
            "policy_action_guard": (
                None if POLICY_ACTION_GUARD == 0.0 else POLICY_ACTION_GUARD
            ),
            "guard_trigger_count": self.guard_trigger_count,
            "guard_reinfer_success_count": self.guard_reinfer_success_count,
            "guard_zero_fallback_count": self.guard_zero_fallback_count,
            "guard_events": self.guard_events,
            "wp29_natural_state_saved": self._wp29_saved,
            "control_semantics": (
                "reset last_action feedback and reinfer only above the guard threshold"
                if POLICY_ACTION_GUARD > 0.0
                else (
                    "unchanged raw official output"
                    if POLICY_ACTION_CLIP == 0.0
                    else "clip raw ONNX output before last_action feedback and action decoding"
                )
            ),
        }
        (DIAGNOSTIC_DIR / "policy_diagnostic_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    def _save_wp29_state(
        self,
        data: mujoco.MjData,
        ids: dict[str, Any],
        command: np.ndarray,
    ) -> None:
        if self._wp29_saved:
            return
        base = data.xpos[ids["base_body_id"]]
        if np.linalg.norm(base[:2] - WP29_XY) > WP29_CAPTURE_RADIUS_M:
            return
        DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            DIAGNOSTIC_DIR / "wp29_natural_state.npz",
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            act=data.act.copy(),
            ctrl=data.ctrl.copy(),
            qacc_warmstart=data.qacc_warmstart.copy(),
            time=np.asarray(float(data.time)),
            last_action=self.last_action.copy(),
            position_target=self.position_target.copy(),
            velocity_target=self.velocity_target.copy(),
            command=np.asarray(command, dtype=np.float32),
            base_xyz=data.xpos[ids["base_body_id"]].copy(),
        )
        self._wp29_saved = True

    def _guard_action(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        observation: np.ndarray,
        raw_action: np.ndarray,
        command: np.ndarray,
    ) -> np.ndarray:
        recovery_observation = observation.copy()
        recovery_observation[-16:] = 0.0
        recovery_action = np.asarray(
            self.session.run(
                ["actions"], {"obs": recovery_observation[None]}
            )[0][0],
            dtype=np.float32,
        )
        recovery_peak = (
            float(np.max(np.abs(recovery_action)))
            if recovery_action.shape == (16,) and np.isfinite(recovery_action).all()
            else None
        )
        recovered = (
            recovery_peak is not None and recovery_peak <= POLICY_ACTION_GUARD
        )
        action = recovery_action if recovered else np.zeros(16, dtype=np.float32)
        self.guard_trigger_count += 1
        if recovered:
            self.guard_reinfer_success_count += 1
        else:
            self.guard_zero_fallback_count += 1
        raw_peak = (
            float(np.max(np.abs(raw_action)))
            if raw_action.shape == (16,) and np.isfinite(raw_action).all()
            else None
        )
        event = {
            "index": self.guard_trigger_count,
            "time_s": float(data.time),
            "raw_action_abs_max": raw_peak,
            "raw_action_finite": bool(
                raw_action.shape == (16,) and np.isfinite(raw_action).all()
            ),
            "recovery_action_abs_max": recovery_peak,
            "recovered_by_zero_last_action_reinference": recovered,
            "fallback": None if recovered else "zero_action",
            "qvel_abs_max": float(np.max(np.abs(data.qvel))),
            "command": np.asarray(command, dtype=np.float32).tolist(),
        }
        self.guard_events.append(event)
        DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
        (DIAGNOSTIC_DIR / "policy_guard_events.json").write_text(
            json.dumps(self.guard_events, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        if self.guard_trigger_count == 1:
            np.savez_compressed(
                DIAGNOSTIC_DIR / "policy_guard_first_trigger.npz",
                observation=observation,
                raw_action=raw_action,
                recovery_observation=recovery_observation,
                recovery_action=recovery_action,
                applied_action=action,
                command=np.asarray(command, dtype=np.float32),
                qpos=data.qpos.copy(),
                qvel=data.qvel.copy(),
                ctrl=data.ctrl.copy(),
                time=np.asarray(float(data.time)),
            )
        print(
            f"[policy-guard] trigger={self.guard_trigger_count} "
            f"sim={data.time:.3f}s raw_peak={raw_peak} "
            f"recovery_peak={recovery_peak} fallback={event['fallback']}",
            flush=True,
        )
        return action.astype(np.float32)

    def _write_failure(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        observation: np.ndarray,
        action: np.ndarray,
        command: np.ndarray,
    ) -> None:
        DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
        previous = list(self._history)
        np.savez_compressed(
            DIAGNOSTIC_DIR / "nonfinite_policy_snapshot.npz",
            current_observation=observation,
            current_action=action,
            current_command=np.asarray(command, dtype=np.float32),
            current_qpos=data.qpos.copy(),
            current_qvel=data.qvel.copy(),
            current_ctrl=data.ctrl.copy(),
            current_qacc=data.qacc.copy(),
            current_time=np.asarray(float(data.time)),
            history_observation=(
                np.stack([item["observation"] for item in previous])
                if previous else np.empty((0, observation.size), dtype=np.float32)
            ),
            history_action=(
                np.stack([item["action"] for item in previous])
                if previous else np.empty((0, 16), dtype=np.float32)
            ),
            history_raw_action=(
                np.stack([item["raw_action"] for item in previous])
                if previous else np.empty((0, 16), dtype=np.float32)
            ),
            history_command=(
                np.stack([item["command"] for item in previous])
                if previous else np.empty((0, 3), dtype=np.float32)
            ),
            history_qpos=(
                np.stack([item["qpos"] for item in previous])
                if previous else np.empty((0, data.qpos.size), dtype=np.float64)
            ),
            history_qvel=(
                np.stack([item["qvel"] for item in previous])
                if previous else np.empty((0, data.qvel.size), dtype=np.float64)
            ),
            history_time=np.asarray([item["time"] for item in previous], dtype=np.float64),
        )
        finite_obs = observation[np.isfinite(observation)]
        finite_action = action[np.isfinite(action)]
        report = {
            "observation_shape": list(observation.shape),
            "observation_finite": bool(np.isfinite(observation).all()),
            "observation_nonfinite_indices": np.flatnonzero(~np.isfinite(observation)).tolist(),
            "observation_finite_min": None if finite_obs.size == 0 else float(np.min(finite_obs)),
            "observation_finite_max": None if finite_obs.size == 0 else float(np.max(finite_obs)),
            "action_shape": list(action.shape),
            "action_nonfinite_indices": np.flatnonzero(~np.isfinite(action)).tolist(),
            "action_finite_min": None if finite_action.size == 0 else float(np.min(finite_action)),
            "action_finite_max": None if finite_action.size == 0 else float(np.max(finite_action)),
            "time_s": float(data.time),
            "qpos_finite": bool(np.isfinite(data.qpos).all()),
            "qvel_finite": bool(np.isfinite(data.qvel).all()),
            "qacc_finite": bool(np.isfinite(data.qacc).all()),
            "qvel_abs_max": float(np.nanmax(np.abs(data.qvel))),
            "qacc_abs_max": float(np.nanmax(np.abs(data.qacc))),
        }
        (DIAGNOSTIC_DIR / "nonfinite_policy_snapshot.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    def update(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ids: dict[str, Any],
        command: np.ndarray,
    ) -> None:
        self._save_wp29_state(data, ids, command)
        angular_local, _, _ = body_velocity(model, data, ids["base_body_id"])
        observation = build_observation(
            angular_local,
            data.qpos[3:7].copy(),
            np.asarray(command, dtype=np.float32),
            data.qpos[7:23].copy(),
            data.qvel[6:22].copy(),
            self.last_action,
        )
        raw_action = np.asarray(
            self.session.run(["actions"], {"obs": observation[None]})[0][0],
            dtype=np.float32,
        )
        raw_action_valid = raw_action.shape == (16,) and np.isfinite(raw_action).all()
        raw_action_peak = (
            float(np.max(np.abs(raw_action))) if raw_action_valid else None
        )
        guard_triggered = POLICY_ACTION_GUARD > 0.0 and (
            not raw_action_valid or raw_action_peak > POLICY_ACTION_GUARD
        )
        if not raw_action_valid and not guard_triggered:
            self._write_failure(model, data, observation, raw_action, command)
            raise RuntimeError(
                "official ONNX emitted invalid action; diagnostic snapshot written "
                f"to {DIAGNOSTIC_DIR}"
            )
        if raw_action_peak is not None:
            self.raw_action_abs_max = max(self.raw_action_abs_max, raw_action_peak)
        if guard_triggered:
            action = self._guard_action(
                model, data, observation, raw_action, command
            )
        elif POLICY_ACTION_CLIP > 0.0:
            self.clipped_value_count += int(
                np.count_nonzero(np.abs(raw_action) > POLICY_ACTION_CLIP)
            )
            action = np.clip(
                raw_action, -POLICY_ACTION_CLIP, POLICY_ACTION_CLIP
            ).astype(np.float32)
        else:
            action = raw_action
        self._history.append(
            {
                "observation": observation.copy(),
                "raw_action": raw_action.copy(),
                "action": action.copy(),
                "command": np.asarray(command, dtype=np.float32).copy(),
                "qpos": data.qpos.copy(),
                "qvel": data.qvel.copy(),
                "time": float(data.time),
            }
        )
        self.last_action = action.copy()
        position, velocity = decode_action(action)
        self.position_target = position.astype(np.float64)
        self.velocity_target = velocity.astype(np.float64)
        self.update_count += 1


# The full-track runner resolves OfficialPolicy from its module namespace.
# Replacing that symbol here leaves its route, timing, physics, and commands
# unchanged while adding only failure evidence.
runner.OfficialPolicy = DiagnosticOfficialPolicy


if __name__ == "__main__":
    try:
        exit_code = runner.main()
    finally:
        if DiagnosticOfficialPolicy.last_instance is not None:
            DiagnosticOfficialPolicy.last_instance.write_run_report()
    raise SystemExit(exit_code)
