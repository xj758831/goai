#!/usr/bin/env python3
"""Read-only review of targeted pose/speed expansion checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import train_s10_m20_curriculum_ppo as base_runner
import train_s10_m20_lidar_curriculum_ppo as lidar_runner
import train_s10_m20_terrain_lidar_pose_speed_expansion_ppo as terrain_runner
from train_s10_m20_lidar_low_torque_stage_ppo import (
    FIXED_STATES,
    S10M20LidarLowTorqueCapabilityEnv,
)
from s10_m20_lidar_bounded_residual_actor import S10LidarBoundedResidualActorCritic
from s10_m20_terrain_lidar_balanced_stage_env import (
    ACTOR_OBS_DIM as TERRAIN_ACTOR_OBS_DIM,
    CRITIC_OBS_DIM as TERRAIN_CRITIC_OBS_DIM,
    S10M20TerrainLidarBalancedStageEnv,
)


DEFAULT_REFERENCE = Path(
    "logs/mujoco/"
    "s10_m20_lidar72_capability_depth021_complementary_distill_base10_alt1_lr1e6_"
    "epoch100_20260809_v1/checkpoint_best_distilled.pt"
)
DEFAULT_TRAINING_DIR = Path(
    "logs/mujoco/s10_lidar_pose_speed_expansion_80iter_8env_20260810_v1"
)

# These are the exact search successes presented to the anchor collector.  With
# the fixed collection seeds, the reference checkpoint reproduces seven of nine.
DEEP_CASES: tuple[tuple[float, float, float, float], ...] = (
    (0.225, 0.70, 0.02, 0.0),
    (0.225, 0.70, -0.03, -2.0),
    (0.230, 0.70, -0.04, 0.0),
    (0.230, 0.70, -0.03, 0.0),
    (0.235, 0.70, -0.02, -1.0),
    (0.240, 0.80, -0.03, -5.0),
    (0.250, 0.70, -0.03, -5.0),
    (0.260, 0.70, -0.03, -3.5),
    (0.270, 0.65, -0.035, -4.5),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--terrain-lidar", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--terrain-input-scale", type=float, default=1.0)
    return parser.parse_args()


def rollout(
    agent: Any,
    env_class: Any,
    *,
    role: str,
    depth_m: float,
    command_mps: float,
    lateral_m: float,
    yaw_deg: float,
    seed: int,
) -> dict[str, Any]:
    state = (lateral_m, yaw_deg)
    env = env_class(
        depth_m=depth_m,
        command_mps=command_mps,
        training_states=(state,),
    )
    episode_return = 0.0
    max_action_abs = 0.0
    max_residual_action_abs = 0.0
    gate_sum = 0.0
    gate_max = 0.0
    gate_active_steps = 0
    measured_gate_steps = 0
    episode_min_lidar_normalized = 1.0
    episode_min_lidar_beam_index = -1
    lidar_below_gate_start_steps = 0
    measured_lidar_steps = 0
    try:
        actor_obs, _, _ = env.reset(seed=seed, initial_state=state)
        final_info: dict[str, Any] | None = None
        for _ in range(env.max_episode_steps):
            with torch.inference_mode():
                actor_tensor = torch.as_tensor(actor_obs, dtype=torch.float32).unsqueeze(0)
                lidar = actor_obs[57:]
                nearest_lidar_index = int(np.argmin(lidar))
                nearest_lidar = float(lidar[nearest_lidar_index])
                if nearest_lidar < episode_min_lidar_normalized:
                    episode_min_lidar_normalized = nearest_lidar
                    episode_min_lidar_beam_index = nearest_lidar_index
                lidar_below_gate_start_steps += int(nearest_lidar < 0.15)
                measured_lidar_steps += 1
                action = (
                    agent.deterministic(actor_tensor)
                    .squeeze(0)
                    .numpy()
                    .astype(np.float32)
                )
                if hasattr(agent.actor, "base") and hasattr(agent.actor, "obstacle_gate"):
                    base_action = agent.actor.base(actor_tensor).squeeze(0).numpy()
                    gate = float(agent.actor.obstacle_gate(actor_tensor).item())
                    max_residual_action_abs = max(
                        max_residual_action_abs,
                        float(np.max(np.abs(action - base_action))),
                    )
                    gate_sum += gate
                    gate_max = max(gate_max, gate)
                    gate_active_steps += int(gate > 0.0)
                    measured_gate_steps += 1
            max_action_abs = max(max_action_abs, float(np.max(np.abs(action))))
            actor_obs, _, reward, terminated, truncated, final_info = env.step(action)
            episode_return += float(reward)
            if terminated or truncated:
                break
        if final_info is None:
            raise RuntimeError("candidate review rollout produced no transition")
        metrics = final_info["metrics"]
        return {
            "role": role,
            "depth_m": depth_m,
            "command_mps": command_mps,
            "lateral_m": lateral_m,
            "yaw_deg": yaw_deg,
            "seed": seed,
            "success": bool(final_info["success"]),
            "termination_reason": final_info["termination_reason"],
            "front_top_latched": bool(metrics["front_top_latched"]),
            "rear_top_latched": bool(metrics["rear_top_latched"]),
            "base_cross_latched": bool(metrics["base_cross_latched"]),
            "episode_return": episode_return,
            "episode_steps": int(final_info["episode_step"]),
            "time_s": float(final_info["time_s"]),
            "max_contact_force_n": float(metrics["episode_max_force_n"]),
            "max_abs_roll_deg": math.degrees(float(metrics["episode_max_roll_rad"])),
            "max_abs_pitch_deg": math.degrees(float(metrics["episode_max_pitch_rad"])),
            "max_action_abs": max_action_abs,
            "max_residual_action_abs": (
                max_residual_action_abs if measured_gate_steps else None
            ),
            "mean_obstacle_gate": (
                gate_sum / measured_gate_steps if measured_gate_steps else None
            ),
            "max_obstacle_gate": gate_max if measured_gate_steps else None,
            "obstacle_gate_active_rate": (
                gate_active_steps / measured_gate_steps if measured_gate_steps else None
            ),
            "episode_min_lidar_normalized": episode_min_lidar_normalized,
            "episode_min_lidar_beam_index": episode_min_lidar_beam_index,
            "any_lidar_below_0p15_rate": lidar_below_gate_start_steps / measured_lidar_steps,
        }
    finally:
        env.close()


def case_specs() -> list[dict[str, Any]]:
    specs = [
        {
            "role": "deep_anchor",
            "depth_m": depth,
            "command_mps": speed,
            "lateral_m": lateral,
            "yaw_deg": yaw,
            "seed": 30_000 + index,
        }
        for index, (depth, speed, lateral, yaw) in enumerate(DEEP_CASES)
    ]
    for role, depth in (("fixed_0p21", 0.21), ("fixed_0p18", 0.18)):
        specs.extend(
            {
                "role": role,
                "depth_m": depth,
                "command_mps": 0.80,
                "lateral_m": lateral,
                "yaw_deg": yaw,
                "seed": 10_000 + index,
            }
            for index, (lateral, yaw) in enumerate(FIXED_STATES)
        )
    return specs


def summarize_role(cases: list[dict[str, Any]], role: str) -> dict[str, Any]:
    selected = [case for case in cases if case["role"] == role]
    successes = [index for index, case in enumerate(selected) if case["success"]]
    return {
        "case_count": len(selected),
        "success_count": len(successes),
        "success_indexes": successes,
        "front_top_count": sum(bool(case["front_top_latched"]) for case in selected),
        "rear_top_count": sum(bool(case["rear_top_latched"]) for case in selected),
        "base_cross_count": sum(bool(case["base_cross_latched"]) for case in selected),
        "max_contact_force_n": max(float(case["max_contact_force_n"]) for case in selected),
    }


def evaluate_checkpoint(
    checkpoint: Path,
    *,
    label: str,
    specs: list[dict[str, Any]],
    workers: int,
    env_class: Any,
    terrain_lidar: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_state = payload.get("model")
    if not isinstance(model_state, dict):
        raise ValueError(f"checkpoint lacks model state: {checkpoint}")
    if terrain_lidar:
        actor_class = terrain_runner.TerrainLidarInitializedActorCritic
    else:
        actor_class = (
            S10LidarBoundedResidualActorCritic
            if any(key.startswith("actor.base.") for key in model_state)
            else lidar_runner.LidarInitializedActorCritic
        )
    agent = actor_class(action_std_scale=1.0)
    agent.load_state_dict(model_state, strict=True)
    agent.eval()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        cases = list(executor.map(lambda spec: rollout(agent, env_class, **spec), specs))
    role_reports = {
        role: summarize_role(cases, role)
        for role in ("deep_anchor", "fixed_0p21", "fixed_0p18")
    }
    deep_success_depths = [
        float(case["depth_m"])
        for case in cases
        if case["role"] == "deep_anchor" and case["success"]
    ]
    return {
        "label": label,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": base_runner.sha256(checkpoint),
        "checkpoint_iteration": payload.get("iteration"),
        "role_reports": role_reports,
        "deepest_success_m": max(deep_success_depths, default=None),
        "cases": cases,
    }


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if not math.isfinite(args.terrain_input_scale) or args.terrain_input_scale <= 0.0:
        raise ValueError("terrain-input-scale must be finite and positive")
    reference = args.reference_checkpoint.expanduser().resolve()
    training_dir = args.training_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    candidates = sorted(training_dir.glob("checkpoint_candidate_iter_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no candidate checkpoints in {training_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    torch.set_num_threads(1)
    lidar_runner.configure_runner()
    if args.terrain_lidar:
        terrain_runner.TERRAIN_INPUT_SCALE = float(args.terrain_input_scale)
        base_runner.ACTOR_OBS_DIM = TERRAIN_ACTOR_OBS_DIM
        base_runner.CRITIC_OBS_DIM = TERRAIN_CRITIC_OBS_DIM
        env_class = S10M20TerrainLidarBalancedStageEnv
    else:
        env_class = S10M20LidarLowTorqueCapabilityEnv
    specs = case_specs()
    policies: list[dict[str, Any]] = []
    checkpoints = [("reference", reference)] + [
        (checkpoint.stem.removeprefix("checkpoint_candidate_"), checkpoint)
        for checkpoint in candidates
    ]
    for label, checkpoint in checkpoints:
        report = evaluate_checkpoint(
            checkpoint,
            label=label,
            specs=specs,
            workers=args.workers,
            env_class=env_class,
            terrain_lidar=args.terrain_lidar,
        )
        policies.append(report)
        roles = report["role_reports"]
        print(
            f"[review] {label}: deep={roles['deep_anchor']['success_count']}/9 "
            f"deepest={report['deepest_success_m']} "
            f"0.21={roles['fixed_0p21']['success_count']}/5 "
            f"0.18={roles['fixed_0p18']['success_count']}/5",
            flush=True,
        )

    reference_report = policies[0]
    reference_deep = {
        index
        for index, case in enumerate(reference_report["cases"][: len(DEEP_CASES)])
        if case["success"]
    }
    for report in policies:
        candidate_deep = {
            index
            for index, case in enumerate(report["cases"][: len(DEEP_CASES)])
            if case["success"]
        }
        report["deep_successes_gained_vs_reference"] = sorted(candidate_deep - reference_deep)
        report["deep_successes_lost_vs_reference"] = sorted(reference_deep - candidate_deep)

    summary = {
        "purpose": "read-only review of targeted pose/speed expansion candidates",
        "terrain_lidar": bool(args.terrain_lidar),
        "official_assets_modified": False,
        "success_gate": "stable physical exit; 500 N report-only; official 50/14 Nm",
        "case_protocol": {
            "deep_anchor_count": len(DEEP_CASES),
            "deep_anchor_reset_seeds": [30_000 + index for index in range(len(DEEP_CASES))],
            "fixed_states": list(FIXED_STATES),
            "fixed_reset_seeds": [10_000 + index for index in range(len(FIXED_STATES))],
            "total_cases_per_policy": len(specs),
        },
        "policy_count": len(policies),
        "policies": policies,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(f"[result] summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
