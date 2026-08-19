#!/usr/bin/env python3
"""Small two-stage learned-policy rescue search.

This is the next isolated experiment after a front-support-only rescue failed.
The locked learned policy is unchanged until a deployable approach estimator
has been above threshold for three steps.  A short front-preparation residual
then fades out; after the deployable front-support estimator latches, a rear
leg/wheel residual is applied.  Hard failures are never CEM elites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import train_s10_m20_curriculum_ppo as base_runner
import train_s10_m20_lidar_curriculum_ppo as lidar_runner
from s10_m20_phase_gated_residual import (
    PHASE_THRESHOLD,
    S10M20PhaseGatedEnv,
)
from train_s10_m20_approach_estimator import ApproachEstimator
from train_s10_m20_curriculum_ppo import GifRecorder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_CHECKPOINT = Path(
    "logs/mujoco/"
    "s10_m20_lidar72_capability_depth021_complementary_distill_base10_alt1_lr1e6_"
    "epoch100_20260809_v1/checkpoint_best_distilled.pt"
)
APPROACH_ESTIMATOR = Path(
    "logs/mujoco/s10_m20_approach_estimator_19cases_lead40_20260810_v2/approach_estimator.pt"
)
PHASE_ESTIMATOR = Path(
    "logs/mujoco/s10_m20_phase_estimator_19cases_20260810_v1/phase_estimator.pt"
)
EXPECTED_REFERENCE_SHA256 = "a04308a3ca40243cd9736b870471dc3ac08b481ed24d3dd71320b4789d535af3"
EXPECTED_APPROACH_SHA256 = "f6375d7d7dd3862c1024356795d95a2247a508576daedd61b2c7f42a9ae34bd2"
EXPECTED_PHASE_SHA256 = "6d497ed2b21765773ab286cebedf459f46a9f7b70ace0d649f63250658e56a66"
OFFICIAL_ASSETS = (
    Path("src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10.xml"),
    Path("src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/scene.xml"),
    Path("src/S10_sdk_deploy/S10_description/s10_mjcf/mjcf/S10_track.xml"),
)

DEPTH_M = 0.21
COMMAND_MPS = 0.8
INITIAL_STATE = (0.0, 0.0)
ROLLOUT_SEED = 10_000
ACTION_DIM = 16
TERRAIN_OBS_DIM = 194
HISTORY_FRAMES = 5
APPROACH_THRESHOLD = 0.80
DWELL_STEPS = 3
CONTROL_DT = 0.02
PRE_ACTIVE = np.arange(0, 6, dtype=np.int64)
POST_ACTIVE = np.asarray([6, 7, 8, 9, 10, 11, 14, 15], dtype=np.int64)
HARD_FAILURES = {
    "body_wall_contact",
    "excessive_contact_force",
    "excessive_roll",
    "excessive_pitch",
    "non_finite_state",
}


@dataclass(frozen=True)
class Limits:
    pre_bound: np.ndarray
    post_bound: np.ndarray
    pre_std: np.ndarray
    post_std: np.ndarray
    rate_per_second: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, default=REFERENCE_CHECKPOINT)
    parser.add_argument("--approach-estimator", type=Path, default=APPROACH_ESTIMATOR)
    parser.add_argument("--phase-estimator", type=Path, default=PHASE_ESTIMATOR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--depth-m", type=float, default=DEPTH_M)
    parser.add_argument("--command-mps", type=float, default=COMMAND_MPS)
    parser.add_argument("--lateral-m", type=float, default=INITIAL_STATE[0])
    parser.add_argument("--yaw-deg", type=float, default=INITIAL_STATE[1])
    parser.add_argument("--rollout-seed", type=int, default=ROLLOUT_SEED)
    parser.add_argument("--approach-gate", choices=("estimator", "terrain"), default="estimator")
    parser.add_argument("--terrain-approach-threshold", type=float, default=0.03)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--cem-update-rate", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--gif-frame-period", type=float, default=0.08)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset_hashes() -> dict[str, str]:
    return {str(path): sha256(path) for path in OFFICIAL_ASSETS}


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.population < 8 or args.iterations < 1 or args.workers < 1:
        raise ValueError("population must be >=8; iterations and workers must be positive")
    if not 0.05 <= args.elite_fraction <= 0.75 or not 0.0 < args.cem_update_rate <= 1.0:
        raise ValueError("invalid CEM parameters")
    reference = args.reference_checkpoint.expanduser().resolve()
    approach = args.approach_estimator.expanduser().resolve()
    phase = args.phase_estimator.expanduser().resolve()
    for path in (reference, approach, phase):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(reference) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("reference checkpoint hash differs from locked checkpoint")
    if sha256(approach) != EXPECTED_APPROACH_SHA256:
        raise RuntimeError("approach estimator hash differs from validated v2 estimator")
    if sha256(phase) != EXPECTED_PHASE_SHA256:
        raise RuntimeError("front-support estimator hash differs from validated estimator")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = PROJECT_ROOT / "logs" / "mujoco" / f"s10_m20_approach_two_stage_rescue_{stamp}"
    else:
        output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return reference, approach, phase, output


def load_reference(path: Path) -> Any:
    lidar_runner.configure_runner()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("reference checkpoint lacks model state")
    agent = lidar_runner.LidarInitializedActorCritic(action_std_scale=1.0)
    agent.load_state_dict(state, strict=True)
    agent.eval()
    return agent


def load_approach(path: Path) -> tuple[ApproachEstimator, torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "s10_deployable_approach_estimator_v1":
        raise ValueError("unsupported approach estimator format")
    if int(payload.get("history_frames", -1)) != HISTORY_FRAMES:
        raise ValueError("approach history length mismatch")
    if int(payload.get("actor_observation_dim", -1)) != TERRAIN_OBS_DIM:
        raise ValueError("approach actor observation dimension mismatch")
    estimator = ApproachEstimator(TERRAIN_OBS_DIM * HISTORY_FRAMES, hidden_dim=256)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("approach estimator lacks model")
    estimator.load_state_dict(state, strict=True)
    estimator.eval()
    mean, std = payload.get("mean"), payload.get("std")
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise ValueError("approach estimator lacks normalization")
    if mean.shape != (TERRAIN_OBS_DIM * HISTORY_FRAMES,) or std.shape != mean.shape:
        raise ValueError("approach normalization shape mismatch")
    return estimator, mean.float(), std.float().clamp_min(1.0e-3)


def limits() -> Limits:
    return Limits(
        pre_bound=np.asarray([0.14] * 6, dtype=np.float64),
        post_bound=np.asarray([0.20] * 6 + [1.0, 1.0], dtype=np.float64),
        pre_std=np.asarray([0.035] * 6, dtype=np.float64),
        post_std=np.asarray([0.055] * 6 + [0.22, 0.22], dtype=np.float64),
        rate_per_second=np.asarray([0.65] * 12 + [0.0, 0.0, 3.0, 3.0], dtype=np.float64),
    )


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def clip_candidate(values: np.ndarray, constraint: Limits) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    bounds = np.concatenate([constraint.pre_bound, constraint.post_bound])
    return np.clip(result, -bounds, bounds)


def structured_seeds(constraint: Limits) -> list[tuple[str, np.ndarray]]:
    def item(name: str, pre: np.ndarray, post: np.ndarray) -> tuple[str, np.ndarray]:
        value = np.concatenate([pre, post]).astype(np.float64)
        value[0:6] *= 0.5
        return name, clip_candidate(value, constraint)

    zero = np.zeros(14, dtype=np.float64)
    seeds = [("zero", zero)]
    pre_a = np.asarray([0.0, -0.08, 0.14, 0.0, -0.08, 0.14])
    post_a = np.asarray([0.0, -0.06, 0.12, 0.0, -0.06, 0.12, 0.40, 0.40])
    seeds.append(item("front_prepare_a", pre_a, post_a))
    seeds.append(item("front_prepare_b", -pre_a, post_a))
    seeds.append(item("front_asymmetric", np.asarray([0.0, -0.08, 0.14, 0.0, 0.08, -0.14]), post_a))
    seeds.append(item("rear_wheels_reverse", pre_a, np.asarray([0.0] * 6 + [-0.60, -0.60])))
    seeds.append(item("rear_wheels_forward", pre_a, np.asarray([0.0] * 6 + [0.60, 0.60])))
    seeds.append(item("front_only", pre_a, np.zeros(8)))
    seeds.append(item("rear_only", np.zeros(6), post_a))
    return seeds


def phase_probability(
    estimator: Any,
    mean: torch.Tensor,
    std: torch.Tensor,
    history: list[np.ndarray],
) -> float:
    features = np.concatenate(history).astype(np.float32)
    with torch.inference_mode():
        return float(
            torch.sigmoid(
                estimator(((torch.as_tensor(features) - mean) / std).unsqueeze(0))
            )[0]
        )


def rank_and_score(summary: dict[str, Any]) -> tuple[int, float]:
    if summary["hard_failure"] or not summary["approach_triggered"]:
        rank = 0
    elif summary["success"]:
        rank = 4
    elif summary["rear_top_latched"]:
        rank = 3
    elif summary["front_top_latched"] and summary["rear_pair_progress_gain_after_front_m"] > 0.01:
        rank = 2
    elif summary["front_top_latched"] and summary["time_s"] > 3.5:
        rank = 1
    else:
        rank = 0
    score = (
        2200.0 * summary["rear_pair_progress_gain_after_front_m"]
        + 400.0 * summary["base_progress_gain_after_approach_m"]
        + 40.0 * summary["front_support_rate_after_approach"]
        + 3.0 * summary["time_s"]
        - 55.0 * math.radians(summary["max_abs_roll_deg"])
        - 2.0 * summary["mean_residual_l2"]
    )
    return rank, float(score)


def rollout(
    agent: Any,
    approach_estimator: Any,
    approach_mean: torch.Tensor,
    approach_std: torch.Tensor,
    phase_path: Path,
    candidate: np.ndarray,
    constraint: Limits,
    *,
    label: str,
    expected_approach_step: int | None = None,
    expected_prefix_sha256: str | None = None,
    gif_path: Path | None = None,
    trace_path: Path | None = None,
    gif_period: float = 0.08,
    depth_m: float = DEPTH_M,
    command_mps: float = COMMAND_MPS,
    initial_state: tuple[float, float] = INITIAL_STATE,
    rollout_seed: int = ROLLOUT_SEED,
    approach_gate: str = "estimator",
    terrain_approach_threshold: float = 0.03,
) -> dict[str, Any]:
    if approach_gate not in {"estimator", "terrain"}:
        raise ValueError(f"unsupported approach gate: {approach_gate}")
    S10M20PhaseGatedEnv.phase_estimator_path = phase_path
    env = S10M20PhaseGatedEnv(
        depth_m=depth_m,
        command_mps=command_mps,
        training_states=(initial_state,),
    )
    recorder = GifRecorder(env.model, gif_period) if gif_path is not None else None
    approach_history: list[np.ndarray] | None = None
    approach_streak = 0
    front_streak = 0
    approach_step: int | None = None
    front_step: int | None = None
    true_front_step: int | None = None
    prefix_actions: list[np.ndarray] = []
    previous_residual = np.zeros(ACTION_DIM, dtype=np.float64)
    post_steps = 0
    post_front_steps = 0
    residual_l2_sum = 0.0
    peak_rear_after_front = 0.0
    initial_rear = 0.0
    approach_base = 0.0
    peak_base_after_approach = -math.inf
    final_info: dict[str, Any] | None = None
    trace: dict[str, list[Any]] = {name: [] for name in (
        "time_s", "approach_probability", "terrain_min", "front_probability", "residual", "final_action",
        "roll_rad", "pitch_rad", "base_progress_m", "wheel_progress_m", "wheel_force_n",
        "front_top", "rear_top", "base_cross",
    )}
    try:
        actor_obs, _, info = env.reset(seed=rollout_seed, initial_state=initial_state)
        terrain_obs = actor_obs[:TERRAIN_OBS_DIM].copy()
        approach_history = [terrain_obs.copy() for _ in range(HISTORY_FRAMES)]
        initial_metrics = info["metrics"]
        initial_rear = float(np.min(initial_metrics["wheel_progress_m"][2:4]))
        peak_rear_after_front = initial_rear
        approach_base = float(initial_metrics["base_progress_m"])
        peak_base_after_approach = approach_base
        if recorder is not None:
            recorder.capture(env)
        for step in range(env.max_episode_steps):
            terrain_min = float(np.min(terrain_obs[129:]))
            if approach_gate == "estimator":
                probability = phase_probability(
                    approach_estimator, approach_mean, approach_std, approach_history
                )
                approach_active = probability >= APPROACH_THRESHOLD
            else:
                approach_active = terrain_min <= terrain_approach_threshold
                probability = float(approach_active)
            approach_streak = approach_streak + 1 if approach_active else 0
            if approach_step is None and approach_streak >= DWELL_STEPS:
                approach_step = step
            front_probability = float(actor_obs[-1])
            front_streak = front_streak + 1 if front_probability >= PHASE_THRESHOLD else 0
            if front_step is None and front_streak >= DWELL_STEPS:
                front_step = step
            with torch.inference_mode():
                nominal = (
                    agent.deterministic(torch.as_tensor(actor_obs[:129], dtype=torch.float32).unsqueeze(0))
                    .squeeze(0).cpu().numpy().astype(np.float32)
                )
            residual = np.zeros(ACTION_DIM, dtype=np.float64)
            if approach_step is None:
                prefix_actions.append(nominal.copy())
            else:
                approach_age = (step - approach_step) * CONTROL_DT
                pre_env = smoothstep(approach_age / 0.15)
                if approach_age > 0.65:
                    pre_env *= 1.0 - smoothstep((approach_age - 0.65) / 0.35)
                residual[PRE_ACTIVE] = candidate[:6] * pre_env
                if front_step is not None:
                    post_age = (step - front_step) * CONTROL_DT
                    residual[POST_ACTIVE] = candidate[6:] * smoothstep(post_age / 0.15)
                maximum_delta = constraint.rate_per_second * CONTROL_DT
                residual = previous_residual + np.clip(
                    residual - previous_residual, -maximum_delta, maximum_delta
                )
                previous_residual = residual
                post_steps += 1
                residual_l2_sum += float(np.linalg.norm(residual))
            final_action = nominal + residual.astype(np.float32)
            actor_obs, _, _, terminated, truncated, final_info = env.step(final_action)
            metrics = final_info["metrics"]
            terrain_obs = actor_obs[:TERRAIN_OBS_DIM].copy()
            approach_history = [*approach_history[1:], terrain_obs.copy()]
            if true_front_step is None and bool(metrics["front_top_latched"]):
                true_front_step = step + 1
            if approach_step is not None:
                post_front_steps += int(bool(metrics["front_top"]))
            if front_step is not None:
                peak_rear_after_front = max(
                    peak_rear_after_front,
                    float(np.min(metrics["wheel_progress_m"][2:4])),
                )
            if approach_step is not None:
                peak_base_after_approach = max(peak_base_after_approach, float(metrics["base_progress_m"]))
            if trace_path is not None:
                values = {
                    "time_s": float(final_info["time_s"]),
                    "approach_probability": probability,
                    "terrain_min": terrain_min,
                    "front_probability": front_probability,
                    "residual": residual.copy(),
                    "final_action": final_action.copy(),
                    "roll_rad": float(metrics["roll_rad"]),
                    "pitch_rad": float(metrics["pitch_rad"]),
                    "base_progress_m": float(metrics["base_progress_m"]),
                    "wheel_progress_m": np.asarray(metrics["wheel_progress_m"]).copy(),
                    "wheel_force_n": np.asarray(metrics["wheel_force_n"]).copy(),
                    "front_top": bool(metrics["front_top"]),
                    "rear_top": bool(metrics["rear_top"]),
                    "base_cross": bool(metrics["base_cross"]),
                }
                for name, value in values.items():
                    trace[name].append(value)
            if recorder is not None:
                recorder.capture(env)
            if terminated or truncated:
                break
        if final_info is None:
            raise RuntimeError("two-stage rollout produced no transition")
        prefix = np.stack(prefix_actions).astype(np.float32).tobytes() if prefix_actions else b""
        prefix_hash = hashlib.sha256(prefix).hexdigest()
        if expected_approach_step is not None and approach_step != expected_approach_step:
            raise RuntimeError(f"candidate {label} changed approach trigger step")
        if expected_prefix_sha256 is not None and prefix_hash != expected_prefix_sha256:
            raise RuntimeError(f"candidate {label} changed reference prefix")
        metrics = final_info["metrics"]
        reason = str(final_info["termination_reason"])
        hard_failure = reason not in {"success", "time_limit", ""}
        report: dict[str, Any] = {
            "label": label,
            "depth_m": depth_m,
            "command_mps": command_mps,
            "lateral_m": initial_state[0],
            "yaw_deg": initial_state[1],
            "seed": rollout_seed,
            "success": bool(final_info["success"]),
            "termination_reason": reason,
            "hard_failure": bool(hard_failure),
            "approach_triggered": approach_step is not None,
            "approach_gate": approach_gate,
            "terrain_approach_threshold": terrain_approach_threshold,
            "approach_trigger_step": approach_step,
            "front_phase_trigger_step": front_step,
            "true_front_latch_step": true_front_step,
            "prefix_action_count": len(prefix_actions),
            "prefix_action_sha256": prefix_hash,
            "pretrigger_max_action_error": 0.0,
            "front_top_latched": bool(metrics["front_top_latched"]),
            "rear_top_latched": bool(metrics["rear_top_latched"]),
            "base_cross_latched": bool(metrics["base_cross_latched"]),
            "time_s": float(final_info["time_s"]),
            "episode_steps": int(final_info["episode_step"]),
            "max_abs_roll_deg": math.degrees(float(metrics["episode_max_roll_rad"])),
            "max_abs_pitch_deg": math.degrees(float(metrics["episode_max_pitch_rad"])),
            "max_contact_force_n": float(metrics["episode_max_force_n"]),
            "contact_force_above_500_n": bool(metrics["episode_max_force_n"] > 500.0),
            "front_support_rate_after_approach": post_front_steps / max(post_steps, 1),
            "rear_pair_progress_gain_after_front_m": peak_rear_after_front - initial_rear,
            "base_progress_gain_after_approach_m": peak_base_after_approach - approach_base,
            "mean_residual_l2": residual_l2_sum / max(post_steps, 1),
            "candidate_pre_vector": candidate[:6].tolist(),
            "candidate_post_vector": candidate[6:].tolist(),
        }
        rank, score = rank_and_score(report)
        report["milestone_rank"] = rank
        report["milestone_name"] = {
            0: "hard_failure_or_no_approach",
            1: "front_support_retained",
            2: "rear_progress_with_front_support",
            3: "rear_top",
            4: "stable_exit",
        }[rank]
        report["tie_break_score"] = score
        if trace_path is not None:
            np.savez_compressed(trace_path, **{name: np.asarray(values) for name, values in trace.items()})
        return report
    finally:
        if recorder is not None and gif_path is not None:
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            recorder.close(gif_path)
        env.close()


def candidate_key(item: tuple[dict[str, Any], np.ndarray]) -> tuple[int, int, float]:
    return (
        int(item[0]["milestone_rank"]),
        int(not item[0]["hard_failure"]),
        float(item[0]["tie_break_score"]),
    )


def main() -> int:
    args = parse_args()
    reference, approach_path, phase_path, output = validate_inputs(args)
    before_assets = asset_hashes()
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    agent = load_reference(reference)
    approach_estimator, approach_mean, approach_std = load_approach(approach_path)
    constraint = limits()
    case_kwargs = {
        "depth_m": float(args.depth_m),
        "command_mps": float(args.command_mps),
        "initial_state": (float(args.lateral_m), float(args.yaw_deg)),
        "rollout_seed": int(args.rollout_seed),
        "approach_gate": args.approach_gate,
        "terrain_approach_threshold": float(args.terrain_approach_threshold),
    }
    zero = np.zeros(14, dtype=np.float64)
    mean = zero.copy()
    std = np.concatenate([constraint.pre_std, constraint.post_std]).astype(np.float64)
    floor = 0.15 * std
    iteration_reports: list[dict[str, Any]] = []
    all_candidates: list[tuple[dict[str, Any], np.ndarray]] = []
    baseline = rollout(
        agent, approach_estimator, approach_mean, approach_std, phase_path, zero, constraint,
        label="reference_zero_two_stage", trace_path=output / "baseline_trace.npz",
        **case_kwargs,
    )
    print(
        f"[baseline] reason={baseline['termination_reason']} approach={baseline['approach_trigger_step']} "
        f"front={baseline['front_phase_trigger_step']} roll={baseline['max_abs_roll_deg']:.2f}deg",
        flush=True,
    )
    seeds = structured_seeds(constraint)
    for iteration in range(1, args.iterations + 1):
        population: list[tuple[str, np.ndarray]] = []
        if iteration == 1:
            population.extend((name, value.copy()) for name, value in seeds)
        else:
            population.extend([("zero", zero.copy()), ("cem_mean", mean.copy())])
        while len(population) < args.population:
            population.append((
                f"sample_{len(population):03d}",
                clip_candidate(mean + std * rng.normal(size=mean.shape), constraint),
            ))
        population = population[:args.population]

        def evaluate(item: tuple[int, tuple[str, np.ndarray]]) -> tuple[dict[str, Any], np.ndarray]:
            index, (name, values) = item
            return rollout(
                agent, approach_estimator, approach_mean, approach_std, phase_path, values, constraint,
                label=f"iter{iteration:02d}_{index:03d}_{name}",
                expected_approach_step=baseline["approach_trigger_step"],
                expected_prefix_sha256=baseline["prefix_action_sha256"],
                **case_kwargs,
            ), values

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            evaluated = list(executor.map(evaluate, enumerate(population)))
        all_candidates.extend(evaluated)
        np.savez_compressed(
            output / f"population_iter_{iteration:04d}.npz",
            labels=np.asarray([item[0]["label"] for item in evaluated]),
            candidates=np.stack([item[1] for item in evaluated]),
        )
        ordered = sorted(evaluated, key=candidate_key, reverse=True)
        eligible = [item for item in ordered if not item[0]["hard_failure"] and item[0]["approach_triggered"]]
        elite_count = max(2, int(math.ceil(args.population * args.elite_fraction)))
        elites = eligible[:elite_count]
        update = len(elites) >= 2
        if update:
            values = np.stack([item[1] for item in elites])
            rate = args.cem_update_rate
            mean = clip_candidate((1.0 - rate) * mean + rate * values.mean(axis=0), constraint)
            std = np.maximum((1.0 - rate) * std + rate * values.std(axis=0), floor)
        best = ordered[0][0]
        report = {
            "iteration": iteration,
            "population": len(evaluated),
            "eligible_count": len(eligible),
            "elite_count": len(elites) if update else 0,
            "cem_update_applied": update,
            "rear_top_count": sum(bool(item[0]["rear_top_latched"]) for item in evaluated),
            "safe_success_count": sum(bool(item[0]["success"]) for item in evaluated),
            "hard_failure_count": sum(bool(item[0]["hard_failure"]) for item in evaluated),
            "termination_reason_counts": {
                reason: sum(item[0]["termination_reason"] == reason for item in evaluated)
                for reason in sorted({item[0]["termination_reason"] for item in evaluated})
            },
            "best": best,
            "candidates": [item[0] for item in evaluated],
        }
        iteration_reports.append(report)
        print(
            f"[cem] iter={iteration}/{args.iterations} rank={best['milestone_rank']} "
            f"eligible={len(eligible)}/{len(evaluated)} rear_top={report['rear_top_count']} update={int(update)}",
            flush=True,
        )
        if not update:
            break
    best_summary, best_candidate = max(all_candidates, key=candidate_key)
    np.savez_compressed(output / "best_candidate.npz", candidate=best_candidate)
    gif_dir = output / "gifs"
    baseline_replay = rollout(
        agent, approach_estimator, approach_mean, approach_std, phase_path, zero, constraint,
        label="reference_zero_two_stage_replay", expected_approach_step=baseline["approach_trigger_step"],
        expected_prefix_sha256=baseline["prefix_action_sha256"], gif_path=gif_dir / "reference_zero_two_stage.gif",
        gif_period=args.gif_frame_period, **case_kwargs,
    )
    best_replay = rollout(
        agent, approach_estimator, approach_mean, approach_std, phase_path, best_candidate, constraint,
        label="best_two_stage_replay", expected_approach_step=baseline["approach_trigger_step"],
        expected_prefix_sha256=baseline["prefix_action_sha256"], gif_path=gif_dir / "best_two_stage.gif",
        gif_period=args.gif_frame_period, trace_path=output / "best_trace.npz", **case_kwargs,
    )
    baseline_replay_match = all(
        baseline_replay[key] == baseline[key]
        for key in ("termination_reason", "episode_steps", "approach_trigger_step", "front_phase_trigger_step")
    )
    best_replay_match = all(
        best_replay[key] == best_summary[key]
        for key in (
            "success", "termination_reason", "episode_steps", "approach_trigger_step",
            "front_phase_trigger_step", "front_top_latched", "rear_top_latched", "base_cross_latched",
        )
    )
    after_assets = asset_hashes()
    if after_assets != before_assets:
        raise RuntimeError("official assets changed during search")
    summary = {
        "purpose": "two-stage approach-preparation plus front-support rear-transfer feasibility pilot",
        "geometry_scope": "official-derived diagonal vertical-wall proxy pit, not full official track mesh",
        "policy_source": "locked learned policy; no official ONNX or teleoperation",
        "reference_checkpoint": str(reference),
        "reference_checkpoint_sha256": sha256(reference),
        "approach_estimator": str(approach_path),
        "approach_estimator_sha256": sha256(approach_path),
        "phase_estimator": str(phase_path),
        "phase_estimator_sha256": sha256(phase_path),
        "case": {
            "depth_m": case_kwargs["depth_m"], "command_mps": case_kwargs["command_mps"],
            "lateral_m": case_kwargs["initial_state"][0], "yaw_deg": case_kwargs["initial_state"][1],
            "seed": case_kwargs["rollout_seed"],
        },
        "constraints": {
            "continuous_torque_limits_nm": {"leg": 50.0, "wheel": 14.0},
            "500_n": "report_only", "1200_n": "hard_failure", "roll_25_deg": "hard_failure",
            "body_wall_contact": "hard_failure", "approach_gate": args.approach_gate,
            "approach_threshold": (
                APPROACH_THRESHOLD if args.approach_gate == "estimator"
                else float(args.terrain_approach_threshold)
            ),
            "approach_dwell_steps": DWELL_STEPS, "front_phase_threshold": PHASE_THRESHOLD,
            "front_phase_dwell_steps": DWELL_STEPS, "pre_active_actions": PRE_ACTIVE.tolist(),
            "post_active_actions": POST_ACTIVE.tolist(), "trigger_prefix": "bit-identical before approach latch",
        },
        "optimizer": {"kind": "ranked diagonal CEM over 14 constant two-stage residual values", "population": args.population, "requested_iterations": args.iterations, "completed_iterations": len(iteration_reports), "workers": args.workers, "seed": args.seed},
        "baseline": baseline, "baseline_replay": baseline_replay, "iterations": iteration_reports,
        "best_search_candidate": best_summary, "best_replay": best_replay,
        "baseline_replay_match": baseline_replay_match,
        "best_replay_match": best_replay_match,
        "rear_top_found": any(bool(item[0]["rear_top_latched"]) for item in all_candidates),
        "stable_exit_found": any(bool(item[0]["success"]) for item in all_candidates),
        "official_assets_modified": False, "official_asset_hashes_before": before_assets, "official_asset_hashes_after": after_assets,
        "files": {"best_candidate": str(output / "best_candidate.npz"), "baseline_trace": str(output / "baseline_trace.npz"), "best_trace": str(output / "best_trace.npz"), "baseline_gif": str(gif_dir / "reference_zero_two_stage.gif"), "best_gif": str(gif_dir / "best_two_stage.gif")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    print(
        f"[result] best={best_replay['milestone_name']} rear_top={int(best_replay['rear_top_latched'])} success={int(best_replay['success'])} baseline_replay={int(baseline_replay_match)} best_replay={int(best_replay_match)} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
