#!/usr/bin/env python3
"""PPO pilot for transferring the M20 actor to the official S10 MuJoCo model.

The M20 actor is copied exactly at initialization.  The actor keeps the
official deployable 57-D observation and raw 16-D action convention, while the
critic may use the environment's privileged 81-D observation.  Every accepted
checkpoint must retain all deterministic validation cases passed by the
initial M20 actor.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.distributions import Normal

from s10_m20_curriculum_env import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    S10M20CurriculumEnv,
    S10M20CurriculumVecEnv,
    VALIDATION_STATES,
)


DEFAULT_M20_CHECKPOINT = Path(
    "/home/xj/Downloads/rl_training-main(1)/rl_training-main/logs/rsl_rl/"
    "deeprobotics_m20_mixed_pit_recovery_fine/"
    "2026-08-09_09-11-45_from_model2100_lr1e5_25/model_2110.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m20-checkpoint", type=Path, default=DEFAULT_M20_CHECKPOINT)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--reset-optimizer-on-resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-m", type=float, default=0.08)
    parser.add_argument("--command-mps", type=float, default=0.8)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5, help="actor learning rate")
    parser.add_argument("--critic-learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--action-std-scale",
        type=float,
        default=0.5,
        help="scale applied to the M20 checkpoint's per-action exploration standard deviations",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.10)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-post-update-kl", type=float, default=0.02)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--min-baseline-successes", type=int, default=4)
    parser.add_argument("--anchor-successful-baseline-actions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anchor-coef", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=377)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--parallel-env-steps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-validation-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--render-final-gifs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gif-frame-period", type=float, default=0.10)
    parser.add_argument(
        "--initial-state",
        action="append",
        default=None,
        metavar="LATERAL_M,YAW_DEG",
        help="training reset state; repeat an entry to give it more sampling weight",
    )
    return parser.parse_args()


def parse_training_states(values: list[str] | None) -> tuple[tuple[float, float], ...] | None:
    if values is None:
        return None
    states: list[tuple[float, float]] = []
    for value in values:
        fields = value.split(",")
        if len(fields) != 2:
            raise ValueError(f"initial state must be LATERAL_M,YAW_DEG, got {value!r}")
        state = (float(fields[0]), float(fields[1]))
        if abs(state[0]) > 0.05 or abs(state[1]) > 5.0:
            raise ValueError(f"initial state exceeds bounded curriculum range: {state}")
        states.append(state)
    if not states:
        raise ValueError("at least one initial state is required")
    return tuple(states)


def layer_init(layer: nn.Linear, std: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.zeros_(layer.bias)
    return layer


class M20InitializedActorCritic(nn.Module):
    def __init__(self, action_std_scale: float) -> None:
        super().__init__()
        self.action_std_scale = float(action_std_scale)
        self.actor = nn.Sequential(
            nn.Linear(ACTOR_OBS_DIM, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, ACTION_DIM),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(CRITIC_OBS_DIM, 512)),
            nn.ELU(),
            layer_init(nn.Linear(512, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))

    def load_m20_actor(self, checkpoint: Path) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = payload.get("model_state_dict")
        if not isinstance(source, dict):
            raise ValueError("M20 checkpoint lacks model_state_dict")
        expected = self.actor.state_dict()
        source_actor = {
            key.removeprefix("actor."): value
            for key, value in source.items()
            if key.startswith("actor.")
        }
        if expected.keys() != source_actor.keys():
            raise ValueError(
                f"M20 actor keys do not match S10 actor: expected={list(expected)}, got={list(source_actor)}"
            )
        for key, target in expected.items():
            if target.shape != source_actor[key].shape:
                raise ValueError(f"M20 actor shape mismatch at {key}: {source_actor[key].shape} != {target.shape}")
        self.actor.load_state_dict(source_actor, strict=True)
        source_log_std = source.get("log_std")
        if not isinstance(source_log_std, torch.Tensor) or source_log_std.shape != self.log_std.shape:
            raise ValueError("M20 checkpoint lacks the expected 16-D log_std")
        with torch.no_grad():
            self.log_std.copy_(source_log_std + math.log(self.action_std_scale))
        for key, target in self.actor.state_dict().items():
            if not torch.equal(target.cpu(), source_actor[key].cpu()):
                raise RuntimeError(f"M20 actor copy verification failed at {key}")

    def distribution(self, actor_obs: torch.Tensor) -> Normal:
        mean = self.actor(actor_obs)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)

    def sample(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(actor_obs)
        action = distribution.sample()
        return action, distribution.log_prob(action).sum(-1), self.value(critic_obs)

    def evaluate(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(actor_obs)
        return distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1), self.value(critic_obs)

    def deterministic(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor(actor_obs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def state_name(state: tuple[float, float]) -> str:
    return f"lateral_{state[0]:+.3f}_yaw_{state[1]:+.1f}"


class GifRecorder:
    def __init__(self, model: mujoco.MjModel, period_s: float) -> None:
        self.period_s = period_s
        self.next_time = 0.0
        self.frames: list[Image.Image] = []
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 3.2
        self.camera.azimuth = 130.0
        self.camera.elevation = -18.0

    def capture(self, env: S10M20CurriculumEnv) -> None:
        if env.data.time + 1.0e-12 < self.next_time:
            return
        self.next_time = float(env.data.time) + self.period_s
        self.camera.lookat[:] = env.data.xpos[env.ids["base_body_id"]]
        self.camera.lookat[2] += 0.12
        self.renderer.update_scene(env.data, camera=self.camera)
        self.frames.append(Image.fromarray(self.renderer.render()))

    def close(self, path: Path) -> str:
        self.renderer.close()
        if not self.frames:
            raise RuntimeError("GIF recorder captured no frames")
        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=max(1, int(round(self.period_s * 1000.0))),
            loop=0,
        )
        return str(path)


def evaluate_policy(
    agent: M20InitializedActorCritic,
    *,
    depth_m: float,
    command_mps: float,
    device: torch.device,
    validation_states: tuple[tuple[float, float], ...] = VALIDATION_STATES,
    gif_dir: Path | None = None,
    gif_frame_period: float = 0.10,
) -> dict[str, Any]:
    was_training = agent.training
    agent.eval()
    cases: list[dict[str, Any]] = []
    if gif_dir is not None:
        gif_dir.mkdir(parents=True, exist_ok=False)
    try:
        for index, state in enumerate(validation_states):
            env = S10M20CurriculumEnv(depth_m=depth_m, command_mps=command_mps, training_states=(state,))
            recorder = GifRecorder(env.model, gif_frame_period) if gif_dir is not None else None
            episode_return = 0.0
            try:
                actor_obs, _, _ = env.reset(seed=10_000 + index, initial_state=state)
                if recorder is not None:
                    recorder.capture(env)
                final_info: dict[str, Any] | None = None
                for _ in range(env.max_episode_steps):
                    with torch.inference_mode():
                        actor_tensor = torch.as_tensor(actor_obs, dtype=torch.float32, device=device).unsqueeze(0)
                        action = agent.deterministic(actor_tensor).squeeze(0).cpu().numpy().astype(np.float32)
                    actor_obs, _, reward, terminated, truncated, final_info = env.step(action)
                    episode_return += reward
                    if recorder is not None:
                        recorder.capture(env)
                    if terminated or truncated:
                        break
                if final_info is None:
                    raise RuntimeError("validation rollout produced no transition")
                metrics = final_info["metrics"]
                gif_path = None
                if recorder is not None:
                    gif_path = recorder.close(gif_dir / f"{index}_{state_name(state)}.gif")
                    recorder = None
                cases.append(
                    {
                        "state": {"lateral_m": state[0], "yaw_deg": state[1]},
                        "safe_success": bool(final_info["success"]),
                        "termination_reason": final_info["termination_reason"],
                        "episode_return": episode_return,
                        "episode_steps": int(final_info["episode_step"]),
                        "time_s": float(final_info["time_s"]),
                        "front_top_latched": bool(metrics["front_top_latched"]),
                        "rear_top_latched": bool(metrics["rear_top_latched"]),
                        "base_cross_latched": bool(metrics["base_cross_latched"]),
                        "max_force_n": float(metrics["episode_max_force_n"]),
                        "max_abs_roll_deg": math.degrees(float(metrics["episode_max_roll_rad"])),
                        "max_abs_pitch_deg": math.degrees(float(metrics["episode_max_pitch_rad"])),
                        "max_action_abs": float(np.max(np.abs(action))),
                        "gif": gif_path,
                    }
                )
            finally:
                if recorder is not None:
                    recorder.renderer.close()
                env.close()
    finally:
        agent.train(was_training)
    success_indexes = [index for index, case in enumerate(cases) if case["safe_success"]]
    return {
        "depth_m": depth_m,
        "cases": cases,
        "safe_success_count": len(success_indexes),
        "safe_success_rate": len(success_indexes) / len(cases),
        "safe_success_indexes": success_indexes,
        "mean_return": mean([float(case["episode_return"]) for case in cases]),
        "max_force_n": max(float(case["max_force_n"]) for case in cases),
    }


def validation_retained(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return set(baseline["safe_success_indexes"]).issubset(candidate["safe_success_indexes"])


def validation_score(report: dict[str, Any]) -> tuple[int, int, float, float]:
    plus_two_cm_success = int(bool(report["cases"][1]["safe_success"]))
    return (
        int(report["safe_success_count"]),
        plus_two_cm_success,
        float(report["mean_return"]),
        -float(report["max_force_n"]),
    )


def collect_successful_baseline_anchors(
    agent: M20InitializedActorCritic,
    *,
    depth_m: float,
    command_mps: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    was_training = agent.training
    agent.eval()
    accepted_observations: list[np.ndarray] = []
    accepted_actions: list[np.ndarray] = []
    accepted_cases: list[int] = []
    try:
        for index, state in enumerate(VALIDATION_STATES):
            env = S10M20CurriculumEnv(depth_m=depth_m, command_mps=command_mps, training_states=(state,))
            observations: list[np.ndarray] = []
            actions: list[np.ndarray] = []
            try:
                actor_obs, _, _ = env.reset(seed=20_000 + index, initial_state=state)
                final_info: dict[str, Any] | None = None
                for _ in range(env.max_episode_steps):
                    with torch.inference_mode():
                        actor_tensor = torch.as_tensor(actor_obs, dtype=torch.float32, device=device).unsqueeze(0)
                        action = agent.deterministic(actor_tensor).squeeze(0).cpu().numpy().astype(np.float32)
                    observations.append(actor_obs.copy())
                    actions.append(action.copy())
                    actor_obs, _, _, terminated, truncated, final_info = env.step(action)
                    if terminated or truncated:
                        break
                if final_info is not None and final_info["success"]:
                    accepted_observations.extend(observations)
                    accepted_actions.extend(actions)
                    accepted_cases.append(index)
            finally:
                env.close()
    finally:
        agent.train(was_training)
    if not accepted_observations:
        raise RuntimeError("baseline policy produced no successful anchor trajectories")
    return (
        np.stack(accepted_observations).astype(np.float32),
        np.stack(accepted_actions).astype(np.float32),
        accepted_cases,
    )


def save_checkpoint(
    path: Path,
    *,
    agent: M20InitializedActorCritic,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    args: argparse.Namespace,
    source_checkpoint: Path,
    resume_checkpoint: Path | None,
    baseline: dict[str, Any],
) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model": agent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "source_m20_checkpoint": str(source_checkpoint),
            "source_m20_checkpoint_sha256": sha256(source_checkpoint),
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "resume_checkpoint_sha256": sha256(resume_checkpoint) if resume_checkpoint is not None else None,
            "actor_observation_dim": ACTOR_OBS_DIM,
            "critic_observation_dim": CRITIC_OBS_DIM,
            "action_dim": ACTION_DIM,
            "actor_action_semantics": "raw official S10 action; no tanh",
            "baseline_validation": baseline,
            "official_assets_modified": False,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    counts = (
        args.num_envs,
        args.max_iterations,
        args.num_steps,
        args.update_epochs,
        args.minibatch_size,
        args.eval_interval,
    )
    if min(counts) <= 0:
        raise ValueError("all PPO counts must be positive")
    batch_size = args.num_envs * args.num_steps
    if args.minibatch_size > batch_size or batch_size % args.minibatch_size:
        raise ValueError("minibatch-size must divide num-envs * num-steps")
    if not 0.0 < args.action_std_scale <= 2.0:
        raise ValueError("action-std-scale must be in (0, 2]")
    if not 0 <= args.min_baseline_successes <= len(VALIDATION_STATES):
        raise ValueError("min-baseline-successes must be between 0 and 5")
    if not math.isfinite(args.anchor_coef) or args.anchor_coef < 0.0:
        raise ValueError("anchor-coef must be finite and non-negative")
    if args.learning_rate <= 0.0 or args.critic_learning_rate <= 0.0 or args.gif_frame_period <= 0.0:
        raise ValueError("learning rates and gif-frame-period must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    source_checkpoint = args.m20_checkpoint.expanduser().resolve()
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    resume_checkpoint = args.resume_checkpoint.expanduser().resolve() if args.resume_checkpoint is not None else None
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise FileNotFoundError(resume_checkpoint)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    training_states = parse_training_states(args.initial_state)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    agent = M20InitializedActorCritic(args.action_std_scale).to(device)
    agent.load_m20_actor(source_checkpoint)
    optimizer = torch.optim.Adam(
        [
            {"params": list(agent.actor.parameters()) + [agent.log_std], "lr": args.learning_rate},
            {"params": agent.critic.parameters(), "lr": args.critic_learning_rate},
        ],
        eps=1.0e-5,
    )
    if resume_checkpoint is not None:
        resume_payload = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        resume_model = resume_payload.get("model")
        resume_optimizer = resume_payload.get("optimizer")
        if not isinstance(resume_model, dict) or not isinstance(resume_optimizer, dict):
            raise ValueError("resume checkpoint lacks model/optimizer state")
        agent.load_state_dict(resume_model, strict=True)
        if not args.reset_optimizer_on_resume:
            optimizer.load_state_dict(resume_optimizer)
        optimizer.param_groups[0]["lr"] = args.learning_rate
        optimizer.param_groups[1]["lr"] = args.critic_learning_rate

    evaluation_started = time.perf_counter()
    baseline = evaluate_policy(agent, depth_m=args.depth_m, command_mps=args.command_mps, device=device)
    baseline_evaluation_seconds = time.perf_counter() - evaluation_started
    print(
        f"[baseline] safe={baseline['safe_success_count']}/5 "
        f"mean_return={baseline['mean_return']:+.3f} elapsed={baseline_evaluation_seconds:.2f}s",
        flush=True,
    )
    if baseline["safe_success_count"] < args.min_baseline_successes:
        raise RuntimeError(
            f"initial policy reproduced only {baseline['safe_success_count']}/5; "
            f"required {args.min_baseline_successes}/5"
        )

    initial_checkpoint = output_dir / "checkpoint_initial.pt"
    save_checkpoint(
        initial_checkpoint,
        agent=agent,
        optimizer=optimizer,
        iteration=0,
        args=args,
        source_checkpoint=source_checkpoint,
        resume_checkpoint=resume_checkpoint,
        baseline=baseline,
    )
    best_model = copy.deepcopy(agent.state_dict())
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_iteration = 0
    best_evaluation = baseline
    best_score = validation_score(baseline)

    anchor_actor_obs: torch.Tensor | None = None
    anchor_actions: torch.Tensor | None = None
    anchor_case_indexes: list[int] = []
    if args.anchor_successful_baseline_actions:
        anchor_obs_np, anchor_actions_np, anchor_case_indexes = collect_successful_baseline_anchors(
            agent,
            depth_m=args.depth_m,
            command_mps=args.command_mps,
            device=device,
        )
        anchor_actor_obs = torch.as_tensor(anchor_obs_np, dtype=torch.float32, device=device)
        anchor_actions = torch.as_tensor(anchor_actions_np, dtype=torch.float32, device=device)
        print(
            f"[anchor] cases={anchor_case_indexes} transitions={anchor_actor_obs.shape[0]} "
            f"coef={args.anchor_coef:g}",
            flush=True,
        )

    env_kwargs: dict[str, Any] = {"depth_m": args.depth_m, "command_mps": args.command_mps}
    if training_states is not None:
        env_kwargs["training_states"] = training_states
    envs = S10M20CurriculumVecEnv(args.num_envs, **env_kwargs)
    executor = ThreadPoolExecutor(max_workers=args.num_envs) if args.parallel_env_steps and args.num_envs > 1 else None
    actor_obs_np, critic_obs_np, reset_infos = envs.reset()
    episode_return = np.zeros(args.num_envs, dtype=np.float64)
    episode_length = np.zeros(args.num_envs, dtype=np.int32)
    total_resets = Counter(state_name((info["initial_state"]["lateral_m"], info["initial_state"]["yaw_deg"])) for info in reset_infos)
    history: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = [{"iteration": 0, "retained": True, "report": baseline}]
    candidate_checkpoints: list[str] = []
    training_started = time.perf_counter()

    try:
        for iteration in range(1, args.max_iterations + 1):
            iteration_started = time.perf_counter()
            actor_store = torch.empty((args.num_steps, args.num_envs, ACTOR_OBS_DIM), device=device)
            critic_store = torch.empty((args.num_steps, args.num_envs, CRITIC_OBS_DIM), device=device)
            action_store = torch.empty((args.num_steps, args.num_envs, ACTION_DIM), device=device)
            logprob_store = torch.empty((args.num_steps, args.num_envs), device=device)
            reward_store = torch.empty((args.num_steps, args.num_envs), device=device)
            done_store = torch.empty((args.num_steps, args.num_envs), device=device)
            value_store = torch.empty((args.num_steps, args.num_envs), device=device)
            completed_returns: list[float] = []
            completed_successes: list[float] = []
            terminations: Counter[str] = Counter()
            step_rewards: list[float] = []

            rollout_started = time.perf_counter()
            for step_index in range(args.num_steps):
                actor_tensor = torch.as_tensor(actor_obs_np, dtype=torch.float32, device=device)
                critic_tensor = torch.as_tensor(critic_obs_np, dtype=torch.float32, device=device)
                with torch.no_grad():
                    actions, logprobs, values = agent.sample(actor_tensor, critic_tensor)
                actions_np = actions.cpu().numpy().astype(np.float32)
                if executor is None:
                    transitions = [env.step(action) for env, action in zip(envs.envs, actions_np)]
                else:
                    transitions = list(executor.map(lambda pair: pair[0].step(pair[1]), zip(envs.envs, actions_np)))

                next_actor = np.empty_like(actor_obs_np)
                next_critic = np.empty_like(critic_obs_np)
                rewards_np = np.empty(args.num_envs, dtype=np.float32)
                dones_np = np.zeros(args.num_envs, dtype=np.float32)
                for env_index, transition in enumerate(transitions):
                    actor_next, critic_next, reward, terminated, truncated, info = transition
                    if not np.isfinite(actor_next).all() or not np.isfinite(critic_next).all() or not math.isfinite(reward):
                        raise FloatingPointError(f"non-finite transition in env {env_index}")
                    episode_return[env_index] += reward
                    episode_length[env_index] += 1
                    step_rewards.append(float(reward))
                    if terminated or truncated:
                        reason = info["termination_reason"] or "time_limit"
                        terminations[reason] += 1
                        completed_returns.append(float(episode_return[env_index]))
                        completed_successes.append(float(info["success"]))
                        episode_return[env_index] = 0.0
                        episode_length[env_index] = 0
                        actor_next, critic_next, reset_info = envs.reset_at(env_index)
                        state = reset_info["initial_state"]
                        total_resets[state_name((state["lateral_m"], state["yaw_deg"]))] += 1
                        dones_np[env_index] = 1.0
                    next_actor[env_index] = actor_next
                    next_critic[env_index] = critic_next
                    rewards_np[env_index] = reward

                actor_store[step_index] = actor_tensor
                critic_store[step_index] = critic_tensor
                action_store[step_index] = actions
                logprob_store[step_index] = logprobs
                reward_store[step_index] = torch.as_tensor(rewards_np, dtype=torch.float32, device=device)
                done_store[step_index] = torch.as_tensor(dones_np, dtype=torch.float32, device=device)
                value_store[step_index] = values
                actor_obs_np, critic_obs_np = next_actor, next_critic
            rollout_seconds = time.perf_counter() - rollout_started

            with torch.no_grad():
                next_value = agent.value(torch.as_tensor(critic_obs_np, dtype=torch.float32, device=device))
                advantages = torch.zeros_like(reward_store)
                gae = torch.zeros(args.num_envs, device=device)
                for step_index in reversed(range(args.num_steps)):
                    next_nonterminal = 1.0 - done_store[step_index]
                    future_value = next_value if step_index == args.num_steps - 1 else value_store[step_index + 1]
                    delta = reward_store[step_index] + args.gamma * future_value * next_nonterminal - value_store[step_index]
                    gae = delta + args.gamma * args.gae_lambda * next_nonterminal * gae
                    advantages[step_index] = gae
                return_targets = advantages + value_store

            flat_actor = actor_store.reshape(-1, ACTOR_OBS_DIM)
            flat_critic = critic_store.reshape(-1, CRITIC_OBS_DIM)
            flat_actions = action_store.reshape(-1, ACTION_DIM)
            flat_logprobs = logprob_store.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = return_targets.reshape(-1)
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std(unbiased=False) + 1.0e-8)
            indexes = np.arange(batch_size)
            losses: dict[str, list[float]] = defaultdict(list)
            for _ in range(args.update_epochs):
                np.random.shuffle(indexes)
                for start in range(0, batch_size, args.minibatch_size):
                    batch = indexes[start : start + args.minibatch_size]
                    new_logprob, entropy, value = agent.evaluate(flat_actor[batch], flat_critic[batch], flat_actions[batch])
                    log_ratio = new_logprob - flat_logprobs[batch]
                    ratio = log_ratio.exp()
                    policy_loss = torch.maximum(
                        -flat_advantages[batch] * ratio,
                        -flat_advantages[batch] * ratio.clamp(1.0 - args.clip_coef, 1.0 + args.clip_coef),
                    ).mean()
                    value_loss = 0.5 * (value - flat_returns[batch]).square().mean()
                    entropy_bonus = entropy.mean()
                    anchor_loss = torch.zeros((), device=device)
                    if anchor_actor_obs is not None and anchor_actions is not None:
                        anchor_indexes = torch.randint(
                            anchor_actor_obs.shape[0], (len(batch),), device=device
                        )
                        anchor_predictions = agent.actor(anchor_actor_obs[anchor_indexes])
                        anchor_loss = (anchor_predictions - anchor_actions[anchor_indexes]).square().mean()
                    loss = (
                        policy_loss
                        + args.value_coef * value_loss
                        - args.entropy_coef * entropy_bonus
                        + args.anchor_coef * anchor_loss
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    approx_kl = ((-log_ratio).exp() - 1.0 + log_ratio).mean()
                    losses["policy"].append(float(policy_loss.detach()))
                    losses["value"].append(float(value_loss.detach()))
                    losses["entropy"].append(float(entropy_bonus.detach()))
                    losses["approx_kl"].append(float(approx_kl.detach()))
                    losses["grad_norm"].append(float(grad_norm.detach()))
                    losses["anchor"].append(float(anchor_loss.detach()))

            with torch.no_grad():
                post_logprob, _, _ = agent.evaluate(flat_actor, flat_critic, flat_actions)
                post_log_ratio = post_logprob - flat_logprobs
                post_update_kl = float((((-post_log_ratio).exp() - 1.0 + post_log_ratio).mean()).detach())
            kl_rollback = post_update_kl > args.max_post_update_kl
            if kl_rollback:
                agent.load_state_dict(best_model)
                optimizer.load_state_dict(best_optimizer)

            iteration_seconds = time.perf_counter() - iteration_started
            row: dict[str, Any] = {
                "iteration": iteration,
                "iteration_seconds": iteration_seconds,
                "rollout_seconds": rollout_seconds,
                "transitions_per_second": batch_size / rollout_seconds,
                "mean_step_reward": mean(step_rewards),
                "completed_episodes": len(completed_returns),
                "mean_episode_return": mean(completed_returns),
                "rollout_success_rate": mean(completed_successes),
                "terminations": dict(terminations),
                "losses": {name: mean(values) for name, values in losses.items()},
                "post_update_kl": post_update_kl,
                "kl_rollback": kl_rollback,
                "validation_retained": None,
                "validation_safe_successes": None,
            }
            should_evaluate = iteration % args.eval_interval == 0 or iteration == args.max_iterations
            if should_evaluate:
                candidate = evaluate_policy(agent, depth_m=args.depth_m, command_mps=args.command_mps, device=device)
                retained = validation_retained(baseline, candidate)
                row["validation_retained"] = retained
                row["validation_safe_successes"] = candidate["safe_success_count"]
                evaluations.append({"iteration": iteration, "retained": retained, "report": candidate})
                if args.save_validation_candidates:
                    candidate_path = output_dir / f"checkpoint_candidate_iter_{iteration:04d}.pt"
                    save_checkpoint(
                        candidate_path,
                        agent=agent,
                        optimizer=optimizer,
                        iteration=iteration,
                        args=args,
                        source_checkpoint=source_checkpoint,
                        resume_checkpoint=resume_checkpoint,
                        baseline=baseline,
                    )
                    candidate_checkpoints.append(str(candidate_path))
                if retained and validation_score(candidate) > best_score:
                    best_model = copy.deepcopy(agent.state_dict())
                    best_optimizer = copy.deepcopy(optimizer.state_dict())
                    best_iteration = iteration
                    best_evaluation = candidate
                    best_score = validation_score(candidate)
                elif not retained:
                    agent.load_state_dict(best_model)
                    optimizer.load_state_dict(best_optimizer)
                    row["validation_rollback"] = True
            history.append(row)
            print(
                f"[train] iter={iteration}/{args.max_iterations} reward={row['mean_step_reward']:+.4f} "
                f"episodes={row['completed_episodes']} sps={row['transitions_per_second']:.1f} "
                f"kl={post_update_kl:.6f} valid={row['validation_safe_successes']}",
                flush=True,
            )

        training_seconds = time.perf_counter() - training_started
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
        envs.close()

    agent.load_state_dict(best_model)
    optimizer.load_state_dict(best_optimizer)
    gif_dir = output_dir / "gifs" if args.render_final_gifs else None
    final_evaluation = evaluate_policy(
        agent,
        depth_m=args.depth_m,
        command_mps=args.command_mps,
        device=device,
        gif_dir=gif_dir,
        gif_frame_period=args.gif_frame_period,
    )
    final_retained = validation_retained(baseline, final_evaluation)
    final_checkpoint = output_dir / f"checkpoint_best_iter_{best_iteration:04d}.pt"
    save_checkpoint(
        final_checkpoint,
        agent=agent,
        optimizer=optimizer,
        iteration=best_iteration,
        args=args,
        source_checkpoint=source_checkpoint,
        resume_checkpoint=resume_checkpoint,
        baseline=baseline,
    )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="ascii") as handle:
        fieldnames = [
            "iteration", "iteration_seconds", "rollout_seconds", "transitions_per_second",
            "mean_step_reward", "completed_episodes", "mean_episode_return", "rollout_success_rate",
            "post_update_kl", "kl_rollback", "validation_retained", "validation_safe_successes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)

    mean_iteration_seconds = mean([float(row["iteration_seconds"]) for row in history])
    mean_rollout_sps = mean([float(row["transitions_per_second"]) for row in history])
    summary = {
        "purpose": f"bounded S10 {args.depth_m:.6f} m M20-initialized full-body PPO pilot",
        "depth_m": args.depth_m,
        "source_m20_checkpoint": str(source_checkpoint),
        "source_m20_checkpoint_sha256": sha256(source_checkpoint),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resume_checkpoint_sha256": sha256(resume_checkpoint) if resume_checkpoint is not None else None,
        "official_assets_modified": False,
        "actor_observation_dim": ACTOR_OBS_DIM,
        "critic_observation_dim": CRITIC_OBS_DIM,
        "action_dim": ACTION_DIM,
        "actor_action_semantics": "raw official S10 action; no tanh",
        "parallel_env_steps": bool(executor is not None),
        "baseline_evaluation_seconds": baseline_evaluation_seconds,
        "training_seconds": training_seconds,
        "mean_iteration_seconds": mean_iteration_seconds,
        "mean_rollout_transitions_per_second": mean_rollout_sps,
        "baseline_evaluation": baseline,
        "evaluations": evaluations,
        "best_iteration": best_iteration,
        "best_evaluation": best_evaluation,
        "final_evaluation": final_evaluation,
        "final_validation_retained": final_retained,
        "target_plus_2cm_safe_success": bool(final_evaluation["cases"][1]["safe_success"]),
        "training_states": training_states,
        "anchor_successful_baseline_actions": args.anchor_successful_baseline_actions,
        "anchor_case_indexes": anchor_case_indexes,
        "anchor_transition_count": int(anchor_actor_obs.shape[0]) if anchor_actor_obs is not None else 0,
        "candidate_checkpoints": candidate_checkpoints,
        "total_resets": dict(total_resets),
        "history": history,
        "files": {
            "initial_checkpoint": str(initial_checkpoint),
            "best_checkpoint": str(final_checkpoint),
            "metrics": str(output_dir / "metrics.csv"),
            "gifs": str(gif_dir) if gif_dir is not None else None,
            "summary": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    return 0 if final_retained else 2


if __name__ == "__main__":
    raise SystemExit(main())
