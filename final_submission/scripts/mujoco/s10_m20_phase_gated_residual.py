#!/usr/bin/env python3
"""Frozen S10 actor with a deployable front-support phase gate."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import train_s10_m20_curriculum_ppo as base_runner
import train_s10_m20_terrain_lidar_pose_speed_expansion_ppo as terrain_runner
from s10_m20_terrain_lidar_balanced_stage_env import (
    ACTOR_OBS_DIM as TERRAIN_ACTOR_OBS_DIM,
    S10M20TerrainLidarBalancedStageEnv,
)


HISTORY_FRAMES = 5
PHASE_OBS_DIM = TERRAIN_ACTOR_OBS_DIM + 1
PHASE_THRESHOLD = 0.50
RESIDUAL_ACTION_LIMIT = 0.05
PHASE_ESTIMATOR_CHECKPOINT = Path(
    "logs/mujoco/s10_m20_phase_estimator_19cases_20260810_v1/phase_estimator.pt"
)
BASE_ACTOR_CRITIC_CLASS = base_runner.M20InitializedActorCritic


class PhaseEstimator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_phase_estimator(checkpoint: Path) -> tuple[PhaseEstimator, torch.Tensor, torch.Tensor]:
    payload = torch.load(checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False)
    if payload.get("format") != "s10_deployable_phase_estimator_v1":
        raise ValueError("unsupported phase estimator checkpoint")
    history_frames = int(payload.get("history_frames", -1))
    actor_observation_dim = int(payload.get("actor_observation_dim", -1))
    if history_frames != HISTORY_FRAMES or actor_observation_dim != TERRAIN_ACTOR_OBS_DIM:
        raise ValueError("phase estimator dimensions do not match terrain actor")
    estimator = PhaseEstimator(TERRAIN_ACTOR_OBS_DIM * HISTORY_FRAMES)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("phase estimator lacks model state")
    estimator.load_state_dict(state, strict=True)
    estimator.eval()
    mean = payload.get("mean")
    std = payload.get("std")
    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise ValueError("phase estimator lacks normalization")
    if mean.shape != (TERRAIN_ACTOR_OBS_DIM * HISTORY_FRAMES,) or std.shape != mean.shape:
        raise ValueError("phase estimator normalization shape mismatch")
    return estimator, mean.float(), std.float().clamp_min(1.0e-3)


class S10M20PhaseGatedEnv(S10M20TerrainLidarBalancedStageEnv):
    """Terrain-lidar environment with one causal front-support probability."""

    actor_observation_dim = PHASE_OBS_DIM
    critic_observation_dim = TERRAIN_ACTOR_OBS_DIM + 24
    phase_estimator_path: Path = PHASE_ESTIMATOR_CHECKPOINT

    def __init__(self, **kwargs: Any) -> None:
        self._phase_estimator, self._phase_mean, self._phase_std = load_phase_estimator(
            self.phase_estimator_path
        )
        self._phase_history: list[np.ndarray] | None = None
        super().__init__(**kwargs)

    def _front_phase_probability(self, terrain_actor_obs: np.ndarray) -> float:
        if self._phase_history is None:
            history = [terrain_actor_obs.copy() for _ in range(HISTORY_FRAMES)]
        else:
            history = self._phase_history
        features = np.concatenate(history).astype(np.float32)
        with torch.inference_mode():
            probability = float(
                torch.sigmoid(
                    self._phase_estimator(
                        ((torch.as_tensor(features) - self._phase_mean) / self._phase_std).unsqueeze(0)
                    )
                )[0, 0]
            )
        self._phase_history = [*history[1:], terrain_actor_obs.copy()]
        return probability

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        self._phase_history = None
        actor_obs, critic_obs, info = super().reset(**kwargs)
        info["front_support_phase_probability"] = float(actor_obs[-1])
        return actor_obs, critic_obs, info

    def _observations(self, metrics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        terrain_actor_obs, critic_obs = super()._observations(metrics)
        probability = self._front_phase_probability(terrain_actor_obs)
        actor_obs = np.concatenate(
            [terrain_actor_obs, np.asarray([probability], dtype=np.float32)]
        ).astype(np.float32)
        if actor_obs.shape != (PHASE_OBS_DIM,) or not np.isfinite(actor_obs).all():
            raise FloatingPointError("invalid phase-gated actor observation")
        return actor_obs, critic_obs

    def _info(self, metrics: dict[str, Any], rewards: dict[str, float], reason: str) -> dict[str, Any]:
        info = super()._info(metrics, rewards, reason)
        info.update(
            {
                "actor_observation_dim": PHASE_OBS_DIM,
                "phase_history_frames": HISTORY_FRAMES,
                "front_support_phase_threshold": PHASE_THRESHOLD,
            }
        )
        return info


class PhaseGatedResidualActor(nn.Module):
    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adapter = nn.Sequential(
            nn.Linear(PHASE_OBS_DIM, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 16),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        base_action = self.base(observation[..., :TERRAIN_ACTOR_OBS_DIM])
        residual = RESIDUAL_ACTION_LIMIT * torch.tanh(self.adapter(observation))
        phase_gate = (observation[..., -1:] >= PHASE_THRESHOLD).to(base_action.dtype)
        return base_action + phase_gate * residual


class PhaseGatedResidualActorCritic(nn.Module):
    """Load the 129-D reference through the bit-exact 194-D terrain extension."""

    def __init__(self, action_std_scale: float) -> None:
        super().__init__()
        self.action_std_scale = float(action_std_scale)
        self.actor = PhaseGatedResidualActor(
            terrain_runner.TerrainSplitActor()
        )
        self.critic = BASE_ACTOR_CRITIC_CLASS(action_std_scale).critic
        self.log_std = nn.Parameter(torch.zeros(16))
        for parameter in self.actor.base.parameters():
            parameter.requires_grad_(False)
        self.log_std.requires_grad_(False)

    def deterministic(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor(actor_obs)

    def sample(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(actor_obs)
        distribution = torch.distributions.Normal(mean, self.log_std.exp())
        action = distribution.rsample()
        return action, distribution.log_prob(action).sum(-1), self.value(critic_obs)

    def evaluate(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(actor_obs)
        distribution = torch.distributions.Normal(mean, self.log_std.exp())
        return distribution.log_prob(action).sum(-1), distribution.entropy().sum(-1), self.value(critic_obs)

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs).squeeze(-1)

    def load_m20_actor(self, checkpoint: Any) -> None:
        temporary = terrain_runner.TerrainLidarInitializedActorCritic(self.action_std_scale)
        temporary.load_m20_actor(checkpoint)
        self.actor.base.load_state_dict(temporary.actor.state_dict(), strict=True)
        with torch.no_grad():
            self.log_std.copy_(temporary.log_std)

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if any(key.startswith("actor.base.") for key in state_dict):
            current = self.state_dict()
            direct = dict(state_dict)
            for key, value in current.items():
                if key.startswith("actor.adapter.") and key not in direct:
                    direct[key] = value
            return nn.Module.load_state_dict(self, direct, *args, **kwargs)
        temporary = terrain_runner.TerrainLidarInitializedActorCritic(self.action_std_scale)
        temporary.load_state_dict(state_dict, strict=True)
        current = self.state_dict()
        converted: dict[str, torch.Tensor] = {}
        for key, value in temporary.state_dict().items():
            if key.startswith("actor."):
                converted[f"actor.base.{key.removeprefix('actor.')}"] = value
            else:
                converted[key] = value
        for key, value in current.items():
            if key.startswith("actor.adapter."):
                converted[key] = value
        return nn.Module.load_state_dict(self, converted, *args, **kwargs)


if __name__ == "__main__":
    print("Phase-gated residual actor library; use the isolated trainer")
