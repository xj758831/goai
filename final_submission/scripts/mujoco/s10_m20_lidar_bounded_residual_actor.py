#!/usr/bin/env python3
"""Frozen S10 lidar actor with a bounded, obstacle-local residual adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

import train_s10_m20_lidar_curriculum_ppo as lidar_runner
from s10_m20_curriculum_env import ACTOR_OBS_DIM as BASE_ACTOR_OBS_DIM
from s10_m20_lidar_curriculum_env import ACTOR_OBS_DIM, CRITIC_OBS_DIM


PROXIMITY_LIDAR_START = BASE_ACTOR_OBS_DIM
PROXIMITY_LIDAR_STOP = ACTOR_OBS_DIM
GATE_START_NORMALIZED = 0.15
GATE_FULL_NORMALIZED = 0.05
RESIDUAL_ACTION_LIMIT = 0.05


class BoundedResidualActor(nn.Module):
    """Preserve the base actor and add a small correction near obstacles."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.adapter = nn.Sequential(
            nn.Linear(ACTOR_OBS_DIM, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 16),
        )
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    @staticmethod
    def obstacle_gate(observation: torch.Tensor) -> torch.Tensor:
        nearest_obstacle = observation[..., PROXIMITY_LIDAR_START:PROXIMITY_LIDAR_STOP].amin(
            dim=-1, keepdim=True
        )
        return torch.clamp(
            (GATE_START_NORMALIZED - nearest_obstacle)
            / (GATE_START_NORMALIZED - GATE_FULL_NORMALIZED),
            0.0,
            1.0,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        correction = RESIDUAL_ACTION_LIMIT * torch.tanh(self.adapter(observation))
        return self.base(observation) + self.obstacle_gate(observation) * correction


class S10LidarBoundedResidualActorCritic(lidar_runner.LidarInitializedActorCritic):
    """Load legacy checkpoints exactly and train only the residual adapter."""

    def __init__(self, action_std_scale: float) -> None:
        super().__init__(action_std_scale)
        base_actor = self.actor
        for parameter in base_actor.parameters():
            parameter.requires_grad_(False)
        self.actor = BoundedResidualActor(base_actor)

    def load_m20_actor(self, checkpoint: Any) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        source = payload.get("model_state_dict")
        if not isinstance(source, dict):
            raise ValueError("M20 checkpoint lacks model_state_dict")
        source_actor = {
            key.removeprefix("actor."): value
            for key, value in source.items()
            if key.startswith("actor.")
        }
        target_actor = self.actor.base.state_dict()
        if target_actor.keys() != source_actor.keys():
            raise ValueError("M20 actor keys do not match the residual base actor")
        expanded: dict[str, torch.Tensor] = {}
        for key, target in target_actor.items():
            source_value = source_actor[key]
            if key == "0.weight":
                if source_value.shape != (target.shape[0], BASE_ACTOR_OBS_DIM):
                    raise ValueError(f"unexpected M20 first layer shape {source_value.shape}")
                value = torch.zeros_like(target)
                value[:, :BASE_ACTOR_OBS_DIM] = source_value
                expanded[key] = value
            elif source_value.shape == target.shape:
                expanded[key] = source_value
            else:
                raise ValueError(f"M20 actor shape mismatch at {key}")
        self.actor.base.load_state_dict(expanded, strict=True)
        source_log_std = source.get("log_std")
        if not isinstance(source_log_std, torch.Tensor) or source_log_std.shape != self.log_std.shape:
            raise ValueError("M20 checkpoint lacks the expected 16-D log_std")
        with torch.no_grad():
            self.log_std.copy_(
                source_log_std
                + torch.log(
                    torch.tensor(
                        self.action_std_scale,
                        dtype=source_log_std.dtype,
                        device=source_log_std.device,
                    )
                )
            )

    def _upgrade_state(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        state = dict(state_dict)
        if "actor.0.weight" in state:
            legacy_actor = {
                key.removeprefix("actor."): value
                for key, value in state.items()
                if key.startswith("actor.")
            }
            for key in [key for key in state if key.startswith("actor.")]:
                del state[key]
            target_actor = self.actor.base.state_dict()
            for key, target in target_actor.items():
                source = legacy_actor[key]
                if key == "0.weight" and source.shape[1] == BASE_ACTOR_OBS_DIM:
                    expanded = torch.zeros_like(target)
                    expanded[:, :BASE_ACTOR_OBS_DIM] = source
                    source = expanded
                if source.shape != target.shape:
                    raise ValueError(
                        f"legacy residual base shape mismatch at {key}: "
                        f"{source.shape} != {target.shape}"
                    )
                state[f"actor.base.{key}"] = source
            current = self.state_dict()
            for key, value in current.items():
                if key.startswith("actor.adapter."):
                    state[key] = value

        critic_key = "critic.0.weight"
        if critic_key in state and state[critic_key].shape[1] != CRITIC_OBS_DIM:
            source = state[critic_key]
            if source.shape[1] != lidar_runner.BASE_CRITIC_OBS_DIM:
                raise ValueError(f"unexpected legacy critic shape {source.shape}")
            target = self.state_dict()[critic_key]
            expanded = torch.zeros_like(target)
            expanded[:, :BASE_ACTOR_OBS_DIM] = source[:, :BASE_ACTOR_OBS_DIM]
            expanded[:, ACTOR_OBS_DIM:] = source[:, BASE_ACTOR_OBS_DIM:]
            state[critic_key] = expanded
        return state

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return nn.Module.load_state_dict(self, self._upgrade_state(state_dict), *args, **kwargs)


if __name__ == "__main__":
    print("Actor library; use the bounded-residual trainer or evaluator")
