#!/usr/bin/env python3
"""Run the bounded S10 PPO pilot with a zero-initialized 72-beam lidar block.

The retained 57-D actor is expanded to 129 inputs without changing its initial
output.  Only the 72 new columns in the first actor layer are trainable; the
existing 57 columns and all downstream actor layers stay frozen.  Validation
rollback and checkpoint handling come from the audited 57-D training runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

import train_s10_m20_curriculum_ppo as base_runner
from s10_m20_curriculum_env import ACTOR_OBS_DIM as BASE_ACTOR_OBS_DIM
from s10_m20_lidar_curriculum_env import (
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    LIDAR_BEAMS,
    S10M20LidarCurriculumEnv,
    S10M20LidarCurriculumVecEnv,
)


BASE_CRITIC_OBS_DIM = base_runner.CRITIC_OBS_DIM
PRIVILEGED_OBS_DIM = BASE_CRITIC_OBS_DIM - BASE_ACTOR_OBS_DIM


class LidarInitializedActorCritic(base_runner.M20InitializedActorCritic):
    """Expand old checkpoints exactly and train only new lidar input columns."""

    def __init__(self, action_std_scale: float) -> None:
        super().__init__(action_std_scale)
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        first_weight = self.actor[0].weight
        first_weight.requires_grad_(True)
        gradient_mask = torch.zeros_like(first_weight)
        gradient_mask[:, BASE_ACTOR_OBS_DIM:] = 1.0
        first_weight.register_hook(lambda gradient: gradient * gradient_mask)

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
        target_actor = self.actor.state_dict()
        if target_actor.keys() != source_actor.keys():
            raise ValueError("M20 actor keys do not match the expanded S10 actor")
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
                raise ValueError(f"M20 actor shape mismatch at {key}: {source_value.shape} != {target.shape}")
        self.actor.load_state_dict(expanded, strict=True)
        source_log_std = source.get("log_std")
        if not isinstance(source_log_std, torch.Tensor) or source_log_std.shape != self.log_std.shape:
            raise ValueError("M20 checkpoint lacks the expected 16-D log_std")
        with torch.no_grad():
            self.log_std.copy_(source_log_std + torch.log(torch.tensor(self.action_std_scale)))
        if not torch.count_nonzero(self.actor[0].weight[:, BASE_ACTOR_OBS_DIM:]).item() == 0:
            raise RuntimeError("lidar actor columns were not initialized to exact zero")

    def _expand_legacy_state(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = dict(state_dict)
        target = super().state_dict()
        actor_key = "actor.0.weight"
        critic_key = "critic.0.weight"
        if actor_key in state and state[actor_key].shape[1] == BASE_ACTOR_OBS_DIM:
            actor = torch.zeros_like(target[actor_key])
            actor[:, :BASE_ACTOR_OBS_DIM] = state[actor_key]
            state[actor_key] = actor
        if critic_key in state and state[critic_key].shape[1] == BASE_CRITIC_OBS_DIM:
            critic = torch.zeros_like(target[critic_key])
            critic[:, :BASE_ACTOR_OBS_DIM] = state[critic_key][:, :BASE_ACTOR_OBS_DIM]
            critic[:, ACTOR_OBS_DIM:] = state[critic_key][:, BASE_ACTOR_OBS_DIM:]
            state[critic_key] = critic
        return state

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return super().load_state_dict(self._expand_legacy_state(state_dict), *args, **kwargs)


def configure_runner() -> None:
    base_runner.ACTOR_OBS_DIM = ACTOR_OBS_DIM
    base_runner.CRITIC_OBS_DIM = CRITIC_OBS_DIM
    base_runner.S10M20CurriculumEnv = S10M20LidarCurriculumEnv
    base_runner.S10M20CurriculumVecEnv = S10M20LidarCurriculumVecEnv
    base_runner.M20InitializedActorCritic = LidarInitializedActorCritic
    original_parse_args = base_runner.parse_args

    def parse_args() -> Any:
        args = original_parse_args()
        args.lidar_beams = LIDAR_BEAMS
        args.lidar_update_hz = 25.0
        args.actor_trainable_scope = "first_layer_lidar_columns_only"
        return args

    base_runner.parse_args = parse_args


def main() -> int:
    configure_runner()
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
