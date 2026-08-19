#!/usr/bin/env python3
"""Targeted pose/speed PPO using the repository's 5x13 terrain-ray layout."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

import train_s10_m20_curriculum_ppo as base_runner
import train_s10_m20_lidar_curriculum_ppo as lidar_runner
import train_s10_m20_lidar_pose_speed_expansion_ppo as expansion_runner
from s10_m20_curriculum_env import ACTOR_OBS_DIM as OFFICIAL_ACTOR_OBS_DIM
from s10_m20_lidar_curriculum_env import ACTOR_OBS_DIM as HORIZONTAL_ACTOR_OBS_DIM
from s10_m20_terrain_lidar_balanced_stage_env import (
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    S10M20TerrainLidarBalancedStageEnv,
    TERRAIN_LIDAR_BEAMS,
    TERRAIN_LIDAR_CHANNELS,
    TERRAIN_LIDAR_HORIZONTAL_BEAMS,
    TERRAIN_LIDAR_HORIZONTAL_FOV_DEG,
    TERRAIN_LIDAR_VERTICAL_FOV_DEG,
)


TERRAIN_INPUT_SCALE = 1.0


class TerrainSplitActor(nn.Module):
    """Keep the 129-D matrix path bit-exact and add a zero 65-D branch."""

    def __init__(self) -> None:
        super().__init__()
        self.base_input = nn.Linear(HORIZONTAL_ACTOR_OBS_DIM, 512)
        self.terrain_input = nn.Linear(TERRAIN_LIDAR_BEAMS, 512, bias=False)
        self.activation = nn.ELU()
        self.downstream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 16),
        )
        nn.init.zeros_(self.terrain_input.weight)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = self.base_input(observation[..., :HORIZONTAL_ACTOR_OBS_DIM])
        terrain = observation[..., HORIZONTAL_ACTOR_OBS_DIM:] * TERRAIN_INPUT_SCALE
        hidden = hidden + self.terrain_input(terrain)
        return self.downstream(self.activation(hidden))


class TerrainLidarInitializedActorCritic(base_runner.M20InitializedActorCritic):
    """Expand the current 129-D policy and train only the 65-D ray branch."""

    LEGACY_TO_SPLIT = {
        "0.weight": "base_input.weight",
        "0.bias": "base_input.bias",
        "2.weight": "downstream.0.weight",
        "2.bias": "downstream.0.bias",
        "4.weight": "downstream.2.weight",
        "4.bias": "downstream.2.bias",
        "6.weight": "downstream.4.weight",
        "6.bias": "downstream.4.bias",
    }

    def __init__(self, action_std_scale: float) -> None:
        super().__init__(action_std_scale)
        self.actor = TerrainSplitActor()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.actor.terrain_input.weight.requires_grad_(True)

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
        if source_actor.keys() != self.LEGACY_TO_SPLIT.keys():
            raise ValueError("M20 actor keys do not match the terrain-lidar actor")
        split_state = self.actor.state_dict()
        for legacy_key, split_key in self.LEGACY_TO_SPLIT.items():
            source_value = source_actor[legacy_key]
            target = split_state[split_key]
            if legacy_key == "0.weight":
                if source_value.shape != (target.shape[0], OFFICIAL_ACTOR_OBS_DIM):
                    raise ValueError(f"unexpected M20 first layer shape {source_value.shape}")
                value = torch.zeros_like(target)
                value[:, :OFFICIAL_ACTOR_OBS_DIM] = source_value
                split_state[split_key] = value
            elif source_value.shape == target.shape:
                split_state[split_key] = source_value
            else:
                raise ValueError(f"M20 actor shape mismatch at {legacy_key}")
        split_state["terrain_input.weight"].zero_()
        self.actor.load_state_dict(split_state, strict=True)
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

    def _expand_state(self, state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = dict(state_dict)
        if "actor.0.weight" in state:
            current = self.state_dict()
            legacy_actor = {
                key.removeprefix("actor."): value
                for key, value in state.items()
                if key.startswith("actor.")
            }
            for key in [key for key in state if key.startswith("actor.")]:
                del state[key]
            for legacy_key, split_key in self.LEGACY_TO_SPLIT.items():
                source = legacy_actor[legacy_key]
                target = current[f"actor.{split_key}"]
                if legacy_key == "0.weight" and source.shape[1] == OFFICIAL_ACTOR_OBS_DIM:
                    expanded = torch.zeros_like(target)
                    expanded[:, :OFFICIAL_ACTOR_OBS_DIM] = source
                    source = expanded
                if source.shape != target.shape:
                    raise ValueError(
                        f"legacy split actor shape mismatch at {legacy_key}: "
                        f"{source.shape} != {target.shape}"
                    )
                state[f"actor.{split_key}"] = source
            state["actor.terrain_input.weight"] = current["actor.terrain_input.weight"]

        critic_key = "critic.0.weight"
        if critic_key in state and state[critic_key].shape[1] != CRITIC_OBS_DIM:
            source = state[critic_key]
            old_actor_dim = source.shape[1] - (base_runner.CRITIC_OBS_DIM - base_runner.ACTOR_OBS_DIM)
            if old_actor_dim not in (OFFICIAL_ACTOR_OBS_DIM, HORIZONTAL_ACTOR_OBS_DIM):
                # Current runner globals already use 194/218, so recognize the
                # two supported legacy critic widths explicitly.
                old_actor_dim = {
                    81: OFFICIAL_ACTOR_OBS_DIM,
                    153: HORIZONTAL_ACTOR_OBS_DIM,
                }.get(source.shape[1], -1)
            if old_actor_dim < 0:
                raise ValueError(f"unexpected legacy critic shape {source.shape}")
            target = self.state_dict()[critic_key]
            expanded = torch.zeros_like(target)
            expanded[:, :old_actor_dim] = source[:, :old_actor_dim]
            expanded[:, ACTOR_OBS_DIM:] = source[:, old_actor_dim:]
            state[critic_key] = expanded
        return state

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return nn.Module.load_state_dict(self, self._expand_state(state_dict), *args, **kwargs)


def configure_runner() -> None:
    # Reuse the audited depth/pose schedules and validation gates, replacing
    # only the observation extension, environment, and actor trainable scope.
    expansion_runner.S10M20LidarBalancedStageEnv = S10M20TerrainLidarBalancedStageEnv
    expansion_runner.configure_runner()
    base_runner.ACTOR_OBS_DIM = ACTOR_OBS_DIM
    base_runner.CRITIC_OBS_DIM = CRITIC_OBS_DIM
    base_runner.S10M20CurriculumEnv = S10M20TerrainLidarBalancedStageEnv
    base_runner.M20InitializedActorCritic = TerrainLidarInitializedActorCritic
    original_parse_args = base_runner.parse_args

    def parse_args() -> Any:
        args = original_parse_args()
        args.actor_trainable_scope = "new 65 terrain-lidar first-layer columns only"
        args.horizontal_lidar_beams = HORIZONTAL_ACTOR_OBS_DIM - OFFICIAL_ACTOR_OBS_DIM
        args.terrain_lidar_beams = TERRAIN_LIDAR_BEAMS
        args.terrain_lidar_channels = TERRAIN_LIDAR_CHANNELS
        args.terrain_lidar_horizontal_beams = TERRAIN_LIDAR_HORIZONTAL_BEAMS
        args.terrain_lidar_vertical_fov_deg = list(TERRAIN_LIDAR_VERTICAL_FOV_DEG)
        args.terrain_lidar_horizontal_fov_deg = list(TERRAIN_LIDAR_HORIZONTAL_FOV_DEG)
        args.terrain_lidar_source = "repository Isaac 5x13 layout; MuJoCo mj_ray implementation"
        args.terrain_input_scale = TERRAIN_INPUT_SCALE
        return args

    base_runner.parse_args = parse_args


def main() -> int:
    global TERRAIN_INPUT_SCALE
    filtered_argv: list[str] = []
    index = 1
    while index < len(sys.argv):
        token = sys.argv[index]
        if token == "--terrain-input-scale":
            if index + 1 >= len(sys.argv):
                raise ValueError("--terrain-input-scale requires a value")
            TERRAIN_INPUT_SCALE = float(sys.argv[index + 1])
            if not math.isfinite(TERRAIN_INPUT_SCALE) or TERRAIN_INPUT_SCALE <= 0.0:
                raise ValueError("--terrain-input-scale must be finite and positive")
            index += 2
            continue
        filtered_argv.append(token)
        index += 1
    sys.argv = [sys.argv[0], *filtered_argv]
    stage_args, remaining = expansion_runner.stage_runner.parse_stage_args(sys.argv[1:])
    if not math.isclose(float(stage_args.target_depth_m), 0.21, abs_tol=1.0e-12):
        raise ValueError("terrain-lidar validation is fixed to --target-depth-m 0.21")
    retention_depths_m = expansion_runner.stage_runner.unique_depths(
        list(stage_args.retention_depth_m)
    )
    if any(depth >= 0.21 for depth in retention_depths_m):
        raise ValueError("retention depths must be shallower than target")
    configure_runner()
    sys.argv = [sys.argv[0], *remaining]
    return base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
