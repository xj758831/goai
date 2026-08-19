#!/usr/bin/env python3
"""Fine-tune 0.21 m lidar capability with downstream actor layers enabled."""

from __future__ import annotations

from typing import Any

import train_s10_m20_lidar_021_capability_with_018_retention_ppo as stage_runner
import train_s10_m20_lidar_022_capability_with_018_retention_ppo as capability_runner
import train_s10_m20_lidar_curriculum_ppo as lidar_runner


class LidarDownstreamActorCritic(lidar_runner.LidarInitializedActorCritic):
    """Keep legacy input columns frozen while allowing downstream adaptation."""

    def __init__(self, action_std_scale: float) -> None:
        super().__init__(action_std_scale)
        for layer_index in (2, 4, 6):
            for parameter in self.actor[layer_index].parameters():
                parameter.requires_grad_(True)


def configure_runner() -> None:
    # The stage module sets 0.21/0.18 before this wrapper installs its gates.
    capability_runner.TARGET_DEPTH_M = stage_runner.capability_runner.TARGET_DEPTH_M
    capability_runner.RETENTION_DEPTH_M = stage_runner.capability_runner.RETENTION_DEPTH_M
    capability_runner.configure_runner()
    capability_runner.base_runner.M20InitializedActorCritic = LidarDownstreamActorCritic
    original_parse_args = capability_runner.base_runner.parse_args

    def parse_args() -> Any:
        args = original_parse_args()
        args.actor_trainable_scope = (
            "first_layer_lidar_columns_plus_downstream_actor_layers"
        )
        return args

    capability_runner.base_runner.parse_args = parse_args


def main() -> int:
    configure_runner()
    return capability_runner.base_runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
