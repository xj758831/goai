#!/usr/bin/env python3
"""Isolated v57 WP29 handoff with an upper-platform activation gate.

v56 used only XY distance to marker 1.  The lower route passes through the
same XY neighborhood before WP29, so that gate could take control too early.
This candidate keeps the v56 controller unchanged and requires the robot to
be on the upper platform before synthesizing the advisory.
"""

from __future__ import annotations

import math

from plan2_pointcloud_wp30_recovery_core import (
    MARKER_1_XY,
    WP30RecoveryConfig,
    WP30PointCloudRecovery,
)


class WP30ClosedLoopV57(WP30PointCloudRecovery):
    """v56 behavior gated by the upper-platform height."""

    UPPER_PLATFORM_MIN_Z_M = 3.50

    def __init__(self, _legacy_config=None) -> None:
        del _legacy_config
        super().__init__(
            WP30RecoveryConfig(
                marker_radius_m=0.15,
                official_wp30_radius_m=0.18,
                reverse_clear_speed=0.0,
                reverse_clear_duration_s=0.20,
                align_stable_s=0.0,
                align_tolerance_rad=math.radians(10.0),
                recenter_timeout_s=5.0,
                align_timeout_s=15.0,
                drive_timeout_s=12.0,
                total_timeout_s=35.0,
                drive_forward_limit=0.48,
                drive_lateral_limit=0.16,
            )
        )
        self.auto_activation_radius_m = 1.45

    def step(self, **kwargs):
        if not self.activated:
            position = kwargs["position_xyz"]
            distance = math.hypot(
                float(position[0]) - MARKER_1_XY[0],
                float(position[1]) - MARKER_1_XY[1],
            )
            if (
                distance <= self.auto_activation_radius_m
                and float(position[2]) >= self.UPPER_PLATFORM_MIN_Z_M
            ):
                kwargs["advisories"] = [
                    {
                        "event": "WP30_STALL_RISK",
                        "source": "v57_marker_handoff_upper_platform_gate",
                        "distance_to_marker_m": distance,
                        "base_z_m": float(position[2]),
                    }
                ]
        return super().step(**kwargs)
