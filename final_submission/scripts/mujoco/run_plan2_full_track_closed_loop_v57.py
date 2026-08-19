#!/usr/bin/env python3
"""Visible isolated full-track v57 wrapper."""

from __future__ import annotations

import run_plan2_full_track_pointcloud_recovery_candidate as base
from plan2_wp30_closed_loop_v57 import WP30ClosedLoopV57


base.WP30PointCloudRecovery = WP30ClosedLoopV57


if __name__ == "__main__":
    raise SystemExit(base.main())
