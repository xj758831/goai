#!/usr/bin/env python3
"""Train the intermediate 0.21 m capability stage with 0.18 m retention."""

from __future__ import annotations

import train_s10_m20_lidar_022_capability_with_018_retention_ppo as capability_runner


# Reuse the audited capability/retention wrapper; only the target depth changes.
capability_runner.TARGET_DEPTH_M = 0.21
capability_runner.RETENTION_DEPTH_M = 0.18


if __name__ == "__main__":
    raise SystemExit(capability_runner.main())
