#!/usr/bin/env python3
"""Read-only 3D ray observer for a live Plan2 MuJoCo model/data pair."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import mujoco
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POINTCLOUD_SCRIPT_DIR = PROJECT_ROOT / "src" / "S10_sdk_deploy" / "scripts"
sys.path.insert(0, str(POINTCLOUD_SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "mujoco"))

from goai_follow_pointcloud_core import (  # noqa: E402
    PointCloudConfig,
    SensorExtrinsic,
    TerrainEventConfig,
    TerrainEventDebouncer,
    build_ray_directions,
    classify_terrain_events,
    extract_terrain_features,
    fuse_sensor_points,
)
from analyze_plan2_pointcloud_regions import build_region_index  # noqa: E402
from plan2_pointcloud_shadow_decision_core import (  # noqa: E402
    ShadowDecisionMonitor,
    load_decision_config,
)
from plan2_local_map_shadow import (  # noqa: E402
    RollingLocalMap,
    select_shadow_candidate,
)


LIDAR_MOUNT_OFFSET = np.asarray([0.22341, 0.0, -0.0001], dtype=np.float64)
LIDAR_GEOMGROUP = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8)


def _pointcloud_config(document: dict[str, Any]) -> PointCloudConfig:
    sensor = document.get("sensor", {})
    if not isinstance(sensor, dict):
        sensor = {}
    return PointCloudConfig(
        front=SensorExtrinsic(
            tuple(sensor.get("front_translation_m", LIDAR_MOUNT_OFFSET.tolist())),
            float(sensor.get("front_yaw_rad", 0.0)),
        ),
        rear=SensorExtrinsic(
            tuple(sensor.get("rear_translation_m", [-0.22341, 0.0, -0.0001])),
            float(sensor.get("rear_yaw_rad", math.pi)),
        ),
        min_range_m=float(sensor.get("min_range_m", 0.25)),
        max_range_m=float(sensor.get("max_range_m", 10.0)),
        roi_x_m=tuple(sensor.get("roi_x_m", [-5.0, 5.0])),
        roi_y_m=tuple(sensor.get("roi_y_m", [-5.0, 5.0])),
        roi_z_m=tuple(sensor.get("roi_z_m", [-2.0, 2.5])),
        voxel_size_m=float(sensor.get("voxel_size_m", 0.05)),
        max_points=int(sensor.get("max_points", 24000)),
        angular_bins=int(sensor.get("angular_bins", 36)),
        front_x_bins=int(sensor.get("front_x_bins", 8)),
        front_x_min_m=float(sensor.get("front_x_min_m", -0.5)),
        front_x_max_m=float(sensor.get("front_x_max_m", 2.5)),
        front_y_half_width_m=float(sensor.get("front_y_half_width_m", 0.65)),
    )


class Plan2PointCloudShadowObserver:
    """Sample terrain features without publishing or changing MuJoCo state."""

    def __init__(
        self,
        output_dir: Path,
        config_path: Path,
        *,
        channels: int = 16,
        horizontal_beams: int = 72,
        vertical_min_deg: float = -45.0,
        vertical_max_deg: float = 45.0,
        rate_hz: float = 10.0,
        region_config_path: Path | None = None,
        decision_config_path: Path | None = None,
    ) -> None:
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.config_path = config_path.resolve()
        self.region_config_path = (
            region_config_path or PROJECT_ROOT / "config" / "plan2_pointcloud_regions.yaml"
        ).resolve()
        self.decision_config_path = (
            decision_config_path
            or PROJECT_ROOT / "config" / "plan2_pointcloud_shadow_decision.yaml"
        ).resolve()
        document = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.config = _pointcloud_config(document)
        event_document = document.get("events", {})
        if not isinstance(event_document, dict):
            event_document = {}
        self.event_config = TerrainEventConfig(**event_document)
        self.debouncer = TerrainEventDebouncer(
            self.event_config.confirm_frames,
            self.event_config.release_frames,
            self.event_config.minimum_confidence,
        )
        region_document = yaml.safe_load(
            self.region_config_path.read_text(encoding="utf-8")
        ) or {}
        decision_document = yaml.safe_load(
            self.decision_config_path.read_text(encoding="utf-8")
        ) or {}
        self.decision_monitor = ShadowDecisionMonitor(
            build_region_index(region_document),
            load_decision_config(decision_document),
        )
        self.local_map = RollingLocalMap()
        self.ray_directions, self.rings = build_ray_directions(
            horizontal_beams,
            channels,
            vertical_min_deg,
            vertical_max_deg,
        )
        self.geom_ids = np.zeros(len(self.ray_directions), dtype=np.int32)
        self.distances = np.full(len(self.ray_directions), -1.0, dtype=np.float64)
        self.period_s = 1.0 / rate_hz
        self.next_sample_time = -math.inf
        self.last_sim_time: float | None = None
        self.started_wall = time.monotonic()
        self.samples = 0
        self.errors = 0
        self.raw_counts: Counter[str] = Counter()
        self.stable_counts: Counter[str] = Counter()
        self.phase_counts: Counter[str] = Counter()
        self.last_status: dict[str, Any] = {}
        self.log_path = self.output_dir / "pointcloud_features.jsonl"
        self.error_path = self.output_dir / "errors.jsonl"
        self.advisory_path = self.output_dir / "advisories.jsonl"
        self.log_file = self.log_path.open("a", encoding="utf-8", buffering=1)
        self.error_file = self.error_path.open("a", encoding="utf-8", buffering=1)
        self.advisory_file = self.advisory_path.open(
            "a", encoding="utf-8", buffering=1
        )
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "purpose": "read-only Plan2 full-track 3D point-cloud shadow",
                    "created_at": datetime.now().astimezone().isoformat(),
                    "control_authority": False,
                    "publishes_cmd_vel": False,
                    "modifies_mujoco_state": False,
                    "config_path": str(self.config_path),
                    "region_config_path": str(self.region_config_path),
                    "decision_config_path": str(self.decision_config_path),
                    "decision_monitor": "read-only terminal and JSONL advisory",
                    "local_map": (
                        "read-only rolling lidar+odometry 2.5-D map with "
                        "hypothetical candidate scoring"
                    ),
                    "rays": len(self.ray_directions),
                    "channels": channels,
                    "horizontal_beams": horizontal_beams,
                    "vertical_range_deg": [vertical_min_deg, vertical_max_deg],
                    "rate_hz_sim_time": rate_hz,
                    "sensor_model": "configurable mj_multiRay proxy; not Airy calibration",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def _sample_points(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        base_body_id: int,
    ) -> tuple[np.ndarray, int]:
        rotation = np.asarray(data.xmat[base_body_id], dtype=np.float64).reshape(3, 3)
        origin = np.asarray(data.xpos[base_body_id], dtype=np.float64) + rotation @ LIDAR_MOUNT_OFFSET
        directions_world = self.ray_directions @ rotation.T
        self.distances.fill(-1.0)
        mujoco.mj_multiRay(
            model,
            data,
            origin,
            directions_world.reshape(-1),
            LIDAR_GEOMGROUP,
            1,
            base_body_id,
            self.geom_ids,
            self.distances,
            None,
            len(self.ray_directions),
            self.config.max_range_m,
        )
        valid = (
            np.isfinite(self.distances)
            & (self.distances >= self.config.min_range_m)
            & (self.distances <= self.config.max_range_m)
        )
        points_sensor = self.ray_directions[valid] * self.distances[valid, None]
        return fuse_sensor_points(points_sensor, None, self.config), int(np.count_nonzero(valid))

    def sample(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        base_body_id: int,
        *,
        phase: str,
        detail: str,
    ) -> bool:
        sim_time = float(data.time)
        if self.last_sim_time is not None and sim_time + 1.0e-9 < self.last_sim_time:
            self.next_sample_time = sim_time
            self.debouncer.reset()
        self.last_sim_time = sim_time
        if sim_time + 1.0e-12 < self.next_sample_time:
            return False
        try:
            points, raw_points = self._sample_points(model, data, base_body_id)
            features = extract_terrain_features(points, self.config, source="plan2_mj_multiRay")
            raw_event = classify_terrain_events(points, self.event_config)
            stable_event = self.debouncer.update(raw_event)
            pose = np.asarray(data.xpos[base_body_id], dtype=np.float64)
            quaternion = np.asarray(data.xquat[base_body_id], dtype=np.float64)
            rotation = np.asarray(
                data.xmat[base_body_id], dtype=np.float64
            ).reshape(3, 3)
            yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
            points_world = points @ rotation.T + pose
            map_update = self.local_map.integrate_world(
                points_world,
                (float(pose[0]), float(pose[1]), yaw),
                sim_time,
            )
            local_map_shadow = {
                **map_update,
                **self.local_map.summary((float(pose[0]), float(pose[1]), yaw)),
            }
            status = {
                "sample": self.samples,
                "sim_time": sim_time,
                "wall_time": datetime.now().astimezone().isoformat(),
                "phase": str(phase),
                "detail": str(detail),
                "pose": pose.tolist(),
                "quaternion_wxyz": quaternion.tolist(),
                "yaw_rad": yaw,
                "raw_points": raw_points,
                "processed_points": int(len(points)),
                "features": features.to_dict(),
                "events": raw_event.to_dict(),
                "stable_event": stable_event.to_dict(),
                "local_map_shadow": local_map_shadow,
                "control_authority": False,
                "publishes_cmd_vel": False,
                "modifies_mujoco_state": False,
            }
            advisories = self.decision_monitor.update(status)
            status["shadow_advisories"] = advisories
            for advisory in advisories:
                advisory["local_map_motion_candidate"] = select_shadow_candidate(
                    local_map_shadow
                )
                self.advisory_file.write(json.dumps(advisory, sort_keys=True) + "\n")
                candidate_name = (
                    "none"
                    if advisory["local_map_motion_candidate"] is None
                    else advisory["local_map_motion_candidate"]["selected_name"]
                )
                print(
                    "[pointcloud-shadow] "
                    f"event={advisory['event']} "
                    f"sim={advisory['sim_time_s']:.3f}s "
                    f"region_elapsed={advisory['region_elapsed_s']:.3f}s "
                    f"hypothetical_candidate={candidate_name} "
                    "advisory_only=true control_authority=false",
                    flush=True,
                )
            self.log_file.write(json.dumps(status, sort_keys=True) + "\n")
            self.samples += 1
            self.raw_counts[raw_event.event] += 1
            self.stable_counts[stable_event.event] += 1
            self.phase_counts[str(phase)] += 1
            self.last_status = status
        except Exception as exc:  # Shadow diagnostics must not interrupt control.
            self.errors += 1
            self.error_file.write(
                json.dumps(
                    {
                        "sim_time": sim_time,
                        "phase": str(phase),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        self.next_sample_time = sim_time + self.period_s
        return True

    def close(self) -> None:
        if not self.log_file.closed:
            self.log_file.close()
        if not self.error_file.closed:
            self.error_file.close()
        if not self.advisory_file.closed:
            self.advisory_file.close()
        decision_summary = self.decision_monitor.summary()
        (self.output_dir / "decision_summary.json").write_text(
            json.dumps(decision_summary, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "control_authority": False,
                    "publishes_cmd_vel": False,
                    "modifies_mujoco_state": False,
                    "samples": self.samples,
                    "errors": self.errors,
                    "wall_elapsed_seconds": time.monotonic() - self.started_wall,
                    "raw_event_counts": dict(sorted(self.raw_counts.items())),
                    "stable_event_counts": dict(sorted(self.stable_counts.items())),
                    "phase_counts": dict(sorted(self.phase_counts.items())),
                    "last_status": self.last_status,
                    "decision_monitor": decision_summary,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
