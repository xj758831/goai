#!/usr/bin/env python3
"""Dependency-light S10 Airy point-cloud adaptation.

The Lightning-LM project provides a useful ROS ``PointCloud2`` contract for
RoboSense sensors, but its M20 extrinsics and 32-line assumptions are not
valid for S10.  This module therefore keeps the reusable parts (xyz fields,
finite-point filtering and deterministic decimation) and owns the S10 frame
conversion and terrain feature definition locally.

This is perception-only.  It does not produce a velocity command.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Optional

import numpy as np


def build_ray_directions(
    horizontal_beams: int,
    channels: int,
    vertical_min_deg: float,
    vertical_max_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sensor-frame unit rays and their vertical channel ids."""

    if horizontal_beams < 4 or channels < 1:
        raise ValueError("horizontal_beams must be >=4 and channels must be positive")
    if not -90.0 < vertical_min_deg <= vertical_max_deg < 90.0:
        raise ValueError("vertical angles must be ordered and inside (-90, 90) degrees")
    horizontal = np.linspace(-math.pi, math.pi, horizontal_beams, endpoint=False)
    vertical = np.radians(np.linspace(vertical_min_deg, vertical_max_deg, channels))
    directions = []
    rings = []
    for ring, elevation in enumerate(vertical):
        cos_elevation = math.cos(float(elevation))
        directions.extend(
            [
                cos_elevation * np.cos(horizontal),
                cos_elevation * np.sin(horizontal),
                np.full(horizontal_beams, math.sin(float(elevation))),
            ]
        )
        rings.extend([ring] * horizontal_beams)
    direction_rows = np.asarray(directions, dtype=np.float64).reshape(
        channels, 3, horizontal_beams
    )
    direction_rows = np.transpose(direction_rows, (0, 2, 1)).reshape(-1, 3)
    return direction_rows, np.asarray(rings, dtype=np.uint16)


def _tuple3(value: Iterable[float], name: str) -> tuple[float, float, float]:
    result = tuple(float(item) for item in value)
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain three finite numbers")
    return result


@dataclass(frozen=True)
class SensorExtrinsic:
    """Planar S10 lidar-to-base transform.

    The current URDF places both Airy units at approximately z=0 and rotates
    the rear unit by pi around z.  A planar yaw is intentional here: the
    sensor mounting roll/pitch must be measured before adding them.
    """

    translation: tuple[float, float, float]
    yaw_rad: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation", _tuple3(self.translation, "translation"))
        if not math.isfinite(self.yaw_rad):
            raise ValueError("yaw_rad must be finite")


@dataclass(frozen=True)
class PointCloudConfig:
    front: SensorExtrinsic = SensorExtrinsic((0.22341, 0.0, -0.0001), 0.0)
    rear: SensorExtrinsic = SensorExtrinsic((-0.22341, 0.0, -0.0001), math.pi)
    min_range_m: float = 0.25
    max_range_m: float = 30.0
    roi_x_m: tuple[float, float] = (-5.0, 5.0)
    roi_y_m: tuple[float, float] = (-5.0, 5.0)
    roi_z_m: tuple[float, float] = (-2.0, 2.5)
    voxel_size_m: float = 0.08
    max_points: int = 24000
    angular_bins: int = 36
    front_x_bins: int = 8
    front_x_min_m: float = -0.5
    front_x_max_m: float = 2.5
    front_y_half_width_m: float = 0.65

    def __post_init__(self) -> None:
        if self.min_range_m <= 0.0 or self.max_range_m <= self.min_range_m:
            raise ValueError("invalid range limits")
        for name in ("roi_x_m", "roi_y_m", "roi_z_m"):
            bounds = tuple(float(item) for item in getattr(self, name))
            if len(bounds) != 2 or bounds[1] <= bounds[0]:
                raise ValueError(f"{name} must be an increasing pair")
            object.__setattr__(self, name, bounds)
        if self.voxel_size_m <= 0.0 or self.max_points <= 0:
            raise ValueError("voxel_size_m and max_points must be positive")
        if self.angular_bins < 4 or self.front_x_bins < 1:
            raise ValueError("feature bin counts are too small")


@dataclass(frozen=True)
class TerrainEventConfig:
    """Conservative thresholds for shadow-only terrain labels."""

    analysis_x_m: tuple[float, float] = (0.25, 1.8)
    baseline_x_m: tuple[float, float] = (0.25, 0.55)
    y_half_width_m: float = 0.65
    minimum_support_points: int = 8
    step_height_threshold_m: float = 0.12
    drop_depth_threshold_m: float = 0.15
    wall_height_threshold_m: float = 0.35
    wall_min_width_m: float = 0.45
    compact_max_width_m: float = 0.35
    compact_height_threshold_m: float = 0.20
    confirm_frames: int = 3
    release_frames: int = 2
    minimum_confidence: float = 0.25

    def __post_init__(self) -> None:
        for name in ("analysis_x_m", "baseline_x_m"):
            bounds = tuple(float(item) for item in getattr(self, name))
            if len(bounds) != 2 or bounds[1] <= bounds[0]:
                raise ValueError(f"{name} must be an increasing pair")
            object.__setattr__(self, name, bounds)
        if self.y_half_width_m <= 0.0 or self.minimum_support_points < 1:
            raise ValueError("invalid terrain event support limits")
        for name in (
            "step_height_threshold_m",
            "drop_depth_threshold_m",
            "wall_height_threshold_m",
            "wall_min_width_m",
            "compact_max_width_m",
            "compact_height_threshold_m",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.confirm_frames < 1 or self.release_frames < 1:
            raise ValueError("confirm_frames and release_frames must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")


def transform_sensor_points(points: np.ndarray, extrinsic: SensorExtrinsic) -> np.ndarray:
    """Transform Nx3 sensor-frame points into S10 ``base_link``."""

    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    c = math.cos(extrinsic.yaw_rad)
    s = math.sin(extrinsic.yaw_rad)
    rotation = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
    return values @ rotation.T + np.asarray(extrinsic.translation, dtype=np.float64)


def _voxel_downsample(points: np.ndarray, voxel_size: float, max_points: int) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    selected = np.sort(first)
    if selected.size > max_points:
        # Keep a deterministic, spatially uniform subset rather than the first
        # points in a driver-dependent packet order.
        positions = np.linspace(0, selected.size - 1, max_points, dtype=np.int64)
        selected = selected[positions]
    return points[selected]


def preprocess_base_points(points: np.ndarray, config: PointCloudConfig) -> np.ndarray:
    """Filter finite, in-range, ROI points and apply deterministic voxelization."""

    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    finite = np.isfinite(values).all(axis=1)
    distance = np.linalg.norm(values, axis=1)
    keep = finite & (distance >= config.min_range_m) & (distance <= config.max_range_m)
    keep &= (values[:, 0] >= config.roi_x_m[0]) & (values[:, 0] <= config.roi_x_m[1])
    keep &= (values[:, 1] >= config.roi_y_m[0]) & (values[:, 1] <= config.roi_y_m[1])
    keep &= (values[:, 2] >= config.roi_z_m[0]) & (values[:, 2] <= config.roi_z_m[1])
    return _voxel_downsample(values[keep], config.voxel_size_m, config.max_points)


def fuse_sensor_points(
    front_points: np.ndarray,
    rear_points: Optional[np.ndarray],
    config: PointCloudConfig,
) -> np.ndarray:
    """Transform and fuse the optional front/rear Airy clouds."""

    front = transform_sensor_points(front_points, config.front)
    clouds = [front]
    if rear_points is not None and np.asarray(rear_points).size:
        clouds.append(transform_sensor_points(rear_points, config.rear))
    return preprocess_base_points(np.concatenate(clouds, axis=0), config)


@dataclass(frozen=True)
class TerrainFeatures:
    """Fixed-size, interpretable features for later policy integration."""

    sector_min_range_m: tuple[float, ...]
    front_height_p90_m: tuple[float, ...]
    front_ground_min_z_m: float
    front_obstacle_max_z_m: float
    front_return_count: int
    total_return_count: int
    return_fraction: float
    closest_distance_m: float
    closest_angle_rad: float
    source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def vector(self) -> np.ndarray:
        """Return a finite vector; this is not yet wired into policy.onnx."""

        return np.asarray(
            self.sector_min_range_m
            + self.front_height_p90_m
            + (
                self.front_ground_min_z_m,
                self.front_obstacle_max_z_m,
                self.return_fraction,
                self.closest_distance_m,
                self.closest_angle_rad,
            ),
            dtype=np.float32,
        )


@dataclass(frozen=True)
class TerrainEvents:
    """Interpretable labels generated from the 3D cloud in shadow mode."""

    event: str
    step_detected: bool
    drop_detected: bool
    wall_detected: bool
    pole_detected: bool
    confidence: float
    baseline_z_m: float
    step_height_m: float
    drop_depth_m: float
    wall_height_m: float
    obstacle_width_m: float
    support_points: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TerrainEventDebounceState:
    """Temporal, advisory state derived from consecutive raw labels."""

    event: str
    raw_event: str
    confirmed: bool
    candidate_streak: int
    clear_streak: int
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TerrainEventDebouncer:
    """Require temporal consistency before exposing a terrain event."""

    _STRUCTURAL_EVENTS = frozenset(("STEP", "DROP", "WALL", "POLE"))

    def __init__(
        self,
        confirm_frames: int = 3,
        release_frames: int = 2,
        minimum_confidence: float = 0.25,
    ) -> None:
        if confirm_frames < 1 or release_frames < 1:
            raise ValueError("confirm_frames and release_frames must be positive")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        self.confirm_frames = int(confirm_frames)
        self.release_frames = int(release_frames)
        self.minimum_confidence = float(minimum_confidence)
        self._candidate: Optional[str] = None
        self._candidate_streak = 0
        self._active: Optional[str] = None
        self._clear_streak = 0

    def reset(self) -> None:
        self._candidate = None
        self._candidate_streak = 0
        self._active = None
        self._clear_streak = 0

    def update(self, raw: TerrainEvents) -> TerrainEventDebounceState:
        eligible = (
            raw.event in self._STRUCTURAL_EVENTS
            and raw.confidence >= self.minimum_confidence
        )
        if eligible and raw.event == self._candidate:
            self._candidate_streak += 1
        elif eligible:
            self._candidate = raw.event
            self._candidate_streak = 1
        else:
            self._candidate = None
            self._candidate_streak = 0

        if self._active is None:
            self._clear_streak = 0
            if self._candidate is not None and self._candidate_streak >= self.confirm_frames:
                self._active = self._candidate
        elif eligible and raw.event == self._active:
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            if self._clear_streak >= self.release_frames:
                self._active = None
                self._clear_streak = 0

        if self._active is None:
            if eligible:
                remaining = max(self.confirm_frames - self._candidate_streak, 0)
                reason = f"candidate {raw.event} needs {remaining} more frame(s)"
            else:
                reason = "no confirmed event"
            return TerrainEventDebounceState(
                event="FLAT_OR_UNKNOWN",
                raw_event=raw.event,
                confirmed=False,
                candidate_streak=self._candidate_streak,
                clear_streak=self._clear_streak,
                confidence=0.0,
                reason=reason,
            )

        confidence = raw.confidence if raw.event == self._active else 0.0
        return TerrainEventDebounceState(
            event=self._active,
            raw_event=raw.event,
            confirmed=True,
            candidate_streak=self._candidate_streak,
            clear_streak=self._clear_streak,
            confidence=confidence,
            reason=f"{self._active} confirmed; release after {self.release_frames} clear frame(s)",
        )


def classify_terrain_events(
    points: np.ndarray,
    config: TerrainEventConfig = TerrainEventConfig(),
) -> TerrainEvents:
    """Classify coarse step/drop/wall/pole events without controlling the robot.

    The baseline is estimated from a near-ground strip in the same sensor frame.
    A raised surface requires its lower percentile to be above the baseline,
    while a wall requires a tall vertical span over a wider lateral extent.
    These guards keep a vertical wall from being mislabeled as a walkable step.
    """

    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        values = np.empty((0, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    analysis = (
        (values[:, 0] >= config.analysis_x_m[0])
        & (values[:, 0] <= config.analysis_x_m[1])
        & (np.abs(values[:, 1]) <= config.y_half_width_m)
    ) if len(values) else np.zeros(0, dtype=bool)
    baseline_mask = (
        (values[:, 0] >= config.baseline_x_m[0])
        & (values[:, 0] <= config.baseline_x_m[1])
        & (np.abs(values[:, 1]) <= config.y_half_width_m)
    ) if len(values) else np.zeros(0, dtype=bool)
    baseline_points = values[baseline_mask]
    candidates = values[analysis]
    if len(baseline_points) < config.minimum_support_points or len(candidates) < config.minimum_support_points:
        return TerrainEvents(
            event="INSUFFICIENT_DATA",
            step_detected=False,
            drop_detected=False,
            wall_detected=False,
            pole_detected=False,
            confidence=0.0,
            baseline_z_m=0.0 if not len(baseline_points) else float(np.median(baseline_points[:, 2])),
            step_height_m=0.0,
            drop_depth_m=0.0,
            wall_height_m=0.0,
            obstacle_width_m=0.0,
            support_points=int(len(candidates)),
            reason="near-field support is insufficient",
        )

    baseline = float(np.median(baseline_points[:, 2]))
    x_edges = np.linspace(config.analysis_x_m[0], config.analysis_x_m[1], 4)
    bin_stats: list[tuple[float, float, int]] = []
    for index in range(3):
        selected = candidates[
            (candidates[:, 0] >= x_edges[index])
            & (candidates[:, 0] < x_edges[index + 1] if index < 2 else candidates[:, 0] <= x_edges[index + 1])
        ]
        if len(selected):
            bin_stats.append(
                (
                    float(np.percentile(selected[:, 2], 10.0)),
                    float(np.percentile(selected[:, 2], 90.0)),
                    int(len(selected)),
                )
            )
        else:
            bin_stats.append((baseline, baseline, 0))

    step_height = max(
        (low - baseline for low, _high, count in bin_stats if count >= config.minimum_support_points),
        default=0.0,
    )
    drop_depth = max(
        (baseline - high for _low, high, count in bin_stats if count >= config.minimum_support_points),
        default=0.0,
    )
    high_span = float(np.percentile(candidates[:, 2], 90.0) - np.percentile(candidates[:, 2], 10.0))
    elevated = candidates[candidates[:, 2] >= baseline + 0.5 * config.compact_height_threshold_m]
    lateral_width = float(np.ptp(elevated[:, 1])) if len(elevated) else 0.0
    longitudinal_width = float(np.ptp(elevated[:, 0])) if len(elevated) else 0.0
    wall = (
        len(elevated) >= config.minimum_support_points
        and high_span >= config.wall_height_threshold_m
        and lateral_width >= config.wall_min_width_m
        and float(np.percentile(candidates[:, 2], 10.0)) <= baseline + 0.12
    )
    compact = (
        len(elevated) >= config.minimum_support_points
        and high_span >= config.compact_height_threshold_m
        and lateral_width <= config.compact_max_width_m
        and longitudinal_width <= config.compact_max_width_m
    )
    step = step_height >= config.step_height_threshold_m and not wall
    drop = drop_depth >= config.drop_depth_threshold_m
    if drop:
        event = "DROP"
        magnitude = drop_depth / config.drop_depth_threshold_m
        reason = "front bins are below the near-field baseline"
    elif step:
        event = "STEP"
        magnitude = step_height / config.step_height_threshold_m
        reason = "front surface is raised above the near-field baseline"
    elif wall:
        event = "WALL"
        magnitude = high_span / config.wall_height_threshold_m
        reason = "tall return span has wall-like lateral width"
    elif compact:
        event = "POLE"
        magnitude = high_span / config.compact_height_threshold_m
        reason = "compact tall return cluster"
    else:
        event = "FLAT_OR_UNKNOWN"
        magnitude = 0.0
        reason = "no conservative event threshold crossed"
    confidence = float(np.clip(min(magnitude, 1.0) * len(candidates) / 40.0, 0.0, 1.0))
    return TerrainEvents(
        event=event,
        step_detected=step,
        drop_detected=drop,
        wall_detected=wall,
        pole_detected=compact,
        confidence=confidence,
        baseline_z_m=baseline,
        step_height_m=float(max(step_height, 0.0)),
        drop_depth_m=float(max(drop_depth, 0.0)),
        wall_height_m=float(max(high_span, 0.0)),
        obstacle_width_m=lateral_width,
        support_points=int(len(candidates)),
        reason=reason,
    )


def extract_terrain_features(
    points: np.ndarray,
    config: PointCloudConfig,
    source: str = "pointcloud",
) -> TerrainFeatures:
    """Extract range sectors and front cross-section height summaries."""

    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        values = np.empty((0, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    n_bins = config.angular_bins
    sector = np.full(n_bins, config.max_range_m, dtype=np.float64)
    if len(values):
        angle = np.mod(np.arctan2(values[:, 1], values[:, 0]), 2.0 * math.pi)
        index = np.floor(angle / (2.0 * math.pi) * n_bins).astype(int)
        index = np.clip(index, 0, n_bins - 1)
        planar = np.hypot(values[:, 0], values[:, 1])
        for bin_index in range(n_bins):
            selected = planar[index == bin_index]
            if selected.size:
                sector[bin_index] = float(np.percentile(selected, 10.0))

    front_mask = (
        (values[:, 0] >= config.front_x_min_m)
        & (values[:, 0] <= config.front_x_max_m)
        & (np.abs(values[:, 1]) <= config.front_y_half_width_m)
    ) if len(values) else np.zeros(0, dtype=bool)
    front = values[front_mask]
    height = np.zeros(config.front_x_bins, dtype=np.float64)
    if len(front):
        edges = np.linspace(config.front_x_min_m, config.front_x_max_m, config.front_x_bins + 1)
        for bin_index in range(config.front_x_bins):
            selected = front[(front[:, 0] >= edges[bin_index]) & (front[:, 0] < edges[bin_index + 1])]
            if selected.size:
                # A high percentile captures a step/wall while ignoring isolated
                # speckles; raw z is preserved in the log for later calibration.
                height[bin_index] = float(np.percentile(selected[:, 2], 90.0))

    if len(front):
        ground_min = float(np.percentile(front[:, 2], 10.0))
        obstacle_max = float(np.percentile(front[:, 2], 90.0))
    else:
        ground_min = 0.0
        obstacle_max = 0.0
    if len(values):
        planar = np.hypot(values[:, 0], values[:, 1])
        nearest = int(np.argmin(planar))
        closest_distance = float(planar[nearest])
        closest_angle = float(math.atan2(values[nearest, 1], values[nearest, 0]))
    else:
        closest_distance = config.max_range_m
        closest_angle = 0.0

    return TerrainFeatures(
        sector_min_range_m=tuple(float(v) for v in sector),
        front_height_p90_m=tuple(float(v) for v in height),
        front_ground_min_z_m=ground_min,
        front_obstacle_max_z_m=obstacle_max,
        front_return_count=int(len(front)),
        total_return_count=int(len(values)),
        return_fraction=float(len(values) / max(config.max_points, 1)),
        closest_distance_m=closest_distance,
        closest_angle_rad=closest_angle,
        source=source,
    )
