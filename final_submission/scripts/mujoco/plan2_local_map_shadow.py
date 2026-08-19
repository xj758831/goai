#!/usr/bin/env python3
"""Read-only rolling 2.5-D map and short-horizon trajectory scoring.

This module deliberately has no ROS or MuJoCo dependency.  It converts each
native lidar cloud from the current base frame into a short-lived world-frame
map using odometry, then scores hypothetical body-frame motions.  The result
is an advisory measurement only; it never emits a command or changes a
simulation state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class LocalMapConfig:
    """Conservative map and candidate-scoring limits."""

    resolution_m: float = 0.05
    retention_s: float = 2.5
    max_range_m: float = 3.0
    footprint_radius_m: float = 0.30
    obstacle_height_m: float = 0.35
    step_height_m: float = 0.30
    candidate_horizon_m: float = 0.90
    candidate_sample_m: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "resolution_m",
            "retention_s",
            "max_range_m",
            "footprint_radius_m",
            "obstacle_height_m",
            "step_height_m",
            "candidate_horizon_m",
            "candidate_sample_m",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.step_height_m > self.obstacle_height_m:
            raise ValueError("step_height_m cannot exceed obstacle_height_m")


@dataclass(frozen=True)
class CandidateMotion:
    """A bounded hypothetical motion in the current body frame."""

    name: str
    lateral_offset_m: float = 0.0
    yaw_offset_rad: float = 0.0
    speed_scale: float = 1.0
    longitudinal_direction: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name must not be empty")
        if not math.isfinite(self.lateral_offset_m) or not math.isfinite(
            self.yaw_offset_rad
        ):
            raise ValueError("candidate offsets must be finite")
        if not 0.0 < float(self.speed_scale) <= 1.0:
            raise ValueError("speed_scale must be in (0, 1]")
        if self.longitudinal_direction not in (-1, 1):
            raise ValueError("longitudinal_direction must be -1 or 1")


DEFAULT_CANDIDATES = (
    CandidateMotion("straight"),
    CandidateMotion("right_offset", lateral_offset_m=-0.16, speed_scale=0.85),
    CandidateMotion("left_offset", lateral_offset_m=0.16, speed_scale=0.85),
    CandidateMotion("right_yaw", yaw_offset_rad=-0.18, speed_scale=0.75),
    CandidateMotion("left_yaw", yaw_offset_rad=0.18, speed_scale=0.75),
    CandidateMotion("slow_straight", speed_scale=0.55),
    CandidateMotion(
        "reverse_clear", speed_scale=0.55, longitudinal_direction=-1
    ),
)


@dataclass(frozen=True)
class CellEvidence:
    """Compact world-map evidence for one occupied xy cell."""

    z_min_m: float
    z_max_m: float
    hits: int
    last_time_s: float


@dataclass(frozen=True)
class CandidateScore:
    name: str
    minimum_clearance_m: float
    initial_clearance_m: float
    final_clearance_m: float
    clearance_gain_m: float
    occupied_samples: int
    high_obstacle_samples: int
    step_like_samples: int
    speed_scale: float
    longitudinal_direction: int

    @property
    def feasible(self) -> bool:
        return self.occupied_samples == 0 and self.high_obstacle_samples == 0

    @property
    def escape_clear(self) -> bool:
        return self.final_clearance_m > self.initial_clearance_m


def _validate_pose(pose_xy_yaw: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in pose_xy_yaw)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("pose_xy_yaw must contain three finite values")
    return values


def _body_to_world(points_xy: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    return np.asarray(points_xy, dtype=np.float64) @ rotation.T + np.asarray((x, y))


def _world_to_body(points_xy: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(((c, s), (-s, c)), dtype=np.float64)
    return (np.asarray(points_xy, dtype=np.float64) - np.asarray((x, y))) @ rotation.T


def _candidate_path_body(
    candidate: CandidateMotion,
    horizon_m: float,
    sample_m: float,
) -> np.ndarray:
    sample_count = max(1, int(math.ceil(horizon_m / sample_m)))
    distances = np.linspace(0.0, horizon_m, sample_count + 1)
    fractions = distances / max(horizon_m, 1.0e-9)
    headings = candidate.yaw_offset_rad * fractions
    return np.column_stack(
        (
            candidate.longitudinal_direction * distances * np.cos(headings),
            candidate.lateral_offset_m * fractions + distances * np.sin(headings),
        )
    )


class RollingLocalMap:
    """A bounded rolling map built solely from lidar points and odometry."""

    def __init__(self, config: LocalMapConfig | None = None) -> None:
        self.config = config or LocalMapConfig()
        self._frames: deque[tuple[float, dict[tuple[int, int], CellEvidence]]] = deque()
        self._cells: dict[tuple[int, int], CellEvidence] = {}
        self.last_time_s: float | None = None
        self.integrated_frames = 0

    def reset(self) -> None:
        self._frames.clear()
        self._cells.clear()
        self.last_time_s = None
        self.integrated_frames = 0

    def _key(self, x: float, y: float) -> tuple[int, int]:
        resolution = self.config.resolution_m
        return (math.floor(x / resolution), math.floor(y / resolution))

    def _expire(self, now: float) -> None:
        cutoff = now - self.config.retention_s
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def _rebuild_cells(self) -> None:
        cells: dict[tuple[int, int], CellEvidence] = {}
        for _, frame_cells in self._frames:
            for key, evidence in frame_cells.items():
                old = cells.get(key)
                if old is None:
                    cells[key] = evidence
                else:
                    cells[key] = CellEvidence(
                        min(old.z_min_m, evidence.z_min_m),
                        max(old.z_max_m, evidence.z_max_m),
                        old.hits + evidence.hits,
                        max(old.last_time_s, evidence.last_time_s),
                    )
        self._cells = cells

    def integrate(
        self,
        points_base: np.ndarray,
        pose_xy_yaw: Iterable[float],
        sim_time_s: float,
    ) -> dict[str, int | float]:
        """Add one cloud; points are Nx3 in the current S10 base frame."""

        now = float(sim_time_s)
        if not math.isfinite(now):
            raise ValueError("sim_time_s must be finite")
        pose = _validate_pose(pose_xy_yaw)
        if self.last_time_s is not None and now + 1.0e-9 < self.last_time_s:
            self.reset()
        values = np.asarray(points_base, dtype=np.float64)
        if values.size == 0:
            values = np.empty((0, 3), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points_base must have shape (N, 3)")
        finite = np.isfinite(values).all(axis=1)
        finite &= np.linalg.norm(values[:, :2], axis=1) <= self.config.max_range_m
        values = values[finite]
        world_xy = _body_to_world(values[:, :2], pose)
        points_world = np.column_stack((world_xy, values[:, 2]))
        return self.integrate_world(points_world, pose, now)

    def integrate_world(
        self,
        points_world: np.ndarray,
        pose_xy_yaw: Iterable[float],
        sim_time_s: float,
    ) -> dict[str, int | float]:
        """Add one cloud already transformed by full odometry into world xyz."""

        now = float(sim_time_s)
        if not math.isfinite(now):
            raise ValueError("sim_time_s must be finite")
        pose = _validate_pose(pose_xy_yaw)
        if self.last_time_s is not None and now + 1.0e-9 < self.last_time_s:
            self.reset()
        values = np.asarray(points_world, dtype=np.float64)
        if values.size == 0:
            values = np.empty((0, 3), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points_world must have shape (N, 3)")
        finite = np.isfinite(values).all(axis=1)
        finite &= np.linalg.norm(
            values[:, :2] - np.asarray(pose[:2], dtype=np.float64), axis=1
        ) <= self.config.max_range_m
        values = values[finite]
        frame_cells: dict[tuple[int, int], CellEvidence] = {}
        for point in values:
            key = self._key(float(point[0]), float(point[1]))
            existing = frame_cells.get(key)
            z = float(point[2])
            if existing is None:
                frame_cells[key] = CellEvidence(z, z, 1, now)
            else:
                frame_cells[key] = CellEvidence(
                    min(existing.z_min_m, z),
                    max(existing.z_max_m, z),
                    existing.hits + 1,
                    now,
                )
        self._frames.append((now, frame_cells))
        self._expire(now)
        self._rebuild_cells()
        self.last_time_s = now
        self.integrated_frames += 1
        return {
            "integrated_frames": self.integrated_frames,
            "active_cells": len(self._cells),
            "frame_points": int(len(values)),
            "sim_time_s": now,
        }

    def _cell_evidence(self, x: float, y: float) -> CellEvidence | None:
        return self._cells.get(self._key(x, y))

    def score_candidate(
        self,
        candidate: CandidateMotion,
        pose_xy_yaw: Iterable[float],
        *,
        horizon_m: float | None = None,
    ) -> CandidateScore:
        """Score a short arc after inflating returns by the S10 footprint."""

        pose = _validate_pose(pose_xy_yaw)
        horizon = float(horizon_m or self.config.candidate_horizon_m)
        if horizon <= 0.0:
            raise ValueError("horizon_m must be positive")
        occupied = 0
        high = 0
        step_like = 0
        minimum_clearance = float("inf")
        blocking_cells: list[tuple[float, float]] = []
        step_cells: list[tuple[float, float]] = []
        resolution = self.config.resolution_m
        for (ix, iy), evidence in self._cells.items():
            cell = ((ix + 0.5) * resolution, (iy + 0.5) * resolution)
            vertical_span = evidence.z_max_m - evidence.z_min_m
            if vertical_span >= self.config.obstacle_height_m:
                blocking_cells.append(cell)
            elif vertical_span <= self.config.step_height_m:
                step_cells.append(cell)
        path_body = _candidate_path_body(
            candidate,
            horizon,
            self.config.candidate_sample_m,
        )
        path_world = _body_to_world(path_body, pose)
        if blocking_cells:
            blockers = np.asarray(blocking_cells, dtype=np.float64)
            clearances = np.linalg.norm(
                path_world[:, None, :] - blockers[None, :, :], axis=2
            ).min(axis=1)
        else:
            clearances = np.full(len(path_world), self.config.max_range_m)
        minimum_clearance = float(np.min(clearances))
        occupied = int(np.count_nonzero(clearances <= self.config.footprint_radius_m))
        high = occupied
        if step_cells:
            steps = np.asarray(step_cells, dtype=np.float64)
            step_clearances = np.linalg.norm(
                path_world[:, None, :] - steps[None, :, :], axis=2
            ).min(axis=1)
            step_like = int(
                np.count_nonzero(step_clearances <= self.config.footprint_radius_m)
            )
        if not math.isfinite(minimum_clearance):
            minimum_clearance = 0.0
        return CandidateScore(
            name=candidate.name,
            minimum_clearance_m=float(minimum_clearance),
            initial_clearance_m=float(clearances[0]),
            final_clearance_m=float(clearances[-1]),
            clearance_gain_m=float(clearances[-1] - clearances[0]),
            occupied_samples=int(occupied),
            high_obstacle_samples=int(high),
            step_like_samples=int(step_like),
            speed_scale=float(candidate.speed_scale),
            longitudinal_direction=int(candidate.longitudinal_direction),
        )

    def score_candidates(
        self,
        pose_xy_yaw: Iterable[float],
        candidates: Iterable[CandidateMotion] = DEFAULT_CANDIDATES,
    ) -> list[CandidateScore]:
        scores = [self.score_candidate(candidate, pose_xy_yaw) for candidate in candidates]
        return sorted(
            scores,
            key=lambda item: (
                not item.feasible,
                not item.escape_clear,
                -item.minimum_clearance_m,
                -item.final_clearance_m,
                item.occupied_samples,
                item.name,
            ),
        )

    def visualization_snapshot(
        self,
        pose_xy_yaw: Iterable[float],
        candidates: Iterable[CandidateMotion] = DEFAULT_CANDIDATES,
    ) -> dict[str, object]:
        """Return a body-frame copy suitable for a read-only dashboard."""

        pose = _validate_pose(pose_xy_yaw)
        candidate_list = tuple(candidates)
        resolution = self.config.resolution_m
        if self._cells:
            keys = list(self._cells)
            centers_world = np.asarray(
                [
                    ((ix + 0.5) * resolution, (iy + 0.5) * resolution)
                    for ix, iy in keys
                ],
                dtype=np.float64,
            )
            centers_body = _world_to_body(centers_world, pose)
        else:
            keys = []
            centers_body = np.empty((0, 2), dtype=np.float64)

        cells = []
        for key, center_body in zip(keys, centers_body):
            evidence = self._cells[key]
            vertical_span = evidence.z_max_m - evidence.z_min_m
            cells.append(
                {
                    "x_m": float(center_body[0]),
                    "y_m": float(center_body[1]),
                    "z_min_m": float(evidence.z_min_m),
                    "z_max_m": float(evidence.z_max_m),
                    "vertical_span_m": float(vertical_span),
                    "hits": int(evidence.hits),
                    "blocking": bool(
                        vertical_span >= self.config.obstacle_height_m
                    ),
                    "step_like": bool(vertical_span <= self.config.step_height_m),
                }
            )

        candidate_paths = []
        for candidate in candidate_list:
            score = self.score_candidate(candidate, pose)
            path = _candidate_path_body(
                candidate,
                self.config.candidate_horizon_m,
                self.config.candidate_sample_m,
            )
            candidate_paths.append(
                {
                    "name": candidate.name,
                    "path_xy_m": path.tolist(),
                    "minimum_clearance_m": score.minimum_clearance_m,
                    "initial_clearance_m": score.initial_clearance_m,
                    "final_clearance_m": score.final_clearance_m,
                    "clearance_gain_m": score.clearance_gain_m,
                    "occupied_samples": score.occupied_samples,
                    "feasible": score.feasible,
                    "escape_clear": score.escape_clear,
                }
            )
        return {
            "frame": "s10_base_xy",
            "sim_time_s": self.last_time_s,
            "active_cells": len(cells),
            "resolution_m": resolution,
            "max_range_m": self.config.max_range_m,
            "footprint_radius_m": self.config.footprint_radius_m,
            "obstacle_height_m": self.config.obstacle_height_m,
            "step_height_m": self.config.step_height_m,
            "cells": cells,
            "candidate_paths": candidate_paths,
            "control_authority": False,
            "publishes_cmd_vel": False,
        }

    def summary(self, pose_xy_yaw: Iterable[float]) -> dict[str, object]:
        scores = self.score_candidates(pose_xy_yaw)
        return {
            "active_cells": len(self._cells),
            "integrated_frames": self.integrated_frames,
            "retention_s": self.config.retention_s,
            "resolution_m": self.config.resolution_m,
            "footprint_radius_m": self.config.footprint_radius_m,
            "control_authority": False,
            "publishes_cmd_vel": False,
            "candidate_scores": [
                {
                    "name": score.name,
                    "minimum_clearance_m": score.minimum_clearance_m,
                    "initial_clearance_m": score.initial_clearance_m,
                    "final_clearance_m": score.final_clearance_m,
                    "clearance_gain_m": score.clearance_gain_m,
                    "occupied_samples": score.occupied_samples,
                    "high_obstacle_samples": score.high_obstacle_samples,
                    "step_like_samples": score.step_like_samples,
                    "speed_scale": score.speed_scale,
                    "longitudinal_direction": score.longitudinal_direction,
                    "feasible": score.feasible,
                    "escape_clear": score.escape_clear,
                }
                for score in scores
            ],
        }


def select_shadow_candidate(local_map_summary: dict[str, object]) -> dict[str, object] | None:
    """Select one hypothetical motion without granting it control authority."""

    raw_scores = local_map_summary.get("candidate_scores", [])
    if not isinstance(raw_scores, list):
        raise ValueError("candidate_scores must be a list")
    eligible = [
        score
        for score in raw_scores
        if isinstance(score, dict)
        and (bool(score.get("feasible")) or bool(score.get("escape_clear")))
    ]
    if not eligible:
        return None
    selected = eligible[0]
    return {
        "event": "LOCAL_MAP_MOTION_CANDIDATE",
        "selected_name": str(selected["name"]),
        "minimum_clearance_m": float(selected["minimum_clearance_m"]),
        "final_clearance_m": float(selected["final_clearance_m"]),
        "clearance_gain_m": float(selected["clearance_gain_m"]),
        "occupied_samples": int(selected["occupied_samples"]),
        "longitudinal_direction": int(selected["longitudinal_direction"]),
        "selection_basis": (
            "rolling lidar+odometry map after a separately confirmed stall"
        ),
        "control_authority": False,
        "publishes_cmd_vel": False,
    }
