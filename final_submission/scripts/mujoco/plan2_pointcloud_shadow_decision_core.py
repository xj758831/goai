#!/usr/bin/env python3
"""Shared state machine for read-only Plan2 point-cloud advisories."""

from __future__ import annotations

from collections import deque
import math
from typing import Any

from analyze_plan2_pointcloud_regions import assign_region


def load_decision_config(document: dict[str, Any]) -> dict[str, Any]:
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("decision config schema_version must be 1")
    if document.get("control_authority") is not False:
        raise ValueError("shadow decision config must disable control authority")
    if document.get("publishes_cmd_vel") is not False:
        raise ValueError("shadow decision config must disable cmd_vel publishing")
    raw_rules = document.get("rules", {})
    if not isinstance(raw_rules, dict) or not raw_rules:
        raise ValueError("decision config requires at least one rule")
    rules: dict[str, dict[str, Any]] = {}
    for rule_name, raw in raw_rules.items():
        if not isinstance(raw, dict):
            raise ValueError(f"{rule_name} must be a mapping")
        raw_fit_run_names = raw.get("fit_run_names")
        if not isinstance(raw_fit_run_names, list):
            raise ValueError(f"fit_run_names for {rule_name} must be a list")
        rule = {
            "event": str(raw["event"]),
            "region_id": str(raw["region_id"]),
            "fit_run_names": [str(name) for name in raw_fit_run_names],
            "minimum_region_elapsed_s": float(raw["minimum_region_elapsed_s"]),
            "motion_window_s": float(raw["motion_window_s"]),
            "maximum_planar_displacement_m": float(
                raw["maximum_planar_displacement_m"]
            ),
            "maximum_closest_distance_m": float(
                raw["maximum_closest_distance_m"]
            ),
            "minimum_front_vertical_span_m": float(
                raw["minimum_front_vertical_span_m"]
            ),
            "confirm_frames": int(raw["confirm_frames"]),
            "evidence": str(raw.get("evidence", "")),
        }
        if not rule["fit_run_names"] or len(set(rule["fit_run_names"])) != len(
            rule["fit_run_names"]
        ):
            raise ValueError(
                f"fit_run_names for {rule_name} must be nonempty and unique"
            )
        positive_names = (
            "minimum_region_elapsed_s",
            "motion_window_s",
            "maximum_planar_displacement_m",
            "maximum_closest_distance_m",
            "minimum_front_vertical_span_m",
        )
        if any(rule[name] <= 0.0 for name in positive_names):
            raise ValueError(
                f"all shadow decision thresholds for {rule_name} must be positive"
            )
        if rule["confirm_frames"] <= 0:
            raise ValueError(f"confirm_frames for {rule_name} must be positive")
        evaluation = raw.get("evaluation_only", {})
        if not isinstance(evaluation, dict):
            raise ValueError(f"evaluation_only for {rule_name} must be a mapping")
        kind = str(evaluation.get("kind", ""))
        if kind == "base_z_below":
            threshold_key = "threshold_m"
        elif kind == "region_duration_exceeds":
            threshold_key = "threshold_s"
        else:
            raise ValueError(f"unsupported evaluation kind for {rule_name}: {kind}")
        threshold = float(evaluation[threshold_key])
        if threshold <= 0.0:
            raise ValueError(f"evaluation threshold for {rule_name} must be positive")
        rule["evaluation_only"] = {
            "kind": kind,
            threshold_key: threshold,
            "note": str(evaluation.get("note", "")),
        }
        rules[str(rule_name)] = rule
    return {
        "interpretation": str(document.get("interpretation", "")),
        "rules": rules,
    }


def front_vertical_span(row: dict[str, Any]) -> float:
    features = row["features"]
    return float(features["front_obstacle_max_z_m"]) - float(
        features["front_ground_min_z_m"]
    )


class ShadowRuleEvaluator:
    """Evaluate one frozen rule in chronological order."""

    def __init__(self, rule: dict[str, Any]) -> None:
        self.rule = rule
        self.reset()

    def reset(self) -> None:
        self.start_time: float | None = None
        self.last_time: float | None = None
        self.history: deque[tuple[float, float, float]] = deque()
        self.streak = 0
        self.matched_frames = 0
        self.samples = 0
        self.first_advisory: dict[str, Any] | None = None

    def update(self, row: dict[str, Any]) -> dict[str, Any] | None:
        now = float(row["sim_time"])
        if self.last_time is not None and now + 1.0e-9 < self.last_time:
            raise ValueError("sim_time moves backward inside the decision region")
        self.last_time = now
        if self.start_time is None:
            self.start_time = now
        self.samples += 1

        x = float(row["pose"][0])
        y = float(row["pose"][1])
        self.history.append((now, x, y))
        cutoff = now - self.rule["motion_window_s"]
        while len(self.history) > 1 and self.history[0][0] < cutoff:
            self.history.popleft()
        _, history_x, history_y = self.history[0]
        displacement = math.hypot(x - history_x, y - history_y)
        closest = float(row["features"]["closest_distance_m"])
        vertical_span = front_vertical_span(row)
        elapsed = now - self.start_time
        matches = (
            elapsed >= self.rule["minimum_region_elapsed_s"]
            and displacement <= self.rule["maximum_planar_displacement_m"]
            and closest <= self.rule["maximum_closest_distance_m"]
            and vertical_span >= self.rule["minimum_front_vertical_span_m"]
        )
        if matches:
            self.streak += 1
            self.matched_frames += 1
        else:
            self.streak = 0
        if self.first_advisory is not None or self.streak < self.rule["confirm_frames"]:
            return None
        self.first_advisory = {
            "event": self.rule["event"],
            "sim_time_s": now,
            "region_elapsed_s": elapsed,
            "motion_window_s": self.rule["motion_window_s"],
            "planar_displacement_m": displacement,
            "closest_distance_m": closest,
            "front_vertical_span_m": vertical_span,
            "confirmed_frames": self.streak,
            "control_authority": False,
            "publishes_cmd_vel": False,
            "modifies_mujoco_state": False,
        }
        return self.first_advisory


class ShadowDecisionMonitor:
    """Route rows to independent advisory evaluators without control authority."""

    def __init__(
        self,
        region_index: dict[str, Any],
        decision_config: dict[str, Any],
    ) -> None:
        self.region_index = region_index
        self.rules = decision_config["rules"]
        self.evaluators = {
            name: ShadowRuleEvaluator(rule) for name, rule in self.rules.items()
        }
        self.last_sim_time: float | None = None
        self.advisories: list[dict[str, Any]] = []

    def reset(self) -> None:
        for evaluator in self.evaluators.values():
            evaluator.reset()
        self.last_sim_time = None
        self.advisories.clear()

    def update(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        sim_time = float(row["sim_time"])
        if self.last_sim_time is not None and sim_time + 1.0e-9 < self.last_sim_time:
            self.reset()
        self.last_sim_time = sim_time
        region = assign_region(row, self.region_index)
        if region is None:
            return []
        emitted: list[dict[str, Any]] = []
        for rule_name, rule in self.rules.items():
            if region["id"] != rule["region_id"]:
                continue
            advisory = self.evaluators[rule_name].update(row)
            if advisory is None:
                continue
            annotated = {
                "rule_name": rule_name,
                "region_id": rule["region_id"],
                **advisory,
            }
            self.advisories.append(annotated)
            emitted.append(annotated)
        return emitted

    def summary(self) -> dict[str, Any]:
        return {
            "control_authority": False,
            "publishes_cmd_vel": False,
            "modifies_mujoco_state": False,
            "advisory_count": len(self.advisories),
            "advisories": self.advisories,
            "rules": {
                name: {
                    "region_id": evaluator.rule["region_id"],
                    "samples": evaluator.samples,
                    "matched_frames": evaluator.matched_frames,
                    "triggered": evaluator.first_advisory is not None,
                    "first_advisory": evaluator.first_advisory,
                }
                for name, evaluator in self.evaluators.items()
            },
        }
