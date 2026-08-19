#!/usr/bin/env python3
"""Compare read-only Plan2 point-cloud features by verified route stage.

The region configuration describes navigation context, not obstacle ground
truth. This tool reads completed shadow logs and never publishes commands or
modifies MuJoCo state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETAIL_PATTERN = re.compile(r"scores=(\d+)/33 target=([^ ]+)")
INITIAL_TARGET_PATTERN = re.compile(r"target score (\d+)")
NUMERIC_FEATURES = (
    "base_z_m",
    "closest_distance_m",
    "front_ground_min_z_m",
    "front_obstacle_max_z_m",
    "front_vertical_span_m",
    "return_fraction",
    "processed_points",
    "step_height_m",
    "drop_depth_m",
    "wall_height_m",
    "obstacle_width_m",
)


def parse_route_context(row: dict[str, Any]) -> tuple[int, int | None] | None:
    detail = str(row.get("detail", ""))
    match = DETAIL_PATTERN.search(detail)
    if match is None:
        initial_match = INITIAL_TARGET_PATTERN.search(detail)
        if initial_match is None:
            return None
        return 0, int(initial_match.group(1))
    reached_count = int(match.group(1))
    target_token = match.group(2)
    target_score = None if target_token == "None" else int(target_token)
    return reached_count, target_score


def build_region_index(document: dict[str, Any]) -> dict[str, Any]:
    if int(document.get("schema_version", 0)) != 1:
        raise ValueError("region config schema_version must be 1")
    regions = document.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("region config must contain a non-empty regions list")

    by_phase: dict[str, dict[str, Any]] = {}
    by_target: dict[int, dict[str, Any]] = {}
    by_reached_without_target: dict[int, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_region in regions:
        if not isinstance(raw_region, dict):
            raise ValueError("each region must be a mapping")
        region_id = str(raw_region.get("id", "")).strip()
        evidence = str(raw_region.get("evidence", "")).strip()
        if not region_id or not evidence:
            raise ValueError("each region requires id and evidence")
        if region_id in seen_ids:
            raise ValueError(f"duplicate region id: {region_id}")
        seen_ids.add(region_id)
        region = {"id": region_id, "evidence": evidence}
        ordered.append(region)

        for phase in raw_region.get("phases", []):
            phase_name = str(phase)
            if phase_name in by_phase:
                raise ValueError(f"phase assigned to multiple regions: {phase_name}")
            by_phase[phase_name] = region
        for score in raw_region.get("target_scores", []):
            target_score = int(score)
            if target_score in by_target:
                raise ValueError(
                    f"target score assigned to multiple regions: {target_score}"
                )
            by_target[target_score] = region
        for count in raw_region.get("reached_counts_without_target", []):
            reached_count = int(count)
            if reached_count in by_reached_without_target:
                raise ValueError(
                    "target-less reached count assigned to multiple regions: "
                    f"{reached_count}"
                )
            by_reached_without_target[reached_count] = region

    return {
        "interpretation": str(document.get("interpretation", "")),
        "sources": document.get("sources", {}),
        "ordered": ordered,
        "by_phase": by_phase,
        "by_target": by_target,
        "by_reached_without_target": by_reached_without_target,
    }


def assign_region(row: dict[str, Any], index: dict[str, Any]) -> dict[str, Any] | None:
    phase = str(row.get("phase", ""))
    if phase in index["by_phase"]:
        return index["by_phase"][phase]
    context = parse_route_context(row)
    if context is None:
        return None
    reached_count, target_score = context
    if target_score is not None:
        return index["by_target"].get(target_score)
    return index["by_reached_without_target"].get(reached_count)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_features(row: dict[str, Any]) -> dict[str, float]:
    features = row.get("features", {})
    events = row.get("events", {})
    pose = row.get("pose", [])
    ground = float(features.get("front_ground_min_z_m", math.nan))
    obstacle = float(features.get("front_obstacle_max_z_m", math.nan))
    values = {
        "base_z_m": float(pose[2]) if len(pose) >= 3 else math.nan,
        "closest_distance_m": float(features.get("closest_distance_m", math.nan)),
        "front_ground_min_z_m": ground,
        "front_obstacle_max_z_m": obstacle,
        "front_vertical_span_m": obstacle - ground,
        "return_fraction": float(features.get("return_fraction", math.nan)),
        "processed_points": float(row.get("processed_points", math.nan)),
        "step_height_m": float(events.get("step_height_m", math.nan)),
        "drop_depth_m": float(events.get("drop_depth_m", math.nan)),
        "wall_height_m": float(events.get("wall_height_m", math.nan)),
        "obstacle_width_m": float(events.get("obstacle_width_m", math.nan)),
    }
    return {name: value for name, value in values.items() if math.isfinite(value)}


def summarize_values(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": percentile(values, 0.0),
        "p10": percentile(values, 0.1),
        "median": percentile(values, 0.5),
        "p90": percentile(values, 0.9),
        "max": percentile(values, 1.0),
    }


def summarize_run(
    rows: list[dict[str, Any]], index: dict[str, Any]
) -> dict[str, Any]:
    region_rows: dict[str, list[dict[str, Any]]] = {
        region["id"]: [] for region in index["ordered"]
    }
    unmapped_contexts: Counter[str] = Counter()
    for row in rows:
        region = assign_region(row, index)
        if region is None:
            context = parse_route_context(row)
            key = f"phase={row.get('phase', '')} context={context}"
            unmapped_contexts[key] += 1
            continue
        region_rows[region["id"]].append(row)

    summaries: dict[str, Any] = {}
    for region in index["ordered"]:
        region_id = region["id"]
        selected = region_rows[region_id]
        stable_counts = Counter(
            str(row.get("stable_event", {}).get("event", "MISSING"))
            for row in selected
        )
        raw_counts = Counter(
            str(row.get("events", {}).get("event", "MISSING")) for row in selected
        )
        numeric: dict[str, list[float]] = {name: [] for name in NUMERIC_FEATURES}
        for row in selected:
            for name, value in numeric_features(row).items():
                numeric[name].append(value)
        summaries[region_id] = {
            "evidence": region["evidence"],
            "samples": len(selected),
            "stable_event_counts": dict(sorted(stable_counts.items())),
            "raw_event_counts": dict(sorted(raw_counts.items())),
            "numeric_features": {
                name: summarize_values(values)
                for name, values in numeric.items()
                if values
            },
        }
    return {
        "input_samples": len(rows),
        "assigned_samples": sum(len(value) for value in region_rows.values()),
        "unmapped_samples": sum(unmapped_contexts.values()),
        "unmapped_contexts": dict(sorted(unmapped_contexts.items())),
        "regions": summaries,
    }


def total_variation(
    first: dict[str, int], second: dict[str, int]
) -> float | None:
    first_total = sum(first.values())
    second_total = sum(second.values())
    if first_total == 0 or second_total == 0:
        return None
    labels = set(first) | set(second)
    return 0.5 * sum(
        abs(first.get(label, 0) / first_total - second.get(label, 0) / second_total)
        for label in labels
    )


def compare_runs(runs: dict[str, dict[str, Any]], region_order: list[str]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    run_names = list(runs)
    for region_id in region_order:
        pairwise: list[dict[str, Any]] = []
        for first_index, first_name in enumerate(run_names):
            for second_name in run_names[first_index + 1 :]:
                first = runs[first_name]["regions"][region_id]
                second = runs[second_name]["regions"][region_id]
                distance = total_variation(
                    first["stable_event_counts"], second["stable_event_counts"]
                )
                pairwise.append(
                    {
                        "first": first_name,
                        "second": second_name,
                        "stable_event_total_variation": distance,
                    }
                )
        finite_distances = [
            float(pair["stable_event_total_variation"])
            for pair in pairwise
            if pair["stable_event_total_variation"] is not None
        ]
        feature_ranges: dict[str, Any] = {}
        for feature in NUMERIC_FEATURES:
            medians = {
                run_name: float(
                    runs[run_name]["regions"][region_id]["numeric_features"][feature][
                        "median"
                    ]
                )
                for run_name in run_names
                if feature
                in runs[run_name]["regions"][region_id]["numeric_features"]
            }
            if medians:
                feature_ranges[feature] = {
                    "medians_by_run": medians,
                    "median_span": max(medians.values()) - min(medians.values()),
                }
        comparisons[region_id] = {
            "sample_counts": {
                run_name: runs[run_name]["regions"][region_id]["samples"]
                for run_name in run_names
            },
            "stable_event_pairwise": pairwise,
            "stable_event_total_variation_max": (
                max(finite_distances) if finite_distances else None
            ),
            "numeric_feature_median_ranges": feature_ranges,
        }
    return comparisons


def analyze_run_directories(
    run_directories: list[Path], config_path: Path
) -> dict[str, Any]:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    index = build_region_index(document)
    runs: dict[str, dict[str, Any]] = {}
    inputs: dict[str, str] = {}
    for run_directory in run_directories:
        resolved = run_directory.resolve()
        run_name = resolved.name
        if run_name in runs:
            raise ValueError(f"duplicate run directory name: {run_name}")
        input_path = resolved / "pointcloud_shadow" / "pointcloud_features.jsonl"
        rows = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        runs[run_name] = summarize_run(rows, index)
        inputs[run_name] = str(input_path)

    region_order = [region["id"] for region in index["ordered"]]
    return {
        "interpretation": index["interpretation"],
        "control_authority": False,
        "publishes_cmd_vel": False,
        "modifies_mujoco_state": False,
        "config_path": str(config_path.resolve()),
        "sources": index["sources"],
        "inputs": inputs,
        "region_order": region_order,
        "runs": runs,
        "comparisons": compare_runs(runs, region_order),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "plan2_pointcloud_regions.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_run_directories(args.run_dir, args.config.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"runs={len(report['runs'])} regions={len(report['region_order'])} "
        f"output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
