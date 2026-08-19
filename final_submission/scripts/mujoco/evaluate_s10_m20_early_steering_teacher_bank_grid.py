#!/usr/bin/env python3
"""Cross-evaluate the early-steering teacher bank on a 0.21 m pose grid."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_s10_m20_two_stage_selector_observability import collect_trigger_history
from evaluate_s10_m20_early_steering_candidate import (
    DEFAULT_CORNER_RESCUE,
    DEFAULT_CORNER_STEERING,
    DEFAULT_POSITIVE_STEERING,
    DEFAULT_STEERING,
)
from search_s10_m20_approach_two_stage_rescue import (
    PHASE_ESTIMATOR,
    REFERENCE_CHECKPOINT,
    asset_hashes,
    load_reference,
    sha256,
)
from search_s10_m20_early_steering_rescue import (
    DEFAULT_RESCUE_CANDIDATE,
    load_candidate,
    rollout,
)


LATERAL_GRID = (-0.030, -0.015, 0.0, 0.015, 0.030)
YAW_GRID = (-3.0, -1.5, 0.0, 1.5, 3.0)


@dataclass(frozen=True)
class Teacher:
    name: str
    candidate: np.ndarray
    steering: np.ndarray
    complexity_rank: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, default=REFERENCE_CHECKPOINT)
    parser.add_argument("--phase-estimator", type=Path, default=PHASE_ESTIMATOR)
    parser.add_argument("--base-rescue", type=Path, default=DEFAULT_RESCUE_CANDIDATE)
    parser.add_argument("--negative-steering", type=Path, default=DEFAULT_STEERING)
    parser.add_argument("--positive-steering", type=Path, default=DEFAULT_POSITIVE_STEERING)
    parser.add_argument("--corner-rescue", type=Path, default=DEFAULT_CORNER_RESCUE)
    parser.add_argument("--corner-steering", type=Path, default=DEFAULT_CORNER_STEERING)
    parser.add_argument(
        "--case15-steering",
        type=Path,
        default=Path(
            "logs/mujoco/s10_m20_early_steering_grid_case15_smoke16x3_20260811_v1/"
            "best_steering.npz"
        ),
    )
    parser.add_argument(
        "--case19-steering",
        type=Path,
        default=Path(
            "logs/mujoco/s10_m20_early_steering_grid_case19_smoke16x3_20260811_v1/"
            "best_steering.npz"
        ),
    )
    parser.add_argument(
        "--case22-steering",
        type=Path,
        default=Path(
            "logs/mujoco/s10_m20_early_steering_grid_case22_smoke16x3_20260811_v1/"
            "best_steering.npz"
        ),
    )
    parser.add_argument(
        "--case23-steering",
        type=Path,
        default=Path(
            "logs/mujoco/s10_m20_early_steering_grid_case23_smoke16x3_20260811_v1/"
            "best_steering.npz"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--suite", choices=("training_grid", "holdout_grid"), default="training_grid"
    )
    parser.add_argument("--depth-m", type=float, default=0.21)
    parser.add_argument("--command-mps", type=float, default=0.8)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument(
        "--lateral-grid",
        type=str,
        default=None,
        help="optional comma-separated lateral positions overriding the named suite",
    )
    parser.add_argument(
        "--yaw-grid",
        type=str,
        default=None,
        help="optional comma-separated yaw angles overriding the named suite",
    )
    parser.add_argument(
        "--teacher-only",
        type=str,
        default=None,
        help="evaluate only one named teacher instead of the full bank",
    )
    parser.add_argument(
        "--teacher-manifest",
        type=Path,
        default=None,
        help="optional JSON manifest of combined candidate+steering NPZ teachers",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-terrain-threshold", type=float, default=0.05)
    parser.add_argument("--approach-terrain-threshold", type=float, default=0.03)
    parser.add_argument(
        "--prelift-from-early",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="start front-leg residual at the early terrain gate",
    )
    parser.add_argument("--episode-seconds", type=float, default=6.0)
    parser.add_argument("--leg-rate-per-second", type=float, default=0.65)
    parser.add_argument(
        "--post-from-approach",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="start rear residual from approach when front phase has not latched",
    )
    parser.add_argument("--post-delay-seconds", type=float, default=0.25)
    return parser.parse_args()


def load_steering(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    steering = np.asarray(payload["steering"], dtype=np.float64)
    if steering.shape != (2,) or not np.isfinite(steering).all():
        raise ValueError(f"invalid steering vector in {path}: {steering}")
    return steering


def case_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if (args.lateral_grid is None) != (args.yaw_grid is None):
        raise ValueError("lateral-grid and yaw-grid must be supplied together")
    if args.lateral_grid is not None:
        lateral_grid = tuple(float(value) for value in args.lateral_grid.split(","))
        yaw_grid = tuple(float(value) for value in args.yaw_grid.split(","))
        if not lateral_grid or not yaw_grid:
            raise ValueError("custom lateral/yaw grids must be non-empty")
        seed_start = 45_000 if args.seed_start is None else args.seed_start
    elif args.suite == "training_grid":
        lateral_grid = LATERAL_GRID
        yaw_grid = YAW_GRID
        seed_start = 43_000 if args.seed_start is None else args.seed_start
    else:
        lateral_grid = (-0.0225, -0.0075, 0.0075, 0.0225)
        yaw_grid = (-2.25, -0.75, 0.75, 2.25)
        seed_start = 44_000 if args.seed_start is None else args.seed_start
    return [
        {
            "case_index": index,
            "depth_m": float(args.depth_m),
            "command_mps": float(args.command_mps),
            "lateral_m": lateral,
            "yaw_deg": yaw,
            "seed": seed_start + index,
        }
        for index, (yaw, lateral) in enumerate(
            (yaw, lateral) for yaw in yaw_grid for lateral in lateral_grid
        )
    ]


def selection_key(teacher: Teacher, report: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(teacher.complexity_rank),
        float(report["max_abs_roll_deg"]),
        float(report["max_contact_force_n"]) / 500.0,
        float(report["mean_residual_l2"]),
    )


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    reference_path = args.reference_checkpoint.expanduser().resolve()
    phase_path = args.phase_estimator.expanduser().resolve()
    base_rescue_path = args.base_rescue.expanduser().resolve()
    negative_path = args.negative_steering.expanduser().resolve()
    positive_path = args.positive_steering.expanduser().resolve()
    corner_rescue_path = args.corner_rescue.expanduser().resolve()
    corner_path = args.corner_steering.expanduser().resolve()
    case15_path = args.case15_steering.expanduser().resolve()
    case19_path = args.case19_steering.expanduser().resolve()
    case22_path = args.case22_steering.expanduser().resolve()
    case23_path = args.case23_steering.expanduser().resolve()
    manifest_path = (
        None
        if args.teacher_manifest is None
        else args.teacher_manifest.expanduser().resolve()
    )
    output = args.output_dir.expanduser().resolve()
    inputs = (
        (reference_path, phase_path, manifest_path)
        if manifest_path is not None
        else (
            reference_path,
            phase_path,
            base_rescue_path,
            negative_path,
            positive_path,
            corner_rescue_path,
            corner_path,
            case15_path,
            case19_path,
            case22_path,
            case23_path,
        )
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)

    before_assets = asset_hashes()
    torch.set_num_threads(1)
    agent = load_reference(reference_path)
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "s10_m20_combined_teacher_bank_v1":
            raise ValueError(f"unsupported teacher manifest: {manifest.get('format')}")
        entries = manifest.get("teachers")
        if not isinstance(entries, list) or not entries:
            raise ValueError("teacher manifest must contain a non-empty teachers list")
        teacher_list: list[Teacher] = []
        teacher_sources: dict[str, dict[str, Any]] = {}
        for item in entries:
            name = str(item["name"])
            path = Path(item["path"]).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if name in teacher_sources:
                raise ValueError(f"duplicate teacher name: {name}")
            teacher_list.append(
                Teacher(
                    name,
                    load_candidate(path),
                    load_steering(path),
                    int(item.get("complexity_rank", 0)),
                )
            )
            teacher_sources[name] = {
                "path": str(path),
                "sha256": sha256(path),
                "combined_candidate_steering": True,
            }
        teachers = tuple(teacher_list)
    else:
        base_rescue = load_candidate(base_rescue_path)
        corner_rescue = load_candidate(corner_rescue_path)
        zero = np.zeros(2, dtype=np.float64)
        negative = load_steering(negative_path)
        positive = load_steering(positive_path)
        corner = load_steering(corner_path)
        case15_steering = load_steering(case15_path)
        case19_steering = load_steering(case19_path)
        case22_steering = load_steering(case22_path)
        case23_steering = load_steering(case23_path)
        teachers = (
            Teacher("base_rescue", base_rescue, zero, 0),
            Teacher("corner_rescue", corner_rescue, zero, 1),
            Teacher("negative_steering", base_rescue, negative, 2),
            Teacher("positive_steering", base_rescue, positive, 2),
            Teacher("corner_steering", corner_rescue, corner, 3),
            Teacher("case15_corner_steering", corner_rescue, case15_steering, 3),
            Teacher("case19_steering", base_rescue, case19_steering, 3),
            Teacher("case22_steering", base_rescue, case22_steering, 3),
            Teacher("case23_steering", base_rescue, case23_steering, 3),
        )
        teacher_sources = {
            "base_rescue": {"path": str(base_rescue_path), "sha256": sha256(base_rescue_path)},
            "negative_steering": {"path": str(negative_path), "sha256": sha256(negative_path)},
            "positive_steering": {"path": str(positive_path), "sha256": sha256(positive_path)},
            "corner_rescue": {"path": str(corner_rescue_path), "sha256": sha256(corner_rescue_path)},
            "corner_steering": {"path": str(corner_path), "sha256": sha256(corner_path)},
            "case15_steering": {"path": str(case15_path), "sha256": sha256(case15_path)},
            "case19_steering": {"path": str(case19_path), "sha256": sha256(case19_path)},
            "case22_steering": {"path": str(case22_path), "sha256": sha256(case22_path)},
            "case23_steering": {"path": str(case23_path), "sha256": sha256(case23_path)},
        }
    if args.teacher_only is not None:
        teachers = tuple(
            teacher for teacher in teachers if teacher.name == args.teacher_only
        )
        if not teachers:
            raise ValueError(f"unknown teacher-only value: {args.teacher_only}")
    specs = case_specs(args)

    histories: list[np.ndarray] = []
    trigger_steps: list[int] = []
    for spec in specs:
        history, trigger_step = collect_trigger_history(
            agent,
            phase_path,
            spec,
            float(args.early_terrain_threshold),
        )
        if history is None or trigger_step is None:
            raise RuntimeError(f"early terrain gate did not trigger for grid case {spec}")
        histories.append(history)
        trigger_steps.append(trigger_step)

    def evaluate(item: tuple[dict[str, Any], Teacher]) -> dict[str, Any]:
        spec, teacher = item
        report = rollout(
            agent,
            phase_path,
            teacher.candidate,
            teacher.steering,
            label=f"grid_{spec['case_index']:02d}_{teacher.name}",
            depth_m=float(spec["depth_m"]),
            command_mps=float(spec["command_mps"]),
            initial_state=(float(spec["lateral_m"]), float(spec["yaw_deg"])),
            rollout_seed=int(spec["seed"]),
            early_threshold=float(args.early_terrain_threshold),
            approach_threshold=float(args.approach_terrain_threshold),
            prelift_from_early=bool(args.prelift_from_early),
            episode_seconds=float(args.episode_seconds),
            leg_rate_per_second=float(args.leg_rate_per_second),
            post_from_approach=bool(args.post_from_approach),
            post_delay_seconds=float(args.post_delay_seconds),
        )
        report["case_index"] = int(spec["case_index"])
        report["teacher"] = teacher.name
        return report

    work = [(spec, teacher) for spec in specs for teacher in teachers]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        evaluated = list(executor.map(evaluate, work))
    by_case: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(specs))}
    for report in evaluated:
        by_case[int(report["case_index"])].append(report)

    teacher_map = {teacher.name: teacher for teacher in teachers}
    cases: list[dict[str, Any]] = []
    selected_labels: list[str] = []
    selected_hard_failures: list[int] = []
    uncovered: list[int] = []
    for spec in specs:
        index = int(spec["case_index"])
        reports = by_case[index]
        safe_successes = [
            report
            for report in reports
            if bool(report["success"]) and not bool(report["hard_failure"])
        ]
        selected: dict[str, Any] | None = None
        if safe_successes:
            selected = min(
                safe_successes,
                key=lambda report: selection_key(
                    teacher_map[str(report["teacher"])], report
                ),
            )
            selected_labels.append(str(selected["teacher"]))
            if bool(selected["hard_failure"]):
                selected_hard_failures.append(index)
        else:
            uncovered.append(index)
            selected_labels.append("uncovered")
        cases.append(
            {
                **spec,
                "trigger_step": trigger_steps[index],
                "safe_success_teachers": sorted(
                    str(report["teacher"]) for report in safe_successes
                ),
                "selected_teacher": None if selected is None else selected["teacher"],
                "selected_report": selected,
                "teacher_reports": reports,
            }
        )
        print(
            f"[grid] case={index:02d} lateral={spec['lateral_m']:+.3f} "
            f"yaw={spec['yaw_deg']:+.1f} options={len(safe_successes)} "
            f"selected={selected_labels[-1]}",
            flush=True,
        )

    teacher_success_counts = {
        teacher.name: sum(
            bool(report["success"]) and not bool(report["hard_failure"])
            for report in evaluated
            if report["teacher"] == teacher.name
        )
        for teacher in teachers
    }
    selected_counts = {
        label: selected_labels.count(label) for label in sorted(set(selected_labels))
    }
    np.savez_compressed(
        output / "selector_training_data.npz",
        histories=np.stack(histories).astype(np.float32),
        labels=np.asarray(selected_labels),
        lateral_m=np.asarray([spec["lateral_m"] for spec in specs], dtype=np.float32),
        yaw_deg=np.asarray([spec["yaw_deg"] for spec in specs], dtype=np.float32),
        seeds=np.asarray([spec["seed"] for spec in specs], dtype=np.int64),
        trigger_steps=np.asarray(trigger_steps, dtype=np.int64),
    )
    after_assets = asset_hashes()
    if before_assets != after_assets:
        raise RuntimeError("official assets changed during teacher-bank grid review")
    summary = {
        "purpose": "teacher-bank coverage and selector-label collection",
        "geometry_scope": "official-derived diagonal vertical-wall proxy pit, not full official track mesh",
        "suite": args.suite,
        "depth_m": float(args.depth_m),
        "command_mps": float(args.command_mps),
        "grid": {
            "lateral_m": sorted({float(spec["lateral_m"]) for spec in specs}),
            "yaw_deg": sorted({float(spec["yaw_deg"]) for spec in specs}),
            "case_count": len(specs),
            "seed_start": int(specs[0]["seed"]),
        },
        "reference_checkpoint": str(reference_path),
        "reference_checkpoint_sha256": sha256(reference_path),
        "phase_estimator": str(phase_path),
        "phase_estimator_sha256": sha256(phase_path),
        "teacher_sources": teacher_sources,
        "teacher_manifest": None if manifest_path is None else str(manifest_path),
        "constraints": {
            "continuous_torque_limits_nm": {"leg": 50.0, "wheel": 14.0},
            "500_n": "report_only",
            "1200_n": "hard_failure",
            "roll_25_deg": "hard_failure",
            "body_wall_contact": "hard_failure",
            "early_terrain_threshold": float(args.early_terrain_threshold),
            "approach_terrain_threshold": float(args.approach_terrain_threshold),
            "prelift_from_early": bool(args.prelift_from_early),
            "episode_seconds": float(args.episode_seconds),
            "leg_rate_per_second": float(args.leg_rate_per_second),
            "post_from_approach": bool(args.post_from_approach),
            "post_delay_seconds": float(args.post_delay_seconds),
        },
        "teacher_safe_success_counts": teacher_success_counts,
        "selected_teacher_counts": selected_counts,
        "covered_count": len(specs) - len(uncovered),
        "uncovered_indexes": uncovered,
        "selected_hard_failure_indexes": selected_hard_failures,
        "promotion_gate": {
            "all_cases_covered": not uncovered,
            "selected_teacher_has_no_hard_failure": not selected_hard_failures,
            "passed": bool(not uncovered and not selected_hard_failures),
        },
        "cases": cases,
        "official_assets_modified": False,
        "official_asset_hashes_before": before_assets,
        "official_asset_hashes_after": after_assets,
        "files": {
            "selector_training_data": str(output / "selector_training_data.npz"),
            "summary": str(output / "summary.json"),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(
        f"[result] covered={summary['covered_count']}/{len(specs)} "
        f"counts={selected_counts} promote={int(summary['promotion_gate']['passed'])} "
        f"summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
