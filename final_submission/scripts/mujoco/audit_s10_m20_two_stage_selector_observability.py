#!/usr/bin/env python3
"""Audit whether causal terrain lidar can select a two-stage rescue teacher.

The audit never changes the reference action.  It records the five terrain-
lidar frames available when the terrain approach gate latches, derives teacher
labels from an existing 19-case review, and reports leave-one-case-out results
for small regularized linear selectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluate_s10_m20_lidar_pose_speed_expansion_candidates import case_specs
from search_s10_m20_approach_two_stage_rescue import (
    PHASE_ESTIMATOR,
    REFERENCE_CHECKPOINT,
    S10M20PhaseGatedEnv,
    asset_hashes,
    load_reference,
    sha256,
)


DEFAULT_REVIEW = Path(
    "logs/mujoco/"
    "s10_m20_approach_teacher_fixed021_index1_terrain_review_19cases_20260810_v1/"
    "summary.json"
)
TERRAIN_START = 129
TERRAIN_STOP = 194
TERRAIN_CHANNELS = 5
TERRAIN_HORIZONTAL_BEAMS = 13
HISTORY_FRAMES = 5
DWELL_STEPS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, default=REFERENCE_CHECKPOINT)
    parser.add_argument("--phase-estimator", type=Path, default=PHASE_ESTIMATOR)
    parser.add_argument("--review-summary", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--candidate-label", default="iteration_01_best")
    parser.add_argument("--terrain-approach-threshold", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def collect_trigger_history(
    agent: Any,
    phase_path: Path,
    spec: dict[str, Any],
    threshold: float,
) -> tuple[np.ndarray | None, int | None]:
    state = (float(spec["lateral_m"]), float(spec["yaw_deg"]))
    S10M20PhaseGatedEnv.phase_estimator_path = phase_path
    env = S10M20PhaseGatedEnv(
        depth_m=float(spec["depth_m"]),
        command_mps=float(spec["command_mps"]),
        training_states=(state,),
    )
    try:
        actor_obs, _, _ = env.reset(seed=int(spec["seed"]), initial_state=state)
        scan = actor_obs[TERRAIN_START:TERRAIN_STOP].copy()
        history = [scan.copy() for _ in range(HISTORY_FRAMES)]
        streak = 0
        for step in range(env.max_episode_steps):
            scan = actor_obs[TERRAIN_START:TERRAIN_STOP].copy()
            history = [*history[1:], scan]
            streak = streak + 1 if float(np.min(scan)) <= threshold else 0
            if streak >= DWELL_STEPS:
                return np.stack(history).astype(np.float64), step
            with torch.inference_mode():
                action = (
                    agent.deterministic(
                        torch.as_tensor(actor_obs[:TERRAIN_START], dtype=torch.float32).unsqueeze(0)
                    )
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            actor_obs, _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    return None, None


def scan_moments(scan: np.ndarray) -> np.ndarray:
    grid = np.asarray(scan, dtype=np.float64).reshape(
        TERRAIN_CHANNELS, TERRAIN_HORIZONTAL_BEAMS
    )
    return np.concatenate(
        [
            grid.mean(axis=1),
            grid.min(axis=1),
            grid.max(axis=1),
            np.quantile(scan, [0.10, 0.25, 0.50, 0.75, 0.90]),
            grid.mean(axis=0),
        ]
    )


def feature_sets(history: np.ndarray) -> dict[str, np.ndarray]:
    moments = np.stack([scan_moments(scan) for scan in history])
    sorted_scans = np.sort(history, axis=1)
    return {
        "moments_current": moments[-1],
        "moments_history": np.concatenate(
            [moments[-1], moments.mean(axis=0), moments[-1] - moments[0]]
        ),
        "sorted_current": sorted_scans[-1],
        "sorted_history": np.concatenate(
            [
                sorted_scans[-1],
                sorted_scans.mean(axis=0),
                sorted_scans[-1] - sorted_scans[0],
            ]
        ),
        "raw_history": history.reshape(-1),
    }


def fit_ridge(
    features: np.ndarray, labels: np.ndarray, regularization: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.maximum(std, 1.0e-6)
    normalized = (features - mean) / std
    design = np.concatenate([normalized, np.ones((len(normalized), 1))], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ labels)
    return mean, std, weights


def predict_ridge(
    features: np.ndarray, mean: np.ndarray, std: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    normalized = (features - mean) / std
    design = np.concatenate([normalized, np.ones((len(normalized), 1))], axis=1)
    return design @ weights


def leave_one_case_out(
    features: np.ndarray, labels: np.ndarray, regularization: float
) -> dict[str, Any]:
    scores = np.zeros(len(labels), dtype=np.float64)
    for held_out in range(len(labels)):
        keep = np.arange(len(labels)) != held_out
        mean, std, weights = fit_ridge(features[keep], labels[keep], regularization)
        scores[held_out] = predict_ridge(
            features[held_out : held_out + 1], mean, std, weights
        )[0]
    predicted = scores >= 0.0
    positive = labels > 0.0
    return {
        "regularization": regularization,
        "accuracy": float(np.mean(predicted == positive)),
        "true_positive_count": int(np.sum(predicted & positive)),
        "false_positive_count": int(np.sum(predicted & ~positive)),
        "true_negative_count": int(np.sum(~predicted & ~positive)),
        "false_negative_count": int(np.sum(~predicted & positive)),
        "scores": scores.tolist(),
        "predicted_alternate": predicted.tolist(),
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.terrain_approach_threshold < 1.0:
        raise ValueError("terrain approach threshold must be in (0, 1)")
    reference_path = args.reference_checkpoint.expanduser().resolve()
    phase_path = args.phase_estimator.expanduser().resolve()
    review_path = args.review_summary.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (reference_path, phase_path, review_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=False)

    before_assets = asset_hashes()
    torch.set_num_threads(1)
    agent = load_reference(reference_path)
    specs = case_specs()
    review = json.loads(review_path.read_text(encoding="utf-8"))
    candidates = {
        candidate["label"]: candidate for candidate in review["candidates"]
    }
    if args.candidate_label not in candidates:
        raise KeyError(f"candidate not found in review: {args.candidate_label}")
    candidate = candidates[args.candidate_label]
    reference_cases = review["reference"]["cases"]
    candidate_cases = candidate["cases"]
    if len(specs) != len(reference_cases) or len(specs) != len(candidate_cases):
        raise RuntimeError("review case count differs from fixed case specification")

    histories: list[np.ndarray] = []
    trigger_steps: list[int | None] = []
    for index, spec in enumerate(specs):
        history, trigger_step = collect_trigger_history(
            agent, phase_path, spec, float(args.terrain_approach_threshold)
        )
        histories.append(
            history
            if history is not None
            else np.full(
                (HISTORY_FRAMES, TERRAIN_STOP - TERRAIN_START),
                np.nan,
                dtype=np.float64,
            )
        )
        trigger_steps.append(trigger_step)
        print(
            f"[collect] case={index:02d} role={spec['role']} depth={spec['depth_m']:.3f} "
            f"trigger={trigger_step if trigger_step is not None else 'none'}",
            flush=True,
        )

    alternate_only = [
        index
        for index, (reference, alternate) in enumerate(
            zip(reference_cases, candidate_cases, strict=True)
        )
        if bool(alternate["success"]) and not bool(reference["success"])
    ]
    reference_only = [
        index
        for index, (reference, alternate) in enumerate(
            zip(reference_cases, candidate_cases, strict=True)
        )
        if bool(reference["success"]) and not bool(alternate["success"])
    ]
    decisive_indexes = alternate_only + reference_only
    missing_decisive = [index for index in decisive_indexes if trigger_steps[index] is None]
    if missing_decisive:
        raise RuntimeError(
            f"teacher outcomes differ despite no terrain trigger: {missing_decisive}"
        )
    labels = np.asarray(
        [1.0 if index in alternate_only else -1.0 for index in decisive_indexes],
        dtype=np.float64,
    )
    all_feature_sets = [
        feature_sets(history) if trigger_steps[index] is not None else None
        for index, history in enumerate(histories)
    ]
    regularizations = (0.01, 0.1, 1.0, 10.0, 100.0)
    models: list[dict[str, Any]] = []
    feature_names = tuple(feature_sets(histories[decisive_indexes[0]]))
    for feature_name in feature_names:
        features = np.stack(
            [
                all_feature_sets[index][feature_name]
                for index in decisive_indexes
                if all_feature_sets[index] is not None
            ]
        )
        for regularization in regularizations:
            report = leave_one_case_out(features, labels, regularization)
            report["feature_name"] = feature_name
            report["feature_dim"] = int(features.shape[1])
            models.append(report)
    best = max(
        models,
        key=lambda report: (
            report["accuracy"],
            -report["false_positive_count"],
            report["true_positive_count"],
            -report["feature_dim"],
            report["regularization"],
        ),
    )
    best_features = np.stack(
        [
            all_feature_sets[index][best["feature_name"]]
            for index in decisive_indexes
            if all_feature_sets[index] is not None
        ]
    )
    mean, std, weights = fit_ridge(
        best_features, labels, float(best["regularization"])
    )
    np.savez_compressed(
        output / "selector_gate.npz",
        feature_name=np.asarray(best["feature_name"]),
        mean=mean,
        std=std,
        weights=weights,
        candidate=np.asarray(candidate["candidate"], dtype=np.float64),
        decisive_indexes=np.asarray(decisive_indexes, dtype=np.int64),
        labels=labels,
    )
    np.savez_compressed(
        output / "trigger_histories.npz",
        histories=np.stack(histories),
        trigger_steps=np.asarray(
            [-1 if step is None else step for step in trigger_steps], dtype=np.int64
        ),
    )
    after_assets = asset_hashes()
    if before_assets != after_assets:
        raise RuntimeError("official assets changed during selector audit")
    summary = {
        "purpose": "causal terrain-lidar observability audit for two-stage teacher selection",
        "reference_checkpoint": str(reference_path),
        "reference_checkpoint_sha256": sha256(reference_path),
        "phase_estimator": str(phase_path),
        "phase_estimator_sha256": sha256(phase_path),
        "review_summary": str(review_path),
        "review_summary_sha256": sha256(review_path),
        "candidate_label": args.candidate_label,
        "candidate_vector": candidate["candidate"],
        "terrain_approach_threshold": float(args.terrain_approach_threshold),
        "case_count": len(specs),
        "alternate_only_indexes": alternate_only,
        "reference_only_indexes": reference_only,
        "decisive_indexes": decisive_indexes,
        "trigger_steps": trigger_steps,
        "models": models,
        "best_model": best,
        "promotion_gate": {
            "requirement": "zero leave-one-case-out false positives and false negatives on decisive cases",
            "passed": bool(
                best["false_positive_count"] == 0
                and best["false_negative_count"] == 0
            ),
        },
        "official_assets_modified": False,
        "official_asset_hashes_before": before_assets,
        "official_asset_hashes_after": after_assets,
        "files": {
            "selector_gate": str(output / "selector_gate.npz"),
            "trigger_histories": str(output / "trigger_histories.npz"),
            "summary": str(output / "summary.json"),
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(
        f"[result] feature={best['feature_name']} lambda={best['regularization']} "
        f"accuracy={best['accuracy']:.3f} fp={best['false_positive_count']} "
        f"fn={best['false_negative_count']} pass={int(summary['promotion_gate']['passed'])} "
        f"summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
