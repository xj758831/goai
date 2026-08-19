#!/usr/bin/env python3
"""Train a deployable estimator for the final approach to front support.

The positive label is a fixed causal-control window immediately before a real
front-top event.  MuJoCo contact is used only to construct offline labels;
inference uses five frames of the deployable 57+72+65 observation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import train_s10_m20_curriculum_ppo as base_runner
import train_s10_m20_lidar_curriculum_ppo as lidar_runner
from evaluate_s10_m20_lidar_pose_speed_expansion_candidates import case_specs
from s10_m20_terrain_lidar_balanced_stage_env import (
    ACTOR_OBS_DIM,
    S10M20TerrainLidarBalancedStageEnv,
)


DEFAULT_CHECKPOINT = Path(
    "logs/mujoco/"
    "s10_m20_lidar72_capability_depth021_complementary_distill_base10_alt1_lr1e6_"
    "epoch100_20260809_v1/checkpoint_best_distilled.pt"
)
HISTORY_FRAMES = 5
DWELL_STEPS = 3
THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lead-steps", type=int, default=40)
    parser.add_argument("--minimum-useful-lead-steps", type=int, default=20)
    parser.add_argument("--maximum-allowed-lead-steps", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser.parse_args()


class ApproachEstimator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation).squeeze(-1)


def load_reference(checkpoint: Path) -> Any:
    lidar_runner.configure_runner()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks model state")
    agent = lidar_runner.LidarInitializedActorCritic(action_std_scale=1.0)
    agent.load_state_dict(state, strict=True)
    agent.eval()
    return agent


def collect_case(agent: Any, spec: dict[str, Any]) -> tuple[np.ndarray, int | None]:
    state = (float(spec["lateral_m"]), float(spec["yaw_deg"]))
    env = S10M20TerrainLidarBalancedStageEnv(
        depth_m=float(spec["depth_m"]),
        command_mps=float(spec["command_mps"]),
        training_states=(state,),
    )
    features: list[np.ndarray] = []
    event_step: int | None = None
    try:
        actor_obs, _, info = env.reset(seed=int(spec["seed"]), initial_state=state)
        history = [actor_obs.copy() for _ in range(HISTORY_FRAMES)]
        for step in range(env.max_episode_steps):
            features.append(np.concatenate(history).astype(np.float32))
            if event_step is None and bool(info["metrics"]["front_top_latched"]):
                event_step = step
            with torch.inference_mode():
                action = (
                    agent.deterministic(
                        torch.as_tensor(actor_obs[:129], dtype=torch.float32).unsqueeze(0)
                    )
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            actor_obs, _, _, terminated, truncated, info = env.step(action)
            history = [*history[1:], actor_obs.copy()]
            if terminated or truncated:
                break
        return np.stack(features), event_step
    finally:
        env.close()


def labels_for_case(length: int, event_step: int | None, lead_steps: int) -> np.ndarray:
    labels = np.zeros(length, dtype=np.float32)
    if event_step is not None:
        start = max(0, event_step - lead_steps)
        labels[start : min(length, event_step + 1)] = 1.0
    return labels


def first_dwell_completion(probability: np.ndarray, threshold: float) -> int | None:
    above = probability >= threshold
    for index in range(DWELL_STEPS - 1, len(above)):
        if bool(np.all(above[index - DWELL_STEPS + 1 : index + 1])):
            return index
    return None


def threshold_report(
    probabilities: list[np.ndarray],
    event_steps: list[int | None],
    specs: list[dict[str, Any]],
    indexes: list[int],
    *,
    threshold: float,
    lead_steps: int,
    minimum_useful_lead_steps: int,
    maximum_allowed_lead_steps: int,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    false_latches = 0
    useful = 0
    detected = 0
    leads: list[int] = []
    for index in indexes:
        event = event_steps[index]
        window_start = max(0, event - lead_steps) if event is not None else None
        earliest_allowed = (
            max(0, event - maximum_allowed_lead_steps) if event is not None else None
        )
        prediction = first_dwell_completion(probabilities[index], threshold)
        false_before = prediction is not None and (
            event is None or prediction < int(earliest_allowed)
        )
        if false_before:
            false_latches += 1
        lead = event - prediction if event is not None and prediction is not None else None
        in_window = bool(
            event is not None
            and prediction is not None
            and int(earliest_allowed) <= prediction <= event
        )
        useful_lead = bool(in_window and lead is not None and lead >= minimum_useful_lead_steps)
        detected += int(in_window)
        useful += int(useful_lead)
        if in_window and lead is not None:
            leads.append(int(lead))
        reports.append(
            {
                "case_index": index,
                "role": specs[index]["role"],
                "depth_m": specs[index]["depth_m"],
                "event_step": event,
                "approach_window_start_step": window_start,
                "earliest_allowed_step": earliest_allowed,
                "predicted_dwell_completion_step": prediction,
                "false_before_window_or_no_event": bool(false_before),
                "lead_steps": lead,
                "detected_inside_window": in_window,
                "minimum_useful_lead": useful_lead,
            }
        )
    event_count = sum(event_steps[index] is not None for index in indexes)
    no_event_count = len(indexes) - event_count
    return {
        "threshold": threshold,
        "case_count": len(indexes),
        "event_case_count": event_count,
        "no_event_case_count": no_event_count,
        "false_latch_cases": false_latches,
        "detected_inside_window_count": detected,
        "minimum_useful_lead_count": useful,
        "lead_steps_min": min(leads) if leads else None,
        "lead_steps_mean": float(np.mean(leads)) if leads else None,
        "lead_steps_max": max(leads) if leads else None,
        "cases": reports,
    }


def choose_threshold(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        reports,
        key=lambda report: (
            int(report["false_latch_cases"] == 0),
            -int(report["false_latch_cases"]),
            int(report["minimum_useful_lead_count"]),
            int(report["detected_inside_window_count"]),
            float(report["lead_steps_mean"] or -1.0),
            float(report["threshold"]),
        ),
    )


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.hidden_dim < 4 or args.batch_size <= 0:
        raise ValueError("invalid training dimensions")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if (
        args.lead_steps <= args.minimum_useful_lead_steps
        or args.minimum_useful_lead_steps < 1
        or args.maximum_allowed_lead_steps < args.lead_steps
    ):
        raise ValueError("lead window must exceed the positive minimum useful lead")
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    agent = load_reference(checkpoint)
    specs = case_specs()
    collected = [collect_case(agent, spec) for spec in specs]
    features = [item[0] for item in collected]
    event_steps = [item[1] for item in collected]
    labels = [labels_for_case(len(item), event, args.lead_steps) for item, event in zip(features, event_steps)]
    validation_indexes = [index for index in range(len(features)) if index % 4 == 0]
    train_indexes = [index for index in range(len(features)) if index not in validation_indexes]
    train_x = np.concatenate([features[index] for index in train_indexes])
    train_y = np.concatenate([labels[index] for index in train_indexes])
    mean = train_x.mean(axis=0).astype(np.float32)
    std = np.maximum(train_x.std(axis=0), 1.0e-3).astype(np.float32)
    train_x_t = torch.as_tensor((train_x - mean) / std, dtype=torch.float32)
    train_y_t = torch.as_tensor(train_y, dtype=torch.float32)
    positive = float(train_y_t.sum())
    negative = float(len(train_y_t) - positive)
    pos_weight = torch.tensor(min(20.0, max(1.0, negative / max(positive, 1.0))))

    estimator = ApproachEstimator(ACTOR_OBS_DIM * HISTORY_FRAMES, args.hidden_dim)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = min(args.batch_size, len(train_x_t))
    generator = torch.Generator().manual_seed(args.seed)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(len(train_x_t), generator=generator)
        loss_sum = 0.0
        estimator.train()
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start : start + batch_size]
            loss = loss_fn(estimator(train_x_t[batch]), train_y_t[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(estimator.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            history.append({"epoch": epoch, "loss": loss_sum / len(train_x_t)})
            print(f"[approach] epoch={epoch}/{args.epochs} loss={history[-1]['loss']:.6f}", flush=True)

    estimator.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for case_features in features:
            normalized = torch.as_tensor((case_features - mean) / std, dtype=torch.float32)
            probabilities.append(torch.sigmoid(estimator(normalized)).numpy())
    validation_reports = [
        threshold_report(
            probabilities,
            event_steps,
            specs,
            validation_indexes,
            threshold=threshold,
            lead_steps=args.lead_steps,
            minimum_useful_lead_steps=args.minimum_useful_lead_steps,
            maximum_allowed_lead_steps=args.maximum_allowed_lead_steps,
        )
        for threshold in THRESHOLDS
    ]
    selected = choose_threshold(validation_reports)
    training_selected = threshold_report(
        probabilities,
        event_steps,
        specs,
        train_indexes,
        threshold=float(selected["threshold"]),
        lead_steps=args.lead_steps,
        minimum_useful_lead_steps=args.minimum_useful_lead_steps,
        maximum_allowed_lead_steps=args.maximum_allowed_lead_steps,
    )
    deployable_gate_pass = bool(
        selected["false_latch_cases"] == 0
        and selected["minimum_useful_lead_count"] == selected["event_case_count"]
    )
    model_path = output / "approach_estimator.pt"
    torch.save(
        {
            "format": "s10_deployable_approach_estimator_v1",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": base_runner.sha256(checkpoint),
            "history_frames": HISTORY_FRAMES,
            "actor_observation_dim": ACTOR_OBS_DIM,
            "lead_steps": args.lead_steps,
            "minimum_useful_lead_steps": args.minimum_useful_lead_steps,
            "maximum_allowed_lead_steps": args.maximum_allowed_lead_steps,
            "dwell_steps": DWELL_STEPS,
            "selected_threshold": selected["threshold"],
            "mean": torch.as_tensor(mean),
            "std": torch.as_tensor(std),
            "model": estimator.state_dict(),
        },
        model_path,
    )
    summary = {
        "purpose": "group-held-out deployable final-approach estimator",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": base_runner.sha256(checkpoint),
        "official_assets_modified": False,
        "history_frames": HISTORY_FRAMES,
        "actor_observation_dim": ACTOR_OBS_DIM,
        "lead_steps": args.lead_steps,
        "lead_seconds": args.lead_steps * 0.02,
        "minimum_useful_lead_steps": args.minimum_useful_lead_steps,
        "minimum_useful_lead_seconds": args.minimum_useful_lead_steps * 0.02,
        "maximum_allowed_lead_steps": args.maximum_allowed_lead_steps,
        "maximum_allowed_lead_seconds": args.maximum_allowed_lead_steps * 0.02,
        "dwell_steps": DWELL_STEPS,
        "case_count": len(features),
        "train_case_indexes": train_indexes,
        "validation_case_indexes": validation_indexes,
        "event_steps": event_steps,
        "positive_transitions_train": int(positive),
        "negative_transitions_train": int(negative),
        "history": history,
        "validation_reports": validation_reports,
        "selected_validation_report": selected,
        "training_report_at_selected_threshold": training_selected,
        "deployable_gate_pass": deployable_gate_pass,
        "deployable": "five-frame causal 57+72+65 observation only; MuJoCo contact used only for offline labels",
        "files": {"model": str(model_path), "summary": str(output / "summary.json")},
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="ascii"
    )
    print(
        f"[result] threshold={selected['threshold']:.2f} false={selected['false_latch_cases']} "
        f"useful={selected['minimum_useful_lead_count']}/{selected['event_case_count']} "
        f"gate_pass={int(deployable_gate_pass)} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
