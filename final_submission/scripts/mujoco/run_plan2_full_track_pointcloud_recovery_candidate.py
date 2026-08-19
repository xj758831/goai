#!/usr/bin/env python3
"""Run the frozen full track with an isolated WP30 point-cloud recovery.

The existing diagnostic runner, route, pit expert, ROS navigator, model, and
policy are imported unchanged.  A bridge subclass keeps the baseline ROS
controller alive and overrides its body command only after the frozen shadow
monitor confirms ``WP30_STALL_RISK``.  Results are experimental and are
written separately from the inherited formal runner summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

import run_plan2_full_track_viewer_policy_diagnostic as diagnostic
import run_plan2_wp29_tail_state_viewer as sync_tail
from plan2_pointcloud_shadow_observer import Plan2PointCloudShadowObserver
from plan2_pointcloud_wp30_recovery_core import (
    WP30PointCloudRecovery,
    WP30RecoveryConfig,
)
from search_s10_m20_approach_two_stage_rescue import asset_hashes


runner = diagnostic.runner
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _candidate_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sim-sync", action="store_true")
    parser.add_argument("--sync-timeout-ms", type=float, default=100.0)
    parser.add_argument("--sync-rate-hz", type=float, default=20.0)
    parser.add_argument("--support-recovery-post-settle", action="store_true")
    parser.add_argument(
        "--disable-recovery-candidate",
        action="store_true",
        help=(
            "A/B baseline: keep sync and shadow observation live but do not pass "
            "advisories to the recovery controller"
        ),
    )
    parser.add_argument(
        "--lifecycle-smoke",
        action="store_true",
        help=(
            "open a real viewer and synchronized ROS bridge for a short cleanup "
            "integration test; never claims a full-track result"
        ),
    )
    parser.add_argument("--smoke-state", type=Path)
    parser.add_argument("--smoke-sim-seconds", type=float, default=0.25)
    parser.add_argument(
        "--clean-hard-exit",
        action="store_true",
        help=(
            "after all explicit cleanup and summary writes, preserve the runner "
            "return code with os._exit to avoid the local GLFW finalizer crash"
        ),
    )
    args, remaining = parser.parse_known_args()
    if (
        args.sync_timeout_ms <= 0.0
        or args.sync_rate_hz <= 0.0
        or args.smoke_sim_seconds <= 0.0
    ):
        raise ValueError("sync timeout and rate must be positive")
    if args.lifecycle_smoke and args.smoke_state is None:
        raise ValueError("--lifecycle-smoke requires --smoke-state")
    sys.argv = [sys.argv[0], *remaining]
    return args


def _output_dir() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args, _ = parser.parse_known_args()
    return args.output_dir.resolve()


CANDIDATE_OPTIONS = _candidate_options()
OUTPUT_DIR = _output_dir()
OBSERVER: Plan2PointCloudShadowObserver | None = None
LAST_BRIDGE: "CandidateRosTailBridge | None" = None
ACTIVE_MODEL: mujoco.MjModel | None = None
ORIGINAL_TRACK_RECORDER = runner.TrackRecorder
ORIGINAL_VISIBLE_RECORDER = runner.VisiblePhaseRecorder
ORIGINAL_ROS_BRIDGE = runner.RosTailBridge


def observer() -> Plan2PointCloudShadowObserver:
    global OBSERVER
    if OBSERVER is None:
        OBSERVER = Plan2PointCloudShadowObserver(
            OUTPUT_DIR / "pointcloud_shadow",
            PROJECT_ROOT / "config/goai_follow_pointcloud_mujoco.yaml",
        )
    return OBSERVER


class CandidateTrackRecorder(ORIGINAL_TRACK_RECORDER):
    def __init__(self, model: mujoco.MjModel, enabled: bool, period_s: float) -> None:
        global ACTIVE_MODEL
        ACTIVE_MODEL = model
        self.candidate_model = model
        observer()
        super().__init__(model, enabled, period_s)

    def capture(
        self,
        data: mujoco.MjData,
        base_body_id: int,
        *,
        phase: str,
        detail: str,
        force: bool = False,
    ) -> None:
        observer().sample(
            self.candidate_model,
            data,
            base_body_id,
            phase=phase,
            detail=detail,
        )
        super().capture(
            data,
            base_body_id,
            phase=phase,
            detail=detail,
            force=force,
        )


class CandidateVisiblePhaseRecorder(ORIGINAL_VISIBLE_RECORDER):
    def capture(self, env: Any) -> None:
        observer().sample(
            env.model,
            env.data,
            env.ids["base_body_id"],
            phase="PIT EXPERT",
            detail="point-cloud candidate observer; locked pit expert remains in control",
        )
        super().capture(env)


class CandidateRosTailBridge(ORIGINAL_ROS_BRIDGE):
    """Keep ROS navigation live while applying one bounded WP30 override."""

    def __init__(self, args: argparse.Namespace) -> None:
        global LAST_BRIDGE
        self.sim_sync = bool(CANDIDATE_OPTIONS.sim_sync)
        if self.sim_sync:
            if Path(args.tail_nav_script).name != "plan2_expert_tail_navigator_sim_sync.py":
                raise ValueError(
                    "--sim-sync requires plan2_expert_tail_navigator_sim_sync.py"
                )
            args.sim_sync = True
            args.sync_timeout_ms = CANDIDATE_OPTIONS.sync_timeout_ms
            args.sync_rate_hz = CANDIDATE_OPTIONS.sync_rate_hz
            args.start_index = 20
            args.initial_command = np.zeros(3, dtype=np.float32)
            sync_tail.IndexedRosTailBridge.__init__(self, args)
        else:
            super().__init__(args)
        controller_config = None
        if CANDIDATE_OPTIONS.support_recovery_post_settle:
            controller_config = WP30RecoveryConfig(
                total_timeout_s=25.0,
                enable_early_progress_retry=True,
                early_progress_check_s=2.0,
                early_progress_required_m=0.20,
                retry_reverse_speed=0.35,
                retry_reverse_duration_s=1.0,
                max_drive_retries=1,
                enable_near_marker_support_recovery=True,
                near_marker_shell_radius_m=0.18,
                support_settle_duration_s=0.50,
                support_settle_timeout_s=1.00,
                fallback_recenter_timeout_s=3.00,
                fallback_position_gain=1.35,
                fallback_yaw_gain=1.20,
                fallback_post_recenter_settle_duration_s=0.50,
                fallback_post_recenter_settle_timeout_s=1.00,
                fallback_align_timeout_s=3.00,
                max_support_recoveries=1,
            )
        self.controller = WP30PointCloudRecovery(controller_config)
        self.recovery_candidate_enabled = not bool(
            CANDIDATE_OPTIONS.disable_recovery_candidate
        )
        self.before_assets = asset_hashes()
        self.trace_path = Path(args.output_dir) / "recovery_candidate_trace.jsonl"
        self.trace_file = self.trace_path.open("w", encoding="utf-8", buffering=1)
        self.next_trace_time = -math.inf
        self.previous_phase = self.controller.phase
        self.max_baseline_abs = 0.0
        self.max_applied_abs = 0.0
        self.closed = False
        LAST_BRIDGE = self

    def step(self, data: mujoco.MjData, ids: dict[str, Any]) -> np.ndarray:
        if ACTIVE_MODEL is None:
            raise RuntimeError("candidate model was not registered by TrackRecorder")
        baseline = (
            sync_tail.IndexedRosTailBridge.step(self, data, ids)
            if self.sim_sync
            else super().step(data, ids)
        )
        position = np.asarray(data.xpos[ids["base_body_id"]], dtype=np.float64)
        yaw = runner.yaw_from_xmat(data.xmat[ids["base_body_id"]])
        advisories = (
            []
            if OBSERVER is None or not self.recovery_candidate_enabled
            else OBSERVER.decision_monitor.advisories
        )
        decision = self.controller.step(
            sim_time_s=float(data.time),
            position_xyz=position,
            yaw_rad=yaw,
            baseline_command=baseline,
            advisories=advisories,
            wheel_contact_count=int(
                np.count_nonzero(
                    runner.contact_metrics(ACTIVE_MODEL, data, ids)[
                        "wheel_contact"
                    ]
                    > 0
                )
            ),
        )
        applied = np.asarray(decision.command, dtype=np.float32)
        self.max_baseline_abs = max(
            self.max_baseline_abs, float(np.max(np.abs(baseline)))
        )
        self.max_applied_abs = max(
            self.max_applied_abs, float(np.max(np.abs(applied)))
        )
        if self.controller.phase != self.previous_phase:
            latest = self.controller.phase_history[-1]
            print(
                f"[wp30-candidate] phase={self.controller.phase} "
                f"sim={data.time:.3f}s base_z={position[2]:.3f} "
                f"reason={latest['reason']}",
                flush=True,
            )
            self.previous_phase = self.controller.phase
        if float(data.time) + 1.0e-12 >= self.next_trace_time:
            self.trace_file.write(
                json.dumps(
                    {
                        "sim_time_s": float(data.time),
                        "base_xyz_m": position.tolist(),
                        "yaw_deg": math.degrees(yaw),
                        "baseline_command": np.asarray(
                            baseline, dtype=np.float32
                        ).tolist(),
                        "applied_command": applied.tolist(),
                        "overridden": decision.overridden,
                        "phase": decision.phase,
                        "reason": decision.reason,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            self.next_trace_time = float(data.time) + 0.10
        return applied

    def _receive_synchronous_reply(
        self, expected_sequence: int, sim_time_s: float
    ) -> None:
        sync_tail.IndexedRosTailBridge._receive_synchronous_reply(
            self, expected_sequence, sim_time_s
        )

    def _write_summary(self) -> None:
        report = self.controller.summary()
        after_assets = asset_hashes()
        report.update(
            {
                "purpose": "isolated full-track WP30 point-cloud recovery candidate",
                "formal_baseline": False,
                "candidate_wrapper": str(Path(__file__).resolve()),
                "baseline_ros_navigator_kept_live": True,
                "max_abs_baseline_command": self.max_baseline_abs,
                "max_abs_applied_command": self.max_applied_abs,
                "trace_path": str(self.trace_path),
                "protected_assets_before": self.before_assets,
                "protected_assets_after": after_assets,
                "protected_assets_unchanged": self.before_assets == after_assets,
                "sim_sync": (
                    sync_tail.IndexedRosTailBridge.sync_summary(self)
                    if self.sim_sync else {"enabled": False}
                ),
                "support_recovery_post_settle_candidate": bool(
                    CANDIDATE_OPTIONS.support_recovery_post_settle
                ),
                "recovery_candidate_enabled": self.recovery_candidate_enabled,
                "clean_hard_exit_after_explicit_cleanup": bool(
                    CANDIDATE_OPTIONS.clean_hard_exit
                ),
                "navigator_return_code_after_close": self.nav_process.returncode,
                "sync_socket_paths": [self.parent_socket, self.child_socket],
                "sync_sockets_removed_after_close": not any(
                    Path(path).exists()
                    for path in (self.parent_socket, self.child_socket)
                ),
                "inherited_summary_warning": (
                    "summary.json is emitted by the unchanged formal runner and says "
                    "lidar_takeover_used=false; use this candidate summary for takeover state"
                ),
            }
        )
        (OUTPUT_DIR / "recovery_candidate_summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.sim_sync:
                sync_tail.IndexedRosTailBridge.close(self)
                sync_tail.IndexedRosTailBridge.write_sync_trace(
                    self, OUTPUT_DIR / "ros_command_trace.jsonl"
                )
            else:
                super().close()
        finally:
            self.trace_file.close()
            self._write_summary()


runner.TrackRecorder = CandidateTrackRecorder
runner.VisiblePhaseRecorder = CandidateVisiblePhaseRecorder
runner.RosTailBridge = CandidateRosTailBridge


def _finalize_candidate_summary() -> None:
    path = OUTPUT_DIR / "recovery_candidate_summary.json"
    if path.is_file():
        report = json.loads(path.read_text(encoding="utf-8"))
    else:
        report = {
            "purpose": "isolated full-track WP30 point-cloud recovery candidate",
            "formal_baseline": False,
            "experimental_control_takeover": False,
            "activated": False,
            "completed": False,
            "aborted": False,
            "phase": "tail_bridge_not_started",
        }
    formal_path = OUTPUT_DIR / "summary.json"
    if formal_path.is_file():
        formal = json.loads(formal_path.read_text(encoding="utf-8"))
        pit = formal.get("pit_expert") or {}
        report["full_track_result"] = {
            "summary_path": str(formal_path),
            "route_complete": bool(formal.get("route_complete", False)),
            "reached_score_count": formal.get("reached_score_count"),
            "last_reached_score": formal.get("last_reached_score"),
            "pit_expert_success": bool(pit.get("success", False)),
            "pit_depth_m": pit.get("depth_m"),
            "formal_summary_lidar_takeover_used": formal.get("lidar_takeover_used"),
        }
    report["pointcloud_samples"] = None if OBSERVER is None else OBSERVER.samples
    report["pointcloud_errors"] = None if OBSERVER is None else OBSERVER.errors
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def _run_lifecycle_smoke() -> int:
    """Exercise the real viewer/observer/sync bridge cleanup without a route claim."""
    args = runner.parse_args()
    if not CANDIDATE_OPTIONS.sim_sync:
        raise ValueError("--lifecycle-smoke requires --sim-sync")
    if not args.ros_tail:
        raise ValueError("--lifecycle-smoke requires --ros-tail")

    state_path = CANDIDATE_OPTIONS.smoke_state.resolve()
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    before_assets = asset_hashes()
    state = np.load(state_path)
    model = mujoco.MjModel.from_xml_path(str(runner.local_path(runner.MJCF_PATH)))
    model.opt.timestep = runner.DT
    runner.configure_continuous_limits(model)
    data = mujoco.MjData(model)
    ids = runner.model_ids(model)
    sync_tail.restore_state(model, data, state)

    args.output_dir = OUTPUT_DIR
    args.state = state_path
    args.start_index = 20
    args.initial_command = np.zeros(3, dtype=np.float32)
    recorder: CandidateTrackRecorder | None = None
    bridge: CandidateRosTailBridge | None = None
    viewer: Any = None
    viewer_opened = False
    viewer_closed = False
    start_time = float(data.time)
    wall_start = time.perf_counter()
    reason = "completed"
    try:
        recorder = CandidateTrackRecorder(model, False, 0.10)
        bridge = CandidateRosTailBridge(args)
        viewer = mujoco.viewer.launch_passive(model, data)
        runner.configure_viewer(viewer, data, ids)
        viewer_opened = bool(viewer.is_running())
        print(
            "[lifecycle-smoke] MuJoCo opened; testing synchronized ROS bridge "
            f"for {CANDIDATE_OPTIONS.smoke_sim_seconds:.3f}s simulated",
            flush=True,
        )
        step = 0
        while (
            float(data.time - start_time) < CANDIDATE_OPTIONS.smoke_sim_seconds
        ):
            if not viewer.is_running():
                reason = "viewer_closed_early"
                break
            bridge.step(data, ids)
            mujoco.mj_step(model, data)
            recorder.capture(
                data,
                ids["base_body_id"],
                phase="LIFECYCLE SMOKE",
                detail="no full-track claim; synchronized bridge cleanup test",
            )
            step += 1
            if step % 20 == 0:
                viewer.sync()
    finally:
        if bridge is not None:
            bridge.close()
        if recorder is not None:
            recorder.close()
        if viewer is not None:
            viewer.close()
            viewer_closed = True

    after_assets = asset_hashes()
    sync_summary = (
        {"enabled": False, "requests": 0, "timeouts": 0}
        if bridge is None
        else sync_tail.IndexedRosTailBridge.sync_summary(bridge)
    )
    socket_paths = [] if bridge is None else [bridge.parent_socket, bridge.child_socket]
    navigator_return_code = (
        None if bridge is None else bridge.nav_process.returncode
    )
    report = {
        "purpose": "candidate wrapper startup/cleanup integration test",
        "full_track_claim": False,
        "route_complete_claim": False,
        "state_path": str(state_path),
        "reason": reason,
        "viewer_opened": viewer_opened,
        "viewer_closed": viewer_closed,
        "elapsed_sim_seconds": float(data.time - start_time),
        "elapsed_wall_seconds": time.perf_counter() - wall_start,
        "ros_command_sync": sync_summary,
        "navigator_return_code_after_close": navigator_return_code,
        "sync_socket_paths": socket_paths,
        "sync_sockets_removed_after_close": not any(
            Path(path).exists() for path in socket_paths
        ),
        "pointcloud_samples_before_final_close": (
            0 if OBSERVER is None else OBSERVER.samples
        ),
        "pointcloud_errors_before_final_close": (
            0 if OBSERVER is None else OBSERVER.errors
        ),
        "protected_assets_unchanged": before_assets == after_assets,
        "clean_hard_exit_requested": bool(CANDIDATE_OPTIONS.clean_hard_exit),
        "recovery_candidate_enabled": bool(
            not CANDIDATE_OPTIONS.disable_recovery_candidate
        ),
    }
    report["internal_gate_passed"] = bool(
        report["reason"] == "completed"
        and report["viewer_opened"]
        and report["viewer_closed"]
        and sync_summary.get("requests", 0) > 0
        and sync_summary.get("timeouts", -1) == 0
        and navigator_return_code is not None
        and report["sync_sockets_removed_after_close"]
        and report["pointcloud_samples_before_final_close"] > 0
        and report["pointcloud_errors_before_final_close"] == 0
        and report["protected_assets_unchanged"]
    )
    (OUTPUT_DIR / "lifecycle_smoke_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[lifecycle-smoke] internal_gate={report['internal_gate_passed']} "
        f"sync={sync_summary.get('requests')}/{sync_summary.get('timeouts')} "
        f"nav_rc={navigator_return_code}",
        flush=True,
    )
    return 0 if report["internal_gate_passed"] else 5


def main() -> int:
    exit_code = 1
    try:
        exit_code = (
            _run_lifecycle_smoke()
            if CANDIDATE_OPTIONS.lifecycle_smoke
            else runner.main()
        )
        return exit_code
    finally:
        if diagnostic.DiagnosticOfficialPolicy.last_instance is not None:
            diagnostic.DiagnosticOfficialPolicy.last_instance.write_run_report()
        if OBSERVER is not None:
            OBSERVER.close()
        _finalize_candidate_summary()


if __name__ == "__main__":
    result = main()
    if CANDIDATE_OPTIONS.clean_hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(result)
    raise SystemExit(result)
