#!/usr/bin/env python3
"""Restore a natural WP29 state and visibly test the unchanged Plan2 tail."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any
import uuid

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np

import run_plan2_full_track_viewer as full
from probe_s10_official_policy_shallow import OfficialPolicy
from s10_fronthook_scripted_probe import contact_metrics, diverged, model_ids
from search_s10_m20_approach_two_stage_rescue import asset_hashes, sha256
from search_s10_m20_front_hook_pull import local_path


DT = 0.001
CONTROL_PERIOD_S = 0.02
OFFICIAL_RADIUS_M = 0.20
START_SCORE = 29
FINAL_SCORE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sim-seconds", type=float, default=120.0)
    parser.add_argument("--viewer-speed", type=float, default=4.0)
    parser.add_argument("--ros-domain", type=int, required=True)
    parser.add_argument("--start-index", type=int, default=38)
    parser.add_argument("--tail-nav-script", type=Path, required=True)
    parser.add_argument("--tail-behavior-config", type=Path, required=True)
    parser.add_argument("--tail-waypoints", type=Path, required=True)
    parser.add_argument(
        "--sim-sync", action="store_true",
        help="audit-only odom->command handshake; unchanged default is asynchronous",
    )
    parser.add_argument("--sync-timeout-ms", type=float, default=100.0)
    parser.add_argument("--sync-rate-hz", type=float, default=20.0)
    parser.add_argument(
        "--clean-hard-exit",
        action="store_true",
        help="after summaries are flushed, skip the known GLFW destructor crash",
    )
    return parser.parse_args()


class IndexedRosTailBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        self.next_publish_time = -math.inf
        sim_sync = bool(getattr(args, "sim_sync", False))
        sync_rate_hz = float(getattr(args, "sync_rate_hz", 20.0))
        sync_timeout_ms = float(getattr(args, "sync_timeout_ms", 100.0))
        self.publish_period = (
            1.0 / sync_rate_hz if sim_sync else CONTROL_PERIOD_S
        )
        self.sim_sync = sim_sync
        self.sync_timeout_s = sync_timeout_ms * 1.0e-3
        self.sequence = 0
        self.sync_requests = 0
        self.sync_timeouts = 0
        self.sync_wait_seconds = 0.0
        self.sync_max_wait_seconds = 0.0
        self.sync_trace: list[dict[str, Any]] = []
        initial_command = getattr(args, "initial_command", None)
        if initial_command is None:
            initial_command = np.load(args.state)["command"]
        self.command = np.asarray(initial_command, dtype=np.float32).copy()
        token = uuid.uuid4().hex
        self.sync_ack_topic = f"/plan2_sim_sync_ack_{token}"
        self.parent_socket = f"/tmp/plan2_ros_tail_parent_{token}.sock"
        self.child_socket = f"/tmp/plan2_ros_tail_child_{token}.sock"
        self.nav_log = args.output_dir / "ros_tail_navigator.log"
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(args.ros_domain)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(self.parent_socket)
        self.socket.settimeout(0.0)
        helper = Path(__file__).with_name("ros_tail_bridge_indexed.py")
        helper_command = [
                "/usr/bin/python3",
                str(helper),
                "--parent-socket",
                self.parent_socket,
                "--child-socket",
                self.child_socket,
                "--nav-script",
                str(args.tail_nav_script),
                "--behavior-config",
                str(args.tail_behavior_config),
                "--waypoints",
                str(args.tail_waypoints),
                "--nav-log",
                str(self.nav_log),
                "--ros-domain",
                str(args.ros_domain),
                "--start-index",
                str(args.start_index),
            ]
        if self.sim_sync:
            helper_command.extend(
                [
                    "--sync-ack-topic",
                    self.sync_ack_topic,
                    "--sync-timeout-ms",
                    str(sync_timeout_ms),
                ]
            )
        self.nav_process = subprocess.Popen(
            helper_command,
            cwd=str(helper.parent.parent.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + (8.0 if self.sim_sync else 3.0)
        while not Path(self.child_socket).exists() and time.monotonic() < deadline:
            if self.nav_process.poll() is not None:
                raise RuntimeError(
                    "indexed ROS bridge exited during startup with code "
                    f"{self.nav_process.returncode}"
                )
            time.sleep(0.02)
        if not Path(self.child_socket).exists():
            raise RuntimeError("indexed ROS bridge did not create its socket")

    def step(self, data: mujoco.MjData, ids: dict[str, Any]) -> np.ndarray:
        if float(data.time) + 1.0e-12 >= self.next_publish_time:
            stamp = float(data.time)
            self.sequence += 1
            state = {
                "sequence": self.sequence,
                "time": stamp,
                "held_command": self.command.tolist(),
                "position": np.asarray(
                    data.xpos[ids["base_body_id"]], dtype=np.float64
                ).tolist(),
                "quaternion": np.asarray(
                    data.xquat[ids["base_body_id"]], dtype=np.float64
                ).tolist(),
                "velocity": np.asarray(data.qvel[:3], dtype=np.float64).tolist(),
            }
            self.socket.sendto(
                json.dumps(state, separators=(",", ":")).encode("utf-8"),
                self.child_socket,
            )
            self.next_publish_time = stamp + self.publish_period
            if self.sim_sync:
                self._receive_synchronous_reply(self.sequence, stamp)
                return self.command.copy()
        try:
            while True:
                reply, _ = self.socket.recvfrom(4096)
                self.command[:] = json.loads(reply.decode("utf-8"))["command"]
        except BlockingIOError:
            pass
        return self.command.copy()

    def _receive_synchronous_reply(
        self, expected_sequence: int, sim_time_s: float
    ) -> None:
        started = time.monotonic()
        deadline = started + self.sync_timeout_s
        self.sync_requests += 1
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            self.socket.settimeout(remaining)
            try:
                reply, _ = self.socket.recvfrom(4096)
            except socket.timeout:
                break
            document = json.loads(reply.decode("utf-8"))
            if document.get("sequence") != expected_sequence:
                continue
            elapsed = time.monotonic() - started
            self.sync_wait_seconds += elapsed
            self.sync_max_wait_seconds = max(self.sync_max_wait_seconds, elapsed)
            self.socket.settimeout(0.0)
            if not document.get("sync_ok", False):
                self.sync_timeouts += 1
                raise RuntimeError(
                    f"no odom-correlated command for sequence {expected_sequence}"
                )
            self.command[:] = document["command"]
            self.sync_trace.append(
                {
                    "sequence": expected_sequence,
                    "sim_time_s": sim_time_s,
                    "command": self.command.tolist(),
                    "command_generation": document.get("command_generation"),
                    "command_published": document.get("command_published"),
                    "ack_stamp_ns": document.get("ack_stamp_ns"),
                    "wait_ms": elapsed * 1.0e3,
                }
            )
            return
        elapsed = time.monotonic() - started
        self.sync_wait_seconds += elapsed
        self.sync_max_wait_seconds = max(self.sync_max_wait_seconds, elapsed)
        self.sync_timeouts += 1
        self.socket.settimeout(0.0)
        raise RuntimeError(
            f"bridge reply timeout for sequence {expected_sequence}"
        )

    def sync_summary(self) -> dict[str, Any]:
        return {
            "enabled": self.sim_sync,
            "state_rate_hz": 1.0 / self.publish_period,
            "timeout_ms": self.sync_timeout_s * 1.0e3,
            "requests": self.sync_requests,
            "timeouts": self.sync_timeouts,
            "mean_wait_ms": (
                self.sync_wait_seconds / self.sync_requests * 1.0e3
                if self.sync_requests else 0.0
            ),
            "max_wait_ms": self.sync_max_wait_seconds * 1.0e3,
        }

    def write_sync_trace(self, path: Path) -> None:
        if not self.sim_sync:
            return
        path.write_text(
            "".join(
                json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n"
                for row in self.sync_trace
            ),
            encoding="utf-8",
        )

    def close(self) -> None:
        try:
            self.socket.sendto(b"__stop__", self.child_socket)
        except OSError:
            pass
        if self.nav_process.poll() is None:
            self.nav_process.send_signal(signal.SIGINT)
            try:
                self.nav_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.nav_process.terminate()
                try:
                    self.nav_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.nav_process.kill()
                    self.nav_process.wait(timeout=2.0)
        self.socket.close()
        for path in (self.parent_socket, self.child_socket):
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass


def restore_state(model: mujoco.MjModel, data: mujoco.MjData, state: Any) -> None:
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    if data.act.size:
        data.act[:] = state["act"]
    data.ctrl[:] = state["ctrl"]
    data.qacc_warmstart[:] = state["qacc_warmstart"]
    data.time = float(state["time"])
    mujoco.mj_forward(model, data)


def main() -> int:
    args = parse_args()
    if args.max_sim_seconds <= 0.0 or args.viewer_speed <= 0.0:
        raise ValueError("simulation duration and viewer speed must be positive")
    if args.sync_timeout_ms <= 0.0 or args.sync_rate_hz <= 0.0:
        raise ValueError("sync timeout and rate must be positive")
    args.state = local_path(args.state)
    args.output_dir = local_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    before_assets = asset_hashes()
    state = np.load(args.state)
    scoring_route = full.PROJECT_ROOT / "config" / args.tail_waypoints.name
    route = full.load_route(scoring_route)
    score_positions = {
        point.score_index: point.xyz.copy()
        for point in route
        if point.score_index is not None
    }

    model = mujoco.MjModel.from_xml_path(str(local_path(full.MJCF_PATH)))
    model.opt.timestep = DT
    full.configure_continuous_limits(model)
    data = mujoco.MjData(model)
    ids = model_ids(model)
    restore_state(model, data, state)
    policy = OfficialPolicy()
    policy.last_action = np.asarray(state["last_action"], dtype=np.float32).copy()
    policy.position_target = np.asarray(
        state["position_target"], dtype=np.float64
    ).copy()
    policy.velocity_target = np.asarray(
        state["velocity_target"], dtype=np.float64
    ).copy()

    reached_scores = [START_SCORE]
    next_score = START_SCORE + 1
    reason = "time_limit"
    score_events: list[dict[str, Any]] = []
    command = np.asarray(state["command"], dtype=np.float32).copy()
    start_time = float(data.time)
    wall_start = time.perf_counter()
    next_control_time = float(data.time)
    next_status_time = float(data.time) + 2.0
    max_force = 0.0
    max_roll = 0.0
    bridge = IndexedRosTailBridge(args)
    viewer = mujoco.viewer.launch_passive(model, data)
    full.configure_viewer(viewer, data, ids)
    print(
        f"[wp29-tail] MuJoCo opened at natural WP29 state; "
        f"path_start={args.start_index} viewer_speed={args.viewer_speed:.1f}x",
        flush=True,
    )
    try:
        step = 0
        sync_stride = max(20, int(round(args.viewer_speed * 20.0)))
        while float(data.time - start_time) < args.max_sim_seconds:
            if not viewer.is_running():
                reason = "viewer_closed"
                break
            base = data.xpos[ids["base_body_id"]].copy()
            if next_score <= FINAL_SCORE:
                distance = float(
                    np.linalg.norm(base[:2] - score_positions[next_score][:2])
                )
                if distance <= OFFICIAL_RADIUS_M:
                    reached_scores.append(next_score)
                    score_events.append(
                        {
                            "score": next_score,
                            "sim_time_s": float(data.time),
                            "distance_m": distance,
                            "base_xyz_m": base.tolist(),
                        }
                    )
                    print(
                        f"[wp29-tail] reached score={next_score} "
                        f"distance={distance:.3f}m sim={data.time:.3f}s",
                        flush=True,
                    )
                    next_score += 1
                    if next_score > FINAL_SCORE:
                        reason = "route_complete"
                        break
            try:
                if float(data.time) + 1.0e-12 >= next_control_time:
                    next_control_time = float(data.time) + CONTROL_PERIOD_S
                    command = bridge.step(data, ids)
                    policy.update(model, data, ids, command)
                else:
                    command = bridge.step(data, ids)
            except RuntimeError as exc:
                reason = f"ros_sync_error:{exc}"
                print(f"[wp29-tail] {reason}", flush=True)
                break
            policy.apply(data)
            mujoco.mj_step(model, data)
            step += 1
            contacts = contact_metrics(model, data, ids)
            max_force = max(max_force, float(contacts["max_contact_force"]))
            rotation = np.asarray(data.xmat[ids["base_body_id"]]).reshape(3, 3)
            roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
            max_roll = max(max_roll, abs(roll))
            bad = diverged(data)
            if bad is not None:
                reason = bad
                break
            if float(data.time) + 1.0e-12 >= next_status_time:
                next_status_time = float(data.time) + 2.0
                target = min(next_score, FINAL_SCORE)
                distance = float(
                    np.linalg.norm(
                        data.xpos[ids["base_body_id"]][:2]
                        - score_positions[target][:2]
                    )
                )
                print(
                    f"[wp29-tail] heartbeat sim={data.time:.2f}s "
                    f"base=({base[0]:.2f},{base[1]:.2f},{base[2]:.2f}) "
                    f"cmd=({command[0]:.2f},{command[1]:.2f},{command[2]:.2f}) "
                    f"target={target} distance={distance:.2f}m",
                    flush=True,
                )
            if step % sync_stride == 0 and not full.sync_visible(
                viewer,
                sim_elapsed_s=float(data.time - start_time),
                wall_start_s=wall_start,
                viewer_speed=args.viewer_speed,
            ):
                reason = "viewer_closed"
                break
    finally:
        bridge.close()
        viewer.close()

    after_assets = asset_hashes()
    if before_assets != after_assets:
        raise RuntimeError("official assets changed during WP29 tail test")
    sync_trace_path = args.output_dir / "ros_command_trace.jsonl"
    bridge.write_sync_trace(sync_trace_path)
    summary = {
        "purpose": "isolated natural-state WP29-to-WP32 visible tail test",
        "reason": reason,
        "route_complete": reached_scores == [29, 30, 31, 32],
        "reached_scores": reached_scores,
        "score_events": score_events,
        "elapsed_sim_seconds": float(data.time - start_time),
        "elapsed_wall_seconds": float(time.perf_counter() - wall_start),
        "final_base_xyz_m": data.xpos[ids["base_body_id"]].tolist(),
        "max_contact_force_n": max_force,
        "max_abs_roll_deg": math.degrees(max_roll),
        "state": {"path": str(args.state), "sha256": sha256(args.state)},
        "tail_nav_script": {
            "path": str(args.tail_nav_script),
            "sha256": sha256(args.tail_nav_script),
        },
        "tail_behavior_config": {
            "path": str(args.tail_behavior_config),
            "sha256": sha256(args.tail_behavior_config),
        },
        "tail_waypoints": {
            "path": str(args.tail_waypoints),
            "sha256": sha256(args.tail_waypoints),
        },
        "start_index": args.start_index,
        "ros_command_sync": bridge.sync_summary(),
        "ros_command_trace": (
            {"path": str(sync_trace_path), "sha256": sha256(sync_trace_path)}
            if args.sim_sync else None
        ),
        "full_track_claim": False,
        "official_assets_modified": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[wp29-tail] complete={summary['route_complete']} "
        f"scores={reached_scores} reason={reason} "
        f"output={args.output_dir}",
        flush=True,
    )
    result = 0 if summary["route_complete"] else 4
    if args.clean_hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
