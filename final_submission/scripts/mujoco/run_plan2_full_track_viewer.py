#!/usr/bin/env python3
"""Run all 33 official waypoints with a visible MuJoCo viewer.

The official 57-D policy follows the repository's production-style waypoint
navigator.  At score 15 the tuned plan2 pit expert takes over without a state
teleport, exits the exact official pit, and hands the same MuJoCo state back to
the official policy.  The run ends only at score 32, a verified failure, the
simulation limit, or when the viewer is closed.

The pit expert still uses oracle simulation events and the research short-peak
torque envelope, so this is a mechanical integration/navigation audit rather
than a deployable competition policy.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import signal
import socket
import subprocess
import sys
import uuid
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw")

import mujoco
import mujoco.viewer
import numpy as np
import torch
from PIL import Image

import evaluate_plan2_full_track_oracle_handoff as oracle_handoff
import evaluate_plan2_official_walk_expert_handoff as expert_handoff
import search_s10_m20_early_steering_rescue as rescue_runner
from evaluate_plan2_full_track_conditioned_handoff import (
    LOCKED_TRACE,
    LOCKED_V4,
    conditioner_cases,
    condition_state,
    configured_expert_kwargs,
    build_conditioner_model,
    plain,
)
from evaluate_plan2_full_track_oracle_handoff import (
    PIT_SCORE_INDEX,
    Navigator,
    TrackRecorder,
    configure_continuous_limits,
    load_route,
    yaw_from_xmat,
)
from evaluate_plan2_official_walk_expert_handoff import (
    HandoffExactPeakEnv,
    PhaseRecorder,
    restore,
    run_expert,
    save_gif,
    snapshot,
)
from probe_s10_official_policy_shallow import OfficialPolicy
from s10_fronthook_scripted_probe import (
    MJCF_PATH,
    contact_metrics,
    diverged,
    model_ids,
)
from s10_speed_baseline import reset_official_start, run_official_standup
from search_s10_m20_approach_two_stage_rescue import (
    PHASE_ESTIMATOR,
    REFERENCE_CHECKPOINT,
    asset_hashes,
    load_reference,
    sha256,
)
from search_s10_m20_front_hook_pull import ORIGINAL_ROOT, local_path


DT = 0.001
CONTROL_PERIOD_S = 0.02
FINAL_SCORE = 32
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIEWER_DISTANCE = 5.0
VIEWER_AZIMUTH = 130.0
VIEWER_ELEVATION = -55.0
MP4_BITRATE = "4M"
MP4_MAX_BYTES = 500_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-sim-seconds", type=float, default=600.0)
    parser.add_argument("--max-forward", type=float, default=1.0)
    parser.add_argument("--viewer-speed", type=float, default=4.0)
    parser.add_argument("--expert-viewer-speed", type=float, default=1.0)
    parser.add_argument("--frame-period", type=float, default=1.0)
    parser.add_argument("--pit-frame-period", type=float, default=0.08)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--mp4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="encode the captured physical render frames as H.264 MP4",
    )
    parser.add_argument(
        "--route-yaml",
        type=Path,
        default=None,
        help="optional path route; used for score bookkeeping and the ROS tail bridge",
    )
    parser.add_argument(
        "--ros-tail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="after the pit expert, feed the exact MuJoCo state to the verified ROS tail navigator",
    )
    parser.add_argument("--ros-domain", type=int, default=118)
    parser.add_argument(
        "--tail-nav-script",
        type=Path,
        default=PROJECT_ROOT
        / "src/S10_sdk_deploy/scripts/plan2_expert_tail_navigator_wp31_drive_through_candidate_left.py",
    )
    parser.add_argument(
        "--tail-behavior-config",
        type=Path,
        default=PROJECT_ROOT
        / "config/plan2_wp25_wp27_behavior_straight_corridor.yaml",
    )
    parser.add_argument(
        "--tail-waypoints",
        type=Path,
        default=PROJECT_ROOT
        / "config/path_plan2_wp24_entry_rightshift_20260815.yaml",
    )
    return parser.parse_args()


class StreamingMp4Sink:
    """Encode frames as they are rendered, without retaining a full run in RAM."""

    def __init__(self, path: Path, fps: float) -> None:
        import imageio.v2 as imageio

        self.path = path
        self.partial_path = path.with_name(f"{path.stem}.partial{path.suffix}")
        self.frame_count = 0
        self.closed = False
        self.finalized = False
        self.writer = imageio.get_writer(
            self.partial_path,
            format="FFMPEG",
            mode="I",
            fps=fps,
            codec="libx264",
            bitrate=MP4_BITRATE,
            macro_block_size=1,
            output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )

    def __call__(self, frame: Image.Image) -> None:
        if self.closed:
            raise RuntimeError("cannot append to a closed MP4 stream")
        self.writer.append_data(np.asarray(frame.convert("RGB")))
        self.frame_count += 1

    def close(self, *, finalize: bool) -> None:
        if not self.closed:
            self.writer.close()
            self.closed = True
        if finalize and not self.finalized:
            if self.frame_count == 0:
                raise RuntimeError("MP4 recorder captured no frames")
            size_bytes = self.partial_path.stat().st_size
            if size_bytes > MP4_MAX_BYTES:
                raise RuntimeError(
                    f"MP4 exceeds 500 MB limit: {size_bytes} bytes"
                )
            self.partial_path.replace(self.path)
            self.finalized = True

    def abort(self) -> None:
        self.close(finalize=False)


def configure_viewer(viewer: Any, data: mujoco.MjData, ids: dict[str, Any]) -> None:
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = ids["base_body_id"]
        viewer.cam.distance = VIEWER_DISTANCE
        viewer.cam.azimuth = VIEWER_AZIMUTH
        viewer.cam.elevation = VIEWER_ELEVATION
        viewer.cam.lookat[:] = data.xpos[ids["base_body_id"]]


def set_viewer_info(viewer: Any, text: str) -> None:
    """Keep simulation time and route state visible in the native viewer."""

    setter = getattr(viewer, "set_texts", None)
    if setter is None:
        return
    try:
        setter(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                text,
                None,
            )
        )
    except (AttributeError, RuntimeError, TypeError):
        # Headless test handles do not implement the native text overlay.
        return


def sync_visible(
    viewer: Any,
    *,
    sim_elapsed_s: float,
    wall_start_s: float,
    viewer_speed: float,
    info_text: str | None = None,
) -> bool:
    if not viewer.is_running():
        return False
    if info_text is not None:
        set_viewer_info(viewer, info_text)
    viewer.sync()
    desired_wall_s = sim_elapsed_s / viewer_speed
    wait_s = desired_wall_s - (time.perf_counter() - wall_start_s)
    if wait_s > 0.0:
        time.sleep(min(wait_s, 0.05))
    return viewer.is_running()


class WaypointOverlay:
    """Hide reached scoring markers in model memory without changing physics."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.geom_ids_by_score: dict[int, list[int]] = {}
        self.hidden_scores: list[int] = []
        prefixes = ("track_waypoint_", "track_height_post_")
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is None:
                continue
            prefix = next((item for item in prefixes if name.startswith(item)), None)
            if prefix is None:
                continue
            score_text = name[len(prefix) : len(prefix) + 3]
            if len(score_text) != 3 or not score_text.isdigit():
                continue
            self.geom_ids_by_score.setdefault(int(score_text), []).append(geom_id)

    def hide(self, score: int) -> None:
        score = int(score)
        if score in self.hidden_scores:
            return
        geom_ids = self.geom_ids_by_score.get(score, [])
        if not geom_ids:
            raise RuntimeError(f"no visual overlay geoms found for score {score}")
        self.model.geom_rgba[geom_ids, 3] = 0.0
        self.hidden_scores.append(score)


def mirror_state_to_display(
    model: mujoco.MjModel,
    target: mujoco.MjData,
    source: mujoco.MjData,
) -> None:
    """Copy one physically simulated state into the persistent viewer data."""

    target.qpos[:] = source.qpos
    target.qvel[:] = source.qvel
    if target.act.size:
        target.act[:] = source.act
    target.ctrl[:] = source.ctrl
    target.qacc_warmstart[:] = source.qacc_warmstart
    target.time = source.time
    mujoco.mj_forward(model, target)


def build_prepared_pit_env(
    locked_report: dict[str, Any],
    phase_path: Path,
) -> HandoffExactPeakEnv:
    """Build the exact expert environment before the visible route starts."""

    kwargs = configured_expert_kwargs(locked_report, True)
    HandoffExactPeakEnv.phase_estimator_path = phase_path
    return HandoffExactPeakEnv(
        depth_m=float(kwargs["depth_m"]),
        command_mps=float(kwargs["command_mps"]),
        training_states=((0.0, 0.0),),
        max_episode_steps=int(round(float(kwargs["episode_seconds"]) / rescue_runner.CONTROL_DT)),
        leg_peak_nm=float(kwargs["leg_peak_nm"]),
        wheel_peak_nm=float(kwargs["wheel_peak_nm"]),
        max_continuous_exceedance_s=float(kwargs["max_continuous_exceedance_s"]),
        allow_body_wall_contact=bool(kwargs["allow_body_wall_contact"]),
    )


class RosTailBridge:
    """Exchange live state with the Python-3.10 ROS2 bridge over Unix datagrams."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.next_publish_time = -math.inf
        self.command = np.zeros(3, dtype=np.float32)
        token = uuid.uuid4().hex
        self.parent_socket = f"/tmp/plan2_ros_tail_parent_{token}.sock"
        self.child_socket = f"/tmp/plan2_ros_tail_child_{token}.sock"
        self.nav_log = Path(args.output_dir) / "ros_tail_navigator.log"
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(args.ros_domain)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket.bind(self.parent_socket)
        self.socket.settimeout(0.0)
        helper = Path(__file__).with_name("ros_tail_bridge.py")
        self.nav_process = subprocess.Popen(
            [
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
            ],
            cwd=str(helper.parent.parent.parent),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 3.0
        while not Path(self.child_socket).exists() and time.monotonic() < deadline:
            if self.nav_process.poll() is not None:
                raise RuntimeError(
                    f"ROS tail bridge exited during startup with code {self.nav_process.returncode}"
                )
            time.sleep(0.02)
        if not Path(self.child_socket).exists():
            raise RuntimeError("ROS tail bridge did not create its socket within 3 seconds")
        time.sleep(0.05)
        if self.nav_process.poll() is not None:
            raise RuntimeError(
                f"ROS tail bridge exited after socket creation with code {self.nav_process.returncode}"
            )

    def step(self, data: mujoco.MjData, ids: dict[str, Any]) -> np.ndarray:
        if float(data.time) + 1.0e-12 >= self.next_publish_time:
            stamp = float(data.time)
            position = np.asarray(data.xpos[ids["base_body_id"]], dtype=np.float64)
            quaternion = np.asarray(data.xquat[ids["base_body_id"]], dtype=np.float64)
            state = {
                "time": stamp,
                "position": position.tolist(),
                "quaternion": quaternion.tolist(),
                "velocity": np.asarray(data.qvel[:3], dtype=np.float64).tolist(),
            }
            self.socket.sendto(
                json.dumps(state, separators=(",", ":")).encode("utf-8"),
                self.child_socket,
            )
            self.next_publish_time = float(data.time) + 0.02
        try:
            while True:
                reply, _ = self.socket.recvfrom(4096)
                self.command[:] = json.loads(reply.decode("utf-8"))["command"]
        except BlockingIOError:
            pass
        return self.command.copy()

    def close(self) -> None:
        if self.nav_process.poll() is None:
            self.nav_process.send_signal(signal.SIGINT)
            try:
                self.nav_process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.nav_process.terminate()
        try:
            self.socket.sendto(b"__stop__", self.child_socket)
        except OSError:
            pass
        try:
            self.nav_process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.nav_process.terminate()
        self.socket.close()
        for path in (self.parent_socket, self.child_socket):
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass


def run_policy_visible(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict[str, Any],
    navigator: Navigator,
    recorder: TrackRecorder,
    viewer: Any,
    *,
    max_sim_seconds: float,
    stop_after_score: int | None,
    viewer_speed: float,
    overall_wall_start: float,
    ros_bridge: RosTailBridge | None = None,
    policy_settle_seconds: float = 0.0,
    waypoint_overlay: WaypointOverlay | None = None,
) -> dict[str, Any]:
    policy = OfficialPolicy()
    start_time = float(data.time)
    segment_wall_start = time.perf_counter()
    next_control_time = float(data.time)
    command = np.zeros(3, dtype=np.float32)
    max_force = 0.0
    max_roll = 0.0
    reason = "time_limit"
    score_events: list[dict[str, Any]] = []
    step = 0
    sync_stride = max(20, int(round(viewer_speed * 20.0)))
    next_status_time = float(data.time) + 5.0
    while float(data.time - start_time) < max_sim_seconds:
        if not viewer.is_running():
            reason = "viewer_closed"
            break
        base = data.xpos[ids["base_body_id"]].copy()
        yaw = yaw_from_xmat(data.xmat[ids["base_body_id"]])
        reached = navigator.update_reached(base, float(data.time))
        for score in reached:
            if waypoint_overlay is not None:
                waypoint_overlay.hide(score)
            point = navigator.current()
            detail = f"reached score {score}; next={None if point is None else point.score_index}"
            recorder.capture(
                data,
                ids["base_body_id"],
                phase="WALK POLICY",
                detail=detail,
                force=True,
            )
            event = {
                "score": int(score),
                "sim_time_s": float(data.time),
                "wall_elapsed_s": float(time.perf_counter() - overall_wall_start),
                "base_xyz_m": base.tolist(),
            }
            score_events.append(event)
            print(
                f"[viewer-track] reached score={score} sim={data.time:.2f}s "
                f"wall={event['wall_elapsed_s']:.2f}s",
                flush=True,
            )
        if stop_after_score is not None and stop_after_score in reached:
            reason = f"reached_score_{stop_after_score}"
            break
        if navigator.current() is None:
            reason = "route_complete"
            break
        if data.time + 1.0e-12 >= next_control_time:
            next_control_time = float(data.time) + CONTROL_PERIOD_S
            if float(data.time - start_time) < policy_settle_seconds:
                command = np.zeros(3, dtype=np.float32)
            elif ros_bridge is None:
                command = navigator.command(base, yaw, float(data.time))
            else:
                command = ros_bridge.step(data, ids)
            policy.update(model, data, ids, command)
        elif ros_bridge is not None:
            command = ros_bridge.step(data, ids)
        policy.apply(data)
        mujoco.mj_step(model, data)
        step += 1
        contacts = contact_metrics(model, data, ids)
        max_force = max(max_force, float(contacts["max_contact_force"]))
        rotation = np.asarray(data.xmat[ids["base_body_id"]]).reshape(3, 3)
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        max_roll = max(max_roll, abs(roll))
        point = navigator.current()
        recorder.capture(
            data,
            ids["base_body_id"],
            phase="WALK POLICY",
            detail=(
                f"scores={len(navigator.reached_scores)}/33 target="
                f"{None if point is None else point.score_index} t={data.time:.1f}s"
            ),
        )
        bad = diverged(data)
        if bad is not None:
            reason = bad
            break
        if float(data.time) + 1.0e-12 >= next_status_time:
            next_status_time = float(data.time) + 5.0
            print(
                f"[viewer-track] heartbeat sim={data.time:.2f}s "
                f"base=({base[0]:.2f},{base[1]:.2f},{base[2]:.2f}) "
                f"yaw={math.degrees(yaw):.1f}deg "
                f"cmd=({command[0]:.2f},{command[1]:.2f},{command[2]:.2f}) "
                f"ctrl_max={float(np.max(np.abs(data.ctrl))):.2f} "
                f"wheel_target={np.round(policy.velocity_target[[3, 7, 11, 15]], 1).tolist()} "
                f"wheel_qvel={np.round(data.qvel[[9, 13, 17, 21]], 1).tolist()} "
                f"target={None if point is None else point.score_index}",
                flush=True,
            )
        if step % sync_stride == 0 and not sync_visible(
            viewer,
            sim_elapsed_s=float(data.time - start_time),
            wall_start_s=segment_wall_start,
            viewer_speed=viewer_speed,
            info_text=(
                f"SIM TIME {data.time:.2f}s | SCORE "
                f"{len(navigator.reached_scores)}/{FINAL_SCORE + 1} | TARGET "
                f"{None if point is None else point.score_index}"
            ),
        ):
            reason = "viewer_closed"
            break
    target = navigator.current()
    return {
        "reason": reason,
        "elapsed_seconds": float(data.time - start_time),
        "wall_elapsed_seconds": float(time.perf_counter() - segment_wall_start),
        "reached_scores": navigator.reached_scores.copy(),
        "new_score_events": score_events,
        "route_index": navigator.index,
        "next_target_score": None if target is None else target.score_index,
        "next_target_kind": None if target is None else target.kind,
        "next_target_xyz_m": None if target is None else target.xyz.tolist(),
        "final_base_xyz_m": data.xpos[ids["base_body_id"]].tolist(),
        "max_contact_force_n": max_force,
        "max_abs_roll_deg": math.degrees(max_roll),
        "command": command.tolist(),
    }


class VisiblePhaseRecorder(PhaseRecorder):
    last_viewer: Any = None
    viewer_speed: float = 1.0
    store_frames: bool = True
    shared_viewer: Any = None
    display_model: mujoco.MjModel | None = None
    display_data: mujoco.MjData | None = None
    display_ids: dict[str, Any] | None = None
    source_overlay: WaypointOverlay | None = None
    frame_overlay: WaypointOverlay | None = None

    def __init__(self, model: mujoco.MjModel, period_s: float) -> None:
        stream_frames = type(self).default_frame_sink is not None
        if self.store_frames or stream_frames:
            super().__init__(model, period_s)
        else:
            self.period_s = float(period_s)
            self.next_time = 0.0
            self.frames = []
            self.frame_sink = None
            self.retain_frames = False
            self.frame_count = 0
            self.robot_inset = None
            self.renderer = None
            self.camera = None
        self.viewer = None
        self.wall_start = 0.0
        self.sim_start = 0.0
        self.capture_count = 0

    @classmethod
    def _mirror_to_display(cls, source: mujoco.MjData) -> None:
        if cls.display_model is None or cls.display_data is None:
            return
        mirror_state_to_display(cls.display_model, cls.display_data, source)

    def capture(self, env: Any) -> None:
        self._mirror_to_display(env.data)
        if self.viewer is None:
            if self.shared_viewer is None:
                self.viewer = mujoco.viewer.launch_passive(env.model, env.data)
                configure_viewer(self.viewer, env.data, env.ids)
            else:
                self.viewer = self.shared_viewer
                if self.display_data is None or self.display_ids is None:
                    raise RuntimeError("persistent viewer has no display state")
            self.wall_start = time.perf_counter()
            self.sim_start = float(env.data.time)
            VisiblePhaseRecorder.last_viewer = self.viewer
        if self.store_frames or self.frame_sink is not None:
            if self.source_overlay is not None:
                if self.frame_overlay is None or self.frame_overlay.model is not env.model:
                    self.frame_overlay = WaypointOverlay(env.model)
                for score in self.source_overlay.hidden_scores:
                    self.frame_overlay.hide(score)
            super().capture(env)
        self.capture_count += 1
        if self.capture_count % 1 == 0:
            sync_visible(
                self.viewer,
                sim_elapsed_s=float(env.data.time - self.sim_start),
                wall_start_s=self.wall_start,
                viewer_speed=VisiblePhaseRecorder.viewer_speed,
                info_text=(
                    f"SIM TIME {env.data.time:.2f}s | SCORE "
                    f"{len(self.source_overlay.hidden_scores) if self.source_overlay is not None else 16}/33 "
                    "| PHASE PIT EXPERT"
                ),
            )

    def close(self, path: Path) -> str:
        if self.store_frames or self.frame_sink is not None:
            return super().close(path)
        PhaseRecorder.last_frames = []
        return str(path)


def uniformly_sample_phase(
    frames: list[Image.Image],
    *,
    source_period_s: float,
    target_period_s: float,
) -> list[Image.Image]:
    """Select real render frames at a common simulation-time interval."""

    if not frames:
        return []
    if source_period_s >= target_period_s - 1.0e-12:
        return frames.copy()
    stride = max(1, int(round(target_period_s / source_period_s)))
    selected = frames[::stride]
    if selected[-1] is not frames[-1]:
        selected.append(frames[-1])
    return selected


def main() -> int:
    args = parse_args()
    if args.max_sim_seconds <= 0.0 or args.max_forward <= 0.0:
        raise ValueError("simulation duration and maximum forward speed must be positive")
    if args.viewer_speed <= 0.0 or args.expert_viewer_speed <= 0.0:
        raise ValueError("viewer speeds must be positive")
    if args.frame_period <= 0.0 or args.pit_frame_period <= 0.0:
        raise ValueError("video frame periods must be positive")
    if args.mp4 and not args.video:
        raise ValueError("--mp4 requires video frame capture; omit --no-video")
    if args.mp4 and not math.isclose(
        args.frame_period,
        args.pit_frame_period,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "streaming MP4 requires equal --frame-period and --pit-frame-period"
        )
    if not math.isclose(args.viewer_speed, args.expert_viewer_speed, rel_tol=0.0, abs_tol=1.0e-9):
        print(
            "[viewer-track] phase-specific expert speed is deprecated; "
            f"using one continuous display speed={args.viewer_speed:.1f}x",
            flush=True,
        )
    # A phase-specific playback rate looks like a pause at the pit handoff.
    # Keep the requested CLI field for compatibility, but use one rate for all
    # visible phases so the display cadence is continuous.
    display_speed = float(args.viewer_speed)
    output = local_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    overall_wall_start = time.perf_counter()

    locked_path = local_path(LOCKED_V4)
    locked_trace_path = local_path(LOCKED_TRACE)
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    locked_report = locked["report"]
    locked_trace = np.load(locked_trace_path)
    locked_reference = {
        "base_progress_m": np.asarray(locked_trace["base_progress_m"][0]),
        "wheel_progress_m": np.asarray(locked_trace["wheel_progress_m"][0]),
        "joint_position_rad": np.asarray(locked_trace["joint_position_rad"][0]),
        "joint_velocity_rad_s": np.asarray(locked_trace["joint_velocity_rad_s"][0]),
    }
    reference_path = local_path(REFERENCE_CHECKPOINT)
    phase_path = local_path(PHASE_ESTIMATOR)
    locked_runtime_inputs = {
        "reference": reference_path,
        "phase": phase_path,
    }
    for name, path in locked_runtime_inputs.items():
        expected_hash = str(locked["inputs"][name]["sha256"])
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"locked {name} input hash mismatch: {actual_hash} != {expected_hash}"
            )
    # Load the expert before opening the viewer so the pit handoff never waits
    # for checkpoint deserialization while the visible simulation is running.
    torch.set_num_threads(1)
    agent = load_reference(reference_path)
    prepared_conditioner_model = build_conditioner_model()
    prepared_pit_env = build_prepared_pit_env(locked_report, phase_path)
    before_assets = asset_hashes()
    locked_hashes = {
        "summary": sha256(locked_path),
        "trace": sha256(locked_trace_path),
    }

    if args.route_yaml is not None:
        route_path = local_path(args.route_yaml)
    elif args.ros_tail:
        route_path = local_path(PROJECT_ROOT / "config" / args.tail_waypoints.name)
    else:
        route_path = None
    route = load_route(route_path)
    model = mujoco.MjModel.from_xml_path(str(local_path(MJCF_PATH)))
    model.opt.timestep = DT
    configure_continuous_limits(model)
    data = mujoco.MjData(model)
    ids = model_ids(model)
    reset_official_start(model, data)
    diverged_stand, stand_reason = run_official_standup(model, data)
    if diverged_stand:
        raise RuntimeError(f"official stand-up failed: {stand_reason}")

    mp4_sink: StreamingMp4Sink | None = None
    mp4_path: Path | None = None
    track_recorder_defaults = (
        oracle_handoff.TrackRecorder.default_frame_sink,
        oracle_handoff.TrackRecorder.default_retain_frames,
        oracle_handoff.TrackRecorder.default_robot_inset_enabled,
    )
    phase_recorder_defaults = (
        expert_handoff.PhaseRecorder.default_frame_sink,
        expert_handoff.PhaseRecorder.default_retain_frames,
        expert_handoff.PhaseRecorder.default_robot_inset_enabled,
    )
    if args.mp4:
        mp4_path = output / "full_track_viewer_rollout.mp4"
        mp4_sink = StreamingMp4Sink(
            mp4_path,
            fps=display_speed / args.frame_period,
        )
        atexit.register(mp4_sink.abort)
        oracle_handoff.TrackRecorder.default_frame_sink = mp4_sink
        oracle_handoff.TrackRecorder.default_retain_frames = False
        oracle_handoff.TrackRecorder.default_robot_inset_enabled = True
        expert_handoff.PhaseRecorder.default_frame_sink = mp4_sink
        expert_handoff.PhaseRecorder.default_retain_frames = False
        expert_handoff.PhaseRecorder.default_robot_inset_enabled = True

    navigator = Navigator(route, data.xpos[ids["base_body_id"]][:2], args.max_forward)
    waypoint_overlay = WaypointOverlay(model)
    pre_recorder = TrackRecorder(model, args.video, args.frame_period)
    pre_recorder.capture(
        data,
        ids["base_body_id"],
        phase="WALK POLICY",
        detail="visible official route; target score 0",
        force=True,
    )
    pre_viewer = mujoco.viewer.launch_passive(model, data)
    configure_viewer(pre_viewer, data, ids)
    set_viewer_info(pre_viewer, f"SIM TIME {data.time:.2f}s | SCORE 0/{FINAL_SCORE + 1} | TARGET 0")
    print(
        f"[viewer-track] MuJoCo window opened; route=0..{FINAL_SCORE} "
        f"viewer_speed={args.viewer_speed:.1f}x",
        flush=True,
    )
    pre_report = run_policy_visible(
        model,
        data,
        ids,
        navigator,
        pre_recorder,
        pre_viewer,
        max_sim_seconds=args.max_sim_seconds,
        stop_after_score=PIT_SCORE_INDEX,
        viewer_speed=display_speed,
        overall_wall_start=overall_wall_start,
        policy_settle_seconds=3.0,
        waypoint_overlay=waypoint_overlay,
    )
    natural_state = snapshot(data)
    np.savez_compressed(
        output / "natural_pit_state.npz",
        qpos=natural_state.qpos,
        qvel=natural_state.qvel,
        act=natural_state.act,
        ctrl=natural_state.ctrl,
        qacc_warmstart=natural_state.qacc_warmstart,
        time=np.asarray(natural_state.time),
    )
    pre_frames = pre_recorder.frames.copy()
    pre_recorder.close()

    pit_report: dict[str, Any] | None = None
    post_report: dict[str, Any] | None = None
    conditioner_report: dict[str, Any] | None = None
    expert_frames: list[Image.Image] = []
    post_frames: list[Image.Image] = []
    conditioner_visible_samples = 0
    if PIT_SCORE_INDEX in pre_report["reached_scores"]:
        conditioner = conditioner_cases(True)[0]
        conditioner_wall_start = time.perf_counter()

        def display_conditioner_state(source: mujoco.MjData) -> None:
            nonlocal conditioner_visible_samples
            mirror_state_to_display(model, data, source)
            conditioner_visible_samples += 1
            if not sync_visible(
                pre_viewer,
                sim_elapsed_s=float(source.time - natural_state.time),
                wall_start_s=conditioner_wall_start,
                viewer_speed=display_speed,
                info_text=f"SIM TIME {source.time:.2f}s | PHASE PIT CONDITIONER",
            ):
                raise RuntimeError("MuJoCo viewer closed during pit conditioner")

        conditioned_state, conditioner_report, conditioner_frames = condition_state(
            natural_state,
            conditioner,
            locked_reference,
            video=args.video,
            period_s=args.pit_frame_period,
            state_observer=display_conditioner_state,
            observer_period_s=0.02,
            hidden_scores=waypoint_overlay.hidden_scores,
            prebuilt_model=prepared_conditioner_model,
        )
        # The observer above has already displayed the isolated conditioner's
        # real states.  This final restore is the same final state, not a jump.
        restore(model, data, conditioned_state)
        pre_viewer.sync()
        original_env = rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv
        original_recorder = rescue_runner.GifRecorder
        rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv = HandoffExactPeakEnv
        VisiblePhaseRecorder.last_frames = []
        VisiblePhaseRecorder.last_viewer = None
        VisiblePhaseRecorder.viewer_speed = display_speed
        VisiblePhaseRecorder.store_frames = args.video
        VisiblePhaseRecorder.shared_viewer = pre_viewer
        VisiblePhaseRecorder.display_model = model
        VisiblePhaseRecorder.display_data = data
        VisiblePhaseRecorder.display_ids = ids
        VisiblePhaseRecorder.source_overlay = waypoint_overlay
        VisiblePhaseRecorder.frame_overlay = None
        rescue_runner.GifRecorder = VisiblePhaseRecorder
        try:
            pit_report = run_expert(
                agent,
                phase_path,
                configured_expert_kwargs(locked_report, True),
                conditioned_state,
                gif_path=output / "expert.gif",
                trace_path=output / "expert_trace.npz",
                frame_period=args.pit_frame_period,
                prebuilt_env=prepared_pit_env,
            )
            expert_frames = VisiblePhaseRecorder.last_frames.copy()
            expert_instance = HandoffExactPeakEnv.last_instance
            if expert_instance is None:
                raise RuntimeError("expert environment instance was not retained")
            expert_viewer = VisiblePhaseRecorder.last_viewer
            if pit_report["success"] and expert_viewer is not None and expert_viewer.is_running():
                # Continue the official policy in the original model/data pair
                # bound to the one persistent viewer.  The copied fields are
                # exactly the fields used by the existing handoff contract.
                restore(model, data, snapshot(expert_instance.data))
                configure_continuous_limits(model)
                post_recorder = TrackRecorder(model, args.video, args.frame_period)
                post_recorder.capture(
                    data,
                    ids["base_body_id"],
                    phase="WALK POLICY RESUMED",
                    detail="pit exit complete; continuing to score 32",
                    force=True,
                )
                remaining_s = max(
                    0.0,
                    args.max_sim_seconds
                    - float(data.time - 3.0),
                )
                ros_bridge = RosTailBridge(args) if args.ros_tail else None
                try:
                    post_report = run_policy_visible(
                        model,
                        data,
                        ids,
                        navigator,
                        post_recorder,
                        expert_viewer,
                        max_sim_seconds=remaining_s,
                        stop_after_score=None,
                        viewer_speed=display_speed,
                        overall_wall_start=overall_wall_start,
                        ros_bridge=ros_bridge,
                        waypoint_overlay=waypoint_overlay,
                    )
                finally:
                    if ros_bridge is not None:
                        ros_bridge.close()
                post_frames = post_recorder.frames.copy()
                post_recorder.close()
        finally:
            rescue_runner.S10M20PhaseGatedTorqueEnvelopeEnv = original_env
            rescue_runner.GifRecorder = original_recorder
            HandoffExactPeakEnv.handoff_state = None
            VisiblePhaseRecorder.shared_viewer = None
            VisiblePhaseRecorder.display_model = None
            VisiblePhaseRecorder.display_data = None
            VisiblePhaseRecorder.display_ids = None
            VisiblePhaseRecorder.source_overlay = None
            VisiblePhaseRecorder.frame_overlay = None
    else:
        conditioner_frames = []

    pre_viewer.close()
    prepared_pit_env.close()

    all_frames = pre_frames + conditioner_frames + expert_frames + post_frames
    if mp4_sink is not None:
        mp4_sink.close(finalize=True)
        oracle_handoff.TrackRecorder.default_frame_sink = track_recorder_defaults[0]
        oracle_handoff.TrackRecorder.default_retain_frames = track_recorder_defaults[1]
        oracle_handoff.TrackRecorder.default_robot_inset_enabled = track_recorder_defaults[2]
        expert_handoff.PhaseRecorder.default_frame_sink = phase_recorder_defaults[0]
        expert_handoff.PhaseRecorder.default_retain_frames = phase_recorder_defaults[1]
        expert_handoff.PhaseRecorder.default_robot_inset_enabled = phase_recorder_defaults[2]
        mp4_frame_count = mp4_sink.frame_count
        mp4_size_bytes = mp4_path.stat().st_size if mp4_path is not None else None
        gif_path = None
    elif args.video and all_frames:
        gif_path = output / "full_track_viewer_rollout.gif"
        save_gif(output / "full_track_viewer_rollout.gif", all_frames, args.pit_frame_period)
        mp4_frame_count = 0
        mp4_size_bytes = None
    else:
        gif_path = None
        mp4_frame_count = 0
        mp4_size_bytes = None

    after_assets = asset_hashes()
    locked_after = {"summary": sha256(locked_path), "trace": sha256(locked_trace_path)}
    if before_assets != after_assets or locked_hashes != locked_after:
        raise RuntimeError("official assets or locked v4 changed during visible full-track run")
    reached_scores = (
        post_report["reached_scores"]
        if post_report is not None
        else pre_report["reached_scores"]
    )
    route_complete = reached_scores == list(range(FINAL_SCORE + 1))
    summary = {
        "purpose": "visible all-33-waypoint official-track navigation audit",
        "route_complete": route_complete,
        "reached_score_count": len(reached_scores),
        "reached_scores": reached_scores,
        "last_reached_score": reached_scores[-1] if reached_scores else None,
        "total_wall_elapsed_seconds": time.perf_counter() - overall_wall_start,
        "final_sim_time_s": (
            post_report["new_score_events"][-1]["sim_time_s"]
            if route_complete and post_report and post_report["new_score_events"]
            else None
        ),
        "pre_track": pre_report,
        "conditioner": conditioner_report,
        "pit_expert": pit_report,
        "post_track": post_report,
        "mechanical_integration_audit_only": True,
        "deployable": False,
        "oracle_phase_events_used": True,
        "lidar_takeover_used": False,
        "state_reset_or_teleport_at_handoff": False,
        "viewer_used": True,
        "single_persistent_viewer": True,
        "phase_camera_reconfigurations": 0,
        "conditioner_display_mode": "physical_states_in_persistent_viewer",
        "conditioner_visible_samples": conditioner_visible_samples,
        "pit_expert_environment_prebuilt_before_viewer": True,
        "conditioner_model_prebuilt_before_viewer": True,
        "waypoint_overlay_runtime_only": True,
        "waypoint_overlay_hidden_scores": waypoint_overlay.hidden_scores,
        "viewer_speed": display_speed,
        "expert_viewer_speed": display_speed,
        "requested_expert_viewer_speed": args.expert_viewer_speed,
        "uniform_viewer_speed": True,
        "viewer_info_overlay": True,
        "viewer_info_fields": ["sim_time_s", "score_progress", "target_score", "phase"],
        "camera_follow": {
            "distance_m": VIEWER_DISTANCE,
            "azimuth_deg": VIEWER_AZIMUTH,
            "elevation_deg": VIEWER_ELEVATION,
            "tracking_body": "base_link",
        },
        "visualization": str(gif_path) if gif_path is not None else None,
        "mp4_visualization": str(mp4_path) if mp4_path is not None else None,
        "mp4_frame_count": mp4_frame_count,
        "mp4_fps": display_speed / args.frame_period if mp4_path is not None else None,
        "mp4_playback_speed": display_speed if mp4_path is not None else None,
        "mp4_interpolated_frames": False,
        "mp4_streaming_encoder": mp4_path is not None,
        "mp4_bitrate": MP4_BITRATE if mp4_path is not None else None,
        "mp4_size_bytes": mp4_size_bytes,
        "mp4_limit_bytes": MP4_MAX_BYTES if mp4_path is not None else None,
        "gif_disabled_for_mp4": bool(args.mp4),
        "robot_visibility_inset": {
            "enabled": bool(args.mp4),
            "position": "bottom_right",
            "camera": "near_vertical_top_down",
            "static_course_hidden_in_inset_only": bool(args.mp4),
            "physics_or_control_modified": False,
        },
        "official_assets_modified": False,
        "locked_v4_modified": False,
        "official_assets_before": before_assets,
        "official_assets_after": after_assets,
        "locked_v4_hashes_before": locked_hashes,
        "locked_v4_hashes_after": locked_after,
        "inputs": {
            "official_track": {
                "path": str(local_path(MJCF_PATH)),
                "sha256": sha256(local_path(MJCF_PATH)),
            },
            "locked_v4": {"path": str(locked_path), "sha256": locked_hashes["summary"]},
            "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
            "phase": {"path": str(phase_path), "sha256": sha256(phase_path)},
        },
        "protected_original_root": str(ORIGINAL_ROOT),
    }
    (output / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(
        f"[viewer-track] complete={route_complete} scores={len(reached_scores)}/33 "
        f"last={None if not reached_scores else reached_scores[-1]} "
        f"wall={summary['total_wall_elapsed_seconds']:.2f}s output={output}",
        flush=True,
    )
    return 0 if route_complete else 4


if __name__ == "__main__":
    raise SystemExit(main())
