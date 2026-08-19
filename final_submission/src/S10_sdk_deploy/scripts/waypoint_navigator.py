#!/usr/bin/env python3
"""Minimal waypoint navigator for the S10 track.

Subscribes to the ground truth pose on /odom, follows the waypoints exported
by export_waypoints.py, and publishes /cmd_vel for the ROS user command
interface in rl_deploy (run it with S10_REMOTE=ros).

The controller is deliberately simple and stateless: turn toward the next
waypoint, drive forward at a speed that falls off with heading error, advance
when inside the reach radius. No integral or derivative terms, so it does not
depend on the simulator's wall-clock rate.

Usage:
    /usr/bin/python3 src/S10_sdk_deploy/scripts/waypoint_navigator.py
    /usr/bin/python3 src/S10_sdk_deploy/scripts/waypoint_navigator.py --max-forward 0.8
    /usr/bin/python3 src/S10_sdk_deploy/scripts/waypoint_navigator.py --no-auto-start
"""

import argparse
import math
import re
from pathlib import Path

import yaml

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String

CURRENT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = (CURRENT_DIR / ".." / ".." / ".." / "config").resolve()
DEFAULT_PATH = CONFIG_DIR / "path.yaml"
FALLBACK_PATH = CONFIG_DIR / "waypoints.yaml"

# Startup sequence states
PHASE_WAIT_LINK = "wait_link"     # waiting for rl_deploy to subscribe
PHASE_STANDING = "standing"       # sent "stand", waiting for the robot to rise
PHASE_ENTER_RL = "enter_rl"       # sent "rl", waiting a moment before driving
PHASE_NAVIGATE = "navigate"
PHASE_DONE = "done"


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract the yaw angle from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class WaypointNavigator(Node):
    def __init__(self, args):
        super().__init__("waypoint_navigator")

        self.args = args
        self.waypoints = self._load_waypoints(args.waypoints)
        self.target_index = max(0, min(args.start_index, len(self.waypoints) - 1))

        self.pose = None          # (x, y, yaw)
        self.sim_time = None      # seconds, from the /odom header stamp
        self.base_z = None
        self.start_sim_time = None
        self.segment_start = None  # (x, y, z) start of the segment currently being tracked

        # A drop is not a normal planar-navigation segment.  While the body is
        # crossing the rim, XY distance can briefly stop improving even though
        # the intended motion is correct.  Treating that as a generic stall
        # starts the reverse/strafe recovery halfway down the pit.  The descent
        # controller deliberately mirrors teleoperation: align once, then hold
        # a forward command and a small heading-hold correction until the
        # bottom waypoint is reached.
        self.descent_target_index = None
        self.descent_started_at = None
        self.descent_heading = None
        self.descent_phase = None
        self.descent_committed = False
        self.descent_start_z = None
        self.descent_drop_seen = False
        self.descent_timeout_reported = False
        self.inbound_heading = None

        self.phase = PHASE_WAIT_LINK if args.auto_start else PHASE_NAVIGATE
        self.phase_started_at = None   # wall clock, set on first timer tick
        self.stand_reference_z = None  # base height just before "stand" was sent
        self.last_rl_request = 0.0
        self.last_progress_time = None
        self.last_report_index = -1
        self.max_cross_track = 0.0

        # Stuck detection and recovery. Without this the controller repeats the
        # same command forever: on the 0.377 m pit it drove forward into the
        # wall for minutes, burning a whole evaluation run.
        self.best_distance = None      # closest approach to the current target
        self.last_improve_time = None
        self.recovery_phase = None     # None | "reverse" | "strafe" | "realign"
        self.recovery_started_at = 0.0
        self.recovery_count = 0        # attempts at the current target
        self.recovery_total = 0
        self.strafe_sign = 1.0

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.mode_pub = self.create_publisher(String, "/robot_mode", 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_callback, 20)

        period = 1.0 / args.rate
        self.timer = self.create_timer(period, self._control_step)

        self.get_logger().info(
            f"Loaded {len(self.waypoints)} waypoints from {args.waypoints}"
        )
        self.get_logger().info(
            f"max_forward={args.max_forward} max_yaw={args.max_yaw} "
            f"reach_radius={args.reach_radius} rate={args.rate}Hz "
            f"auto_start={'on' if args.auto_start else 'off'}"
        )

    # ---------------- setup ----------------

    def _load_waypoints(self, path: Path):
        """Load either the planned path (waypoints + via points) or the plain
        waypoint list. Returns a list of
        ``(x, y, z, kind, score_index, radius_override)``.

        The original navigator discarded waypoint Z because ordinary segments
        are tracked in XY.  Keeping it lets the controller recognise a real
        platform-to-pit descent without hard-coding a map coordinate.
        """
        if not path.is_file():
            raise FileNotFoundError(
                f"Cannot find {path}. Run plan_path.py (or export_waypoints.py) first."
            )
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))

        if "path" in doc:
            points = [(float(e["pos"][0]), float(e["pos"][1]),
                       float(e["pos"][2]) if len(e["pos"]) >= 3 else 0.0,
                       e.get("kind", "waypoint"), e.get("index"),
                       (float(e["radius"]) if e.get("radius") is not None
                        else None))
                      for e in doc["path"]]
        elif "waypoints" in doc:
            points = [(float(w["pos"][0]), float(w["pos"][1]),
                       float(w["pos"][2]) if len(w["pos"]) >= 3 else 0.0,
                       "waypoint", self._score_index_from_name(w.get("name")),
                       (float(w["radius"]) if w.get("radius") is not None
                        else None))
                      for w in doc["waypoints"]]
        else:
            raise ValueError(f"{path} has neither a 'path' nor a 'waypoints' key")

        if not points:
            raise ValueError(f"No points in {path}")
        return points

    @staticmethod
    def _score_index_from_name(name):
        """Recover export_waypoints.py's scoring index from a point name."""
        if not name:
            return None
        match = re.search(r"track_waypoint_(\d+)_", str(name))
        return int(match.group(1)) if match else None

    def _target_label(self, index: int) -> str:
        kind = self.waypoints[index][3]
        score_index = self.waypoints[index][4]
        if score_index is not None:
            return f"{kind} score={int(score_index)} path={index}"
        return f"{kind} path={index}"

    def _target(self, index: int):
        return self.waypoints[index][0], self.waypoints[index][1]

    def _target_z(self, index: int) -> float:
        return self.waypoints[index][2]

    def _radius(self, index: int) -> float:
        """Scoring waypoints must be entered within the simulator's radius; via
        points only exist to steer through a gap, so nailing them exactly would
        just cost time."""
        kind = self.waypoints[index][3]
        radius_override = self.waypoints[index][5]
        if radius_override is not None:
            return radius_override
        return self.args.reach_radius if kind == "waypoint" else self.args.via_radius

    # ---------------- callbacks ----------------

    def _odom_callback(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_from_quaternion(q.x, q.y, q.z, q.w))
        self.base_z = p.z
        self.sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    # ---------------- startup sequence ----------------

    def _now(self) -> float:
        """Wall clock seconds; the startup sequence is timed in real time
        because rl_deploy's command watchdog is too."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _enter_phase(self, phase: str):
        self.phase = phase
        self.phase_started_at = self._now()

    def _publish_mode(self, mode: str):
        msg = String()
        msg.data = mode
        self.mode_pub.publish(msg)
        self.get_logger().info(f"-> /robot_mode '{mode}'")

    def _run_startup(self) -> bool:
        """Drive the stand -> rl sequence. Returns True once navigation may start."""
        if self.phase_started_at is None:
            self.phase_started_at = self._now()
        elapsed = self._now() - self.phase_started_at

        if self.phase == PHASE_WAIT_LINK:
            # A message published before rl_deploy's subscription is matched is
            # simply dropped, so wait for the connection before sending "stand".
            if self.mode_pub.get_subscription_count() > 0:
                self.stand_reference_z = self.base_z
                self._publish_mode("stand")
                self._enter_phase(PHASE_STANDING)
            elif elapsed > 20.0:
                self.get_logger().error(
                    "No subscriber on /robot_mode after 20s. "
                    "Is rl_deploy running with S10_REMOTE=ros?"
                )
                self.phase_started_at = self._now()  # keep waiting, warn again later
            return False

        if self.phase == PHASE_STANDING:
            # The FSM does not publish its state, so use the base height from
            # /odom as the standing-up confirmation. It has to be measured as a
            # RISE from where the robot started, not against an absolute z: on
            # a raised part of the track the base clears any fixed threshold
            # while still lying down, and "rl" then gets sent -- and rejected --
            # before the robot has actually stood up.
            risen = (self.base_z is not None and self.stand_reference_z is not None
                     and self.base_z - self.stand_reference_z > self.args.stand_rise)
            if risen:
                self._publish_mode("rl")
                self.last_rl_request = self._now()
                self._enter_phase(PHASE_ENTER_RL)
            elif elapsed > 15.0:
                rise = None if self.base_z is None or self.stand_reference_z is None \
                    else self.base_z - self.stand_reference_z
                self.get_logger().error(
                    f"Robot did not rise by {self.args.stand_rise} m within 15s "
                    f"(rise={rise}). Giving up on auto start."
                )
                self._enter_phase(PHASE_NAVIGATE)
            return False

        if self.phase == PHASE_ENTER_RL:
            # rl_deploy rejects "rl" unless the FSM has actually reached
            # StandingUp, and it publishes no state we could wait on, so repeat
            # the request until the settle window is over.
            if self._now() - self.last_rl_request > 0.5:
                self._publish_mode("rl")
                self.last_rl_request = self._now()
            if elapsed > self.args.rl_settle_time:
                self.get_logger().info("Starting navigation")
                self._enter_phase(PHASE_NAVIGATE)
            return False

        return True

    # ---------------- control ----------------

    def _control_step(self):
        if self.phase == PHASE_DONE:
            self.cmd_pub.publish(Twist())
            return

        if self.pose is None:
            # Keep the watchdog fed with zeros until odometry arrives.
            self.cmd_pub.publish(Twist())
            return

        if self.args.auto_start and not self._run_startup():
            self.cmd_pub.publish(Twist())
            return

        if self.start_sim_time is None:
            self.start_sim_time = self.sim_time
        if self.segment_start is None:
            # The first segment runs from wherever the robot actually is.
            self.segment_start = (self.pose[0], self.pose[1],
                                  self.base_z if self.base_z is not None else 0.0)

        x, y, yaw = self.pose
        tx, ty = self._target(self.target_index)

        distance = math.hypot(tx - x, ty - y)
        if distance <= self._radius(self.target_index):
            self._advance_waypoint(distance)
            return

        heading_to_target = wrap_to_pi(math.atan2(ty - y, tx - x) - yaw)

        # ---- dedicated platform-to-pit descent ----
        if self._is_descent_segment():
            self._run_descent_controller(distance, yaw)
            return

        # ---- stuck detection / recovery ----
        if self._update_recovery(distance, heading_to_target):
            return

        # ---- pure pursuit along the segment, not straight at the waypoint ----
        # Steering at the waypoint alone only corrects heading, so a small
        # heading error on a 13 m leg becomes a large lateral offset by the end
        # of it -- which is exactly what walks the robot into a door frame.
        ax, ay, _ = self.segment_start
        sx, sy = tx - ax, ty - ay
        seg_len = math.hypot(sx, sy)

        if seg_len < 1e-6:
            carrot_x, carrot_y = tx, ty
            cross_track = 0.0
        else:
            ux, uy = sx / seg_len, sy / seg_len
            # Projection of the robot onto the segment
            proj = (x - ax) * ux + (y - ay) * uy
            proj = max(0.0, min(seg_len, proj))
            # Signed lateral offset from the segment (left of travel is positive)
            cross_track = -(x - ax) * uy + (y - ay) * ux
            # The carrot rides ahead of the projection, so being off the line
            # produces a heading command that pulls back onto it.
            carrot = min(proj + self.args.lookahead, seg_len)
            carrot_x, carrot_y = ax + ux * carrot, ay + uy * carrot

        heading_error = wrap_to_pi(math.atan2(carrot_y - y, carrot_x - x) - yaw)

        yaw_cmd = self.args.yaw_gain * heading_error
        yaw_cmd = max(-self.args.max_yaw, min(self.args.max_yaw, yaw_cmd))

        if abs(heading_error) > self.args.turn_in_place_angle:
            # Too far off course to make useful forward progress; rotate first.
            fwd_cmd = 0.0
        else:
            # Fall off with heading error, and ease off when close to the target
            # so the robot does not overshoot the 0.2 m scoring radius.
            fwd_cmd = self.args.max_forward * math.cos(heading_error)
            approach = distance / max(self.args.slowdown_distance, 1e-6)
            fwd_cmd *= min(1.0, approach)
            # Off the line by a lot: slow down and let the steering recover
            # before covering more ground.
            if abs(cross_track) > self.args.cross_track_slow:
                fwd_cmd *= self.args.cross_track_slow / abs(cross_track)
            fwd_cmd = max(self.args.min_forward, min(self.args.max_forward, fwd_cmd))

        cmd = Twist()
        cmd.linear.x = float(fwd_cmd)
        cmd.angular.z = float(yaw_cmd)
        self.cmd_pub.publish(cmd)

        self._report(distance, heading_error, fwd_cmd, yaw_cmd, cross_track)

    def _is_descent_segment(self) -> bool:
        """Whether the current route segment has a meaningful downward step."""
        # target 0 is the route start marker, not a segment from a previous
        # waypoint.  Its YAML z is 0.0, so comparing it with the live standing
        # base height would falsely enable descent mode at startup.
        if (not self.args.enable_descent_mode or self.segment_start is None
                or self.target_index <= 0):
            return False
        target_z = self._target_z(self.target_index)
        return (target_z <= self.args.descent_target_max_z
                and target_z <= self.segment_start[2] - self.args.descent_min_drop)

    def _run_descent_controller(self, distance: float, yaw: float):
        """Cross a descending rim, then navigate to its bottom marker.

        The target marker is normally displaced sideways from the entry line.
        Chasing it before crossing the rim steers a wheel into the side wall.
        Entry therefore holds the *inbound* heading from the preceding route
        segment, just as a driver holding ``W`` would.  Only after the base has
        dropped to the pit floor does the controller steer toward the bottom
        marker.  Both phases suppress generic reverse/strafe recovery.
        """
        now = self._now()
        tx, ty = self._target(self.target_index)

        if self.descent_target_index != self.target_index:
            sx, sy, _ = self.segment_start
            self.descent_target_index = self.target_index
            self.descent_started_at = now
            target_heading = math.atan2(ty - sy, tx - sx)
            # Use the path=18->19 segment direction, not the previous segment's
            # inbound heading.  The inbound heading (17->18) is nearly due east
            # while path=19 lies ~10 deg south, producing 1.47 m of lateral
            # error at the pit rim when using the inbound angle.  Aligning to
            # target_heading on the flat ground before the pit reduces that
            # error to ~0.09 m so the floor phase needs only a tiny adjustment.
            self.descent_heading = target_heading
            self.descent_phase = "entry"
            self.descent_committed = False
            self.descent_start_z = self.base_z
            self.descent_drop_seen = False
            self.descent_timeout_reported = False
            self.recovery_phase = None
            self._reset_progress_tracking(distance)
            self.get_logger().info(
                f"[NAV] Descent mode -> {self._target_label(self.target_index)}: "
                f"route_z={self.segment_start[2]:.3f}->{self._target_z(self.target_index):.3f}, "
                f"entry_heading={math.degrees(self.descent_heading):.1f}deg, "
                f"bottom_heading={math.degrees(target_heading):.1f}deg"
            )

        # Keep the body aligned to the entry lane even after the floor is
        # reached.  A bottom marker can be substantially sideways from that
        # lane; using it as a yaw target while the wheels are contact-loaded
        # caused the half-pit turn-and-stall observed in the GUI.
        heading_error = wrap_to_pi(self.descent_heading - yaw)
        if not self.descent_committed:
            # Establish the entry heading before crossing the rim.  The
            # committed phase holds this *route* heading; it does not chase
            # the moving pit-bottom marker and therefore cannot introduce the
            # large asymmetric pure-pursuit turns seen at the rim.
            if abs(heading_error) > self.args.descent_entry_heading:
                cmd = Twist()
                cmd.angular.z = float(max(-self.args.descent_align_yaw,
                                          min(self.args.descent_align_yaw,
                                              self.args.yaw_gain * heading_error)))
                self.cmd_pub.publish(cmd)
                self._report(distance, heading_error, 0.0, cmd.angular.z, 0.0)
                return

            self.descent_committed = True
            self.get_logger().info(
                f"[NAV] Descent committed: forward={self.args.descent_forward:.2f}, "
                "bounded route-heading hold, recovery suppressed"
            )

        if (self.base_z is not None and self.descent_start_z is not None
                and self.base_z <= self.descent_start_z - self.args.descent_drop_detect):
            self.descent_drop_seen = True

        if (self.descent_phase == "entry" and self.base_z is not None
                and self.descent_start_z is not None
                and self.base_z <= self.descent_start_z - self.args.descent_floor_drop):
            self.descent_phase = "floor"
            self.get_logger().info(
                "[NAV] Descent floor detected: switching from inbound heading hold "
                "to low-speed bottom-marker alignment"
            )

        elapsed = now - self.descent_started_at
        if elapsed > self.args.descent_timeout and not self.descent_timeout_reported:
            self.descent_timeout_reported = True
            self.get_logger().warn(
                f"[NAV] Descent target {self.target_index} has not been reached after "
                f"{elapsed:.1f}s (drop_seen={self.descent_drop_seen}); holding straight "
                "forward command rather than reversing on the ledge."
            )

        cmd = Twist()
        if self.descent_phase == "entry":
            cmd.linear.x = float(min(self.args.max_forward, self.args.descent_forward))
        else:
            dx, dy = tx - self.pose[0], ty - self.pose[1]
            # On the pit floor, steer toward the pit-bottom marker using a
            # direction-normalised command so the robot always has enough
            # velocity to physically enter the 0.20 m scoring radius.
            # Forward *and* backward are allowed so an overshoot can be
            # recovered without a yaw turn (yaw turns at contact caused
            # turn-and-stall in earlier tests).
            # Speed tapers from full descent_floor_forward to 40 % over the
            # last 0.5 m, but is held at 40 % minimum once inside that
            # window so the robot never stalls just outside the radius.
            if distance > 1e-3:
                fwd_err  =  math.cos(yaw) * dx + math.sin(yaw) * dy
                side_err = -math.sin(yaw) * dx + math.cos(yaw) * dy
                ux = fwd_err  / distance   # body-frame unit vector toward target
                uy = side_err / distance
                # No speed taper: the scoring check (distance <= 0.20 m) already
                # fires before this controller runs each tick, so there is no
                # overshoot risk.  Decelerating near the radius only causes the
                # robot to stall just outside the 0.20 m boundary.
                speed = self.args.descent_floor_forward
                cmd.linear.x = float(max(-self.args.descent_floor_forward,
                                         min(self.args.descent_floor_forward,
                                             ux * speed)))
                cmd.linear.y = float(max(-self.args.descent_floor_side,
                                         min(self.args.descent_floor_side,
                                             uy * speed)))
        if abs(heading_error) >= self.args.descent_heading_deadband:
            cmd.angular.z = float(max(-self.args.descent_hold_yaw,
                                      min(self.args.descent_hold_yaw,
                                          self.args.descent_yaw_gain * heading_error)))
        # There is intentionally no lateral command.  The yaw hold merely
        # counters contact-induced rotation while preserving a direct rim
        # crossing.  At the floor, lateral velocity (not a turn in place)
        # closes the marker's sideways offset.
        self.cmd_pub.publish(cmd)
        self._report(distance, heading_error, cmd.linear.x, cmd.angular.z, 0.0)

    def _reset_progress_tracking(self, distance: float):
        self.best_distance = distance
        self.last_improve_time = self._now()

    def _update_recovery(self, distance: float, heading_error: float) -> bool:
        """Detect a stall and drive the recovery manoeuvre.

        Returns True if recovery published a command this tick, meaning the
        normal pure-pursuit path should be skipped.

        The stock controller has no way out of a stall: it keeps issuing the
        same forward command, which on a ledge just grinds the wheels into the
        wall. Backing straight out is what a human does on the teleop keys, and
        the track ledges are clean vertical drops with no overhang to catch on.
        """
        now = self._now()
        if self.best_distance is None:
            self._reset_progress_tracking(distance)
            return False

        # ---- already recovering: run the manoeuvre to completion ----
        if self.recovery_phase is not None:
            elapsed = now - self.recovery_started_at
            cmd = Twist()

            if self.recovery_phase == "reverse":
                if elapsed < self.args.reverse_time:
                    cmd.linear.x = -abs(self.args.reverse_speed)
                    self.cmd_pub.publish(cmd)
                    return True
                self._enter_recovery_phase("strafe" if self.recovery_count >= 2 else "realign")
                return True

            if self.recovery_phase == "strafe":
                # Only after a plain reverse has already failed once: shuffle
                # sideways to get off whatever edge the wheels are caught on.
                if elapsed < self.args.strafe_time:
                    cmd.linear.y = self.strafe_sign * abs(self.args.strafe_speed)
                    self.cmd_pub.publish(cmd)
                    return True
                self.strafe_sign = -self.strafe_sign
                self._enter_recovery_phase("realign")
                return True

            if self.recovery_phase == "realign":
                aligned = abs(heading_error) < self.args.realign_tolerance
                if not aligned and elapsed < self.args.realign_timeout:
                    yaw_cmd = self.args.yaw_gain * heading_error
                    cmd.angular.z = float(max(-self.args.max_yaw,
                                              min(self.args.max_yaw, yaw_cmd)))
                    self.cmd_pub.publish(cmd)
                    return True
                self.get_logger().info(
                    f"[NAV] Recovery finished (attempt {self.recovery_count}), resuming"
                )
                self.recovery_phase = None
                # Measure progress afresh: after backing up the distance is
                # worse than the old best, which would re-trigger immediately.
                self._reset_progress_tracking(distance)
                self.segment_start = (self.pose[0], self.pose[1],
                                      self.base_z if self.base_z is not None else 0.0)
                return False

        # ---- not recovering: look for a stall ----
        if distance < self.best_distance - self.args.progress_epsilon:
            self._reset_progress_tracking(distance)
            return False

        if now - self.last_improve_time > self.args.stuck_time:
            self.recovery_count += 1
            self.recovery_total += 1
            self.get_logger().warn(
                f"[NAV] Stuck at target {self.target_index}: no progress for "
                f"{self.args.stuck_time:.0f}s (distance={distance:.2f}m, "
                f"best={self.best_distance:.2f}m). Recovery attempt {self.recovery_count}."
            )
            self._enter_recovery_phase("reverse")
            return True

        return False

    def _enter_recovery_phase(self, phase: str):
        self.recovery_phase = phase
        self.recovery_started_at = self._now()
        self.get_logger().info(f"[NAV] Recovery -> {phase}")

    def _advance_waypoint(self, distance: float):
        reached = self.target_index
        elapsed = 0.0
        if self.start_sim_time is not None and self.sim_time is not None:
            elapsed = self.sim_time - self.start_sim_time

        self.get_logger().info(
            f"[NAV] Reached {self._target_label(reached)} "
            f"(distance={distance:.3f}m, sim_time={self.sim_time:.2f}s, elapsed={elapsed:.2f}s, "
            f"max_cross_track={self.max_cross_track:.2f}m)"
        )
        self.max_cross_track = 0.0

        sx, sy, _ = self.segment_start
        rx, ry = self.waypoints[reached][:2]
        segment_dx, segment_dy = rx - sx, ry - sy
        if math.hypot(segment_dx, segment_dy) > 0.10:
            self.inbound_heading = math.atan2(segment_dy, segment_dx)

        self.target_index += 1
        self.last_progress_time = self._now()
        self.segment_start = self.waypoints[reached][:3]
        self.recovery_phase = None
        self.recovery_count = 0
        self.best_distance = None
        self.descent_target_index = None
        self.descent_started_at = None
        self.descent_heading = None
        self.descent_phase = None
        self.descent_committed = False

        if self.target_index >= len(self.waypoints):
            self.get_logger().info(
                f"[NAV] Course complete in {elapsed:.2f}s of sim time. "
                f"{self.recovery_total} recovery manoeuvre(s) used."
            )
            self.phase = PHASE_DONE
            self.cmd_pub.publish(Twist())

    def _report(self, distance: float, heading_error: float, fwd: float, yaw: float,
                cross_track: float):
        now = self._now()
        if self.last_progress_time is None:
            self.last_progress_time = now

        if self.target_index != self.last_report_index:
            self.last_report_index = self.target_index
            tx, ty = self._target(self.target_index)
            self.get_logger().info(
                f"[NAV] Heading to {self._target_label(self.target_index)} "
                f"({self.target_index}/{len(self.waypoints) - 1}) "
                f"target_xy=({tx:.3f},{ty:.3f}) distance={distance:.2f}m"
            )

        self.max_cross_track = max(self.max_cross_track, abs(cross_track))

        # Stuck detection: warn, but keep trying. The run is only valid if every
        # waypoint is actually entered, so giving up would be worse than looping.
        if now - self.last_progress_time > self.args.stuck_warn_time:
            self.get_logger().warn(
                f"[NAV] No waypoint reached for {self.args.stuck_warn_time:.0f}s. "
                f"target={self.target_index} distance={distance:.2f}m "
                f"cross_track={cross_track:+.2f}m "
                f"heading_error={math.degrees(heading_error):.1f}deg "
                f"cmd=({fwd:.2f}, {yaw:.2f})"
            )
            self.last_progress_time = now


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--waypoints", type=Path, default=None,
                        help=f"Path YAML to follow. Defaults to {DEFAULT_PATH} "
                             f"if it exists, otherwise {FALLBACK_PATH}")
    parser.add_argument("--via-radius", type=float, default=0.5,
                        help="Reach radius for inserted via points; they are steering "
                             "aids, not scoring points")
    parser.add_argument("--rate", type=float, default=20.0,
                        help="/cmd_vel publish rate in Hz; must stay well above "
                             "the 300ms watchdog in rl_deploy")
    parser.add_argument("--max-forward", type=float, default=0.6,
                        help="Forward velocity limit (rl_deploy clamps at 1.0)")
    parser.add_argument("--min-forward", type=float, default=0.0,
                        help="Lower bound on forward velocity while driving")
    parser.add_argument("--max-yaw", type=float, default=0.8,
                        help="Yaw rate limit (rl_deploy clamps at 1.0)")
    parser.add_argument("--yaw-gain", type=float, default=1.2,
                        help="Proportional gain from heading error to yaw rate")
    parser.add_argument("--lookahead", type=float, default=1.2,
                        help="Pure pursuit lookahead along the segment; smaller tracks the "
                             "line more tightly but oscillates more")
    parser.add_argument("--cross-track-slow", type=float, default=0.4,
                        help="Slow down when lateral offset from the segment exceeds this")
    parser.add_argument("--turn-in-place-angle", type=float, default=0.9,
                        help="Heading error in radians above which forward speed is zero")
    parser.add_argument("--slowdown-distance", type=float, default=1.5,
                        help="Start easing off the throttle within this distance of the target")
    parser.add_argument("--reach-radius", type=float, default=0.2,
                        help="Must match the simulator's scoring radius")
    parser.add_argument("--enable-descent-mode", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use straight, recovery-free control on route segments whose "
                             "target Z drops by --descent-min-drop")
    parser.add_argument("--descent-min-drop", type=float, default=0.15,
                        help="Minimum waypoint Z decrease that enables descent mode (m)")
    parser.add_argument("--descent-target-max-z", type=float, default=0.20,
                        help="Only apply descent mode to targets at or below this world Z (m)")
    parser.add_argument("--descent-forward", type=float, default=1.0,
                        help="Fixed forward command after aligning with a descent segment")
    parser.add_argument("--descent-entry-heading", type=float, default=0.18,
                        help="Maximum heading error before committing to a straight descent (rad)")
    parser.add_argument("--descent-align-yaw", type=float, default=0.30,
                        help="Yaw-rate limit used only before descent is committed (rad/s)")
    parser.add_argument("--descent-yaw-gain", type=float, default=0.8,
                        help="Route-heading hold gain after descent is committed")
    parser.add_argument("--descent-hold-yaw", type=float, default=0.25,
                        help="Yaw-rate limit for committed descent heading hold (rad/s)")
    parser.add_argument("--descent-heading-deadband", type=float, default=0.025,
                        help="Do not correct heading inside this committed-descent error (rad)")
    parser.add_argument("--descent-drop-detect", type=float, default=0.08,
                        help="Base-height reduction recorded as an observed descent (m)")
    parser.add_argument("--descent-floor-drop", type=float, default=0.32,
                        help="Base-height reduction that permits pit-bottom marker alignment (m)")
    parser.add_argument("--descent-floor-forward", type=float, default=0.45,
                        help="Forward command used after the pit floor is detected")
    parser.add_argument("--descent-floor-side", type=float, default=0.30,
                        help="Maximum lateral command used to align with the pit-bottom marker")
    parser.add_argument("--descent-floor-side-gain", type=float, default=0.50,
                        help="Lateral gain toward the pit-bottom marker after floor detection")
    parser.add_argument("--descent-timeout", type=float, default=20.0,
                        help="Warn after this many seconds in descent mode; never reverses")
    parser.add_argument("--stand-rise", type=float, default=0.20,
                        help="Base must rise by this much above its pre-stand height "
                             "to count as standing; relative so it works on raised terrain")
    parser.add_argument("--rl-settle-time", type=float, default=3.0,
                        help="Seconds to wait after entering RL control before driving")
    parser.add_argument("--stuck-warn-time", type=float, default=20.0,
                        help="Warn if no waypoint is reached for this many seconds")
    parser.add_argument("--stuck-time", type=float, default=10.0,
                        help="Trigger recovery after this long without closing on the target")
    parser.add_argument("--progress-epsilon", type=float, default=0.05,
                        help="Distance improvement that counts as progress")
    parser.add_argument("--reverse-speed", type=float, default=0.45,
                        help="Backward speed during recovery")
    parser.add_argument("--reverse-time", type=float, default=2.5,
                        help="How long to back up")
    parser.add_argument("--strafe-speed", type=float, default=0.4,
                        help="Sideways speed, used only after a plain reverse failed")
    parser.add_argument("--strafe-time", type=float, default=2.0,
                        help="How long to shuffle sideways")
    parser.add_argument("--realign-tolerance", type=float, default=0.2,
                        help="Heading error in radians that counts as re-aimed")
    parser.add_argument("--realign-timeout", type=float, default=6.0,
                        help="Give up re-aiming after this long and drive anyway")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Begin from this index in the path list. Tuning aid for "
                             "hard sections; a scored run must start from 0")
    parser.add_argument("--no-auto-start", dest="auto_start", action="store_false",
                        help="Do not send stand/rl; assume the robot is already in RL control")
    parser.set_defaults(auto_start=True)
    args, ros_args = parser.parse_known_args()
    if args.waypoints is None:
        args.waypoints = DEFAULT_PATH if DEFAULT_PATH.is_file() else FALLBACK_PATH
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = WaypointNavigator(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
