#!/usr/bin/env python3
"""Isolated behavior controller for the narrow WP25 -> WP27 stair landing.

The stock navigator remains in full control through the stair climb. A pose and
height trigger, which is not a navigation target and therefore cannot cause
approach slowdown, switches to a stop/centre/turn/traverse state machine only
after the final stair has been climbed.
"""

import argparse
import math
from pathlib import Path
import sys

import rclpy
import yaml
from geometry_msgs.msg import Twist

from waypoint_navigator import (
    PHASE_DONE,
    PHASE_NAVIGATE,
    WaypointNavigator,
    parse_args as parse_stock_args,
    wrap_to_pi,
)


CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_BEHAVIOR_CONFIG = (
    CURRENT_DIR / ".." / ".." / ".." / "config" /
    "path_wp25_wp27_behavior_test.yaml"
).resolve()

STATE_STOCK_CLIMB = "stock_climb"
STATE_FINAL_STAIR_PUSH = "final_stair_push"
STATE_SETTLE_WP26 = "settle_wp26"
STATE_TURN_EAST = "turn_east"
STATE_PLATFORM = "platform"
STATE_BRAKE_WP27 = "brake_wp27"
STATE_HOLD_WP27 = "hold_wp27"
STATE_STOCK_CONTINUE = "stock_continue"
STATE_FAILED = "failed"


def clamp(value, limit):
    return max(-limit, min(limit, value))


class WP25WP27BehaviorNavigator(WaypointNavigator):
    def __init__(self, args):
        self.behavior = self._load_behavior(args.behavior_config)
        self.behavior_state = STATE_STOCK_CLIMB
        self.state_started_at = None
        self.condition_since = None
        self.platform_point_index = 0
        self.previous_odom = None
        self.planar_speed = 0.0
        self.vertical_speed = 0.0
        self.body_forward_speed = 0.0
        self.body_lateral_speed = 0.0
        self.angular_speed = 0.0
        self.unsafe_angular_since = None
        self.wp27_success_reported = False
        self.final_stair_started_at = None
        self.last_stock_profile_index = None
        self.last_climb_profile_index = None
        self.wp25_clearance_active = False
        self.wp25_clearance_started_at = None
        super().__init__(args)

        post = self.behavior.get("post_wp27", {})
        continuation_index = int(post.get("stock_target_index", 1 << 30))
        if (post.get("enabled", False) and
                self.target_index >= continuation_index):
            self.behavior_state = STATE_STOCK_CONTINUE
            previous_index = max(0, self.target_index - 1)
            self.segment_start = self.waypoints[previous_index][:3]
            self.get_logger().info(
                f"[BEHAVIOR] Isolated continuation start at target "
                f"{self.target_index}; previous path point={previous_index}"
            )
        self.get_logger().info(
            f"[BEHAVIOR] Loaded isolated WP25->WP27 config from "
            f"{args.behavior_config}"
        )

    @staticmethod
    def _load_behavior(path):
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find behavior config: {path}")
        config = yaml.safe_load(path.read_text(encoding="ascii"))
        required = (
            "stock_path_indices", "landing_trigger", "final_stair_push",
            "turn_east", "platform", "wp27_stop",
        )
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"Behavior config is missing: {', '.join(missing)}")
        if not config["platform"].get("points"):
            raise ValueError("Behavior config has no platform points")
        return config

    def _odom_callback(self, msg):
        p = msg.pose.pose.position
        twist = msg.twist.twist
        sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.previous_odom is not None:
            previous_time, previous_x, previous_y, previous_z = self.previous_odom
            dt = sim_time - previous_time
            if 1e-4 < dt < 0.2:
                measured_planar = math.hypot(p.x - previous_x, p.y - previous_y) / dt
                measured_vertical = (p.z - previous_z) / dt
                # Suppress single contact impulses without hiding sustained motion.
                self.planar_speed = 0.7 * self.planar_speed + 0.3 * measured_planar
                self.vertical_speed = 0.7 * self.vertical_speed + 0.3 * measured_vertical
        # MuJoCo publishes twist in the base frame. Keep signed components so
        # the WP27 stop phase can actively cancel residual forward drift.
        self.body_forward_speed = float(twist.linear.x)
        self.body_lateral_speed = float(twist.linear.y)
        self.angular_speed = math.sqrt(
            float(twist.angular.x) ** 2 +
            float(twist.angular.y) ** 2 +
            float(twist.angular.z) ** 2
        )
        self.previous_odom = (sim_time, p.x, p.y, p.z)
        super()._odom_callback(msg)

    def _sim_elapsed(self):
        if self.state_started_at is None or self.sim_time is None:
            return 0.0
        return self.sim_time - self.state_started_at

    def _enter_behavior_state(self, state, message):
        self.behavior_state = state
        self.state_started_at = self.sim_time
        self.condition_since = None
        self.cmd_pub.publish(Twist())
        x, y, _ = self.pose
        self.get_logger().info(
            f"[BEHAVIOR] {message}; state={state}, pose=({x:.3f}, {y:.3f}, "
            f"{self.base_z:.3f}), speed={self.planar_speed:.3f}m/s"
        )

    def _condition_stable(self, condition, duration):
        if not condition:
            self.condition_since = None
            return False
        if self.condition_since is None:
            self.condition_since = self.sim_time
            return False
        return self.sim_time - self.condition_since >= duration

    def _landing_triggered(self):
        trigger = self.behavior["landing_trigger"]
        x, y, _ = self.pose
        condition = (
            y >= float(trigger["min_y"])
            and self.base_z >= float(trigger["min_base_z"])
            and abs(self.vertical_speed) <= float(trigger["max_abs_vertical_speed"])
        )
        return self._condition_stable(condition, float(trigger["stable_time"]))

    def _platform_resume_ready(self):
        """Resume an isolated platform test without repeating the stair climb."""
        # A continuation run must start from the verified WP27 scoring circle.
        # Allowing the older platform-resume shortcut here lets a robot that
        # drifted while waiting for RL startup re-enter at x~33.8 and replay
        # platform points, which looks like a retreat to the previous target.
        if (self._post_wp27_enabled() or self.args.auto_start or
                self.pose is None or self.base_z is None):
            return False
        platform = self.behavior["platform"]
        return (
            self.pose[0] >= float(platform["resume_min_x"]) and
            abs(self.pose[1] - float(platform["center_y"])) <=
            float(platform["max_center_error"]) and
            self.base_z >= float(platform["resume_min_base_z"])
        )

    def _post_wp27_enabled(self):
        return bool(self.behavior.get("post_wp27", {}).get("enabled", False))

    def _post_wp27_resume_ready(self):
        """Start a continuation-only test from a verified WP27 spawn pose."""
        if (not self._post_wp27_enabled() or self.phase != PHASE_NAVIGATE or
                self.pose is None or self.base_z is None):
            return False
        post = self.behavior["post_wp27"]
        target = tuple(float(value) for value in self.behavior["wp27_stop"]["target"])
        return (
            math.hypot(target[0] - self.pose[0], target[1] - self.pose[1]) <=
            float(post["resume_radius"]) and
            self.base_z >= float(post["min_resume_base_z"])
        )

    def _handoff_to_stock_target(self, message):
        """Give WP28 and every later waypoint back to the stock navigator."""
        post = self.behavior["post_wp27"]
        target_index = int(post["stock_target_index"])
        target = self._target(target_index)
        self.behavior_state = STATE_STOCK_CONTINUE
        self.target_index = target_index
        self.phase = PHASE_NAVIGATE
        stock_start = tuple(float(value) for value in post["stock_segment_start"])
        self.segment_start = (
            stock_start[0], stock_start[1],
            self.base_z if self.base_z is not None else 0.0,
        )
        self.recovery_phase = None
        self.recovery_count = 0
        self.best_distance = None
        self.last_progress_time = self._now()
        self.get_logger().info(
            f"[BEHAVIOR] {message}; stock target index={target_index}, "
            f"target=({target[0]:.3f}, {target[1]:.3f})"
        )
        # Publish the stock target-following command in this same control tick;
        # do not insert a zero-command pause between WP27 and WP28.
        self._stock_continue_step()

    def _stock_continue_step(self):
        """Follow the current path target with its configured speed profile."""
        if self.phase == PHASE_DONE or self.target_index >= len(self.waypoints):
            self.cmd_pub.publish(Twist())
            return
        post = self.behavior["post_wp27"]
        default = post.get("stock_speed_default", {})
        profiles = post.get("stock_speed_profiles", {})
        profile = profiles.get(self.target_index, profiles.get(str(self.target_index), {}))

        saved = {
            "max_forward": self.args.max_forward,
            "max_yaw": self.args.max_yaw,
            "slowdown_distance": self.args.slowdown_distance,
            "lookahead": self.args.lookahead,
            "turn_in_place_angle": self.args.turn_in_place_angle,
            "cross_track_slow": self.args.cross_track_slow,
        }
        via = post.get("stock_via_speed", {})
        is_via = self.waypoints[self.target_index][3] == "via"
        for name in saved:
            source = via if is_via else profile
            value = source.get(name, default.get(name))
            if value is not None:
                setattr(self.args, name, float(value))

        if self.last_stock_profile_index != self.target_index:
            self.get_logger().info(
                f"[BEHAVIOR] Target {self.target_index} speed profile: "
                f"forward={self.args.max_forward:.2f}, yaw={self.args.max_yaw:.2f}, "
                f"slowdown={self.args.slowdown_distance:.2f}m"
                + (", via alignment mode" if is_via else "")
            )
            self.last_stock_profile_index = self.target_index
        try:
            if is_via:
                self._via_continue_step()
            else:
                super()._control_step()
        finally:
            for name, value in saved.items():
                setattr(self.args, name, value)

    def _stock_climb_step(self):
        """Apply momentum-preserving profiles only on known stair targets."""
        profiles = self.behavior.get("stock_climb_speed_profiles", {})
        profile = profiles.get(
            self.target_index,
            profiles.get(str(self.target_index), {}),
        )
        if not profile:
            super()._control_step()
            return

        saved = {
            "max_forward": self.args.max_forward,
            "slowdown_distance": self.args.slowdown_distance,
        }
        for name in saved:
            if name in profile:
                setattr(self.args, name, float(profile[name]))
        if self.last_climb_profile_index != self.target_index:
            self.get_logger().info(
                f"[BEHAVIOR] Stair target {self.target_index}: keeping "
                f"forward={self.args.max_forward:.2f} through "
                f"slowdown={self.args.slowdown_distance:.2f}m"
            )
            self.last_climb_profile_index = self.target_index
        try:
            super()._control_step()
        finally:
            for name, value in saved.items():
                setattr(self.args, name, value)

    def _via_continue_step(self):
        """Move toward a via point while turning, instead of stopping to spin.

        The policy accepts forward, lateral, and yaw commands.  The stock
        waypoint controller only publishes forward plus yaw and sets forward
        to zero when the heading error is large.  That combination leaves S10
        stationary at a side-corridor waypoint because the learned controller
        does not reliably rotate in place.  A bounded body-frame vector keeps
        the robot making progress toward the world-frame via point while the
        yaw command brings the chassis around.  Negative forward commands are
        suppressed so this recovery cannot send the robot back over a ledge.
        """
        if self.pose is None:
            self.cmd_pub.publish(Twist())
            return

        x, y, yaw = self.pose
        tx, ty = self._target(self.target_index)
        dx, dy = tx - x, ty - y
        distance = math.hypot(dx, dy)
        if distance <= self._radius(self.target_index):
            self._advance_waypoint(distance)
            return
        desired_yaw = math.atan2(dy, dx)
        heading_error = wrap_to_pi(desired_yaw - yaw)

        # Keep the bounded lateral recovery as a last resort.  It publishes
        # its own command while active, so the normal guidance is skipped.
        if self._update_recovery(distance, heading_error):
            return

        c, s = math.cos(yaw), math.sin(yaw)
        body_forward = c * dx + s * dy
        body_lateral = -s * dx + c * dy
        norm = max(distance, 1e-6)

        slowdown = max(float(self.args.slowdown_distance), 1e-6)
        speed = float(self.args.max_forward) * min(1.0, distance / slowdown)
        # Stay responsive in the final approach without jumping across the
        # small via radius.  The target-radius check runs before this method.
        speed = max(0.06, speed)

        command = Twist()
        command.linear.x = float(clamp(
            max(0.0, speed * body_forward / norm),
            float(self.args.max_forward),
        ))
        command.linear.y = float(clamp(
            speed * body_lateral / norm,
            float(self.args.max_forward),
        ))
        command.angular.z = float(clamp(
            self.args.yaw_gain * heading_error,
            float(self.args.max_yaw),
        ))
        self.cmd_pub.publish(command)

        self._report(
            distance,
            heading_error,
            command.linear.x,
            command.angular.z,
            0.0,
        )

    def _in_final_stair_zone(self):
        if self.pose is None or self.base_z is None:
            return False
        zone = self.behavior["final_stair"]
        wp26_index = int(self.behavior["stock_path_indices"]["wp26"])
        return (
            self.target_index == wp26_index
            and self.pose[1] >= float(zone["min_y"])
            and self.base_z >= float(zone["min_base_z"])
        )

    def _update_recovery(self, distance, heading_error):
        """Keep the stock climb command continuous while approaching WP26.

        The parent detector only watches XY distance. On each riser the base
        gains height while XY progress can pause, so it falsely starts a
        reverse recovery and backs away from an otherwise valid climb.
        """
        wp26_index = int(self.behavior["stock_path_indices"]["wp26"])
        if (self.behavior_state == STATE_STOCK_CLIMB and
                self.target_index == wp26_index):
            if self._in_final_stair_zone() and self.final_stair_started_at is None:
                self.final_stair_started_at = self.sim_time
                self.get_logger().info(
                    "[BEHAVIOR] Entered final stair zone; continuous WP26 climb "
                    "remains active"
                )
            self.recovery_phase = None
            self.recovery_count = 0
            self._reset_progress_tracking(distance)
            return False
        post = self.behavior.get("post_wp27", {})
        pole_recovery_targets = {
            int(index) for index in post.get("pole_recovery_target_indices", [])
        }
        # The same forward-yaw escape is also useful for a marked pole before
        # WP25.  The verified baseline leaves this list empty, so its behavior
        # is unchanged.
        if self.target_index in pole_recovery_targets:
            return self._update_pole_recovery(distance)

        if self.behavior_state == STATE_STOCK_CONTINUE:
            via_recovery_from = int(post.get("via_recovery_from_index", 1 << 30))
            is_via = (
                self.target_index < len(self.waypoints) and
                self.waypoints[self.target_index][3] == "via"
            )
            if self.target_index >= via_recovery_from and is_via:
                return self._update_via_recovery(distance, heading_error)
            if self.target_index < int(post["allow_recovery_from_index"]):
                # Turning toward or climbing WP28 can leave XY distance nearly
                # unchanged.  The generic detector would interpret that as a
                # stall and command reverse, sending the robot back to WP27.
                self.recovery_phase = None
                self.recovery_count = 0
                self._reset_progress_tracking(distance)
                return False
        return super()._update_recovery(distance, heading_error)

    def _update_pole_recovery(self, distance):
        """Keep moving while alternating yaw to release a shallow pole contact."""
        config = self.behavior["post_wp27"]["pole_recovery"]
        now = self._now()

        if self.recovery_phase == "pole_wiggle":
            elapsed = now - self.recovery_started_at
            if elapsed < float(config["wiggle_time"]):
                command = Twist()
                command.linear.x = float(config["forward_speed"])
                command.angular.z = float(
                    self.strafe_sign * abs(float(config["yaw_speed"]))
                )
                self.cmd_pub.publish(command)
                return True
            self.strafe_sign = -self.strafe_sign
            self.get_logger().info(
                f"[BEHAVIOR] Pole wiggle {self.recovery_count} complete; "
                "re-aiming at WP30 from the current pose"
            )
            self.recovery_phase = None
            self._reset_progress_tracking(distance)
            self.segment_start = (
                self.pose[0], self.pose[1],
                self.base_z if self.base_z is not None else 0.0,
            )
            return False

        if self.best_distance is None:
            self._reset_progress_tracking(distance)
            return False
        if distance < self.best_distance - float(config["progress_epsilon"]):
            self._reset_progress_tracking(distance)
            return False
        if now - self.last_improve_time <= float(config["stuck_time"]):
            return False

        self.recovery_count += 1
        self.recovery_total += 1
        if self.recovery_count > int(config["max_attempts"]):
            self._fail(
                f"WP30 remained blocked after {config['max_attempts']} "
                "bounded forward-yaw wiggles"
            )
            return True
        self.get_logger().warn(
            f"[BEHAVIOR] WP30 pole contact at {distance:.2f}m; "
            f"forward-yaw wiggle {self.recovery_count}/"
            f"{config['max_attempts']} (no reverse)"
        )
        self._enter_recovery_phase("pole_wiggle")
        return True

    def _update_via_recovery(self, distance, heading_error):
        """Use a short lateral shuffle at a stalled via point, never reverse."""
        config = self.behavior["post_wp27"]["via_recovery"]
        now = self._now()

        if self.recovery_phase == "via_strafe":
            elapsed = now - self.recovery_started_at
            if elapsed < float(config["strafe_time"]):
                command = Twist()
                command.linear.y = float(
                    self.strafe_sign * abs(float(config["strafe_speed"]))
                )
                self.cmd_pub.publish(command)
                return True
            self.strafe_sign = -self.strafe_sign
            self.get_logger().info(
                f"[BEHAVIOR] Via recovery {self.recovery_count} complete; "
                "resuming moving target guidance without an in-place turn"
            )
            self.recovery_phase = None
            self._reset_progress_tracking(distance)
            self.segment_start = (
                self.pose[0], self.pose[1],
                self.base_z if self.base_z is not None else 0.0,
            )
            return False

        if self.best_distance is None:
            self._reset_progress_tracking(distance)
            return False
        if distance < self.best_distance - float(config["progress_epsilon"]):
            self._reset_progress_tracking(distance)
            return False
        if now - self.last_improve_time <= float(config["stuck_time"]):
            return False

        self.recovery_count += 1
        self.recovery_total += 1
        if self.recovery_count > int(config["max_attempts"]):
            self._fail(
                f"via target {self.target_index} remained blocked after "
                f"{config['max_attempts']} bounded lateral recoveries"
            )
            return True
        self.get_logger().warn(
            f"[BEHAVIOR] Via target {self.target_index} stalled at "
            f"{distance:.2f}m; lateral recovery {self.recovery_count}/"
            f"{config['max_attempts']} (no reverse)"
        )
        self._enter_recovery_phase("via_strafe")
        return True

    def _publish_target_command(self, target, max_forward, min_forward, max_yaw,
                                slowdown_distance=0.35):
        x, y, yaw = self.pose
        tx, ty = target
        distance = math.hypot(tx - x, ty - y)
        desired_yaw = math.atan2(ty - y, tx - x)
        heading_error = wrap_to_pi(desired_yaw - yaw)

        command = Twist()
        command.angular.z = float(clamp(1.2 * heading_error, max_yaw))
        if abs(heading_error) <= 0.35:
            speed = max_forward * min(1.0, distance / slowdown_distance)
            command.linear.x = float(max(min_forward, speed))
        self.cmd_pub.publish(command)
        return distance, heading_error

    def _advance_waypoint(self, distance):
        """Keep driving straight after WP25 until the rear wheels clear the pole."""
        reached = self.target_index
        super()._advance_waypoint(distance)

        clearance = self.behavior.get("wp25_clearance", {})
        if (
            self.behavior_state == STATE_STOCK_CLIMB
            and bool(clearance.get("enabled", False))
            and reached == int(clearance.get("path_index", -1))
            and self.phase != PHASE_DONE
        ):
            self.wp25_clearance_active = True
            self.wp25_clearance_started_at = self.sim_time
            target = tuple(float(value) for value in clearance["target"])
            self.get_logger().info(
                f"[BEHAVIOR] WP25 scored; driving straight through to chassis "
                f"clearance target=({target[0]:.3f}, {target[1]:.3f}) before "
                "turning toward WP26"
            )

    def _wp25_clearance_step(self):
        clearance = self.behavior["wp25_clearance"]
        target = tuple(float(value) for value in clearance["target"])
        distance = math.hypot(target[0] - self.pose[0], target[1] - self.pose[1])

        if distance <= float(clearance["radius"]):
            self.wp25_clearance_active = False
            self.wp25_clearance_started_at = None
            self.segment_start = (
                self.pose[0], self.pose[1],
                self.base_z if self.base_z is not None else 0.0,
            )
            self.recovery_phase = None
            self.recovery_count = 0
            self.best_distance = None
            self.get_logger().info(
                f"[BEHAVIOR] WP25 chassis clearance complete at "
                f"distance={distance:.3f}m; turning toward WP26"
            )
            super()._control_step()
            return

        elapsed = self.sim_time - self.wp25_clearance_started_at
        if elapsed > float(clearance["max_time"]):
            self._fail(
                f"WP25 straight clearance exceeded {clearance['max_time']:.1f}s; "
                "stopping instead of reversing into the pole"
            )
            return

        self._publish_target_command(
            target,
            float(clearance["max_forward"]),
            float(clearance["min_forward"]),
            float(clearance["max_yaw"]),
            float(clearance["slowdown_distance"]),
        )

    def _world_hold_command(self, target, position_gain, max_speed,
                            allow_reverse=True):
        """Return a bounded body-frame correction toward a world-frame point."""
        x, y, yaw = self.pose
        dx_world = float(target[0]) - x
        dy_world = float(target[1]) - y
        c, s = math.cos(yaw), math.sin(yaw)
        forward = position_gain * (c * dx_world + s * dy_world)
        lateral = position_gain * (-s * dx_world + c * dy_world)
        command = Twist()
        command.linear.x = float(clamp(forward, max_speed))
        if not allow_reverse:
            command.linear.x = max(0.0, command.linear.x)
        command.linear.y = float(clamp(lateral, max_speed))
        return command

    def _publish_wp27_hold_command(self, target, config):
        """Apply a very small body-frame correction while holding WP27.

        A zero command alone can leave residual dynamics on the raised platform.
        The correction is deliberately capped below the normal traverse speed.
        """
        x, y, yaw = self.pose
        dx_world = float(target[0]) - x
        dy_world = float(target[1]) - y
        c, s = math.cos(yaw), math.sin(yaw)
        error_forward = c * dx_world + s * dy_world
        error_lateral = -s * dx_world + c * dy_world
        command = Twist()
        gain = float(config["position_gain"])
        command.linear.x = float(clamp(gain * error_forward, float(config["max_hold_speed"])))
        command.linear.y = float(clamp(gain * error_lateral, float(config["max_hold_speed"])))
        # Damp any remaining body-frame velocity without commanding a large
        # correction that could push the robot across the platform edge.
        command.linear.x += float(clamp(
            -float(config["velocity_gain"]) * self.body_forward_speed,
            float(config["max_hold_speed"]),
        ))
        command.linear.y += float(clamp(
            -float(config["velocity_gain"]) * self.body_lateral_speed,
            float(config["max_hold_speed"]),
        ))
        command.linear.x = float(clamp(command.linear.x, float(config["max_hold_speed"])))
        command.linear.y = float(clamp(command.linear.y, float(config["max_hold_speed"])))
        self.cmd_pub.publish(command)

    def _wp27_stop_step(self):
        stop = self.behavior["wp27_stop"]
        target = tuple(float(value) for value in stop["target"])
        elapsed = self._sim_elapsed()
        command = Twist()
        if self.behavior_state == STATE_BRAKE_WP27:
            # Brake only the signed forward component. Lateral damping keeps a
            # diagonal arrival from sliding toward the platform edge.
            forward = max(0.0, self.body_forward_speed)
            command.linear.x = float(-min(
                float(stop["max_reverse"]),
                float(stop["brake_gain"]) * forward,
            ))
            command.linear.y = float(clamp(
                -float(stop["lateral_damping"]) * self.body_lateral_speed,
                float(stop["max_reverse"]),
            ))
            self.cmd_pub.publish(command)
            speed_ok = self.planar_speed <= float(stop["max_stop_speed"])
            if (elapsed >= float(stop["min_brake_time"]) and
                    self._condition_stable(speed_ok, float(stop["stable_time"]))):
                self._enter_behavior_state(
                    STATE_HOLD_WP27,
                    "WP27 residual motion was actively braked",
                )
            elif elapsed >= float(stop["max_brake_time"]):
                self._enter_behavior_state(
                    STATE_HOLD_WP27,
                    "WP27 brake timeout reached; switching to bounded hold",
                )
            return

        self._publish_wp27_hold_command(target, stop)
        distance = math.hypot(target[0] - self.pose[0], target[1] - self.pose[1])
        stable = (
            distance <= float(stop["hold_radius"]) and
            self.planar_speed <= float(stop["max_hold_stop_speed"]) and
            abs(self.vertical_speed) <= float(stop["max_abs_vertical_speed"])
        )
        if (not self.wp27_success_reported and
                self._condition_stable(stable, float(stop["hold_stable_time"]))):
            self.wp27_success_reported = True
            self.get_logger().info(
                f"[BEHAVIOR] SUCCESS: WP27 held for "
                f"{stop['hold_stable_time']:.2f}s at distance={distance:.3f}m"
            )
            if self._post_wp27_enabled():
                self._handoff_to_stock_target(
                    "continuing after WP27 with stock waypoint navigation",
                )
            else:
                self.phase = PHASE_DONE

    def _fail(self, reason):
        self.behavior_state = STATE_FAILED
        self.phase = PHASE_DONE
        self.cmd_pub.publish(Twist())
        x, y, _ = self.pose
        self.get_logger().error(
            f"[BEHAVIOR] SAFETY STOP: {reason}; pose=({x:.3f}, {y:.3f}, "
            f"{self.base_z:.3f})"
        )

    def _control_step(self):
        if self.behavior_state == STATE_FAILED:
            self.cmd_pub.publish(Twist())
            return

        # A continuation-only test may begin at a via point. Let the stock
        # startup sequence complete before any specialized via command is sent.
        if self.args.auto_start and self.phase != PHASE_NAVIGATE:
            super()._control_step()
            return

        if self.wp25_clearance_active:
            if self.pose is None or self.sim_time is None:
                self.cmd_pub.publish(Twist())
                return
            self._wp25_clearance_step()
            return

        post = self.behavior.get("post_wp27", {})
        check_angular_safety = (
            self.behavior_state == STATE_STOCK_CONTINUE and
            self.target_index >= int(post.get("safety_check_from_index", 1 << 30))
        )
        angular_unsafe = (
            check_angular_safety and
            self.angular_speed > float(post.get("max_safe_angular_speed", 5.0))
        )
        if angular_unsafe:
            if self.unsafe_angular_since is None:
                self.unsafe_angular_since = self.sim_time
            elif (self.sim_time is not None and
                    self.sim_time - self.unsafe_angular_since >=
                    float(post.get("max_unsafe_angular_duration", 0.20))):
                self._fail(
                    f"angular speed stayed above the safety limit "
                    f"({self.angular_speed:.2f}rad/s); stopping before the "
                    "robot is thrown by an obstacle"
                )
                return
        else:
            self.unsafe_angular_since = None

        if self.behavior_state == STATE_STOCK_CLIMB and self._post_wp27_resume_ready():
            self.wp27_success_reported = True
            self._handoff_to_stock_target(
                "resuming at verified WP27 pose with stock waypoint navigation",
            )
            return

        if self.behavior_state == STATE_STOCK_CLIMB and self._platform_resume_ready():
            self._enter_behavior_state(
                STATE_PLATFORM,
                "resuming isolated test from a verified upper-platform pose",
            )
            return

        # Keep the stock controller, with target-local speed profiles, in charge
        # through WP25 and the stair climb before the measured landing event.
        if self.behavior_state == STATE_STOCK_CLIMB:
            if self.phase == PHASE_NAVIGATE and self.pose is not None:
                wp26_index = int(self.behavior["stock_path_indices"]["wp26"])
                if self._in_final_stair_zone() and self.final_stair_started_at is not None:
                    max_climb_time = float(self.behavior["final_stair"]["max_climb_time"])
                    if (max_climb_time > 0.0 and
                            self.sim_time - self.final_stair_started_at > max_climb_time):
                        self._fail(
                            f"final stair climb exceeded {max_climb_time:.1f}s; "
                            "reverse recovery intentionally not used"
                        )
                        return
                if self.target_index == wp26_index and self._landing_triggered():
                    self._enter_behavior_state(
                        STATE_FINAL_STAIR_PUSH,
                        "final stair zone detected; continuing north until rear wheels land",
                    )
                    return
                if (self.target_index == wp26_index and
                        self._in_final_stair_zone() and
                        bool(self.behavior["final_stair"].get(
                            "takeover_on_zone", False
                        ))):
                    self._enter_behavior_state(
                        STATE_FINAL_STAIR_PUSH,
                        "front wheels reached the final stair; applying the "
                        "straight landing push",
                    )
                    return
                if self.target_index > wp26_index:
                    self._enter_behavior_state(
                        STATE_FINAL_STAIR_PUSH,
                        "fallback takeover after stock controller advanced near WP26",
                    )
                    return
            self._stock_climb_step()
            return

        if self.pose is None:
            self.cmd_pub.publish(Twist())
            return

        if self.behavior_state == STATE_STOCK_CONTINUE:
            self._stock_continue_step()
            return

        push = self.behavior["final_stair_push"]
        push_target = tuple(float(value) for value in push["target"])
        if self.behavior_state == STATE_FINAL_STAIR_PUSH:
            elapsed = self._sim_elapsed()
            if self.pose[1] > float(push["max_y"]):
                self._fail("northward drift during final stair push")
                return
            distance, _ = self._publish_target_command(
                push_target,
                float(push["max_forward"]),
                float(push["min_forward"]),
                float(push["max_yaw"]),
                float(push["slowdown_distance"]),
            )
            fully_landed = (
                self.pose[1] >= float(push["min_landed_y"]) and
                self.base_z >= float(push["min_base_z"]) and
                distance <= float(push["radius"])
            )
            if fully_landed:
                self._enter_behavior_state(
                    STATE_SETTLE_WP26,
                    "rear wheels reached the upper platform inside WP26",
                )
            elif elapsed >= float(push["max_time"]):
                self._fail("final stair push timed out before rear-wheel landing")
            return

        if self.behavior_state == STATE_SETTLE_WP26:
            command = self._world_hold_command(
                push_target,
                float(push["hold_position_gain"]),
                float(push["max_hold_speed"]),
                allow_reverse=False,
            )
            self.cmd_pub.publish(command)
            settled = self._condition_stable(
                self.planar_speed <= float(push["max_settle_speed"]),
                float(push["settle_stable_time"]),
            )
            if (self._sim_elapsed() >= float(push["min_settle_time"]) and settled):
                self._enter_behavior_state(STATE_TURN_EAST, "WP26 settled; turning east in place")
            elif self._sim_elapsed() >= float(push["max_settle_time"]):
                self._enter_behavior_state(
                    STATE_TURN_EAST,
                    "WP26 settle timeout reached; turning while holding platform pose",
                )
            return

        if self.behavior_state in (STATE_BRAKE_WP27, STATE_HOLD_WP27):
            self._wp27_stop_step()
            return

        platform = self.behavior["platform"]
        center_error = self.pose[1] - float(platform["center_y"])
        if abs(center_error) > float(platform["max_center_error"]):
            self._fail(f"platform centre error {center_error:+.3f}m")
            return

        if self.behavior_state == STATE_TURN_EAST:
            turn = self.behavior["turn_east"]
            yaw_error = wrap_to_pi(float(turn["yaw"]) - self.pose[2])
            command = self._world_hold_command(
                push_target,
                float(push["turn_hold_position_gain"]),
                float(push["turn_max_hold_speed"]),
                allow_reverse=False,
            )
            command.angular.z = float(clamp(1.0 * yaw_error, float(turn["max_yaw"])))
            self.cmd_pub.publish(command)
            aligned = abs(yaw_error) <= float(turn["tolerance"])
            if self._condition_stable(aligned, float(turn["stable_time"])):
                self._enter_behavior_state(STATE_PLATFORM, "east alignment stable")
            return

        if self.behavior_state == STATE_PLATFORM:
            point = platform["points"][self.platform_point_index]
            target = tuple(float(value) for value in point["pos"])
            distance, _ = self._publish_target_command(
                target,
                float(platform["max_forward"]),
                float(platform["min_forward"]),
                float(platform["max_yaw"]),
            )
            if distance > float(point["radius"]):
                return

            if point.get("stop_after"):
                if self._post_wp27_enabled():
                    self.wp27_success_reported = True
                    self._handoff_to_stock_target(
                        f"entered WP{point.get('scoring_waypoint', 27)} circle; "
                        "continuing with stock waypoint navigation",
                    )
                else:
                    self._enter_behavior_state(
                        STATE_BRAKE_WP27,
                        f"entered WP{point.get('scoring_waypoint', 27)} circle; braking",
                    )
                return

            self.get_logger().info(
                f"[BEHAVIOR] Reached platform point {self.platform_point_index + 1}/"
                f"{len(platform['points'])} at distance={distance:.3f}m"
            )
            self.platform_point_index += 1
            return


def parse_args():
    behavior_parser = argparse.ArgumentParser(add_help=False)
    behavior_parser.add_argument(
        "--behavior-config", type=Path, default=DEFAULT_BEHAVIOR_CONFIG,
    )
    behavior_args, remaining = behavior_parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args, ros_args = parse_stock_args()
    finally:
        sys.argv = original_argv
    args.behavior_config = behavior_args.behavior_config.resolve()
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = WP25WP27BehaviorNavigator(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
