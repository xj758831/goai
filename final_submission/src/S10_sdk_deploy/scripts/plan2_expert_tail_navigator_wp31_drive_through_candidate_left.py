#!/usr/bin/env python3
"""Plan2 navigator with the user's demonstrated WP31/WP32 manoeuvre.

The normal waypoint navigator remains in charge through path index 41.  The
tail controller then reproduces the structure of the 2026-08-14 expert run as
a closed-loop manoeuvre instead of replaying timestamps or joint commands.
"""

import argparse
import math
from pathlib import Path
import sys

from geometry_msgs.msg import Twist

import waypoint_navigator as base
import waypoint_navigator_wp25_wp27_behavior as platform


HANDOFF_INDEX = 41
TAIL_PATH_START_INDEX = 38
POST_PIT_ENTRY_TARGET_INDEX = 21
POST_PIT_STAIRS_TARGET_INDEX = 22

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_BEHAVIOR_CONFIG = (
    CURRENT_DIR / ".." / ".." / ".." / "config" /
    "plan2_wp25_wp27_behavior.yaml"
).resolve()

WP31 = (29.9175, 20.6175)
WP32 = (32.9250, 18.4500)
OFFICIAL_RADIUS = 0.20

# Short dwell windows preserve command settling without spending several
# seconds stopped at every transition.  The higher tail minimum prevents the
# low-level policy from creeping through the final few centimetres to a via.
ALIGN_SETTLE_TIME = 0.15
WP31_SETTLE_TIME = 0.25
WP32_APPROACH_SETTLE_TIME = 0.40
TAIL_MIN_FORWARD = 0.20
EXPERT_FORWARD_LIMIT = 0.85
POST_PIT_ENTRY_MAX_FORWARD = 0.55
POST_PIT_STAIRS_MIN_FORWARD = 0.65
POST_PIT_MIN_FORWARD = 0.20
WP24_EXIT_VIA_INDEX = 31
WP24_EXIT_VIA_RADIUS = 0.22
PLATFORM_TURN_GATE = (34.1800, 21.2500)
PLATFORM_TURN_TRANSLATION_LIMIT = 0.34
PLATFORM_TURN_POSITION_GAIN = 0.90
PLATFORM_TURN_YAW_LIMIT = 1.00
PLATFORM_TURN_TOLERANCE = math.radians(10.0)
PLATFORM_TURN_STABLE_TIME = 0.18
PLATFORM_TURN_TIMEOUT = 8.0

# Candidate only: continue through the official WP31 circle instead of stopping
# at a near-edge inner point.  The line from the right gate crosses the official
# circle near x=30.08, leaving useful margin to the unchanged WP31 centre.
WP31_INNER = (30.0750, 20.6000)
WP31_DRIVE_THROUGH = (29.9700, 20.9000)
# Hold a little farther right while clearing the wall, then converge into the
# unchanged WP31 scoring circle.  The expert line was near x=30.10 here.
WP31_RIGHT_GATE = (30.2000, 19.3500)
TURN_ANCHOR = (30.1000, 18.5000)
TURN_ANCHOR_REACH_RADIUS = 0.10
WP32_APPROACH = (31.4500, 18.3600)
# Keep aiming beyond WP32 so the controller crosses the scoring circle on a
# straight eastbound line instead of turning back as soon as it passes it.
WP32_DRIVE_THROUGH = (33.3000, 18.3600)
WP32_HOLD_TARGET = (33.2000, 18.4500)
WP31_NO_PROGRESS_TIMEOUT = 8.0
WP31_RECOVERY_ALIGN_TIMEOUT = 12.0
WP31_RECOVERY_REVERSE_TIMEOUT = 12.0
WP31_MAX_RETRIES = 3

PHASE_ALIGN_WP31 = "align_wp31"
PHASE_DRIVE_WP31_GATE = "drive_wp31_right_gate"
PHASE_ALIGN_WP31_FINAL = "align_wp31_final"
PHASE_DRIVE_WP31 = "drive_wp31"
PHASE_RECOVER_WP31_ALIGN = "recover_wp31_align"
PHASE_RECOVER_WP31_REVERSE = "recover_wp31_reverse"
PHASE_SETTLE_WP31 = "settle_wp31"
PHASE_REVERSE = "reverse_to_anchor"
PHASE_ALIGN_WP32 = "align_wp32"
PHASE_DRIVE_WP32_APPROACH = "drive_wp32_approach"
PHASE_SETTLE_WP32_APPROACH = "settle_wp32_approach"
PHASE_ALIGN_WP32_FINAL = "align_wp32_final"
PHASE_DRIVE_WP32 = "drive_wp32"
PHASE_COMPLETE = "complete"


def clip(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class Plan2ExpertTailNavigator(platform.WP25WP27BehaviorNavigator):
    def __init__(self, args):
        self.expert_mode = False
        self.expert_phase = None
        self.expert_phase_started = None
        self.align_stable_since = None
        self.wp31_reached = False
        self.wp31_retry_count = 0
        self.wp32_reached = False
        self.wp32_hold_active = False
        self.last_expert_report = -1e9
        self.phase_best_error = float("inf")
        self.phase_last_progress = None
        self.wp25_corridor_active = False
        self.wp25_corridor_phase = None
        self.wp25_corridor_started_at = None
        self.wp25_corridor_aligned_since = None
        self.wp25_corridor_scored = False
        self.wp24_retrace_active = False
        self.wp24_retrace_phase = None
        self.wp24_retrace_started_at = None
        self.wp24_retrace_aligned_since = None
        self.wp24_retrace_last_report = -1e9
        self.wp24_retrace_attempt = 1
        self.wp24_approach_active = False
        self.wp24_approach_phase = None
        self.wp24_approach_started_at = None
        self.wp24_approach_aligned_since = None
        self.wp24_approach_last_report = -1e9
        super().__init__(args)
        self.get_logger().info(
            f"Plan2 expert tail armed at path index {HANDOFF_INDEX}"
        )
        if args.start_index >= 43:
            self.expert_mode = True
            self.wp31_reached = True
            self._enter_expert_phase(PHASE_ALIGN_WP32)
            self.get_logger().info(
                "PLAN2 EXPERT RESUME: continuing from the post-WP31 turn"
            )
        elif args.start_index == 42:
            self.expert_mode = True
            self._enter_expert_phase(PHASE_ALIGN_WP31)
            self.get_logger().info(
                "PLAN2 EXPERT RESUME: continuing from the WP31 approach"
            )

    def _advance_waypoint(self, distance: float):
        reached = self.target_index
        super()._advance_waypoint(distance)

        approach = self.behavior.get("wp24_approach", {})
        if (
            bool(approach.get("enabled", False))
            and reached == int(approach.get("after_path_index", -1))
        ):
            self.wp24_approach_active = True
            self.wp24_approach_phase = "align_right"
            self.wp24_approach_started_at = self.sim_time
            self.wp24_approach_aligned_since = None
            self.target_index = int(approach["target_path_index"])
            self.segment_start = self.waypoints[reached][:3]
            self.recovery_phase = None
            self.recovery_count = 0
            self.best_distance = None
            self.get_logger().info(
                "[PLAN2 WP24 APPROACH] first gate reached; aligning right "
                "before advancing to the narrow-mouth gate"
            )

        retrace = self.behavior.get("wp24_retrace", {})
        if (
            bool(retrace.get("enabled", False))
            and reached == int(retrace.get("entry_path_index", -1))
        ):
            self.wp24_retrace_active = True
            self.wp24_retrace_phase = "align_entry"
            self.wp24_retrace_started_at = self.sim_time
            self.wp24_retrace_aligned_since = None
            self.wp24_retrace_attempt = 1
            self.target_index = int(retrace["wp24_path_index"])
            self.segment_start = self.waypoints[reached][:3]
            self.recovery_phase = None
            self.recovery_count = 0
            self.best_distance = None
            self.get_logger().info(
                "[PLAN2 WP24 RETRACE] entry gate reached; aligning for a "
                "straight entry before scoring WP24"
            )

        corridor = self.behavior.get("wp25_straight_corridor", {})
        if (
            bool(corridor.get("enabled", False))
            and reached == int(corridor.get("entry_path_index", -1))
        ):
            self.wp25_corridor_active = True
            self.wp25_corridor_phase = "align"
            self.wp25_corridor_started_at = self.sim_time
            self.wp25_corridor_aligned_since = None
            self.wp25_corridor_scored = False
            self.target_index = int(corridor["wp25_path_index"])
            self.segment_start = self.waypoints[reached][:3]
            self.recovery_phase = None
            self.recovery_count = 0
            self.best_distance = None
            self.get_logger().info(
                "[PLAN2 WP25 CORRIDOR] entry reached; generic recovery disabled, "
                "aligning before straight traversal"
            )

        straight = self.behavior.get("straight_wp25_entry", {})
        if (
            bool(straight.get("enabled", False))
            and reached == int(straight.get("after_path_index", -1))
            and self.target_index == int(straight.get("skip_path_index", -1))
        ):
            skipped = self.target_index
            self.target_index = int(straight["target_path_index"])
            self.segment_start = self.waypoints[reached][:3]
            self.recovery_phase = None
            self.recovery_count = 0
            self.best_distance = None
            self.get_logger().info(
                f"[PLAN2] Skipping temporary via {skipped}; driving straight "
                f"from via {reached} to official WP25 index={self.target_index}"
            )
        if reached == HANDOFF_INDEX:
            self.expert_mode = True
            self._enter_expert_phase(PHASE_ALIGN_WP31)
            self.get_logger().info(
                "PLAN2 EXPERT HANDOFF: temporary point 3 reached; "
                "starting closed-loop WP31/WP32 manoeuvre"
            )

    def _radius(self, index: int) -> float:
        radius = super()._radius(index)
        if index == WP24_EXIT_VIA_INDEX:
            return max(radius, WP24_EXIT_VIA_RADIUS)
        return radius

    def _plan2_platform_turn_resume_ready(self):
        wp26_index = int(self.behavior["stock_path_indices"]["wp26"])
        if (self.args.start_index != wp26_index or
                self.behavior_state != platform.STATE_STOCK_CLIMB or
                self.phase != base.PHASE_NAVIGATE or
                self.pose is None or self.base_z is None):
            return False
        turn = self.behavior["turn_east"]
        platform_config = self.behavior["platform"]
        return (
            self.pose[0] >= float(turn.get("resume_min_x", 33.55)) and
            abs(self.pose[1] - float(platform_config["center_y"])) <=
            float(platform_config["max_center_error"]) and
            self.base_z >= float(platform_config["resume_min_base_z"])
        )

    def _plan2_platform_turn_step(self):
        """Reproduce the expert's moving clockwise turn toward WP27."""
        if self.pose is None:
            self.cmd_pub.publish(Twist())
            return

        platform_config = self.behavior["platform"]
        center_error = self.pose[1] - float(platform_config["center_y"])
        if abs(center_error) > float(platform_config["max_center_error"]):
            self._fail(f"platform centre error {center_error:+.3f}m")
            return
        if self._sim_elapsed() > PLATFORM_TURN_TIMEOUT:
            self._fail(
                f"expert moving turn exceeded {PLATFORM_TURN_TIMEOUT:.1f}s"
            )
            return

        dx = PLATFORM_TURN_GATE[0] - self.pose[0]
        dy = PLATFORM_TURN_GATE[1] - self.pose[1]
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        body_forward = c * dx + s * dy
        body_lateral = -s * dx + c * dy
        yaw_error = base.wrap_to_pi(-self.pose[2])

        command = Twist()
        command.linear.x = float(clip(
            max(0.0, PLATFORM_TURN_POSITION_GAIN * body_forward),
            PLATFORM_TURN_TRANSLATION_LIMIT,
        ))
        command.linear.y = float(clip(
            PLATFORM_TURN_POSITION_GAIN * body_lateral,
            PLATFORM_TURN_TRANSLATION_LIMIT,
        ))
        command.angular.z = float(clip(
            2.0 * yaw_error,
            PLATFORM_TURN_YAW_LIMIT,
        ))

        self.cmd_pub.publish(command)
        aligned = abs(yaw_error) <= PLATFORM_TURN_TOLERANCE
        if self._condition_stable(aligned, PLATFORM_TURN_STABLE_TIME):
            self.platform_point_index = 0
            for index, point in enumerate(platform_config["points"]):
                if self.pose[0] >= (
                    float(point["pos"][0]) - float(point["radius"])
                ):
                    self.platform_point_index = min(
                        index + 1, len(platform_config["points"]) - 1
                    )
            self._enter_behavior_state(
                platform.STATE_PLATFORM,
                "expert moving turn aligned; traversing directly to WP27",
            )

    def _wp25_corridor_step(self):
        """Cross the WP25 mouth on one heading; turn only after chassis exit."""
        if self.pose is None or self.sim_time is None:
            self._publish()
            return

        config = self.behavior["wp25_straight_corridor"]
        if self.wp25_corridor_started_at is None:
            self.wp25_corridor_started_at = self.sim_time
        elapsed = self.sim_time - self.wp25_corridor_started_at
        if elapsed > float(config["max_time"]):
            self.wp25_corridor_active = False
            self._fail(
                f"WP25 straight corridor exceeded {config['max_time']:.1f}s"
            )
            return

        x, y, yaw = self.pose
        target_yaw = math.radians(float(config.get("heading_deg", 0.0)))
        yaw_error = base.wrap_to_pi(target_yaw - yaw)
        tolerance = math.radians(float(config["align_tolerance_deg"]))

        if self.wp25_corridor_phase == "align":
            self._publish(
                yaw=clip(
                    float(config["align_yaw_gain"]) * yaw_error,
                    float(config["align_max_yaw"]),
                )
            )
            if abs(yaw_error) <= tolerance:
                if self.wp25_corridor_aligned_since is None:
                    self.wp25_corridor_aligned_since = self.sim_time
                elif (self.sim_time - self.wp25_corridor_aligned_since >=
                      float(config["align_stable_time"])):
                    self.wp25_corridor_phase = "drive"
                    self.get_logger().info(
                        "[PLAN2 WP25 CORRIDOR] aligned; driving straight through "
                        "WP25 before any turn"
                    )
            else:
                self.wp25_corridor_aligned_since = None
            return

        wp25_index = int(config["wp25_path_index"])
        wp25 = self._target(wp25_index)
        wp25_distance = math.hypot(wp25[0] - x, wp25[1] - y)
        if not self.wp25_corridor_scored and wp25_distance <= OFFICIAL_RADIUS:
            self.wp25_corridor_scored = True
            self._advance_waypoint(wp25_distance)
            self.get_logger().info(
                f"[PLAN2 WP25 CORRIDOR] official WP25 reached at "
                f"{wp25_distance:.3f}m; continuing straight to chassis exit"
            )

        if x >= float(config["exit_x"]):
            if not self.wp25_corridor_scored:
                self.wp25_corridor_active = False
                self._fail("crossed WP25 corridor exit without scoring WP25")
                return
            self.wp25_corridor_active = False
            self.wp25_corridor_phase = None
            self.segment_start = (
                x, y, self.base_z if self.base_z is not None else 0.0
            )
            self.recovery_phase = None
            self.recovery_count = 0
            self.best_distance = None
            self.last_progress_time = self._now()
            self.get_logger().info(
                f"[PLAN2 WP25 CORRIDOR] chassis clear at x={x:.3f}; "
                "turning toward WP26 is now allowed"
            )
            super()._control_step()
            return

        center_error = float(config["center_y"]) - y
        if abs(center_error) > float(config["max_center_error"]):
            self.wp25_corridor_active = False
            self._fail(
                f"WP25 corridor centre error {center_error:+.3f}m"
            )
            return

        yaw_command = (
            float(config["drive_yaw_gain"]) * yaw_error
            + float(config["center_yaw_gain"]) * center_error
        )
        forward = float(config["forward_speed"])
        if abs(yaw_error) > math.radians(float(config["drive_stop_angle_deg"])):
            forward = 0.0
        self._publish(
            forward=forward,
            yaw=clip(yaw_command, float(config["drive_max_yaw"])),
        )

    def _wp24_retrace_step(self):
        """Enter WP24 head-on, then reverse along the same line before turning."""
        if self.pose is None or self.sim_time is None:
            self._publish()
            return

        config = self.behavior["wp24_retrace"]
        if self.wp24_retrace_started_at is None:
            self.wp24_retrace_started_at = self.sim_time
        elapsed = self.sim_time - self.wp24_retrace_started_at
        x, y, yaw = self.pose
        if elapsed > float(config["max_time"]):
            max_attempts = int(config.get("max_attempts", 0))
            if (
                self.wp24_retrace_phase in ("align_entry", "drive_in")
                and (max_attempts <= 0 or self.wp24_retrace_attempt < max_attempts)
            ):
                self.wp24_retrace_attempt += 1
                self.wp24_retrace_phase = "retry_reverse"
                self.wp24_retrace_started_at = self.sim_time
                self.get_logger().warn(
                    f"[PLAN2 WP24 RETRACE] attempt "
                    f"{self.wp24_retrace_attempt - 1}/"
                    f"{'unbounded' if max_attempts <= 0 else max_attempts} missed "
                    "WP24; reversing outside for another aligned entry"
                )
                return
            else:
                self.wp24_retrace_active = False
                self._fail(
                    f"WP24 retrace phase {self.wp24_retrace_phase} exceeded "
                    f"{config['max_time']:.1f}s after "
                    f"{self.wp24_retrace_attempt} attempt(s)"
                )
                return

        center_x = float(config["center_x"])
        target_yaw = math.radians(float(config.get("heading_deg", 90.0)))
        yaw_error = base.wrap_to_pi(target_yaw - yaw)
        tolerance = math.radians(float(config["align_tolerance_deg"]))

        if (
            self.wp24_retrace_phase == "drive_in"
            and abs(x - center_x) > float(config["max_center_error"])
        ):
            self.wp24_retrace_attempt += 1
            self.wp24_retrace_phase = "retry_reverse"
            self.wp24_retrace_started_at = self.sim_time
            self.wp24_retrace_aligned_since = None
            self.get_logger().warn(
                f"[PLAN2 WP24 RETRACE] centre error {x - center_x:+.3f}m; "
                "reversing outside to recenter instead of safety-stopping"
            )
            return

        if self.wp24_retrace_phase == "align_entry":
            hold_dx = center_x - x
            hold_dy = float(config["align_hold_y"]) - y
            body_hold_error = (
                math.cos(yaw) * hold_dx + math.sin(yaw) * hold_dy
            )
            body_lateral_error = (
                -math.sin(yaw) * hold_dx + math.cos(yaw) * hold_dy
            )
            hold_forward = clip(
                float(config["align_creep_bias"])
                + float(config["align_position_gain"]) * body_hold_error,
                float(config["align_max_forward"]),
            )
            hold_lateral = clip(
                float(config["align_lateral_gain"]) * body_lateral_error,
                float(config["align_max_side"]),
            )
            self._publish(
                forward=hold_forward,
                side=hold_lateral,
                yaw=clip(
                    float(config["align_yaw_gain"]) * yaw_error,
                    float(config["align_max_yaw"]),
                )
            )
            centered = abs(x - center_x) <= float(config["align_center_tolerance"])
            if abs(yaw_error) <= tolerance and centered:
                if self.wp24_retrace_aligned_since is None:
                    self.wp24_retrace_aligned_since = self.sim_time
                elif (self.sim_time - self.wp24_retrace_aligned_since >=
                      float(config["align_stable_time"])):
                    self.wp24_retrace_phase = "drive_in"
                    self.wp24_retrace_started_at = self.sim_time
                    self.get_logger().info(
                        "[PLAN2 WP24 RETRACE] aligned north; driving straight "
                        "into WP24"
                    )
            else:
                self.wp24_retrace_aligned_since = None
            if self.sim_time - self.wp24_retrace_last_report >= 1.0:
                self.wp24_retrace_last_report = self.sim_time
                self.get_logger().info(
                    f"[PLAN2 WP24 RETRACE] phase={self.wp24_retrace_phase} "
                    f"pose=({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg) "
                    f"hold=({hold_forward:+.2f},{hold_lateral:+.2f}) "
                    f"heading_error={math.degrees(yaw_error):+.1f}deg"
                )
            return

        yaw_command = clip(
            float(config["drive_yaw_gain"]) * yaw_error,
            float(config["drive_max_yaw"]),
        )
        if self.wp24_retrace_phase == "drive_in":
            wp24_index = int(config["wp24_path_index"])
            wp24 = self._target(wp24_index)
            wp24_distance = math.hypot(wp24[0] - x, wp24[1] - y)
            forward = float(config["forward_speed"])
            if abs(yaw_error) > math.radians(
                float(config["drive_stop_angle_deg"])
            ):
                forward = 0.0
            self._publish(forward=forward, yaw=yaw_command)
            if self.sim_time - self.wp24_retrace_last_report >= 1.0:
                self.wp24_retrace_last_report = self.sim_time
                self.get_logger().info(
                    f"[PLAN2 WP24 RETRACE] phase={self.wp24_retrace_phase} "
                    f"pose=({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg) "
                    f"wp24_distance={wp24_distance:.3f} "
                    f"heading_error={math.degrees(yaw_error):+.1f}deg"
                )
            if (
                wp24_distance <= OFFICIAL_RADIUS
                and y >= float(config["score_min_y"])
            ):
                self._advance_waypoint(wp24_distance)
                self.wp24_retrace_phase = "settle"
                self.wp24_retrace_started_at = self.sim_time
                self.get_logger().info(
                    f"[PLAN2 WP24 RETRACE] official WP24 reached at "
                    f"{wp24_distance:.3f}m; reversing on the entry heading"
                )
            return

        if self.wp24_retrace_phase == "settle":
            self._publish()
            if elapsed >= float(config["settle_time"]):
                self.wp24_retrace_phase = "reverse_out"
                self.wp24_retrace_started_at = self.sim_time
            return

        if self.wp24_retrace_phase == "retry_reverse":
            self._publish(
                forward=-abs(float(config["retry_reverse_speed"])),
                yaw=yaw_command,
            )
            if self.sim_time - self.wp24_retrace_last_report >= 1.0:
                self.wp24_retrace_last_report = self.sim_time
                self.get_logger().info(
                    f"[PLAN2 WP24 RETRACE] phase={self.wp24_retrace_phase} "
                    f"attempt={self.wp24_retrace_attempt}/"
                    f"{'unbounded' if int(config['max_attempts']) <= 0 else int(config['max_attempts'])} "
                    f"pose=({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg)"
                )
            if y <= float(config["retry_exit_y"]):
                self.wp24_retrace_phase = "retry_recenter"
                self.wp24_retrace_started_at = self.sim_time
                self.wp24_retrace_aligned_since = None
                self.get_logger().info(
                    f"[PLAN2 WP24 RETRACE] retry "
                    f"{self.wp24_retrace_attempt}/"
                    f"{'unbounded' if int(config['max_attempts']) <= 0 else int(config['max_attempts'])}: "
                    "outside the mouth; recentering laterally before re-entry"
                )
            return

        if self.wp24_retrace_phase == "retry_recenter":
            target_x = center_x
            target_y = float(config["retry_center_y"])
            hold_dx = target_x - x
            hold_dy = target_y - y
            body_forward_error = (
                math.cos(yaw) * hold_dx + math.sin(yaw) * hold_dy
            )
            body_lateral_error = (
                -math.sin(yaw) * hold_dx + math.cos(yaw) * hold_dy
            )
            self._publish(
                forward=clip(
                    float(config["retry_center_position_gain"])
                    * body_forward_error,
                    float(config["retry_center_max_forward"]),
                ),
                side=clip(
                    float(config["retry_center_lateral_gain"])
                    * body_lateral_error,
                    float(config["retry_center_max_side"]),
                ),
                yaw=yaw_command,
            )
            centered = (
                abs(x - target_x) <=
                float(config["retry_center_tolerance"])
                and abs(y - target_y) <=
                float(config["retry_center_y_tolerance"])
                and abs(yaw_error) <= tolerance
            )
            if centered:
                if self.wp24_retrace_aligned_since is None:
                    self.wp24_retrace_aligned_since = self.sim_time
                elif (
                    self.sim_time - self.wp24_retrace_aligned_since >=
                    float(config["retry_center_stable_time"])
                ):
                    self.wp24_retrace_phase = "align_entry"
                    self.wp24_retrace_started_at = self.sim_time
                    self.wp24_retrace_aligned_since = None
                    self.get_logger().info(
                        f"[PLAN2 WP24 RETRACE] retry "
                        f"{self.wp24_retrace_attempt}: centered outside at "
                        f"x={x:.3f}; starting aligned re-entry"
                    )
            else:
                self.wp24_retrace_aligned_since = None
            if self.sim_time - self.wp24_retrace_last_report >= 1.0:
                self.wp24_retrace_last_report = self.sim_time
                self.get_logger().info(
                    f"[PLAN2 WP24 RETRACE] phase={self.wp24_retrace_phase} "
                    f"pose=({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg) "
                    f"center_error={x - target_x:+.3f}m"
                )
            return

        if self.wp24_retrace_phase == "reverse_out":
            self._publish(
                forward=-abs(float(config["reverse_speed"])),
                yaw=yaw_command,
            )
            if self.sim_time - self.wp24_retrace_last_report >= 1.0:
                self.wp24_retrace_last_report = self.sim_time
                self.get_logger().info(
                    f"[PLAN2 WP24 RETRACE] phase={self.wp24_retrace_phase} "
                    f"pose=({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg) "
                    f"heading_error={math.degrees(yaw_error):+.1f}deg"
                )
            if y <= float(config["exit_y"]):
                next_index = int(config["next_path_index"])
                skipped_index = int(config["exit_path_index"])
                self.wp24_retrace_active = False
                self.wp24_retrace_phase = None
                self.target_index = next_index
                self.segment_start = (
                    x, y, self.base_z if self.base_z is not None else 0.0
                )
                self.recovery_phase = None
                self.recovery_count = 0
                self.best_distance = None
                self.last_progress_time = self._now()
                self.get_logger().info(
                    f"[PLAN2 WP24 RETRACE] chassis clear at y={y:.3f}; "
                    f"exit via {skipped_index} already retraced, turning toward "
                    f"path index {next_index} is now allowed"
                )
            return

        self.wp24_retrace_active = False
        self._fail(f"unknown WP24 retrace phase: {self.wp24_retrace_phase}")

    def _wp24_approach_step(self):
        """Finish the rightward approach on a stable heading before WP24."""
        if self.pose is None or self.sim_time is None:
            self._publish()
            return

        config = self.behavior["wp24_approach"]
        elapsed = self.sim_time - self.wp24_approach_started_at
        if elapsed > float(config["max_time"]):
            self.wp24_approach_active = False
            self._fail(
                f"WP24 rightward approach exceeded {config['max_time']:.1f}s"
            )
            return

        x, y, yaw = self.pose
        target = (float(config["target_x"]), float(config["center_y"]))
        target_yaw = math.radians(float(config.get("heading_deg", 0.0)))
        yaw_error = base.wrap_to_pi(target_yaw - yaw)
        tolerance = math.radians(float(config["align_tolerance_deg"]))

        if self.wp24_approach_phase == "align_right":
            self._publish(
                forward=float(config.get("align_forward", 0.0)),
                yaw=clip(
                    float(config["align_yaw_gain"]) * yaw_error,
                    float(config["align_max_yaw"]),
                )
            )
            if abs(yaw_error) <= tolerance:
                if self.wp24_approach_aligned_since is None:
                    self.wp24_approach_aligned_since = self.sim_time
                elif (self.sim_time - self.wp24_approach_aligned_since >=
                      float(config["align_stable_time"])):
                    self.wp24_approach_phase = "drive_right"
                    self.wp24_approach_started_at = self.sim_time
                    self.get_logger().info(
                        "[PLAN2 WP24 APPROACH] aligned right; holding the "
                        "corridor centre until the narrow-mouth gate"
                    )
            else:
                self.wp24_approach_aligned_since = None
            if self.sim_time - self.wp24_approach_last_report >= 1.0:
                self.wp24_approach_last_report = self.sim_time
                self.get_logger().info(
                    f"[PLAN2 WP24 APPROACH] phase={self.wp24_approach_phase} "
                    f"pose=({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg) "
                    f"heading_error={math.degrees(yaw_error):+.1f}deg"
                )
            return

        if self.wp24_approach_phase != "drive_right":
            self.wp24_approach_active = False
            self._fail(
                f"unknown WP24 approach phase: {self.wp24_approach_phase}"
            )
            return

        center_error = float(config["center_y"]) - y
        if abs(center_error) > float(config["max_center_error"]):
            self.wp24_approach_active = False
            self._fail(
                f"WP24 approach centre error {center_error:+.3f}m"
            )
            return

        distance = math.hypot(target[0] - x, target[1] - y)
        yaw_command = (
            float(config["drive_yaw_gain"]) * yaw_error
            + float(config["center_yaw_gain"]) * center_error
        )
        forward = float(config["forward_speed"])
        if abs(yaw_error) > math.radians(
            float(config["drive_stop_angle_deg"])
        ):
            forward = 0.0
        self._publish(
            forward=forward,
            yaw=clip(yaw_command, float(config["drive_max_yaw"])),
        )

        if self.sim_time - self.wp24_approach_last_report >= 1.0:
            self.wp24_approach_last_report = self.sim_time
            self.get_logger().info(
                f"[PLAN2 WP24 APPROACH] phase={self.wp24_approach_phase} "
                f"pose=({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg) "
                f"target_distance={distance:.3f} "
                f"heading_error={math.degrees(yaw_error):+.1f}deg"
            )

        if distance <= float(config["target_radius"]):
            via_index = int(config["target_path_index"])
            via = self._target(via_index)
            via_distance = math.hypot(via[0] - x, via[1] - y)
            if via_distance > self._radius(via_index):
                self.wp24_approach_active = False
                self._fail(
                    f"right-shifted gate missed path via {via_index} by "
                    f"{via_distance:.3f}m"
                )
                return
            self.wp24_approach_active = False
            self.wp24_approach_phase = None
            self._advance_waypoint(via_distance)
            self.get_logger().info(
                f"[PLAN2 WP24 APPROACH] right-shifted gate reached; original "
                f"via distance={via_distance:.3f}m"
            )

    def _control_step(self):
        if self.behavior_state == platform.STATE_FAILED:
            self._publish()
            return
        if self.expert_mode:
            if self.args.auto_start and not self._run_startup():
                self._publish()
                return
            self._expert_step()
        else:
            if self.wp24_approach_active:
                self._wp24_approach_step()
                return
            if self.wp24_retrace_active:
                self._wp24_retrace_step()
                return
            if self.wp25_corridor_active:
                self._wp25_corridor_step()
                return
            # Keep the successful WP27 hold alive after the behavior controller
            # marks the phase done. Otherwise the startup gate falls through to
            # WaypointNavigator, which publishes zero and lets the policy creep
            # backwards on the raised platform.
            if (self.behavior_state == platform.STATE_HOLD_WP27 and
                    self.wp27_success_reported and self.pose is not None):
                self._wp27_stop_step()
                return
            if self._plan2_platform_turn_resume_ready():
                self._enter_behavior_state(
                    platform.STATE_TURN_EAST,
                    "resuming Plan2 from a verified WP26 upper-platform pose",
                )
            if self.behavior_state == platform.STATE_TURN_EAST:
                self._plan2_platform_turn_step()
                return

            configured_min_forward = self.args.min_forward
            configured_max_forward = self.args.max_forward
            if POST_PIT_ENTRY_TARGET_INDEX <= self.target_index <= HANDOFF_INDEX:
                self.args.min_forward = max(
                    configured_min_forward, POST_PIT_MIN_FORWARD
                )
            if self.target_index == POST_PIT_ENTRY_TARGET_INDEX:
                self.args.max_forward = min(
                    configured_max_forward, POST_PIT_ENTRY_MAX_FORWARD
                )
            if self.target_index == POST_PIT_STAIRS_TARGET_INDEX:
                self.args.min_forward = max(
                    configured_min_forward, POST_PIT_STAIRS_MIN_FORWARD
                )
            if TAIL_PATH_START_INDEX <= self.target_index <= HANDOFF_INDEX:
                self.args.min_forward = max(
                    configured_min_forward, TAIL_MIN_FORWARD
                )
            try:
                super()._control_step()
            finally:
                self.args.min_forward = configured_min_forward
                self.args.max_forward = configured_max_forward

    def _stock_continue_step(self):
        """Return WP28 onward to Plan2's existing waypoint behavior."""
        post = self.behavior.get("post_wp27", {})
        is_via = (
            self.target_index < len(self.waypoints) and
            self.waypoints[self.target_index][3] == "via"
        )
        # The baseline has no continuation profiles; keep its exact stock
        # path behavior. Candidate profiles opt ordinary waypoints into the
        # behavior base class. Vias retain Plan2's already-verified approach
        # unless an explicit via profile is provided.
        use_profiled_control = (
            (not is_via and (
                post.get("stock_speed_default") or
                post.get("stock_speed_profiles")
            )) or
            (is_via and post.get("stock_via_speed"))
        )
        if use_profiled_control:
            super()._stock_continue_step()
        else:
            base.WaypointNavigator._control_step(self)

    def _enter_expert_phase(self, phase: str):
        self.expert_phase = phase
        self.expert_phase_started = self._now()
        self.align_stable_since = None
        self.phase_best_error = float("inf")
        self.phase_last_progress = self.expert_phase_started
        self.get_logger().info(f"[PLAN2 EXPERT] phase -> {phase}")

    def _publish(
        self,
        forward: float = 0.0,
        yaw: float = 0.0,
        side: float = 0.0,
    ):
        cmd = Twist()
        cmd.linear.x = float(clip(forward, EXPERT_FORWARD_LIMIT))
        cmd.linear.y = float(clip(side, 0.60))
        cmd.angular.z = float(clip(yaw, 1.00))
        self.cmd_pub.publish(cmd)

    def _distance(self, target):
        return math.hypot(target[0] - self.pose[0], target[1] - self.pose[1])

    def _official_distances(self):
        return self._distance(WP31), self._distance(WP32)

    def _align(
        self,
        desired_yaw: float,
        next_phase: str,
        tolerance_deg: float = 8.0,
        gain: float = 1.5,
        max_yaw: float = 0.62,
    ):
        error = base.wrap_to_pi(desired_yaw - self.pose[2])
        now = self._now()
        if abs(error) <= math.radians(tolerance_deg):
            if self.align_stable_since is None:
                self.align_stable_since = now
            self._publish()
            if now - self.align_stable_since >= ALIGN_SETTLE_TIME:
                self._enter_expert_phase(next_phase)
            return abs(error)

        self.align_stable_since = None
        self._publish(yaw=clip(gain * error, max_yaw))
        return abs(error)

    def _drive_to(
        self,
        target,
        direction: float,
        max_speed: float,
        heading_stop_deg: float = 28.0,
        yaw_limit: float = 0.58,
        yaw_gain: float = 1.45,
    ):
        x, y, yaw = self.pose
        dx = target[0] - x
        dy = target[1] - y
        distance = math.hypot(dx, dy)
        movement_heading = math.atan2(dy, dx)
        body_heading = movement_heading if direction > 0.0 else base.wrap_to_pi(
            movement_heading + math.pi
        )
        heading_error = base.wrap_to_pi(body_heading - yaw)

        yaw_cmd = clip(yaw_gain * heading_error, yaw_limit)
        if abs(heading_error) > math.radians(heading_stop_deg):
            speed = 0.0
        else:
            speed_limit = min(max_speed, max(TAIL_MIN_FORWARD, 0.75 * distance))
            speed = math.copysign(speed_limit, direction)
            speed *= max(0.35, math.cos(heading_error))
        self._publish(speed, yaw_cmd)
        return distance, heading_error

    def _track_progress(self, error: float, timeout: float = 12.0):
        now = self._now()
        if error < self.phase_best_error - 0.025:
            self.phase_best_error = error
            self.phase_last_progress = now
            return False
        if now - self.phase_last_progress > timeout:
            self.get_logger().warn(
                f"[PLAN2 EXPERT] no progress for {timeout:.0f}s in "
                f"{self.expert_phase}; "
                f"error={error:.3f} best={self.phase_best_error:.3f}"
            )
            self.phase_last_progress = now
            return True
        return False

    def _start_wp31_recovery(self):
        if self.wp31_retry_count >= WP31_MAX_RETRIES:
            self._publish()
            self._fail(
                f"WP31 remained outside the official {OFFICIAL_RADIUS:.2f}m "
                f"radius after {self.wp31_retry_count} recovery attempts"
            )
            return
        self.wp31_retry_count += 1
        self.get_logger().warn(
            f"[PLAN2 EXPERT] WP31 attempt stalled outside the official "
            f"scoring circle; recovery {self.wp31_retry_count}/"
            f"{WP31_MAX_RETRIES}: align north, reverse clear, then retry"
        )
        self._enter_expert_phase(PHASE_RECOVER_WP31_ALIGN)

    def _expert_step(self):
        if self.pose is None:
            self._publish()
            return

        now = self._now()
        d31, d32 = self._official_distances()
        phase_error = 0.0
        heading_error = 0.0

        if self.expert_phase == PHASE_ALIGN_WP31:
            desired = math.atan2(
                WP31_RIGHT_GATE[1] - self.pose[1],
                WP31_RIGHT_GATE[0] - self.pose[0],
            )
            phase_error = self._align(
                desired,
                PHASE_DRIVE_WP31_GATE,
                tolerance_deg=10.0,
                gain=2.0,
                max_yaw=0.90,
            )

        elif self.expert_phase == PHASE_DRIVE_WP31_GATE:
            phase_error, heading_error = self._drive_to(
                WP31_RIGHT_GATE,
                +1.0,
                0.50,
                heading_stop_deg=16.0,
                yaw_limit=0.80,
                yaw_gain=1.60,
            )
            if phase_error <= 0.16:
                self._enter_expert_phase(PHASE_ALIGN_WP31_FINAL)

        elif self.expert_phase == PHASE_ALIGN_WP31_FINAL:
            desired = math.atan2(
                WP31_INNER[1] - self.pose[1], WP31_INNER[0] - self.pose[0]
            )
            phase_error = self._align(
                desired,
                PHASE_DRIVE_WP31,
                tolerance_deg=8.0,
                gain=1.8,
                max_yaw=0.75,
            )

        elif self.expert_phase == PHASE_DRIVE_WP31:
            # Do not turn around when the near-edge helper point passes behind
            # the chassis.  Keep the northbound line until the official circle
            # is entered, then reuse the existing settle/reverse sequence.
            phase_error, heading_error = self._drive_to(
                WP31_DRIVE_THROUGH, +1.0, 0.46
            )
            if d31 <= OFFICIAL_RADIUS:
                self.wp31_reached = True
            if self.wp31_reached:
                self.get_logger().info(
                    f"OFFICIAL WP31 REACHED distance={d31:.3f}m; "
                    "settling before demonstrated reverse"
                )
                self._enter_expert_phase(PHASE_SETTLE_WP31)

        elif self.expert_phase == PHASE_RECOVER_WP31_ALIGN:
            phase_error = abs(base.wrap_to_pi(math.pi / 2.0 - self.pose[2]))
            if now - self.expert_phase_started > WP31_RECOVERY_ALIGN_TIMEOUT:
                self._publish()
                self._fail(
                    f"WP31 recovery {self.wp31_retry_count} could not align "
                    "north inside the timeout"
                )
                self.expert_phase = PHASE_COMPLETE
                return
            self._align(
                math.pi / 2.0,
                PHASE_RECOVER_WP31_REVERSE,
                tolerance_deg=12.0,
                gain=1.8,
                max_yaw=0.72,
            )

        elif self.expert_phase == PHASE_RECOVER_WP31_REVERSE:
            phase_error, heading_error = self._drive_to(
                WP31_RIGHT_GATE,
                -1.0,
                0.62,
                heading_stop_deg=12.0,
                yaw_limit=0.52,
                yaw_gain=1.55,
            )
            if now - self.expert_phase_started > WP31_RECOVERY_REVERSE_TIMEOUT:
                self._publish()
                self._fail(
                    f"WP31 recovery {self.wp31_retry_count} could not reverse "
                    "clear of the mouth inside the timeout"
                )
                self.expert_phase = PHASE_COMPLETE
                return
            if phase_error <= 0.16:
                self._publish()
                self._enter_expert_phase(PHASE_ALIGN_WP31_FINAL)

        elif self.expert_phase == PHASE_SETTLE_WP31:
            self._publish()
            phase_error = d31
            if now - self.expert_phase_started >= WP31_SETTLE_TIME:
                self._enter_expert_phase(PHASE_REVERSE)

        elif self.expert_phase == PHASE_REVERSE:
            phase_error, heading_error = self._drive_to(TURN_ANCHOR, -1.0, 0.80)
            if phase_error <= TURN_ANCHOR_REACH_RADIUS:
                self._publish()
                self._enter_expert_phase(PHASE_ALIGN_WP32)

        elif self.expert_phase == PHASE_ALIGN_WP32:
            # The expert demonstration used a full-rate clockwise turn from
            # about 85 degrees to due east.  A weak turn drifts too far north.
            phase_error = self._align(
                0.0,
                PHASE_DRIVE_WP32_APPROACH,
                tolerance_deg=12.0,
                gain=2.0,
                max_yaw=0.95,
            )

        elif self.expert_phase == PHASE_DRIVE_WP32_APPROACH:
            phase_error, heading_error = self._drive_to(
                WP32_APPROACH,
                +1.0,
                0.48,
                heading_stop_deg=14.0,
                yaw_limit=0.80,
                yaw_gain=1.60,
            )
            if phase_error <= 0.12:
                self._publish()
                self._enter_expert_phase(PHASE_SETTLE_WP32_APPROACH)

        elif self.expert_phase == PHASE_SETTLE_WP32_APPROACH:
            self._publish()
            phase_error = self._distance(WP32_APPROACH)
            if now - self.expert_phase_started >= WP32_APPROACH_SETTLE_TIME:
                self._enter_expert_phase(PHASE_ALIGN_WP32_FINAL)

        elif self.expert_phase == PHASE_ALIGN_WP32_FINAL:
            phase_error = self._align(
                0.0,
                PHASE_DRIVE_WP32,
                tolerance_deg=8.0,
                gain=2.0,
                max_yaw=0.85,
            )

        elif self.expert_phase == PHASE_DRIVE_WP32:
            phase_error, heading_error = self._drive_to(
                WP32_DRIVE_THROUGH,
                +1.0,
                0.48,
                heading_stop_deg=14.0,
                yaw_limit=0.80,
                yaw_gain=1.60,
            )
            if d32 <= OFFICIAL_RADIUS:
                self.wp32_reached = True
                self.get_logger().info(
                    f"OFFICIAL WP32 REACHED distance={d32:.3f}m; "
                    "PLAN2 EXPERT TAIL COMPLETE"
                )
                self.expert_phase = PHASE_COMPLETE
                self.phase = base.PHASE_DONE
                self._publish()

        elif self.expert_phase == PHASE_COMPLETE:
            # A zero command lets this policy creep backwards.  Use a small,
            # forward-only hysteresis correction to remain near the finish.
            if self.pose[0] < 32.98:
                self.wp32_hold_active = True
            elif self.pose[0] >= 33.05:
                self.wp32_hold_active = False

            if self.wp32_hold_active:
                self._drive_to(
                    WP32_HOLD_TARGET,
                    +1.0,
                    0.18,
                    heading_stop_deg=14.0,
                    yaw_limit=0.60,
                    yaw_gain=1.50,
                )
            else:
                self._publish()
            return

        else:
            self.get_logger().error(f"Unknown expert phase: {self.expert_phase}")
            self._publish()
            self.phase = base.PHASE_DONE
            return

        stalled = self._track_progress(
            phase_error,
            timeout=(
                WP31_NO_PROGRESS_TIMEOUT
                if self.expert_phase == PHASE_DRIVE_WP31
                else 12.0
            ),
        )
        if (
            stalled
            and self.expert_phase == PHASE_DRIVE_WP31
            and not self.wp31_reached
        ):
            self._start_wp31_recovery()
        if now - self.last_expert_report >= 1.0:
            self.last_expert_report = now
            self.get_logger().info(
                f"[PLAN2 EXPERT] phase={self.expert_phase} "
                f"pose=({self.pose[0]:.3f},{self.pose[1]:.3f},"
                f"{math.degrees(self.pose[2]):.1f}deg) "
                f"d31={d31:.3f} d32={d32:.3f} phase_error={phase_error:.3f} "
                f"heading_error={math.degrees(heading_error):+.1f}deg"
            )


def parse_args():
    behavior_parser = argparse.ArgumentParser(add_help=False)
    behavior_parser.add_argument(
        "--behavior-config", type=Path, default=DEFAULT_BEHAVIOR_CONFIG,
    )
    behavior_args, remaining = behavior_parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args, ros_args = base.parse_args()
    finally:
        sys.argv = original_argv
    args.behavior_config = behavior_args.behavior_config.resolve()
    return args, ros_args


def main():
    args, ros_args = parse_args()
    base.rclpy.init(args=ros_args)
    node = Plan2ExpertTailNavigator(args)
    try:
        base.rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            for _ in range(5):
                node._publish()
        except Exception:
            pass
        node.destroy_node()
        if base.rclpy.ok():
            base.rclpy.shutdown()


if __name__ == "__main__":
    main()
