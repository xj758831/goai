#!/usr/bin/env python3
"""Simulation-time-synchronous wrapper for the isolated Plan2 tail audit.

This is not a production navigator.  It disables the ROS wall/executor timer
and runs one high-level control step from each acknowledged odometry sample.
The acknowledgement is published only after that control step has produced a
command, allowing the MuJoCo audit bridge to reject stale commands.
"""

from __future__ import annotations

import argparse
import json
import sys

from geometry_msgs.msg import Twist
from std_msgs.msg import String

import plan2_expert_tail_navigator_wp31_drive_through_candidate_left as candidate


class CapturingPublisher:
    """Delegate ROS publishing while exposing this control step's command."""

    def __init__(self, publisher) -> None:
        self.publisher = publisher
        self.command = None

    def publish(self, msg: Twist) -> None:
        self.command = [
            float(msg.linear.x),
            float(msg.linear.y),
            float(msg.angular.z),
        ]
        self.publisher.publish(msg)

    def __getattr__(self, name):
        return getattr(self.publisher, name)


class SimSynchronousPlan2Navigator(candidate.Plan2ExpertTailNavigator):
    def __init__(self, args: argparse.Namespace, ack_topic: str) -> None:
        super().__init__(args)
        if not self.destroy_timer(self.timer):
            raise RuntimeError("failed to disable the asynchronous control timer")
        self.timer = None
        self.cmd_pub = CapturingPublisher(self.cmd_pub)
        self.sync_ack_pub = self.create_publisher(String, ack_topic, 10)
        self.get_logger().info(
            f"SIM-SYNC AUDIT enabled: odom-driven control, ack={ack_topic}"
        )

    def _now(self) -> float:
        # All audit state-machine durations must advance with MuJoCo, not with
        # the host scheduler.  The odometry callback sets sim_time first.
        if self.sim_time is not None:
            return float(self.sim_time)
        return 0.0

    def _odom_callback(self, msg) -> None:
        super()._odom_callback(msg)
        self.cmd_pub.command = None
        self._control_step()
        ack = String()
        ack.data = json.dumps(
            {
                "stamp_ns": (
                    int(msg.header.stamp.sec) * 1_000_000_000
                    + int(msg.header.stamp.nanosec)
                ),
                "published": self.cmd_pub.command is not None,
                "command": self.cmd_pub.command,
            },
            separators=(",", ":"),
        )
        self.sync_ack_pub.publish(ack)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sync-ack-topic", required=True)
    sync_args, remaining = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining]
        args, ros_args = candidate.parse_args()
    finally:
        sys.argv = original_argv
    args.sync_ack_topic = sync_args.sync_ack_topic
    return args, ros_args


def main() -> None:
    args, ros_args = parse_args()
    candidate.base.rclpy.init(args=ros_args)
    node = SimSynchronousPlan2Navigator(args, args.sync_ack_topic)
    try:
        candidate.base.rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if candidate.base.rclpy.ok():
            candidate.base.rclpy.shutdown()


if __name__ == "__main__":
    main()
