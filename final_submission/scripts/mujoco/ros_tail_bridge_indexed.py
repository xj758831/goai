#!/usr/bin/env python3
"""System-Python ROS2 bridge with an explicit tail path start index."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-socket", required=True)
    parser.add_argument("--child-socket", required=True)
    parser.add_argument("--nav-script", type=Path, required=True)
    parser.add_argument("--behavior-config", type=Path, required=True)
    parser.add_argument("--waypoints", type=Path, required=True)
    parser.add_argument("--nav-log", type=Path, required=True)
    parser.add_argument("--ros-domain", type=int, required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--sync-ack-topic")
    parser.add_argument("--sync-timeout-ms", type=float, default=100.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain)
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock
    from std_msgs.msg import String

    class BridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("plan2_mujoco_ros_tail_indexed_bridge")
            self.command = [0.0, 0.0, 0.0]
            self.command_generation = 0
            self.ack_stamp_ns = -1
            self.ack_command = None
            self.ack_published = False
            self.clock_pub = self.create_publisher(Clock, "/clock", 10)
            self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
            self.cmd_sub = self.create_subscription(
                Twist, "/cmd_vel", self._cmd_callback, 20
            )
            self.ack_sub = None
            if args.sync_ack_topic:
                self.ack_sub = self.create_subscription(
                    String, args.sync_ack_topic, self._ack_callback, 20
                )

        def _cmd_callback(self, msg: Twist) -> None:
            self.command = [
                float(msg.linear.x),
                float(msg.linear.y),
                float(msg.angular.z),
            ]
            self.command_generation += 1

        def _ack_callback(self, msg: String) -> None:
            document = json.loads(msg.data)
            self.ack_stamp_ns = int(document["stamp_ns"])
            self.ack_published = bool(document["published"])
            self.ack_command = document.get("command")

        def publish_state(self, state: dict) -> None:
            stamp = float(state["time"])
            sec = int(stamp)
            nanosec = int(round((stamp - sec) * 1.0e9))
            if nanosec >= 1_000_000_000:
                sec += 1
                nanosec -= 1_000_000_000
            clock = Clock()
            clock.clock.sec = sec
            clock.clock.nanosec = nanosec
            self.clock_pub.publish(clock)
            msg = Odometry()
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
            msg.header.frame_id = "odom"
            msg.child_frame_id = "base_link"
            position = state["position"]
            quaternion = state["quaternion"]
            msg.pose.pose.position.x = float(position[0])
            msg.pose.pose.position.y = float(position[1])
            msg.pose.pose.position.z = float(position[2])
            msg.pose.pose.orientation.w = float(quaternion[0])
            msg.pose.pose.orientation.x = float(quaternion[1])
            msg.pose.pose.orientation.y = float(quaternion[2])
            msg.pose.pose.orientation.z = float(quaternion[3])
            velocity = state.get("velocity", [0.0, 0.0, 0.0])
            msg.twist.twist.linear.x = float(velocity[0])
            msg.twist.twist.linear.y = float(velocity[1])
            msg.twist.twist.linear.z = float(velocity[2])
            self.odom_pub.publish(msg)

    args.nav_log.parent.mkdir(parents=True, exist_ok=True)
    nav_log = args.nav_log.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(args.ros_domain)
    nav_command = [
            "/usr/bin/python3",
            str(args.nav_script),
            "--behavior-config",
            str(args.behavior_config),
            "--waypoints",
            str(args.waypoints),
            "--start-index",
            str(args.start_index),
            "--max-forward",
            "0.6",
            "--stand-rise",
            "0.15",
            "--no-auto-start",
            "--ros-args",
            "-p",
            "use_sim_time:=true",
        ]
    if args.sync_ack_topic:
        nav_command.extend(["--sync-ack-topic", args.sync_ack_topic])
    nav = subprocess.Popen(
        nav_command,
        cwd=str(args.nav_script.parent.parent.parent.parent),
        env=env,
        stdout=nav_log,
        stderr=subprocess.STDOUT,
    )

    rclpy.init(args=None)
    node = BridgeNode()
    sock = None
    try:
        if args.sync_ack_topic:
            ready_deadline = time.monotonic() + 5.0
            while rclpy.ok() and time.monotonic() < ready_deadline:
                if nav.poll() is not None:
                    raise RuntimeError(
                        f"synchronous navigator exited during discovery: {nav.returncode}"
                    )
                rclpy.spin_once(node, timeout_sec=0.02)
                if (
                    node.odom_pub.get_subscription_count() > 0
                    and node.count_publishers(args.sync_ack_topic) > 0
                ):
                    break
            else:
                raise RuntimeError(
                    "synchronous navigator ROS endpoints were not ready within 5 seconds"
                )
        # The socket path is the parent-side readiness signal.  Expose it only
        # after ROS discovery, so sequence 1 cannot race navigator startup.
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sock.bind(args.child_socket)
        sock.settimeout(0.25)
        while rclpy.ok():
            try:
                packet, _ = sock.recvfrom(65536)
            except socket.timeout:
                if nav.poll() is not None:
                    break
                rclpy.spin_once(node, timeout_sec=0.0)
                continue
            if packet == b"__stop__":
                break
            state = json.loads(packet.decode("utf-8"))
            node.publish_state(state)
            sync_ok = True
            if args.sync_ack_topic:
                stamp_ns = int(round(float(state["time"]) * 1.0e9))
                deadline = time.monotonic() + args.sync_timeout_ms * 1.0e-3
                while rclpy.ok() and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.002)
                    if node.ack_stamp_ns == stamp_ns:
                        break
                sync_ok = node.ack_stamp_ns == stamp_ns
                if sync_ok:
                    if node.ack_published:
                        node.command = list(node.ack_command)
                    else:
                        node.command = list(state["held_command"])
            else:
                rclpy.spin_once(node, timeout_sec=0.0)
            reply = json.dumps(
                {
                    "command": node.command,
                    "sequence": state.get("sequence"),
                    "sync_ok": sync_ok,
                    "ack_stamp_ns": node.ack_stamp_ns,
                    "command_published": node.ack_published,
                    "command_generation": node.command_generation,
                },
                separators=(",", ":"),
            ).encode("utf-8")
            sock.sendto(reply, args.parent_socket)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if nav.poll() is None:
            nav.send_signal(signal.SIGINT)
            try:
                nav.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                nav.terminate()
        nav_log.close()
        if sock is not None:
            sock.close()
        for path in (args.parent_socket, args.child_socket):
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
