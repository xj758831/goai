#!/usr/bin/env python3
"""System-Python ROS2 side of the plan2 MuJoCo tail bridge."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain)
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rosgraph_msgs.msg import Clock

    class BridgeNode(Node):
        def __init__(self) -> None:
            super().__init__("plan2_mujoco_ros_tail_bridge")
            self.command = [0.0, 0.0, 0.0]
            self.clock_pub = self.create_publisher(Clock, "/clock", 10)
            self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
            self.cmd_sub = self.create_subscription(
                Twist, "/cmd_vel", self._cmd_callback, 20
            )

        def _cmd_callback(self, msg: Twist) -> None:
            self.command = [
                float(msg.linear.x),
                float(msg.linear.y),
                float(msg.angular.z),
            ]

        def publish_state(self, state: dict) -> None:
            msg = Odometry()
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
    nav = subprocess.Popen(
        [
            "/usr/bin/python3",
            str(args.nav_script),
            "--behavior-config",
            str(args.behavior_config),
            "--waypoints",
            str(args.waypoints),
            "--start-index",
            "20",
            "--max-forward",
            "0.6",
            "--stand-rise",
            "0.15",
            "--no-auto-start",
            "--ros-args",
            "-p",
            "use_sim_time:=true",
        ],
        cwd=str(args.nav_script.parent.parent.parent.parent),
        env=env,
        stdout=nav_log,
        stderr=subprocess.STDOUT,
    )

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(args.child_socket)
    sock.settimeout(0.25)
    rclpy.init(args=None)
    node = BridgeNode()
    try:
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
            rclpy.spin_once(node, timeout_sec=0.0)
            reply = json.dumps({"command": node.command}, separators=(",", ":")).encode("utf-8")
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
        sock.close()
        for path in (args.parent_socket, args.child_socket):
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
