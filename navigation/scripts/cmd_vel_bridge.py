#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions


class CmdVelBridge(Node):
    def __init__(self):
        super().__init__('cmd_vel_bridge')
        self.declare_parameter('in_topic', 'cmd_vel')
        self.declare_parameter('out_topic', '/a200_0000/cmd_vel')
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('max_linear', 0.35)
        self.declare_parameter('max_angular', 0.35)
        self.declare_parameter('stop_pub_timeout', 2.0)
        gp = self.get_parameter
        self.frame_id = gp('frame_id').value
        self.max_v = float(gp('max_linear').value)
        self.max_w = float(gp('max_angular').value)
        self.stop_timeout = Duration(seconds=float(gp('stop_pub_timeout').value))
        self.prev_zero = False
        self.stop_time = None
        out_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(TwistStamped, gp('out_topic').value, out_qos)
        self.create_subscription(Twist, gp('in_topic').value, self.cb, 10)
        self.sent = 0
        self.dropped = 0
        self.get_logger().info(
            f"{gp('in_topic').value} -> {gp('out_topic').value} (TwistStamped with stamp 0, "
            f"stamped by the Raspberry Pi), limits {self.max_v} m/s, {self.max_w} rad/s")

    @staticmethod
    def clamp(v, m):
        return max(-m, min(m, v))

    def cb(self, msg):
        vals = (msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z)
        if not all(math.isfinite(v) for v in vals):
            self.dropped += 1
            self.get_logger().warn(f'Non-finite command dropped ({self.dropped})')
            return
        zero = msg.linear.x == 0.0 and msg.linear.y == 0.0 and msg.angular.z == 0.0
        if zero:
            now = self.get_clock().now()
            if not self.prev_zero:
                self.stop_time = now
            elif now - self.stop_time > self.stop_timeout:
                return
        self.prev_zero = zero
        out = TwistStamped()
        out.header.frame_id = self.frame_id
        out.twist.linear.x = self.clamp(msg.linear.x, self.max_v)
        out.twist.angular.z = self.clamp(msg.angular.z, self.max_w)
        self.pub.publish(out)
        self.sent += 1


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = CmdVelBridge()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop = TwistStamped()
        stop.header.frame_id = node.frame_id
        for _ in range(3):
            node.pub.publish(stop)
        node.get_logger().info(f'Closed: {node.sent} commands forwarded, {node.dropped} dropped, stop sent')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
