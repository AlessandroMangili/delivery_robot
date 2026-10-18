#!/usr/bin/env python3
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformListener


class GoalFoxglove(Node):
    def __init__(self):
        super().__init__('goal_foxglove')
        self.declare_parameter('in_topic', '/goal_foxglove')
        self.declare_parameter('out_topic', '/goal_pose')
        self.declare_parameter('frame', 'map')
        self.declare_parameter('distanza_max_m', 150.0)
        gp = self.get_parameter
        self.frame = gp('frame').value
        self.dmax = float(gp('distanza_max_m').value)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pub = self.create_publisher(PoseStamped, gp('out_topic').value, 10)
        self.create_subscription(PoseStamped, gp('in_topic').value, self.cb, 10)
        self.get_logger().info(f"{gp('in_topic').value} -> {gp('out_topic').value} nel frame {self.frame}, ora azzerata")

    def cb(self, msg):
        sorgente = msg.header.frame_id or self.frame
        ora = self.get_clock().now().nanoseconds * 1e-9
        anticipo = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - ora
        try:
            tf = self.buffer.lookup_transform(self.frame, sorgente, Time())
        except Exception as e:
            self.get_logger().warn(f'goal scartato: niente trasformata {sorgente} -> {self.frame} ({e})')
            return
        posa = do_transform_pose(msg.pose, tf)
        if not all(math.isfinite(v) for v in (posa.position.x, posa.position.y)) or math.hypot(posa.position.x, posa.position.y) > self.dmax:
            self.get_logger().warn(f'goal scartato: ({posa.position.x:.2f}; {posa.position.y:.2f}) fuori dalla mappa')
            return
        out = PoseStamped()
        out.header.frame_id = self.frame
        out.pose = posa
        out.pose.position.z = 0.0
        self.pub.publish(out)
        yaw = math.degrees(2.0 * math.atan2(posa.orientation.z, posa.orientation.w))
        self.get_logger().info(
            f'goal da Foxglove: frame {sorgente}, ora {anticipo:+.1f} s rispetto alla Jetson -> '
            f'{self.frame} ({posa.position.x:.2f}; {posa.position.y:.2f}; {yaw:.0f} gradi)')


def main():
    rclpy.init()
    node = GoalFoxglove()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
