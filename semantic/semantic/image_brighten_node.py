#!/usr/bin/env python3
"""Schiarisce uno stream camera sottoesposto: in_topic -> out_topic.

    out = clip( (in/255)^(1/gamma) * gain , 0, 1 ) * 255

Stessa formula di brighten_extract.py, applicata con una LUT a 256 voci (un
lookup per pixel, economico anche a 1920x1280). Header, encoding e risoluzione
restano identici all'ingresso, quindi la CameraInfo originale resta valida.
Funziona uguale su camera live e su `ros2 bag play`.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def gamma_lut(gamma, gain):
    x = np.arange(256, dtype=np.float32) / 255.0
    return (np.clip(np.power(x, 1.0 / gamma) * gain, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


class ImageBrightenNode(Node):
    def __init__(self):
        super().__init__('image_brighten_node')
        self.declare_parameter('in_topic', '/camera2/image_raw')
        self.declare_parameter('out_topic', '/camera2/image_bright')
        self.declare_parameter('gamma', 2.2)
        self.declare_parameter('gain', 1.0)

        gp = self.get_parameter
        self.lut = gamma_lut(float(gp('gamma').value), float(gp('gain').value))

        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(Image, gp('out_topic').value, 5)
        self.create_subscription(Image, gp('in_topic').value, self.cb, sub_qos)
        self.get_logger().info(
            f'{gp("in_topic").value} -> {gp("out_topic").value} '
            f'(gamma={gp("gamma").value}, gain={gp("gain").value})')

    def cb(self, msg):
        if msg.encoding.lower() not in ('rgb8', 'bgr8', 'mono8'):
            self.get_logger().warn(f'encoding non gestito: {msg.encoding}',
                                   throttle_duration_sec=5.0)
            return
        out = Image()
        out.header = msg.header
        out.height, out.width = msg.height, msg.width
        out.encoding, out.is_bigendian, out.step = msg.encoding, msg.is_bigendian, msg.step
        out.data = self.lut[np.frombuffer(msg.data, dtype=np.uint8)].tobytes()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ImageBrightenNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
