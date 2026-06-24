#!/usr/bin/env python3
"""
Relay segmentazione ground-truth (Gazebo) -> ID Cityscapes.

La segmentation camera di Gazebo pubblica /semantic/labels_map dove ogni pixel
e' la label assegnata nel world (label = ID_Cityscapes + 1, perche' 0 = sfondo).
Questo nodo riconverte (cs = gz - 1) e pubblica /semantic/segmentation (mono8)
ESATTAMENTE come farebbe la rete: cosi' semantic_costmap_node.py non cambia.

Usa questo nodo AL POSTO di segmentation_node.py quando vuoi la verita' a terra
(niente inferenza, frame rate pieno, marciapiede sempre corretto).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)


class GtSegRelay(Node):
    def __init__(self):
        super().__init__('gt_segmentation_relay')

        self.declare_parameter('labels_topic', '/semantic/labels_map')
        self.declare_parameter('out_topic', '/semantic/segmentation')
        self.declare_parameter('publish_overlay', True)

        lt = self.get_parameter('labels_topic').value
        ot = self.get_parameter('out_topic').value
        self.publish_overlay = self.get_parameter('publish_overlay').value

        self.bridge = CvBridge()

        # LUT: gz_label -> ID Cityscapes (= gz-1). 0 (sfondo) -> 255 = "ignora"
        lut = np.arange(256, dtype=np.int16) - 1
        lut[0] = 255
        self.lut = np.clip(lut, 0, 255).astype(np.uint8)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(Image, lt, self.cb, qos)
        self.pub = self.create_publisher(Image, ot, qos)
        self.pub_ov = self.create_publisher(Image, '/semantic/overlay', qos)

        self.get_logger().info(f'GT relay: {lt} -> {ot} (ID Cityscapes)')

    def cb(self, msg: Image):
        lab = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if lab.ndim == 3:               # alcune build pubblicano 3 canali: prendine uno
            lab = lab[..., 0]
        lab = lab.astype(np.uint8)

        cs = self.lut[lab]              # ID Cityscapes per pixel

        out = self.bridge.cv2_to_imgmsg(cs, 'mono8')
        out.header = msg.header
        self.pub.publish(out)

        if self.publish_overlay:
            color = CITYSCAPES_PALETTE[np.clip(cs, 0, 18)]
            color[cs > 18] = (0, 0, 0)  # sfondo/ignora -> nero
            ov = self.bridge.cv2_to_imgmsg(color, 'rgb8')
            ov.header = msg.header
            self.pub_ov.publish(ov)


def main():
    rclpy.init()
    node = GtSegRelay()
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