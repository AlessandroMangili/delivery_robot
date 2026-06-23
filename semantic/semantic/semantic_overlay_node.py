#!/usr/bin/env python3
"""
Overlay di DEBUG: fonde la camera RGB con la segmentazione (GT) e pubblica
/semantic/overlay_rgb in RELIABLE, cosi' RViz lo vede senza toccare il QoS.

NON serve al funzionamento della pipeline: e' solo un aiuto visivo per
controllare quanto bene le etichette si sovrappongono alla scena.

Lancio:
  ros2 run semantic semantic_overlay_node --ros-args -p use_sim_time:=true
In RViz: display Image sul topic /semantic/overlay_rgb
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import message_filters

CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)


class SemanticOverlayNode(Node):
    def __init__(self):
        super().__init__('semantic_overlay_node')

        self.declare_parameter('rgb_topic', '/camera/image')
        self.declare_parameter('seg_topic', '/semantic/segmentation')
        self.declare_parameter('out_topic', '/semantic/overlay_rgb')
        rgb_topic = self.get_parameter('rgb_topic').value
        seg_topic = self.get_parameter('seg_topic').value
        out_topic = self.get_parameter('out_topic').value

        self.bridge = CvBridge()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        rgb = message_filters.Subscriber(self, Image, rgb_topic, qos_profile=qos)
        seg = message_filters.Subscriber(self, Image, seg_topic, qos_profile=qos)
        # i due topic possono avere rate/stamp leggermente diversi -> slop generoso
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb, seg], queue_size=5, slop=0.2)
        self.sync.registerCallback(self.cb)

        # pubblica RELIABLE: RViz lo vede senza cambiare nulla
        self.pub = self.create_publisher(Image, out_topic, 10)
        self.get_logger().info(f'Overlay RGB+seg -> {out_topic}')

    def cb(self, rgb_msg, seg_msg):
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'rgb8')
        seg = self.bridge.imgmsg_to_cv2(seg_msg, 'mono8')
        if seg.shape[:2] != rgb.shape[:2]:
            seg = cv2.resize(seg, (rgb.shape[1], rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        color = CITYSCAPES_PALETTE[np.clip(seg, 0, 18)]
        color[seg > 18] = 0                      # sfondo/ignora -> nero
        out = (rgb * 0.55 + color * 0.45).astype(np.uint8)
        msg = self.bridge.cv2_to_imgmsg(out, 'rgb8')
        msg.header = rgb_msg.header
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SemanticOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()