#!/usr/bin/env python3

import os

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


def resize_label_map(label_map, target_w, target_h):
    lm = np.asarray(label_map)
    h, w = lm.shape[:2]
    if (w, h) == (int(target_w), int(target_h)):
        return lm.copy()
    return cv2.resize(lm, (int(target_w), int(target_h)),
                      interpolation=cv2.INTER_NEAREST)


def to_mono8(label_map):
    lm = np.asarray(label_map)
    if lm.ndim != 2:
        raise ValueError('expected a 2D label map, got shape %s' % (lm.shape,))
    if lm.dtype != np.uint8:
        if lm.min() < 0 or lm.max() > 255:
            raise ValueError('label ids outside [0, 255] do not fit in uint8')
        lm = lm.astype(np.uint8)
    return np.ascontiguousarray(lm)


class SegInferNode(Node):
    def __init__(self):
        super().__init__('seg_infer_node')

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('seg_topic', '/semantic/segmentation')
        self.declare_parameter('model_path', 'semantic.pt')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('image_encoding', 'bgr8')
        self.declare_parameter('log_period_s', 5.0)

        gp = self.get_parameter
        self.image_topic = gp('image_topic').value
        self.seg_topic = gp('seg_topic').value
        self.model_path = self.resolve_model_path(gp('model_path').value)
        self.device = gp('device').value
        self.imgsz = int(gp('imgsz').value)
        self.encoding = gp('image_encoding').value
        self.log_period = float(gp('log_period_s').value)

        self.bridge = CvBridge()

        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        self.get_logger().info('Segmentation model loaded: %s' % self.model_path)

        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self._infer(dummy)
        self.get_logger().info('Warm-up inference done.')

        pub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(Image, self.seg_topic, pub_qos)
        self.create_subscription(Image, self.image_topic, self.image_cb, sub_qos)

        self._n = 0
        self._t_last_log = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info(
            'Segmentation ready: %s -> %s (imgsz %d, device %s)'
            % (self.image_topic, self.seg_topic, self.imgsz, self.device))

    def resolve_model_path(self, model_path):
        if os.path.isabs(model_path):
            return model_path
        try:
            from ament_index_python.packages import get_package_share_directory
            candidate = os.path.join(get_package_share_directory('semantic'), 'models', model_path)
            if os.path.isfile(candidate):
                return candidate
        except Exception:
            pass
        return model_path

    def _infer(self, bgr):
        res = self.model.predict(bgr, imgsz=self.imgsz, device=self.device,
                                 verbose=False)
        cls = res[0].semantic_mask.data
        return cls.detach().cpu().numpy()

    def image_cb(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.encoding)
        h_in, w_in = bgr.shape[:2]

        cls = to_mono8(self._infer(bgr))
        cls = to_mono8(resize_label_map(cls, w_in, h_in))

        out = self.bridge.cv2_to_imgmsg(cls, encoding='mono8')
        out.header = msg.header
        self.pub.publish(out)

        self._n += 1
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._t_last_log >= self.log_period:
            hz = self._n / max(1e-6, now - self._t_last_log)
            seen = sorted(np.unique(cls).tolist())
            self.get_logger().info(
                'segmentation %dx%d at about %.1f Hz (classes: %s)'
                % (w_in, h_in, hz, seen[:12]))
            self._n = 0
            self._t_last_log = now


def main():
    rclpy.init()
    node = SegInferNode()
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
