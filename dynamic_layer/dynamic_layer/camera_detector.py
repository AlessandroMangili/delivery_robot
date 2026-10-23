#!/usr/bin/env python3

import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge

NOMI_COCO = {0: 'person', 1: 'bicycle', 3: 'motorcycle'}


class CameraDetector(Node):
    def __init__(self):
        super().__init__('camera_detector')
        d = self.declare_parameter
        d('image_topic', '/c920/image_raw')
        d('detections_topic', '/camera_detections')
        d('model_path', 'yolo26n.pt')
        d('device', 'cuda:0')
        d('imgsz', 640)
        d('conf', 0.35)
        d('classes', [0, 1, 3])
        d('max_hz', 10.0)
        d('log_period_s', 10.0)
        d('overlay_topic', '/camera_detections/immagine')

        g = lambda k: self.get_parameter(k).value
        self.image_topic = g('image_topic')
        self.model_path = self.resolve_model_path(g('model_path'))
        self.device = g('device')
        self.imgsz = int(g('imgsz'))
        self.conf = float(g('conf'))
        self.classes = [int(c) for c in g('classes')]
        self.min_period = 1.0 / max(float(g('max_hz')), 0.1)
        self.log_period = float(g('log_period_s'))

        self.bridge = CvBridge()
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        self._infer(np.zeros((480, 640, 3), dtype=np.uint8))
        self.get_logger().info(f'Modello di detection caricato: {self.model_path}')

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(Detection2DArray, g('detections_topic'), 10)
        self.pub_img = self.create_publisher(Image, g('overlay_topic'), 1)
        if self.image_topic.endswith('/compressed'):
            self.create_subscription(CompressedImage, self.image_topic, self.on_compressed, qos)
        else:
            self.create_subscription(Image, self.image_topic, self.on_image, qos)

        self._t_ultimo = 0.0
        self._n = 0
        self._n_det = 0
        self._t_infer = 0.0
        self._t_log = time.monotonic()
        self.get_logger().info(
            f'camera_detector pronto: {self.image_topic} -> {g("detections_topic")} '
            f'(classi {self.classes}, conf {self.conf}, max {1.0 / self.min_period:.0f} Hz)')

    def _overlay(self, bgr, xyxy, cls, conf, header):
        import cv2
        img = bgr.copy()
        for (x0, y0, x1, y1), c, s in zip(xyxy, cls, conf):
            p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
            cv2.rectangle(img, p0, p1, (0, 0, 0), 4)
            cv2.rectangle(img, p0, p1, (255, 255, 255), 2)
            cv2.putText(img, f'{NOMI_COCO.get(int(c), str(int(c)))} {s:.2f}', (p0[0], max(p0[1] - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header = header
        self.pub_img.publish(msg)

    def resolve_model_path(self, model_path):
        if os.path.isabs(model_path):
            return model_path
        try:
            from ament_index_python.packages import get_package_share_directory
            candidate = os.path.join(get_package_share_directory('dynamic_layer'), 'models', model_path)
            if os.path.isfile(candidate):
                return candidate
        except Exception:
            pass
        return model_path

    def _infer(self, bgr):
        res = self.model.predict(bgr, imgsz=self.imgsz, device=self.device, conf=self.conf,
                                 classes=self.classes, verbose=False)
        b = res[0].boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)
        return (b.xyxy.detach().cpu().numpy(), b.cls.detach().cpu().numpy().astype(int),
                b.conf.detach().cpu().numpy())

    def _pronto(self):
        ora = time.monotonic()
        if ora - self._t_ultimo < self.min_period:
            return False
        self._t_ultimo = ora
        return True

    def on_image(self, msg):
        if not self._pronto():
            return
        self._elabora(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'), msg.header)

    def on_compressed(self, msg):
        if not self._pronto():
            return
        self._elabora(self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8'), msg.header)

    def _elabora(self, bgr, header):
        t0 = time.monotonic()
        xyxy, cls, conf = self._infer(bgr)
        self._t_infer += time.monotonic() - t0
        out = Detection2DArray()
        out.header = header
        for (x0, y0, x1, y1), c, s in zip(xyxy, cls, conf):
            det = Detection2D()
            det.header = header
            det.bbox.center.position.x = float((x0 + x1) / 2)
            det.bbox.center.position.y = float((y0 + y1) / 2)
            det.bbox.size_x = float(x1 - x0)
            det.bbox.size_y = float(y1 - y0)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = NOMI_COCO.get(int(c), str(int(c)))
            hyp.hypothesis.score = float(s)
            det.results.append(hyp)
            out.detections.append(det)
        self.pub.publish(out)
        if self.pub_img.get_subscription_count() > 0:
            self._overlay(bgr, xyxy, cls, conf, header)
        self._n += 1
        self._n_det += len(out.detections)
        ora = time.monotonic()
        if ora - self._t_log >= self.log_period:
            dt = ora - self._t_log
            self.get_logger().info(
                f'detection: {self._n / dt:.1f} immagini/s, {self._n_det} oggetti, '
                f'{1000 * self._t_infer / max(self._n, 1):.0f} ms per immagine')
            self._n = 0
            self._n_det = 0
            self._t_infer = 0.0
            self._t_log = ora


def main(args=None):
    rclpy.init(args=args)
    node = CameraDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
