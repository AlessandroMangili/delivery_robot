#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

from semantic.seg_utils import build_remap_lut, apply_remap, resize_label_map, to_mono8


class SegInferNode(Node):
    def __init__(self):
        super().__init__('seg_infer_node')

        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('seg_topic', '/semantic/segmentation')
        self.declare_parameter('model_path', 'yolo26n-sem.pt')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('imgsz', 1024)
        # half: NON passato alla predict (deprecato in questa versione: usa
        # 'quantize'). La fp16 la fissi all'EXPORT dell'engine, non qui.
        self.declare_parameter('half', True)
        self.declare_parameter('image_encoding', 'bgr8')
        # remap come stringa "src dst, src dst, ..." per evitare il typing
        # ambiguo delle liste vuote nei parametri ROS2. Vuoto = identita'.
        self.declare_parameter('label_remap', '')
        self.declare_parameter('ignore_id', 255)
        self.declare_parameter('publish_at_input_res', True)
        self.declare_parameter('warmup', True)
        self.declare_parameter('log_period_s', 5.0)

        gp = self.get_parameter
        self.image_topic = gp('image_topic').value
        self.seg_topic = gp('seg_topic').value
        self.model_path = self.resolve_model_path(gp('model_path').value)
        self.device = gp('device').value
        self.imgsz = int(gp('imgsz').value)
        self.half = bool(gp('half').value)
        self.encoding = gp('image_encoding').value
        self.ignore_id = int(gp('ignore_id').value)
        self.at_input_res = bool(gp('publish_at_input_res').value)
        self.do_warmup = bool(gp('warmup').value)
        self.log_period = float(gp('log_period_s').value)

        raw = str(gp('label_remap').value).strip()
        toks = [int(x) for x in raw.replace(',', ' ').split()] if raw else []
        if len(toks) % 2 != 0:
            raise ValueError('label_remap deve avere un numero pari di interi: '
                             '"src dst, src dst, ..."')
        pairs = [(toks[i], toks[i + 1]) for i in range(0, len(toks), 2)]
        self.remap_lut = build_remap_lut(pairs, self.ignore_id)
        self.identity_remap = (len(pairs) == 0)

        self.bridge = CvBridge()

        # Ultralytics importato qui cosi' il modulo si introspeziona senza torch
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        self.get_logger().info('Modello semantico caricato: %s' % self.model_path)

        if self.do_warmup:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self._infer(dummy)
            self.get_logger().info('Warmup inference eseguita.')

        pub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(Image, self.seg_topic, pub_qos)
        self.create_subscription(Image, self.image_topic, self.image_cb, sub_qos)

        self._n = 0
        self._t_last_log = self.get_clock().now().nanoseconds * 1e-9
        self.get_logger().info(
            'seg_infer pronto: %s -> %s (imgsz=%d, device=%s)'
            % (self.image_topic, self.seg_topic, self.imgsz, self.device))

    def resolve_model_path(self, model_path):
        """Path assoluto -> usato cosi' com'e'. Path relativo -> cercato prima
        in share/semantic/models/ (installato via setup.py data_files), poi
        nella directory corrente (comportamento storico dello script standalone)."""
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
        """Esegue il modello su un'immagine BGR numpy, ritorna la mappa (H,W)."""
        res = self.model.predict(bgr, imgsz=self.imgsz, device=self.device,
                                 verbose=False)
        cls = res[0].semantic_mask.data          # tensor torch (H,W), intero
        return cls.detach().cpu().numpy()

    def image_cb(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.encoding)
        h_in, w_in = bgr.shape[:2]

        cls = self._infer(bgr)
        if not self.identity_remap:
            cls = apply_remap(cls, self.remap_lut)
        else:
            cls = to_mono8(cls)                   # garantisce uint8/contiguo

        if self.at_input_res:
            cls = resize_label_map(cls, w_in, h_in)
        cls = to_mono8(cls)

        out = self.bridge.cv2_to_imgmsg(cls, encoding='mono8')
        out.header = msg.header                  # STESSO stamp + frame del RGB
        self.pub.publish(out)

        self._n += 1
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._t_last_log >= self.log_period:
            hz = self._n / max(1e-6, now - self._t_last_log)
            seen = sorted(np.unique(cls).tolist())
            self.get_logger().info(
                'seg out %dx%d @ ~%.1f Hz (classi: %s)'
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
