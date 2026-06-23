#!/usr/bin/env python3
"""
Nodo di segmentazione semantica per la navigazione su marciapiede.

Modello: SegFormer-B0 fine-tuned su Cityscapes (HuggingFace transformers).
  - leggero (~3.7M param backbone) e ADDESTRATO sulle classi giuste:
    gli ID di uscita 0..18 sono i train-id di Cityscapes
    (0=road, 1=sidewalk, 2=building, ... 8=vegetation, 9=terrain, 11=person ...),
    quindi combaciano gia' con la LUT di semantic_costmap_node.py.

Ingresso: topic camera, RAW (sensor_msgs/Image) o COMPRESSED
  (sensor_msgs/CompressedImage) a seconda del parametro use_compressed.
Uscita:  /semantic/segmentation  (mono8, train-id per pixel, a risoluzione piena)
         /semantic/overlay       (rgb8, opzionale, per debug in RViz)

QoS: best-effort depth=1 -> si elabora SEMPRE il frame piu' recente e si
scartano quelli vecchi, cosi' l'inferenza lenta non accumula ritardo.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from transformers import SegformerForSemanticSegmentation


CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

# normalizzazione ImageNet (richiesta da SegFormer)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SegmentationNode(Node):

    def __init__(self):
        super().__init__('segmentation_node')

        self.declare_parameter('image_topic', '/camera/image')
        self.declare_parameter('use_compressed', False)
        self.declare_parameter('model_name', 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024')
        self.declare_parameter('infer_width', 512)
        self.declare_parameter('infer_height', 256)
        self.declare_parameter('publish_overlay', True)
        self.declare_parameter('device', 'auto')   # auto | cuda | cpu

        self.topic = self.get_parameter('image_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.model_name = self.get_parameter('model_name').value
        self.iw = int(self.get_parameter('infer_width').value)
        self.ih = int(self.get_parameter('infer_height').value)
        self.publish_overlay = self.get_parameter('publish_overlay').value

        dev = self.get_parameter('device').value
        if dev == 'auto':
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = dev
        self.use_half = (self.device == 'cuda')

        if self.device == 'cpu':
            # su CPU il throughput dipende dai thread: usali tutti
            torch.set_num_threads(max(1, torch.get_num_threads()))
        else:
            torch.backends.cudnn.benchmark = True

        self.bridge = CvBridge()

        self.get_logger().info(
            f'Caricamento {self.model_name} su {self.device.upper()} '
            f'(infer {self.iw}x{self.ih}, half={self.use_half})')
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        self.model.to(self.device).eval()
        if self.use_half:
            self.model.half()
        self._mean = _MEAN.to(self.device)
        self._std = _STD.to(self.device)

        # QoS: tieni solo l'ultimo frame -> niente coda, niente lag accumulato
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        if self.use_compressed:
            self.sub = self.create_subscription(CompressedImage, self.topic, self.cb_compressed, qos)
            self.get_logger().info(f'Sottoscritto (COMPRESSED) a {self.topic}')
        else:
            self.sub = self.create_subscription(Image, self.topic, self.cb_raw, qos)
            self.get_logger().info(f'Sottoscritto (RAW) a {self.topic}')

        self.pub_seg = self.create_publisher(Image, '/semantic/segmentation', qos)
        self.pub_ov = self.create_publisher(Image, '/semantic/overlay', qos)

        self._busy = False
        self._n = 0
        self._t0 = self.get_clock().now()
        self.get_logger().info('SegFormer pronto.')

    # ---------- ingressi ----------
    def cb_raw(self, msg: Image):
        if self._busy:
            return
        rgb = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        self._process(rgb, msg.header)

    def cb_compressed(self, msg: CompressedImage):
        if self._busy:
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            self.get_logger().warn('imdecode fallita su frame compresso')
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._process(rgb, msg.header)

    # ---------- inferenza ----------
    @torch.no_grad()
    def _process(self, rgb, header):
        self._busy = True
        try:
            h, w = rgb.shape[:2]

            small = cv2.resize(rgb, (self.iw, self.ih), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(small).to(self.device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            x = (x - self._mean) / self._std
            if self.use_half:
                x = x.half()

            logits = self.model(pixel_values=x).logits           # (1,19,ih/4,iw/4)
            logits = F.interpolate(logits.float(), size=(h, w),
                                   mode='bilinear', align_corners=False)
            seg = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()

            seg_msg = self.bridge.cv2_to_imgmsg(seg, 'mono8')
            seg_msg.header = header
            self.pub_seg.publish(seg_msg)

            if self.publish_overlay:
                color = CITYSCAPES_PALETTE[np.clip(seg, 0, 18)]
                overlay = (rgb * 0.55 + color * 0.45).astype(np.uint8)
                ov_msg = self.bridge.cv2_to_imgmsg(overlay, 'rgb8')
                ov_msg.header = header
                self.pub_ov.publish(ov_msg)

            # log del rate effettivo ogni ~30 frame
            self._n += 1
            if self._n % 30 == 0:
                now = self.get_clock().now()
                dt = (now - self._t0).nanoseconds * 1e-9
                self._t0 = now
                self.get_logger().info(f'segmentazione ~{30.0 / dt:.1f} Hz')
        finally:
            self._busy = False


def main():
    rclpy.init()
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()