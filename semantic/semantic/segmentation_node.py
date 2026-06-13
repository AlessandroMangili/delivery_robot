#!/usr/bin/env python3
"""
Nodo di segmentazione semantica.
Sottoscrive le immagini della camera, esegue SegFormer-B0 (Cityscapes, 19 classi
urbane tra cui 'sidewalk') e pubblica:
  - /semantic/segmentation  (mono8, valore = id classe Cityscapes)
  - /semantic/overlay       (rgb8, immagine colorata per debug in RViz)

Cityscapes trainId rilevanti:
  0 road, 1 sidewalk, 2 building, 8 vegetation, 9 terrain,
  11 person, 13 car, 18 bicycle ...
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# palette Cityscapes (RGB) per i 19 trainId, solo per il debug visivo
CITYSCAPES_PALETTE = np.array([
    [128, 64,128],   # 0 road
    [244, 35,232],   # 1 sidewalk
    [ 70, 70, 70],   # 2 building
    [102,102,156],   # 3 wall
    [190,153,153],   # 4 fence
    [153,153,153],   # 5 pole
    [250,170, 30],   # 6 traffic light
    [220,220,  0],   # 7 traffic sign
    [107,142, 35],   # 8 vegetation
    [152,251,152],   # 9 terrain
    [ 70,130,180],   # 10 sky
    [220, 20, 60],   # 11 person
    [255,  0,  0],   # 12 rider
    [  0,  0,142],   # 13 car
    [  0,  0, 70],   # 14 truck
    [  0, 60,100],   # 15 bus
    [  0, 80,100],   # 16 train
    [  0,  0,230],   # 17 motorcycle
    [119, 11, 32],   # 18 bicycle
], dtype=np.uint8)


class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_name', 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024')
        self.declare_parameter('infer_width', 1024)   # ridimensiona per velocita'
        self.declare_parameter('infer_height', 512)
        self.declare_parameter('publish_overlay', True)

        image_topic = self.get_parameter('image_topic').value
        model_name = self.get_parameter('model_name').value
        self.iw = self.get_parameter('infer_width').value
        self.ih = self.get_parameter('infer_height').value
        self.pub_overlay = self.get_parameter('publish_overlay').value

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Carico {model_name} su {self.device} ...')
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device).eval()
        self.get_logger().info('Modello caricato.')

        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, image_topic, self.cb, 1)
        self.pub_seg = self.create_publisher(Image, '/semantic/segmentation', 1)
        if self.pub_overlay:
            self.pub_ov = self.create_publisher(Image, '/semantic/overlay', 1)

    def cb(self, msg: Image):
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        h0, w0 = rgb.shape[:2]

        # inferenza
        inputs = self.processor(images=rgb, return_tensors='pt',
                                size={'height': self.ih, 'width': self.iw}).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits  # [1, C, h/4, w/4]

        # upsample alla risoluzione originale e argmax -> id classe per pixel
        up = torch.nn.functional.interpolate(
            logits, size=(h0, w0), mode='bilinear', align_corners=False)
        seg = up.argmax(dim=1)[0].to('cpu').numpy().astype(np.uint8)

        seg_msg = self.bridge.cv2_to_imgmsg(seg, encoding='mono8')
        seg_msg.header = msg.header
        self.pub_seg.publish(seg_msg)

        if self.pub_overlay:
            color = CITYSCAPES_PALETTE[np.clip(seg, 0, 18)]
            blend = (0.5 * rgb + 0.5 * color).astype(np.uint8)
            ov = self.bridge.cv2_to_imgmsg(blend, encoding='rgb8')
            ov.header = msg.header
            self.pub_ov.publish(ov)


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
