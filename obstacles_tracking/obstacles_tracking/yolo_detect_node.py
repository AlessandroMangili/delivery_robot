#!/usr/bin/env python3
"""
yolo_detect_node.py -- gira YOLO (Ultralytics) sul flusso camera e pubblica:

  /yolo/detections_image  sensor_msgs/Image             immagine annotata (per RViz)
  /yolo/detections        vision_msgs/Detection2DArray  bbox strutturate (per la pipeline)

La /yolo/detections (Detection2DArray) e' la sorgente consumata da
bbox_lidar_detector.py: bbox in pixel, person_only, con l'header COPIATO
dall'immagine sorgente (quindi lo stamp e' quello di cattura -> il sync col
LiDAR nel detector regge a prescindere da use_sim_time).

NB (cambio rispetto alla versione precedente): prima il nodo pubblicava SOLO
l'immagine annotata, e per errore la metteva su '/yolo/detections'. Cosi' su quel
nome girava un sensor_msgs/Image, non le bbox -> il detector (che aspetta
Detection2DArray) non riceveva nulla. Ora i due tipi stanno su due nomi separati.
In RViz punta il display Image su '/yolo/detections_image'.
"""

import numpy as np


# ===========================================================================
# FUNZIONE PURA (numpy) -- testabile senza ROS/ultralytics.
# Sopra il marker @@@ROS_BOUNDARY@@@ non si importa nulla di ROS.
# ===========================================================================
def _to_np(x):
    """Tensor torch / ndarray / lista -> ndarray. None -> None."""
    if x is None:
        return None
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):
        x = x.numpy()
    return np.asarray(x)


def extract_boxes(result):
    """Da un result Ultralytics estrae le bbox come lista di
    (cx, cy, w, h, conf, cls_id) in pixel (coordinate immagine originale).
    Robusto: boxes assenti/vuote -> []. Funzione pura (duck-typed) -> testabile
    con un mock, senza ultralytics."""
    boxes = getattr(result, 'boxes', None)
    if boxes is None:
        return []
    xywh = _to_np(getattr(boxes, 'xywh', None))
    if xywh is None or len(xywh) == 0:
        return []
    conf = _to_np(getattr(boxes, 'conf', None))
    cls = _to_np(getattr(boxes, 'cls', None))
    out = []
    for i in range(len(xywh)):
        cx, cy, w, h = (float(v) for v in xywh[i][:4])
        c = float(conf[i]) if conf is not None and i < len(conf) else 0.0
        k = int(cls[i]) if cls is not None and i < len(cls) else 0
        out.append((cx, cy, w, h, c, k))
    return out


# @@@ROS_BOUNDARY@@@  (i test isolati eseguono solo cio' che sta sopra questa riga)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import (Detection2DArray, Detection2D, BoundingBox2D,
                             ObjectHypothesisWithPose)
from cv_bridge import CvBridge
import cv2


def set_bbox_center(bb, cx, cy):
    """Imposta il centro della BoundingBox2D robustamente alle due varianti:
    vision_msgs/Pose2D (.position.x, humble 4.x) o geometry_msgs/Pose2D (.x)."""
    try:
        bb.center.position.x = float(cx)
        bb.center.position.y = float(cy)
    except AttributeError:
        bb.center.x = float(cx)
        bb.center.y = float(cy)


def make_detection(header, cx, cy, w, h, conf, cls_id):
    """Costruisce un vision_msgs/Detection2D dalla bbox (pixel)."""
    d = Detection2D()
    d.header = header
    bb = BoundingBox2D()
    set_bbox_center(bb, cx, cy)
    bb.size_x = float(w)
    bb.size_y = float(h)
    d.bbox = bb
    # ipotesi best-effort: il detector la ignora, ma serve per debug / futuro EagerMOT.
    # guardata: differenze di versione non devono mai rompere la pubblicazione.
    try:
        hyp = ObjectHypothesisWithPose()
        try:
            hyp.hypothesis.class_id = str(cls_id)     # vision_msgs 4.x (humble)
            hyp.hypothesis.score = float(conf)
        except AttributeError:
            hyp.id = str(cls_id)                       # vision_msgs 3.x
            hyp.score = float(conf)
        d.results.append(hyp)
    except Exception:
        pass
    return d


class YoloDetectNode(Node):
    def __init__(self):
        super().__init__('yolo_detect_node')

        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('image_topic', '/camera/image')
        # DUE uscite distinte (prima ce n'era una sola, ed era l'immagine su
        # '/yolo/detections' -> tipo sbagliato per la pipeline).
        self.declare_parameter('image_out_topic', '/yolo/detections_image')  # Image (RViz)
        self.declare_parameter('det_out_topic', '/yolo/detections')          # Detection2DArray
        self.declare_parameter('conf', 0.5)          # soglia confidenza
        self.declare_parameter('imgsz', 320)         # risoluzione (come il training)
        self.declare_parameter('person_only', True)  # tieni solo classe person (COCO id 0)
        self.declare_parameter('device', 'cpu')      # 'cpu' o '0' se avessi una GPU
        self.declare_parameter('process_every_n', 1) # 1 = ogni frame; 2 = uno su due

        gp = self.get_parameter
        model_path = gp('model_path').value
        self.conf = float(gp('conf').value)
        self.imgsz = int(gp('imgsz').value)
        self.person_only = bool(gp('person_only').value)
        self.device = gp('device').value
        self.every_n = max(1, int(gp('process_every_n').value))
        self._frame_i = 0

        try:
            from ultralytics import YOLO
        except Exception as e:
            self.get_logger().error(
                f'ultralytics non importabile: {e}. Installa con: pip install ultralytics')
            raise
        self.get_logger().info(f'Carico modello YOLO: {model_path} (device={self.device})')
        self.model = YOLO(model_path)

        self.bridge = CvBridge()
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(
            Image, gp('image_topic').value, self.cb, qos)
        # due publisher: immagine annotata + bbox strutturate
        self.pub_img = self.create_publisher(Image, gp('image_out_topic').value, qos)
        self.pub_det = self.create_publisher(Detection2DArray, gp('det_out_topic').value, qos)

        self._t_ema = None
        self.get_logger().info(
            f"YOLO detector pronto. In: {gp('image_topic').value} -> "
            f"img:{gp('image_out_topic').value}  det:{gp('det_out_topic').value}. "
            f"conf={self.conf}, imgsz={self.imgsz}, person_only={self.person_only}")

    def cb(self, msg: Image):
        # skip frame per alleggerire la CPU se richiesto
        self._frame_i += 1
        if (self._frame_i % self.every_n) != 0:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        classes = [0] if self.person_only else None  # 0 = person (COCO e tuo dataset)

        import time
        t0 = time.time()
        res = self.model.predict(img, conf=self.conf, imgsz=self.imgsz,
                                 classes=classes, device=self.device, verbose=False)
        dt = time.time() - t0
        self._t_ema = dt if self._t_ema is None else 0.9 * self._t_ema + 0.1 * dt

        result = res[0]
        dets = extract_boxes(result)   # [(cx,cy,w,h,conf,cls), ...] in pixel

        # --- 1. Detection2DArray su /yolo/detections (per la pipeline) ---
        # header COPIATO dall'immagine -> stesso stamp del frame catturato.
        da = Detection2DArray()
        da.header = msg.header
        for (cx, cy, w, h, conf, cls_id) in dets:
            da.detections.append(
                make_detection(msg.header, cx, cy, w, h, conf, cls_id))
        self.pub_det.publish(da)   # pubblica sempre (anche vuoto): ritmo costante

        # --- 2. immagine annotata su /yolo/detections_image (per RViz) ---
        annotated = result.plot()
        n = len(dets)
        fps = 1.0 / self._t_ema if self._t_ema and self._t_ema > 0 else 0.0
        cv2.putText(annotated, f'det:{n}  {fps:4.1f} fps (CPU)', (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        out = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out.header = msg.header
        self.pub_img.publish(out)


def main():
    rclpy.init()
    node = YoloDetectNode()
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
