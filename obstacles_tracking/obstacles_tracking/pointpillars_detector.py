#!/usr/bin/env python3
"""Detection 3D con PointPillars, drop-in di pointcloud_detector.py.

Pubblica la STESSA interfaccia del detector geometrico:
  - PoseArray  su /detections          (centroidi, verso il tracker)
  - MarkerArray su /detections/markers (box per RViz)
cosi' il tracker non va toccato e i due detector sono intercambiabili
cambiando solo quale nodo lanci: e' l'A/B che serve in tesi.

DEVE girare con il python del venv, che ha torch e pcdet:
  /mnt/2tbpartition/mangili/pcdet_venv/bin/python3 pointpillars_detector.py --ros-args ...

Il config usa _BASE_CONFIG_ con percorso RELATIVO, quindi il nodo si sposta da
solo in OpenPCDet/tools prima di caricarlo (vedi cfg_dir).
"""

import os
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)


# ===================== lettura nuvola (vista strutturata) =====================
# codici datatype di sensor_msgs/PointField
_PF = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def dtype_da_campi(fields, point_step, is_bigendian=False):
    """dtype strutturato che descrive UN punto del messaggio, padding compreso.
    Funzione PURA. Ritorna None se il layout non e' interpretabile."""
    voci = []
    offset = 0
    for f in sorted(fields, key=lambda f: f.offset):
        if f.datatype not in _PF or getattr(f, 'count', 1) != 1:
            return None
        if f.offset < offset:
            return None
        if f.offset > offset:
            voci.append((f'_pad{offset}', np.uint8, f.offset - offset))
        voci.append((f.name, _PF[f.datatype]))
        offset = f.offset + np.dtype(_PF[f.datatype]).itemsize
    if offset > point_step:
        return None
    if offset < point_step:
        voci.append(('_padfine', np.uint8, point_step - offset))
    dt = np.dtype(voci)
    return dt.newbyteorder('>') if is_bigendian else dt


def cloud_to_xyzi(msg):
    """PointCloud2 -> ndarray (N,4) float32 [x, y, z, intensity], senza NaN/inf.

    La quarta colonna e' forzata a ZERO anche se il messaggio porta
    un'intensita' vera: nel dataset di training aveva un solo valore distinto
    (tutta zero), quindi passarne una diversa metterebbe il modello fuori
    distribuzione su una feature che la PillarVFE riceve davvero in ingresso.
    """
    dt = dtype_da_campi(msg.fields, msg.point_step,
                        getattr(msg, 'is_bigendian', False))
    if dt is None or not all(n in dt.names for n in ('x', 'y', 'z')):
        return np.zeros((0, 4), dtype=np.float32)
    n = len(msg.data) // msg.point_step
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    rec = np.frombuffer(bytes(msg.data), dtype=dt, count=n)
    out = np.zeros((n, 4), dtype=np.float32)
    out[:, 0] = rec['x']
    out[:, 1] = rec['y']
    out[:, 2] = rec['z']
    return out[np.isfinite(out).all(axis=1)]


# ===================== nodo =====================
class PointPillarsDetector(Node):
    def __init__(self):
        super().__init__('pointpillars_detector')

        p = self.declare_parameter
        p('cloud_topic', '/scan/points')
        p('detections_topic', '/detections')
        p('markers_topic', '/detections/markers')
        p('detections3d_topic', '/detections3d')   # box 3D complete, per EagerMOT
        p('cfg_file', 'cfgs/custom_models/pointpillar.yaml')
        p('cfg_dir', '/mnt/2tbpartition/mangili/OpenPCDet/tools')
        p('ckpt', '/home/mangili/Downloads/checkpoint_epoch_80.pth')
        p('score_thresh', 0.30)        # soglia di confidenza per accettare una box
        p('debug_show_rejected', False)
        p('diag_period_frames', 20)

        g = lambda k: self.get_parameter(k).value
        self.score_thresh = float(g('score_thresh'))
        self.debug_show_rejected = bool(g('debug_show_rejected'))
        self.diag_period_frames = int(g('diag_period_frames'))
        self._frame = 0

        self._carica_modello(str(g('cfg_dir')), str(g('cfg_file')), str(g('ckpt')))

        self.pub_det = self.create_publisher(PoseArray, g('detections_topic'), 10)
        self.pub_mrk = self.create_publisher(MarkerArray, g('markers_topic'), 10)
        # Il PoseArray perde dimensioni, yaw e score: basta al tracker vecchio
        # ma non a EagerMOT, che sulla distanza scalata usa anche le
        # dimensioni. Quindi pubblichiamo ANCHE la box 3D completa.
        self.pub_d3 = self.create_publisher(
            Detection3DArray, g('detections3d_topic'), 10)
        cloud_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PointCloud2, g('cloud_topic'),
                                 self.on_cloud, cloud_qos)

        self.get_logger().info(
            f"PointPillars detector pronto | classi={self.class_names} "
            f"| soglia={self.score_thresh} | {g('cloud_topic')} -> "
            f"{g('detections_topic')}")

    def _carica_modello(self, cfg_dir, cfg_file, ckpt):
        """Import di pcdet e costruzione della rete. L'import sta QUI e non in
        cima al file perche' tira dentro torch/spconv/numba: se fallisce, il
        messaggio d'errore arriva a nodo gia' avviato ed e' leggibile nei log."""
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network
        from pcdet.utils import common_utils

        # _BASE_CONFIG_ nel yaml e' relativo alla CWD: mi ci sposto.
        vecchia = os.getcwd()
        os.chdir(cfg_dir)
        try:
            cfg_from_yaml_file(cfg_file, cfg)
        finally:
            os.chdir(vecchia)

        self.torch = torch
        self.class_names = list(cfg.CLASS_NAMES)
        self.pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)

        class _Uno(DatasetTemplate):
            """Riusa il preprocessing del config: stesso identico trattamento
            del training (mask fuori range, voxelizzazione)."""
            def __init__(s, dcfg, names, logger):
                super().__init__(dataset_cfg=dcfg, class_names=names,
                                 training=False, root_path=None, logger=logger)
            def __len__(s):
                return 1
            def prepara(s, punti):
                d = s.prepare_data(data_dict={'points': punti, 'frame_id': 0})
                return s.collate_batch([d])

        logger = common_utils.create_logger()
        self.ds = _Uno(cfg.DATA_CONFIG, self.class_names, logger)
        self.model = build_network(model_cfg=cfg.MODEL,
                                   num_class=len(self.class_names),
                                   dataset=self.ds)
        self.model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
        self.model.cuda().eval()

        from pcdet.models import load_data_to_gpu
        self._to_gpu = load_data_to_gpu

    # -----------------------------------------------------------------------
    def on_cloud(self, msg):
        self._frame += 1
        t0 = time.perf_counter()
        punti = cloud_to_xyzi(msg)
        t_read = time.perf_counter() - t0
        if len(punti) == 0:
            return

        t0 = time.perf_counter()
        with self.torch.no_grad():
            batch = self.ds.prepara(punti)
            self._to_gpu(batch)
            pred, _ = self.model.forward(batch)
        box = pred[0]['pred_boxes'].cpu().numpy()      # (M,7) x y z dx dy dz yaw
        score = pred[0]['pred_scores'].cpu().numpy()
        t_inf = time.perf_counter() - t0

        tenute = score >= self.score_thresh
        self.publish(msg.header, box, score, tenute)

        if self.diag_period_frames > 0 and self._frame % self.diag_period_frames == 0:
            tot = t_read + t_inf
            self.get_logger().info(
                f"[pp] punti={len(punti)} box={len(box)} "
                f"sopra_soglia={int(tenute.sum())} "
                f"score_max={float(score.max()) if len(score) else 0.0:.2f} | "
                f"read={t_read*1000:.0f} inf={t_inf*1000:.0f} "
                f"TOT={tot*1000:.0f} ms ({1/max(tot,1e-9):.1f} Hz)")

    def publish(self, header, box, score, tenute):
        # PoseArray dei centroidi accettati (stessa interfaccia del geometrico).
        # L'orientamento porta lo yaw della box: il tracker oggi lo ignora, ma
        # cosi' non si perde quando passeremo alla strada completa.
        # --- Detection3DArray: box completa (centro, taglia, yaw, score) ---
        d3 = Detection3DArray()
        d3.header = header
        for i in np.flatnonzero(tenute):
            det = Detection3D()
            det.header = header
            det.bbox.center.position.x = float(box[i, 0])
            det.bbox.center.position.y = float(box[i, 1])
            det.bbox.center.position.z = float(box[i, 2])
            yaw = float(box[i, 6])
            det.bbox.center.orientation.z = float(np.sin(yaw / 2.0))
            det.bbox.center.orientation.w = float(np.cos(yaw / 2.0))
            det.bbox.size.x = float(box[i, 3])
            det.bbox.size.y = float(box[i, 4])
            det.bbox.size.z = float(box[i, 5])
            ip = ObjectHypothesisWithPose()
            # vision_msgs cambia layout fra le distribuzioni: in Humble lo
            # score sta in .hypothesis, altrove direttamente sull'oggetto.
            if hasattr(ip, 'hypothesis'):
                ip.hypothesis.class_id = 'Person'
                ip.hypothesis.score = float(score[i])
            else:
                ip.score = float(score[i])
            det.results.append(ip)
            d3.detections.append(det)
        self.pub_d3.publish(d3)

        pa = PoseArray()
        pa.header = header
        for i in np.flatnonzero(tenute):
            ps = Pose()
            ps.position.x = float(box[i, 0])
            ps.position.y = float(box[i, 1])
            ps.position.z = float(box[i, 2])
            yaw = float(box[i, 6])
            ps.orientation.z = float(np.sin(yaw / 2.0))
            ps.orientation.w = float(np.cos(yaw / 2.0))
            pa.poses.append(ps)
        self.pub_det.publish(pa)

        ma = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        mid = 0
        for i in range(len(box)):
            ok = bool(tenute[i])
            if (not ok) and (not self.debug_show_rejected):
                continue
            mk = Marker()
            mk.header = header
            mk.ns = 'detections'
            mk.id = mid
            mid += 1
            mk.type = Marker.CUBE
            mk.action = Marker.ADD
            mk.pose.position.x = float(box[i, 0])
            mk.pose.position.y = float(box[i, 1])
            mk.pose.position.z = float(box[i, 2])
            yaw = float(box[i, 6])
            mk.pose.orientation.z = float(np.sin(yaw / 2.0))
            mk.pose.orientation.w = float(np.cos(yaw / 2.0))
            mk.scale.x = max(float(box[i, 3]), 0.05)
            mk.scale.y = max(float(box[i, 4]), 0.05)
            mk.scale.z = max(float(box[i, 5]), 0.05)
            if ok:
                mk.color = ColorRGBA(r=0.10, g=0.90, b=0.20, a=0.55)
            else:
                mk.color = ColorRGBA(r=0.90, g=0.20, b=0.10, a=0.30)
            ma.markers.append(mk)

            # etichetta con lo score, per leggere la confidenza in RViz
            tx = Marker()
            tx.header = header
            tx.ns = 'det_score'
            tx.id = mid
            mid += 1
            tx.type = Marker.TEXT_VIEW_FACING
            tx.action = Marker.ADD
            tx.pose.position.x = float(box[i, 0])
            tx.pose.position.y = float(box[i, 1])
            tx.pose.position.z = float(box[i, 2]) + 1.1
            tx.pose.orientation.w = 1.0
            tx.scale.z = 0.25
            tx.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9)
            tx.text = f'{score[i]:.2f}'
            ma.markers.append(tx)

        self.pub_mrk.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = PointPillarsDetector()
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
