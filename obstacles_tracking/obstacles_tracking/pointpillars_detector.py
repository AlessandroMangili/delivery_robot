import os
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import (Detection3D, Detection3DArray,
                             ObjectHypothesisWithPose)


_PF = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def dtype_from_fields(fields, point_step, is_bigendian=False):
    entries = []
    offset = 0
    for f in sorted(fields, key=lambda f: f.offset):
        if f.datatype not in _PF or getattr(f, 'count', 1) != 1:
            return None
        if f.offset < offset:
            return None
        if f.offset > offset:
            entries.append((f'_pad{offset}', np.uint8, f.offset - offset))
        entries.append((f.name, _PF[f.datatype]))
        offset = f.offset + np.dtype(_PF[f.datatype]).itemsize
    if offset > point_step:
        return None
    if offset < point_step:
        entries.append(('_padend', np.uint8, point_step - offset))
    dt = np.dtype(entries)
    return dt.newbyteorder('>') if is_bigendian else dt


def cloud_to_xyzi(msg):
    dt = dtype_from_fields(msg.fields, msg.point_step,
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


class PointPillarsDetector(Node):
    def __init__(self):
        super().__init__('pointpillars_detector')

        p = self.declare_parameter
        p('cloud_topic', '/scan/points')
        p('markers_topic', '/detections/markers')
        p('detections3d_topic', '/detections3d')
        p('cfg_file', 'cfgs/custom_models/pointpillar.yaml')
        p('cfg_dir', '/mnt/2tbpartition/mangili/OpenPCDet/tools')
        p('ckpt', '/home/mangili/Downloads/checkpoint_epoch_80.pth')
        p('score_thresh', 0.30)
        p('debug_show_rejected', False)
        p('diag_period_frames', 20)

        g = lambda k: self.get_parameter(k).value
        self.score_thresh = float(g('score_thresh'))
        self.debug_show_rejected = bool(g('debug_show_rejected'))
        self.diag_period_frames = int(g('diag_period_frames'))
        self._frame = 0

        self._load_model(str(g('cfg_dir')), str(g('cfg_file')), str(g('ckpt')))

        self.pub_mrk = self.create_publisher(MarkerArray, g('markers_topic'), 10)
        self.pub_d3 = self.create_publisher(
            Detection3DArray, g('detections3d_topic'), 10)
        cloud_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PointCloud2, g('cloud_topic'),
                                 self.on_cloud, cloud_qos)

        self.get_logger().info(
            f"PointPillars detector ready | classes={self.class_names} "
            f"| threshold={self.score_thresh} | {g('cloud_topic')} -> "
            f"{g('detections3d_topic')}")

    def _load_model(self, cfg_dir, cfg_file, ckpt):
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import DatasetTemplate
        from pcdet.models import build_network
        from pcdet.utils import common_utils

        previous_cwd = os.getcwd()
        os.chdir(cfg_dir)
        try:
            cfg_from_yaml_file(cfg_file, cfg)
        finally:
            os.chdir(previous_cwd)

        self.torch = torch
        self.class_names = list(cfg.CLASS_NAMES)
        self.pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)

        class _SingleFrame(DatasetTemplate):
            def __init__(s, dcfg, names, logger):
                super().__init__(dataset_cfg=dcfg, class_names=names,
                                 training=False, root_path=None, logger=logger)
            def __len__(s):
                return 1
            def prepare_points(s, points):
                d = s.prepare_data(data_dict={'points': points, 'frame_id': 0})
                return s.collate_batch([d])

        logger = common_utils.create_logger()
        self.ds = _SingleFrame(cfg.DATA_CONFIG, self.class_names, logger)
        self.model = build_network(model_cfg=cfg.MODEL,
                                   num_class=len(self.class_names),
                                   dataset=self.ds)
        self.model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=False)
        self.model.cuda().eval()

        from pcdet.models import load_data_to_gpu
        self._to_gpu = load_data_to_gpu

    def on_cloud(self, msg):
        self._frame += 1
        t0 = time.perf_counter()
        points = cloud_to_xyzi(msg)
        t_read = time.perf_counter() - t0
        if len(points) == 0:
            return

        t0 = time.perf_counter()
        with self.torch.no_grad():
            batch = self.ds.prepare_points(points)
            self._to_gpu(batch)
            pred, _ = self.model.forward(batch)
        box = pred[0]['pred_boxes'].cpu().numpy()
        score = pred[0]['pred_scores'].cpu().numpy()
        t_inf = time.perf_counter() - t0

        kept = score >= self.score_thresh
        self.publish(msg.header, box, score, kept)

        if self.diag_period_frames > 0 and self._frame % self.diag_period_frames == 0:
            tot = t_read + t_inf
            self.get_logger().info(
                f"[pp] points={len(points)} box={len(box)} "
                f"above_thresh={int(kept.sum())} "
                f"score_max={float(score.max()) if len(score) else 0.0:.2f} | "
                f"read={t_read*1000:.0f} inf={t_inf*1000:.0f} "
                f"TOT={tot*1000:.0f} ms ({1/max(tot,1e-9):.1f} Hz)")

    def publish(self, header, box, score, kept):
        d3 = Detection3DArray()
        d3.header = header
        for i in np.flatnonzero(kept):
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
            if hasattr(ip, 'hypothesis'):
                ip.hypothesis.class_id = 'Person'
                ip.hypothesis.score = float(score[i])
            else:
                ip.score = float(score[i])
            det.results.append(ip)
            d3.detections.append(det)
        self.pub_d3.publish(d3)

        ma = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        mid = 0
        for i in range(len(box)):
            ok = bool(kept[i])
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
