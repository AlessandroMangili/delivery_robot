#!/usr/bin/env python3

import math
import os
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)

from sensor_msgs.msg import PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

import tf2_ros
from tf2_ros import TransformException


# ---------------------------------------------------------------------------
# Funzioni geometriche pure (testabili in isolamento, senza ROS)
# ---------------------------------------------------------------------------
def cloud_to_xyz(msg):
    """PointCloud2 -> ndarray (N,3) float64. Robusto tra versioni di sensor_msgs_py."""
    campi = {f.name: f for f in msg.fields}
    if (all(k in campi and campi[k].datatype == 7 for k in 'xyz') and not msg.is_bigendian
            and msg.point_step > 0 and len(msg.data) >= msg.point_step):
        n = msg.width * msg.height
        raw = np.frombuffer(msg.data, dtype=np.uint8, count=n * msg.point_step).reshape(n, msg.point_step)
        xyz = np.stack([raw[:, campi[k].offset:campi[k].offset + 4].copy().view(np.float32).ravel()
                        for k in 'xyz'], axis=1).astype(np.float64)
        return xyz[np.isfinite(xyz).all(axis=1)]
    try:
        arr = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z'], skip_nans=True)
        xyz = np.asarray(arr, dtype=np.float64).reshape(-1, 3)
    except Exception:
        pts = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        xyz = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)
    if xyz.size == 0:
        return xyz.reshape(0, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


def voxel_downsample(xyz, vox):
    if vox <= 0.0 or len(xyz) == 0:
        return xyz
    k = np.floor(xyz / vox).astype(np.int64)
    k -= k.min(axis=0)
    span = k.max(axis=0) + 1
    lin = (k[:, 0] * span[1] + k[:, 1]) * span[2] + k[:, 2]
    _, idx = np.unique(lin, return_index=True)
    return xyz[np.sort(idx)]


def fit_ground_plane(xyz, dist_thresh, max_iter, normal_z_min, seed=0, sample=3000, fit_range=0.0):
    N = len(xyz)
    if N < 10:
        return None, np.zeros(N, dtype=bool)
    rng = np.random.default_rng(seed)
    C = xyz[np.hypot(xyz[:, 0], xyz[:, 1]) < fit_range] if fit_range > 0.0 else xyz
    if len(C) < 10:
        C = xyz
    S = C if len(C) <= sample else C[rng.integers(0, len(C), sample)]
    M = len(S)
    tri = rng.integers(0, M, (max_iter, 3))
    p0, p1, p2 = S[tri[:, 0]], S[tri[:, 1]], S[tri[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-6
    n[ok] /= nn[ok, None]
    ok &= np.abs(n[:, 2]) >= normal_z_min
    if not ok.any():
        return None, np.zeros(N, dtype=bool)
    n = n[ok]
    d = -np.einsum('ij,ij->i', n, p0[ok])
    cnt = (np.abs(S @ n.T + d) < dist_thresh).sum(axis=0)
    b = int(np.argmax(cnt))
    nb, db = n[b], float(d[b])
    inl = np.abs(xyz @ nb + db) < dist_thresh
    P = xyz[inl]
    if len(P) >= 50:
        c = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        nr = vt[2]
        if abs(nr[2]) >= normal_z_min:
            nb, db = nr, -float(nr.dot(c))
            inl = np.abs(xyz @ nb + db) < dist_thresh
    return (nb, db), inl


def euclidean_cluster(xyz, tol, min_pts, max_pts):
    """Estrazione cluster euclidei (algoritmo PCL) via cKDTree.
    Ritorna lista di array di indici in xyz."""
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    N = len(xyz)
    if N == 0:
        return []
    pairs = cKDTree(xyz).query_pairs(tol, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(N, N))
    n_comp, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind='stable')
    counts = np.bincount(labels, minlength=n_comp)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return [order[starts[c]:starts[c] + counts[c]].astype(np.int64)
            for c in range(n_comp) if min_pts <= counts[c] <= max_pts]


def merge_clusters(cluster_pts, merge_dist):
    n = len(cluster_pts)
    if n <= 1:
        return cluster_pts
    cen = np.array([c[:, :2].mean(axis=0) for c in cluster_pts])
    D = np.hypot(cen[:, None, 0] - cen[None, :, 0], cen[:, None, 1] - cen[None, :, 1])
    ii, jj = np.nonzero(np.triu(D < merge_dist, 1))
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(ii, jj):
        parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [np.vstack([cluster_pts[i] for i in g]) if len(g) > 1 else cluster_pts[g[0]]
            for g in groups.values()]


def lzf_decompress(inp, n):
    out = bytearray(n)
    ip = op = 0
    L = len(inp)
    while ip < L:
        c = inp[ip]
        ip += 1
        if c < 32:
            k = c + 1
            out[op:op + k] = inp[ip:ip + k]
            ip += k
            op += k
        else:
            k = c >> 5
            ref = op - ((c & 31) << 8) - 1
            if k == 7:
                k += inp[ip]
                ip += 1
            ref -= inp[ip]
            ip += 1
            k += 2
            if ref + k <= op:
                out[op:op + k] = out[ref:ref + k]
            else:
                for j in range(k):
                    out[op + j] = out[ref + j]
            op += k
    return bytes(out)


def load_pcd_xyz(path):
    raw = open(path, 'rb').read()
    head = {}
    pos = 0
    while True:
        end = raw.index(b'\n', pos)
        line = raw[pos:end].decode('ascii', 'ignore').strip()
        pos = end + 1
        if not line or line.startswith('#'):
            continue
        key, *val = line.split()
        head[key.upper()] = val
        if key.upper() == 'DATA':
            break
    names = head['FIELDS']
    sizes = [int(s) for s in head['SIZE']]
    types = head['TYPE']
    counts = [int(c) for c in head.get('COUNT', ['1'] * len(names))]
    n = int(head['POINTS'][0])
    kind = {'F': 'f', 'I': 'i', 'U': 'u'}
    mode = head['DATA'][0].lower()
    if mode == 'ascii':
        A = np.loadtxt(raw[pos:].decode().splitlines(), ndmin=2)
        cols = np.cumsum([0] + counts)
        xyz = np.stack([A[:, cols[names.index(k)]] for k in 'xyz'], axis=1)
    elif mode == 'binary':
        dt = np.dtype([(nm, f'<{kind[t]}{s}', (c,)) if c > 1 else (nm, f'<{kind[t]}{s}')
                       for nm, s, t, c in zip(names, sizes, types, counts)])
        A = np.frombuffer(raw, dtype=dt, count=n, offset=pos)
        xyz = np.stack([A[k].astype(np.float64) for k in 'xyz'], axis=1)
    elif mode == 'binary_compressed':
        cs, us = np.frombuffer(raw[pos:pos + 8], dtype='<u4')
        d = lzf_decompress(raw[pos + 8:pos + 8 + int(cs)], int(us))
        off = 0
        col = {}
        for nm, s, t, c in zip(names, sizes, types, counts):
            nb = n * s * c
            col[nm] = np.frombuffer(d[off:off + nb], dtype=f'<{kind[t]}{s}').reshape(n, c)[:, 0]
            off += nb
        xyz = np.stack([col[k].astype(np.float64) for k in 'xyz'], axis=1)
    else:
        raise ValueError(f'formato PCD non gestito: {mode}')
    return xyz[np.isfinite(xyz).all(axis=1)]


def quat_to_rot(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


# ---------------------------------------------------------------------------
# Nodo ROS2
# ---------------------------------------------------------------------------
def camera_rescue(rejected, accepted_c, boxes, K, wh, R_cv, t_cv, min_pts,
                  min_height, max_height, max_footprint, depth_band):
    if len(boxes) == 0 or len(rejected) == 0:
        return []
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w, h = wh
    taken = np.zeros(len(boxes), dtype=bool)
    for c in accepted_c:
        pc = R_cv @ c + t_cv
        if pc[2] <= 0.3:
            continue
        u, v = fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy
        taken |= ((boxes[:, 0] <= u) & (u <= boxes[:, 2]) &
                  (boxes[:, 1] <= v) & (v <= boxes[:, 3]))
    proj = []
    for pts in rejected:
        pc = pts @ R_cv.T + t_cv
        z = pc[:, 2]
        front = z > 0.3
        zs = np.where(front, z, 1.0)
        proj.append((fx * pc[:, 0] / zs + cx, fy * pc[:, 1] / zs + cy, z, front))
    out = []
    for b, (x0, y0, x1, y1) in enumerate(boxes):
        if taken[b]:
            continue
        best = None
        for pts, (u, v, z, front) in zip(rejected, proj):
            inside = front & (u >= max(x0, 0)) & (u <= min(x1, w)) & (v >= max(y0, 0)) & (v <= min(y1, h))
            if inside.sum() < min_pts:
                continue
            d = z[inside]
            sub = pts[inside][d <= np.percentile(d, 20) + depth_band]
            if len(sub) < min_pts:
                continue
            size = sub.max(axis=0) - sub.min(axis=0)
            if not (min_height <= size[2] <= max_height and math.hypot(size[0], size[1]) <= max_footprint):
                continue
            if best is None or len(sub) > best[2]:
                best = (sub.mean(axis=0), size, len(sub))
        if best is not None:
            out.append(best[:2])
    return out


class PointCloudDetector(Node):
    def __init__(self):
        super().__init__('pointcloud_detector')

        # --- parametri (dichiarati, poi letti) ---
        p = self.declare_parameter
        p('input_topic', '/velodyne_points')
        p('detections_topic', '/detections')
        p('markers_topic', '/detections/markers')

        # crop
        p('min_range', 0.30)        # scarta il corpo del robot / punti troppo vicini
        p('max_range', 12.0)        # oltre, il LiDAR è rado/rumoroso
        p('crop_min_z', -1.0)       # box in z, frame sensore
        p('crop_max_z', 2.5)

        # downsample
        p('voxel_size', 0.05)

        # ground removal (RANSAC)
        p('ground_dist_thresh', 0.08)
        p('ground_max_iter', 60)
        p('ground_normal_z_min', 0.85)   # |n_z| min per considerare il piano orizzontale
        p('remove_below_ground', True)
        p('ground_fit_range', 0.0)

        # clustering
        p('cluster_tolerance', 0.30)
        p('cluster_min_points', 6)
        p('cluster_max_points', 800)

        # merge in world-space: unisce cluster vicini (cura la frammentazione:
        # le gambe di una persona spezzate in 2 cluster tornano una detection sola)
        p('merge_enabled', True)
        p('merge_dist', 0.45)       # < distanza tra 2 persone, > tra le gambe di una

        # filtro geometrico "persona" (permissivo: la camera rifinisce a valle in Step 3)
        p('min_height', 0.20)
        p('max_height', 2.20)
        p('max_footprint', 1.00)

        p('map_filter', False)
        p('map_pcd', '')
        p('map_filter_dist', 0.20)
        p('map_filter_frac', 0.50)
        p('map_filter_min_dist', 0.0)
        p('map_frame', 'map')
        p('odom_frame', 'odom')
        p('map_tf_timeout', 0.1)

        p('camera_rescue', False)
        p('yolo_topic', '/camera_detections')
        p('camera_info_topic', '/c920/camera_info')
        p('camera_frame', 'c920_optical')
        p('rescue_classes', ['person'])
        p('rescue_box_margin', 0.10)
        p('rescue_max_age', 0.25)
        p('rescue_depth_band', 0.40)
        p('rescue_max_footprint', 1.0)

        # diagnostica / debug
        p('debug_show_rejected', True)   # mostra in rosso i cluster scartati (per tarare)
        p('diag_period_frames', 20)
        p('coda_nuvole', 5)

        g = lambda k: self.get_parameter(k).value
        self.input_topic = g('input_topic')
        self.detections_topic = g('detections_topic')
        self.markers_topic = g('markers_topic')
        self.min_range = float(g('min_range'))
        self.max_range = float(g('max_range'))
        self.crop_min_z = float(g('crop_min_z'))
        self.crop_max_z = float(g('crop_max_z'))
        self.voxel_size = float(g('voxel_size'))
        self.ground_dist_thresh = float(g('ground_dist_thresh'))
        self.ground_max_iter = int(g('ground_max_iter'))
        self.ground_normal_z_min = float(g('ground_normal_z_min'))
        self.remove_below_ground = bool(g('remove_below_ground'))
        self.ground_fit_range = float(g('ground_fit_range'))
        self.cluster_tolerance = float(g('cluster_tolerance'))
        self.cluster_min_points = int(g('cluster_min_points'))
        self.cluster_max_points = int(g('cluster_max_points'))
        self.merge_enabled = bool(g('merge_enabled'))
        self.merge_dist = float(g('merge_dist'))
        self.min_height = float(g('min_height'))
        self.max_height = float(g('max_height'))
        self.max_footprint = float(g('max_footprint'))
        self.debug_show_rejected = bool(g('debug_show_rejected'))
        self.diag_period_frames = int(g('diag_period_frames'))
        self.coda_nuvole = max(1, int(g('coda_nuvole')))

        self.map_tree = None
        self.map_filter_dist = float(g('map_filter_dist'))
        self.map_filter_frac = float(g('map_filter_frac'))
        self.map_filter_min_dist = float(g('map_filter_min_dist'))
        self.map_frame = g('map_frame')
        self.odom_frame = g('odom_frame')
        self.map_tf_timeout = float(g('map_tf_timeout'))
        self._n_map_removed = 0
        self._n_map_no_tf = 0
        self.camera_rescue = bool(g('camera_rescue'))
        self.tf_buffer = None
        if bool(g('map_filter')) or self.camera_rescue:
            self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, None, spin_thread=True)
        if bool(g('map_filter')):
            from scipy.spatial import cKDTree
            from ament_index_python.packages import get_package_share_directory
            path = g('map_pcd') or os.path.join(
                get_package_share_directory('localization'), 'maps', 'mappa.pcd')
            xyz_map = load_pcd_xyz(path)
            self.map_tree = cKDTree(xyz_map)
            self.get_logger().info(
                f"filtro mappa: {len(xyz_map)} punti da {path}, scarto i cluster con almeno il "
                f"{100 * self.map_filter_frac:.0f}% dei punti entro {self.map_filter_dist:.2f} m "
                f"dalla mappa (frame {self.map_frame}) oltre {self.map_filter_min_dist:.1f} m dal LiDAR")

        self.camera_frame = g('camera_frame')
        self.rescue_classes = set(str(c) for c in g('rescue_classes'))
        self.rescue_box_margin = float(g('rescue_box_margin'))
        self.rescue_max_age = float(g('rescue_max_age'))
        self.rescue_depth_band = float(g('rescue_depth_band'))
        self.rescue_max_footprint = float(g('rescue_max_footprint'))
        self.K = None
        self.cam_wh = None
        self.yolo = deque(maxlen=20)
        self._n_rescued = 0
        self._n_rescue_no_box = 0
        self._n_rescue_no_tf = 0
        if self.camera_rescue:
            from vision_msgs.msg import Detection2DArray
            self.sub_yolo = self.create_subscription(
                Detection2DArray, g('yolo_topic'), self.on_yolo, 10)
            self.sub_info = self.create_subscription(
                CameraInfo, g('camera_info_topic'), self.on_caminfo, 1)
            self.get_logger().info(
                f"recupero con la camera: cluster scartati dentro i riquadri "
                f"{sorted(self.rescue_classes)} di {g('yolo_topic')}")

        # QoS Best Effort: i cloud dal bridge sono Best Effort
        qos = QoSProfile(depth=self.coda_nuvole,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE)

        self.pub_det = self.create_publisher(PoseArray, self.detections_topic, 10)
        self.pub_mrk = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.sub = self.create_subscription(
            PointCloud2, self.input_topic, self.on_cloud, qos)

        self._frame = 0
        self.get_logger().info(
            f"pointcloud_detector avviato: input={self.input_topic} "
            f"-> {self.detections_topic}")

    # -----------------------------------------------------------------------
    def on_cloud(self, msg):
        self._frame += 1
        xyz = cloud_to_xyz(msg)
        n_in = len(xyz)
        if n_in == 0:
            return

        # crop radiale + box in z
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        m = ((r >= self.min_range) & (r <= self.max_range) &
             (xyz[:, 2] >= self.crop_min_z) & (xyz[:, 2] <= self.crop_max_z))
        xyz = xyz[m]
        n_crop = len(xyz)
        if n_crop == 0:
            return

        # downsample
        xyz = voxel_downsample(xyz, self.voxel_size)

        # rimozione piano di terra
        plane, ground = fit_ground_plane(
            xyz, self.ground_dist_thresh, self.ground_max_iter,
            self.ground_normal_z_min, fit_range=self.ground_fit_range)
        if plane is None:
            self.get_logger().warn(
                "RANSAC: nessun piano di terra trovato, uso tutti i punti")
            nonground = xyz
        else:
            n, d = plane
            keep = ~ground
            if self.remove_below_ground:
                signed = xyz.dot(n) + d
                if n[2] < 0:               # orienta la normale verso l'alto
                    signed = -signed
                keep = keep & (signed > -self.ground_dist_thresh)
            nonground = xyz[keep]
        n_ng = len(nonground)

        # clustering
        clusters = euclidean_cluster(
            nonground, self.cluster_tolerance,
            self.cluster_min_points, self.cluster_max_points)

        # merge in world-space: riunisce cluster frammentati (gambe di una persona)
        cluster_pts = [nonground[idx] for idx in clusters]
        n_pre = len(cluster_pts)
        if self.merge_enabled and len(cluster_pts) > 1:
            cluster_pts = merge_clusters(cluster_pts, self.merge_dist)

        # feature per cluster + filtro geometrico
        results = []   # (centroid(3,), size(3,), ok, rescued)
        rejected = []
        tf_map = self._map_tf(msg.header) if self.map_tree is not None else None
        n_map = 0
        for pts in cluster_pts:
            c = pts.mean(axis=0)
            size = pts.max(axis=0) - pts.min(axis=0)
            height = float(size[2])
            footprint = float(math.hypot(size[0], size[1]))
            ok = ((self.min_height <= height <= self.max_height) and
                  (footprint <= self.max_footprint))
            if (ok and tf_map is not None and math.hypot(c[0], c[1]) > self.map_filter_min_dist and
                    self._map_fraction(pts, tf_map) >= self.map_filter_frac):
                ok = False
                n_map += 1
            results.append((c, size, ok, False))
            if not ok:
                rejected.append(pts)
        self._n_map_removed = n_map
        if self.camera_rescue and rejected:
            rescued = self._rescue(msg.header, rejected, [c for c, _, ok, _ in results if ok])
            self._n_rescued += len(rescued)
            results.extend((c, size, True, True) for c, size in rescued)

        n_ok = sum(1 for _, _, ok, _ in results if ok)
        self.publish(msg.header, results)

        if self.diag_period_frames > 0 and self._frame % self.diag_period_frames == 0:
            self.get_logger().info(
                f"[det] in={n_in} crop={n_crop} nonground={n_ng} "
                f"cluster={n_pre}->{len(cluster_pts)} persone={n_ok}"
                + (f" scartati_mappa={self._n_map_removed} senza_tf={self._n_map_no_tf}"
                   if self.map_tree is not None else '')
                + (f" salvati_camera={self._n_rescued} senza_riquadri={self._n_rescue_no_box} "
                   f"senza_tf_camera={self._n_rescue_no_tf}" if self.camera_rescue else ''))

    def _map_tf(self, header):
        stamp = rclpy.time.Time.from_msg(header.stamp)
        try:
            return self._rt(self.tf_buffer.lookup_transform(
                self.map_frame, header.frame_id, stamp, timeout=Duration(seconds=0.0)))
        except TransformException:
            pass
        try:
            R1, t1 = self._rt(self.tf_buffer.lookup_transform(
                self.map_frame, self.odom_frame, rclpy.time.Time(), timeout=Duration(seconds=0.0)))
            R2, t2 = self._rt(self.tf_buffer.lookup_transform(
                self.odom_frame, header.frame_id, stamp, timeout=Duration(seconds=self.map_tf_timeout)))
        except TransformException as e:
            self._n_map_no_tf += 1
            self.get_logger().warn(f'TF {self.map_frame} <- {header.frame_id} non disponibile, '
                                   f'cluster non filtrati ({e})', throttle_duration_sec=5.0)
            return None
        return R1 @ R2, R1 @ t2 + t1

    def on_caminfo(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.cam_wh = (int(msg.width), int(msg.height))

    def on_yolo(self, msg):
        box = []
        for det in msg.detections:
            if not det.results or det.results[0].hypothesis.class_id not in self.rescue_classes:
                continue
            cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
            mw = det.bbox.size_x * (0.5 + self.rescue_box_margin)
            mh = det.bbox.size_y * (0.5 + self.rescue_box_margin)
            box.append((cx - mw, cy - mh, cx + mw, cy + mh))
        self.yolo.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                          np.array(box, dtype=float).reshape(-1, 4), msg.header.stamp))

    def _rescue(self, header, rejected, accepted_c):
        if self.K is None or not self.yolo:
            self._n_rescue_no_box += 1
            return []
        t_c = header.stamp.sec + header.stamp.nanosec * 1e-9
        cand = [y for y in self.yolo if y[0] - t_c <= 0.05 and t_c - y[0] <= self.rescue_max_age]
        if not cand:
            self._n_rescue_no_box += 1
            return []
        _, boxes, stamp_y = min(cand, key=lambda y: abs(y[0] - t_c))
        if len(boxes) == 0:
            return []
        try:
            tf = self.tf_buffer.lookup_transform_full(
                self.camera_frame, rclpy.time.Time.from_msg(stamp_y),
                header.frame_id, rclpy.time.Time.from_msg(header.stamp),
                self.odom_frame, timeout=Duration(seconds=0.05))
        except TransformException as e:
            self._n_rescue_no_tf += 1
            self.get_logger().warn(f'TF {self.camera_frame} <- {header.frame_id} non disponibile, '
                                   f'nessun recupero ({e})', throttle_duration_sec=5.0)
            return []
        R, t = self._rt(tf)
        return camera_rescue(rejected, accepted_c, boxes, self.K, self.cam_wh, R, t,
                             self.cluster_min_points, self.min_height, self.max_height,
                             self.rescue_max_footprint, self.rescue_depth_band)

    @staticmethod
    def _rt(tf):
        q = tf.transform.rotation
        tr = tf.transform.translation
        return quat_to_rot(q.x, q.y, q.z, q.w), np.array([tr.x, tr.y, tr.z])

    def _map_fraction(self, pts, tf_map):
        R, t = tf_map
        d, _ = self.map_tree.query(pts @ R.T + t, distance_upper_bound=self.map_filter_dist)
        return float(np.isfinite(d).mean())

    # -----------------------------------------------------------------------
    def publish(self, header, results):
        # PoseArray dei centroidi accettati (interfaccia verso Step 2)
        pa = PoseArray()
        pa.header = header
        for c, _size, ok, _rescued in results:
            if not ok:
                continue
            ps = Pose()
            ps.position.x = float(c[0])
            ps.position.y = float(c[1])
            ps.position.z = float(c[2])
            ps.orientation.w = 1.0
            pa.poses.append(ps)
        self.pub_det.publish(pa)

        # MarkerArray per RViz (verde = persona, rosso = scartato)
        ma = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        mid = 0
        for c, size, ok, rescued in results:
            if (not ok) and (not self.debug_show_rejected):
                continue
            mk = Marker()
            mk.header = header
            mk.ns = 'detections'
            mk.id = mid
            mid += 1
            mk.type = Marker.CUBE
            mk.action = Marker.ADD
            mk.pose.position.x = float(c[0])
            mk.pose.position.y = float(c[1])
            mk.pose.position.z = float(c[2])
            mk.pose.orientation.w = 1.0
            mk.scale.x = max(float(size[0]), 0.05)
            mk.scale.y = max(float(size[1]), 0.05)
            mk.scale.z = max(float(size[2]), 0.05)
            if rescued:
                mk.color = ColorRGBA(r=0.10, g=0.50, b=0.95, a=0.60)
            elif ok:
                mk.color = ColorRGBA(r=0.10, g=0.90, b=0.20, a=0.55)
            else:
                mk.color = ColorRGBA(r=0.90, g=0.20, b=0.10, a=0.30)
            ma.markers.append(mk)
        self.pub_mrk.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudDetector()
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