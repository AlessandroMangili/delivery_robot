#!/usr/bin/env python3
"""
pointcloud_detector.py — Stadio di DETECTION per il tracker AB3DMOT (LiDAR 3D).

Step 1 della pipeline. Puramente geometrico, nessuna rete neurale:

    /scan/points (PointCloud2, 16 corone)
        -> crop (range radiale + box in z)
        -> voxel downsample
        -> rimozione piano di terra (RANSAC, normale ~ verticale)
        -> clustering euclideo (cKDTree region growing, algoritmo PCL)
        -> filtro geometrico "persona" (altezza / footprint / n. punti)
        -> centroidi = detection

Uscite:
    /detections           geometry_msgs/PoseArray          (centroidi accettati)
    /detections/markers   visualization_msgs/MarkerArray   (verifica in RViz)

Frame: le detection escono NEL FRAME DEL CLOUD (base_scan). La trasformazione
in odom la fa il tracker a valle (Step 2), non questo nodo: qui restiamo
puramente geometrici, come il front-end di AB3DMOT.

Dipendenze: numpy, scipy (cKDTree). Se manca scipy:
    pip install scipy    (oppure: sudo apt install python3-scipy)
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA


# ---------------------------------------------------------------------------
# Funzioni geometriche pure (testabili in isolamento, senza ROS)
# ---------------------------------------------------------------------------
def cloud_to_xyz(msg):
    """PointCloud2 -> ndarray (N,3) float64. Robusto tra versioni di sensor_msgs_py."""
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
    """Un punto per voxel. Tiene basso il numero di punti per il clustering."""
    if vox <= 0.0 or len(xyz) == 0:
        return xyz
    keys = np.floor(xyz / vox).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def fit_ground_plane(xyz, dist_thresh, max_iter, normal_z_min, seed=0):
    """RANSAC per un piano ~orizzontale (normale vicina a z).
    Ritorna ((n, d), inlier_mask). Scarta i piani verticali (muri)."""
    N = len(xyz)
    if N < 10:
        return None, np.zeros(N, dtype=bool)
    rng = np.random.default_rng(seed)
    best_count = 0
    best_inliers = None
    best_plane = None
    for _ in range(max_iter):
        idx = rng.choice(N, 3, replace=False)
        p0, p1, p2 = xyz[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn
        if abs(n[2]) < normal_z_min:      # non orizzontale -> non è terra
            continue
        d = -float(n.dot(p0))
        dist = np.abs(xyz.dot(n) + d)
        inliers = dist < dist_thresh
        c = int(inliers.sum())
        if c > best_count:
            best_count = c
            best_inliers = inliers
            best_plane = (n, d)
    if best_inliers is None:
        return None, np.zeros(N, dtype=bool)
    return best_plane, best_inliers


def euclidean_cluster(xyz, tol, min_pts, max_pts):
    """Estrazione cluster euclidei (algoritmo PCL) via cKDTree.
    Ritorna lista di array di indici in xyz."""
    from scipy.spatial import cKDTree
    N = len(xyz)
    if N == 0:
        return []
    tree = cKDTree(xyz)
    visited = np.zeros(N, dtype=bool)
    clusters = []
    for i in range(N):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        comp = [i]
        while queue:
            j = queue.pop()
            for k in tree.query_ball_point(xyz[j], tol):
                if not visited[k]:
                    visited[k] = True
                    queue.append(k)
                    comp.append(k)
        if min_pts <= len(comp) <= max_pts:
            clusters.append(np.asarray(comp, dtype=np.int64))
    return clusters


# ---------------------------------------------------------------------------
# Nodo ROS2
# ---------------------------------------------------------------------------
class PointCloudDetector(Node):
    def __init__(self):
        super().__init__('pointcloud_detector')

        # --- parametri (dichiarati, poi letti) ---
        p = self.declare_parameter
        p('input_topic', '/scan/points')
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

        # clustering
        p('cluster_tolerance', 0.30)
        p('cluster_min_points', 6)
        p('cluster_max_points', 800)

        # filtro geometrico "persona" (permissivo: la camera rifinisce a valle in Step 3)
        p('min_height', 0.20)
        p('max_height', 2.20)
        p('max_footprint', 1.00)

        # diagnostica / debug
        p('debug_show_rejected', True)   # mostra in rosso i cluster scartati (per tarare)
        p('diag_period_frames', 20)

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
        self.cluster_tolerance = float(g('cluster_tolerance'))
        self.cluster_min_points = int(g('cluster_min_points'))
        self.cluster_max_points = int(g('cluster_max_points'))
        self.min_height = float(g('min_height'))
        self.max_height = float(g('max_height'))
        self.max_footprint = float(g('max_footprint'))
        self.debug_show_rejected = bool(g('debug_show_rejected'))
        self.diag_period_frames = int(g('diag_period_frames'))

        # QoS Best Effort: i cloud dal bridge sono Best Effort
        qos = QoSProfile(depth=5,
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
            self.ground_normal_z_min)
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

        # feature per cluster + filtro geometrico
        results = []   # (centroid(3,), size(3,), ok)
        for idx in clusters:
            pts = nonground[idx]
            c = pts.mean(axis=0)
            size = pts.max(axis=0) - pts.min(axis=0)
            height = float(size[2])
            footprint = float(math.hypot(size[0], size[1]))
            ok = ((self.min_height <= height <= self.max_height) and
                  (footprint <= self.max_footprint))
            results.append((c, size, ok))

        n_ok = sum(1 for _, _, ok in results if ok)
        self.publish(msg.header, results)

        if self.diag_period_frames > 0 and self._frame % self.diag_period_frames == 0:
            self.get_logger().info(
                f"[det] in={n_in} crop={n_crop} nonground={n_ng} "
                f"cluster={len(clusters)} persone={n_ok}")

    # -----------------------------------------------------------------------
    def publish(self, header, results):
        # PoseArray dei centroidi accettati (interfaccia verso Step 2)
        pa = PoseArray()
        pa.header = header
        for c, _size, ok in results:
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
        for c, size, ok in results:
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
            if ok:
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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
