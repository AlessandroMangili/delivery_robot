#!/usr/bin/env python3
"""
bbox_lidar_detector.py -- detector FRUSTUM bbox->LiDAR (ramo camera-primary).

Rimpiazza il clustering geometrico class-agnostic (pointcloud_detector.py) con
detection guidata dalla bbox YOLO: la camera dice DOVE c'e' una persona (bbox),
il LiDAR dice A CHE DISTANZA. La classe e' intrinseca (YOLO gira person_only) ->
in uscita ci sono SOLO persone, quindi la separazione persona/non-persona avviene
PRIMA del Kalman (niente gate semantico a valle da rifinire).

Sottoscrive:
  /yolo/detections     vision_msgs/Detection2DArray  (bbox in pixel; person_only)
  /scan/points         sensor_msgs/PointCloud2        (nuvola, frame base_scan)
  /camera/camera_info  sensor_msgs/CameraInfo         (intrinseci K, w, h)

Pubblica:
  /detections          geometry_msgs/PoseArray        (centroidi, frame = cloud)
  /detections/markers  visualization_msgs/MarkerArray (verde = persona, RViz)

INTERFACCIA INVARIATA verso lidar3d_tracker.py: stesso topic /detections, stesso
tipo PoseArray, stesso frame (quello della nuvola, base_scan) e stesso stamp (della
nuvola) -> il tracker fa la sua TF a odom senza modifiche.

PIPELINE (per frame di detection):
  1. Prende la nuvola con lo stamp piu' vicino a quello della bbox (buffer + sync).
  2. crop radiale + voxel downsample (frame sensore).
  3. RANSAC ground removal GLOBALE (una volta): stabile, a differenza del RANSAC
     per-frustum su pochi punti.
  4. Trasforma i punti non-terra nel frame ottico camera (TF) e li proietta in pixel.
  5. Per ogni bbox (con shrink dei bordi) seleziona i punti proiettati dentro.
  6. FRUSTUM -> PERSONA: cluster euclideo sui punti del frustum, prende il cluster
     PIU' VICINO al sensore (la persona e' in primo piano; il muro dietro proietta
     nella stessa bbox ma e' un cluster piu' lontano -> scartato). Unifica le gambe
     assorbendo i cluster entro merge_dist dal cluster scelto (esclude il muro, che
     e' oltre). Centroide = detection.

Nessun filtro geometrico persona (min_height/footprint): lo faceva YOLO. Resta solo
min_points_in_frustum: se una bbox non ha ritorni LiDAR non e' localizzabile ->
si salta quel frame (il tracker fa coasting).
"""

import math
import numpy as np


# ===========================================================================
# FUNZIONI PURE (numpy, niente ROS) -- testabili in isolamento.
# Tutto cio' che sta SOPRA il marker @@@ROS_BOUNDARY@@@ non importa ROS.
# ===========================================================================
def cloud_to_xyz(msg_read):
    """Wrapper attorno a read_points_numpy gia' fatto (vedi nodo). Qui la funzione
    pura lavora su un ndarray gia' estratto; la lascio per simmetria/test."""
    xyz = np.asarray(msg_read, dtype=np.float64).reshape(-1, 3)
    if xyz.size == 0:
        return xyz.reshape(0, 3)
    return xyz[np.isfinite(xyz).all(axis=1)]


def voxel_downsample(xyz, vox):
    """Un punto per voxel. Tiene basso il numero di punti."""
    if vox <= 0.0 or len(xyz) == 0:
        return xyz
    keys = np.floor(xyz / vox).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def fit_ground_plane(xyz, dist_thresh, max_iter, normal_z_min, seed=0):
    """RANSAC per un piano ~orizzontale (normale vicina a z). Scarta i muri.
    Ritorna ((n, d), inlier_mask)."""
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
        if abs(n[2]) < normal_z_min:
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


def remove_ground(xyz, plane, ground_mask, dist_thresh, remove_below):
    """Ritorna i punti non-terra. Se remove_below, toglie anche quelli sotto il piano."""
    if plane is None:
        return xyz
    n, d = plane
    keep = ~ground_mask
    if remove_below:
        signed = xyz.dot(n) + d
        if n[2] < 0:
            signed = -signed
        keep = keep & (signed > -dist_thresh)
    return xyz[keep]


def quat_to_rot(x, y, z, w):
    """Quaternione -> matrice di rotazione 3x3."""
    nrm = x * x + y * y + z * z + w * w
    if nrm < 1e-12:
        return np.eye(3)
    s = 2.0 / nrm
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1 - (yy + zz), xy - wz,       xz + wy],
        [xy + wz,       1 - (xx + zz), yz - wx],
        [xz - wy,       yz + wx,       1 - (xx + yy)]])


def transform_points(R, t, xyz):
    """Applica (R, t) a (N,3): p' = R @ p + t. Ritorna (N,3)."""
    if len(xyz) == 0:
        return xyz.reshape(0, 3)
    return (R @ xyz.T).T + np.asarray(t, dtype=float).reshape(1, 3)


def project_points(K, xyz_cam, front_z_min):
    """Proietta (N,3) del frame OTTICO camera (Z avanti) in pixel (u,v).
    Ritorna (u, v, valid): u,v float (N,), valid bool (N,) True se davanti."""
    n = len(xyz_cam)
    if n == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0, dtype=bool))
    z = xyz_cam[:, 2]
    valid = z > front_z_min
    zz = np.where(valid, z, 1.0)                 # evita /0 sui non validi
    u = K[0, 0] * xyz_cam[:, 0] / zz + K[0, 2]
    v = K[1, 1] * xyz_cam[:, 1] / zz + K[1, 2]
    return u, v, valid


def points_in_bbox(u, v, valid, box):
    """Maschera dei punti (validi) il cui pixel cade dentro box=(u0,v0,u1,v1)."""
    u0, v0, u1, v1 = box
    return valid & (u >= u0) & (u < u1) & (v >= v0) & (v < v1)


def shrink_box(cx, cy, sx, sy, frac):
    """bbox (centro,size) -> (u0,v0,u1,v1) ristretto di 'frac' per lato.
    Lo shrink evita di pescare lo sfondo che sborda ai bordi del box YOLO."""
    hx = 0.5 * sx * (1.0 - frac)
    hy = 0.5 * sy * (1.0 - frac)
    return (cx - hx, cy - hy, cx + hx, cy + hy)


def euclidean_cluster(xyz, tol, min_pts, max_pts):
    """Cluster euclidei (PCL) via cKDTree. Lista di array di indici in xyz."""
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


def select_foreground(xyz_s, tol, min_pts, max_pts, merge_dist):
    """FRUSTUM -> PERSONA. Dai punti del frustum (frame sensore, origine = LiDAR):
      1. cluster euclidei;
      2. sceglie il cluster PIU' VICINO al sensore (persona in primo piano; il muro
         dietro e' un cluster piu' lontano);
      3. unifica le gambe: assorbe i cluster il cui centroide XY dista < merge_dist
         da quello scelto (le due gambe si riuniscono; il muro, oltre merge_dist,
         resta escluso).
    Ritorna (centroid(3,), size(3,), n_points) oppure None se nessun cluster valido.
    Funzione pura -> testabile senza ROS."""
    clusters = euclidean_cluster(xyz_s, tol, min_pts, max_pts)
    if not clusters:
        return None
    cents = [xyz_s[idx].mean(axis=0) for idx in clusters]
    rng = [float(np.linalg.norm(c)) for c in cents]   # distanza dall'origine sensore
    k = int(np.argmin(rng))
    c0 = cents[k]
    chosen = [clusters[k]]
    for i, idx in enumerate(clusters):
        if i == k:
            continue
        if math.hypot(cents[i][0] - c0[0], cents[i][1] - c0[1]) < merge_dist:
            chosen.append(idx)
    pts = xyz_s[np.concatenate(chosen)]
    centroid = pts.mean(axis=0)
    size = pts.max(axis=0) - pts.min(axis=0)
    return centroid, size, int(len(pts))


# @@@ROS_BOUNDARY@@@  (i test isolati eseguono solo cio' che sta sopra questa riga)

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)

from sensor_msgs.msg import PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import Detection2DArray

import tf2_ros
from tf2_ros import TransformException


def read_cloud_xyz(msg):
    """PointCloud2 -> ndarray (N,3). Robusto tra versioni di sensor_msgs_py."""
    try:
        arr = pc2.read_points_numpy(msg, field_names=['x', 'y', 'z'], skip_nans=True)
        return cloud_to_xyz(arr)
    except Exception:
        pts = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        return cloud_to_xyz(np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64))


def bbox_center_size(det):
    """Estrae (cx, cy, sx, sy) da un vision_msgs/Detection2D, robusto alle due
    varianti di BoundingBox2D.center (vision_msgs/Pose2D con .position.x, oppure
    geometry_msgs/Pose2D con .x diretto)."""
    b = det.bbox
    c = b.center
    if hasattr(c, 'position'):
        cx, cy = float(c.position.x), float(c.position.y)
    else:
        cx, cy = float(c.x), float(c.y)
    return cx, cy, float(b.size_x), float(b.size_y)


class BboxLidarDetector(Node):
    def __init__(self):
        super().__init__('bbox_lidar_detector')

        p = self.declare_parameter
        p('points_topic', '/scan/points')
        p('detections_in_topic', '/yolo/detections')
        p('camera_info_topic', '/camera/camera_info')
        p('output_topic', '/detections')
        p('markers_topic', '/detections/markers')
        p('camera_frame', 'camera_rgb_optical_frame')

        # sync nuvola <-> bbox
        p('cloud_buffer', 15)          # quante nuvole recenti tenere
        p('sync_tolerance', 0.20)      # [s] max |t_bbox - t_cloud| per accettare la coppia

        # crop + downsample (frame sensore)
        p('min_range', 0.30)
        p('max_range', 12.0)
        p('voxel_size', 0.05)

        # ground removal (RANSAC, frame sensore dove z ~ su)
        p('ground_dist_thresh', 0.08)
        p('ground_max_iter', 60)
        p('ground_normal_z_min', 0.85)
        p('remove_below_ground', True)

        # frustum
        p('bbox_shrink', 0.10)         # restringe ogni lato del box del 10%
        p('front_z_min', 0.05)         # [m] Z ottica minima (davanti alla camera)
        p('min_points_in_frustum', 4)  # sotto -> bbox non localizzabile, si salta

        # frustum -> persona (cluster piu' vicino + unificazione gambe)
        p('cluster_tolerance', 0.30)
        p('cluster_min_points', 4)
        p('cluster_max_points', 2000)
        p('merge_dist', 0.45)          # < distanza tra 2 persone, > tra le gambe di una

        p('diag_period_frames', 20)

        g = lambda k: self.get_parameter(k).value
        self.points_topic = g('points_topic')
        self.detections_in_topic = g('detections_in_topic')
        self.camera_info_topic = g('camera_info_topic')
        self.output_topic = g('output_topic')
        self.markers_topic = g('markers_topic')
        self.camera_frame = g('camera_frame')
        self.cloud_buffer = int(g('cloud_buffer'))
        self.sync_tolerance = float(g('sync_tolerance'))
        self.min_range = float(g('min_range'))
        self.max_range = float(g('max_range'))
        self.voxel_size = float(g('voxel_size'))
        self.ground_dist_thresh = float(g('ground_dist_thresh'))
        self.ground_max_iter = int(g('ground_max_iter'))
        self.ground_normal_z_min = float(g('ground_normal_z_min'))
        self.remove_below_ground = bool(g('remove_below_ground'))
        self.bbox_shrink = float(g('bbox_shrink'))
        self.front_z_min = float(g('front_z_min'))
        self.min_points_in_frustum = int(g('min_points_in_frustum'))
        self.cluster_tolerance = float(g('cluster_tolerance'))
        self.cluster_min_points = int(g('cluster_min_points'))
        self.cluster_max_points = int(g('cluster_max_points'))
        self.merge_dist = float(g('merge_dist'))
        self.diag_period_frames = int(g('diag_period_frames'))

        # tutte le sub in BEST_EFFORT: un sub best-effort e' compatibile sia con
        # publisher reliable sia best-effort -> nessun mismatch di QoS.
        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE)

        # gruppi callback: le nuvole su un gruppo, le bbox+caminfo su un altro ->
        # la callback pesante (bbox: RANSAC+cluster) non serializza con l'ingresso
        # nuvole ne' affama il feed TF.
        self.cb_cloud = MutuallyExclusiveCallbackGroup()
        self.cb_work = MutuallyExclusiveCallbackGroup()

        self.pub_det = self.create_publisher(PoseArray, self.output_topic, 10)
        self.pub_mrk = self.create_publisher(MarkerArray, self.markers_topic, 10)

        self.sub_cloud = self.create_subscription(
            PointCloud2, self.points_topic, self.on_cloud, qos,
            callback_group=self.cb_cloud)
        self.sub_det = self.create_subscription(
            Detection2DArray, self.detections_in_topic, self.on_detections, qos,
            callback_group=self.cb_work)
        self.sub_cam = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.on_caminfo, qos,
            callback_group=self.cb_work)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)

        self._clouds = []          # buffer: lista di (t_sec, msg)
        self.K = None
        self.img_w = None
        self.img_h = None
        self._frame = 0

        # --- diagnostica heartbeat: contatori per capire quale cancello e' chiuso ---
        self._n_det_rx = 0         # detection bbox ricevute (top di on_detections)
        self._n_cloud_rx = 0       # nuvole ricevute (on_cloud)
        self._n_published = 0      # frame che arrivano a publish()
        self._last_skip = 'avvio'  # ultimo motivo di uscita anticipata
        self._last_nbbox = 0
        self._last_nhit = 0
        # timer sul gruppo nuvole (non affamato dalla callback bbox pesante)
        self.hb_timer = self.create_timer(
            1.0, self._heartbeat, callback_group=self.cb_cloud)

        self.get_logger().info(
            f"bbox_lidar_detector avviato: {self.detections_in_topic} + "
            f"{self.points_topic} -> {self.output_topic}")

    def _heartbeat(self):
        """Ogni secondo dice a colpo d'occhio quale cancello e' chiuso."""
        self.get_logger().info(
            f"[hb] det_rx={self._n_det_rx} cloud_rx={self._n_cloud_rx} "
            f"buf={len(self._clouds)} K={'set' if self.K is not None else 'None'} "
            f"pubblicati={self._n_published} ultimo_skip={self._last_skip} "
            f"(bbox={self._last_nbbox} localizzate={self._last_nhit})")

    # -----------------------------------------------------------------------
    def on_caminfo(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.img_w = int(msg.width)
        self.img_h = int(msg.height)

    def on_cloud(self, msg):
        """Bufferizza le nuvole; la processa solo quando arriva una bbox (sync)."""
        self._n_cloud_rx += 1
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._clouds.append((t, msg))
        if len(self._clouds) > self.cloud_buffer:
            self._clouds.pop(0)

    def _nearest_cloud(self, t_bbox):
        """Nuvola con stamp piu' vicino a t_bbox, se entro sync_tolerance."""
        if not self._clouds:
            return None
        best = min(self._clouds, key=lambda tc: abs(tc[0] - t_bbox))
        if abs(best[0] - t_bbox) > self.sync_tolerance:
            return None
        return best[1]

    def _lookup_cam(self, cloud_frame):
        """TF cloud_frame -> camera_frame. Montaggio rigido (TF statica) -> l'ultima
        e' esatta; timeout piccolo, niente blocco."""
        try:
            return self.tf_buffer.lookup_transform(
                self.camera_frame, cloud_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
        except TransformException:
            return None

    # -----------------------------------------------------------------------
    def on_detections(self, msg):
        self._frame += 1
        self._n_det_rx += 1
        if self.K is None:
            self._last_skip = 'K_none (camera_info assente)'
            return

        t_bbox = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        cloud = self._nearest_cloud(t_bbox)
        if cloud is None:
            self._last_skip = 'no_cloud_sync (stamp fuori tolleranza o buffer vuoto)'
            self.get_logger().warn(
                "nessuna nuvola entro sync_tolerance per questa bbox, salto il frame",
                throttle_duration_sec=2.0)
            return

        cloud_frame = cloud.header.frame_id
        tf_cam = self._lookup_cam(cloud_frame)
        if tf_cam is None:
            self._last_skip = f'tf_fail ({self.camera_frame}<-{cloud_frame})'
            self.get_logger().warn(
                f"TF {self.camera_frame} <- {cloud_frame} non disponibile, salto",
                throttle_duration_sec=2.0)
            return

        # 1. nuvola -> xyz, crop radiale, downsample (frame sensore)
        xyz = read_cloud_xyz(cloud)
        if len(xyz) == 0:
            self._last_skip = 'empty_cloud'
            return
        r = np.hypot(xyz[:, 0], xyz[:, 1])
        xyz = xyz[(r >= self.min_range) & (r <= self.max_range)]
        xyz = voxel_downsample(xyz, self.voxel_size)
        if len(xyz) == 0:
            self._last_skip = 'empty_after_crop'
            return

        # 2. ground removal globale
        plane, ground = fit_ground_plane(
            xyz, self.ground_dist_thresh, self.ground_max_iter,
            self.ground_normal_z_min)
        nonground = remove_ground(
            xyz, plane, ground, self.ground_dist_thresh, self.remove_below_ground)
        if len(nonground) == 0:
            self._last_skip = 'all_ground'
            return

        # 3. non-terra -> frame ottico camera -> pixel
        q = tf_cam.transform.rotation
        tr = tf_cam.transform.translation
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        xyz_cam = transform_points(R, [tr.x, tr.y, tr.z], nonground)
        u, v, valid = project_points(self.K, xyz_cam, self.front_z_min)

        # 4. per ogni bbox: frustum -> persona
        results = []       # (centroid(3,), size(3,), n)
        n_bbox = len(msg.detections)
        n_hit = 0
        for det in msg.detections:
            cx, cy, sx, sy = bbox_center_size(det)
            box = shrink_box(cx, cy, sx, sy, self.bbox_shrink)
            in_box = points_in_bbox(u, v, valid, box)
            if int(in_box.sum()) < self.min_points_in_frustum:
                continue
            fg = select_foreground(
                nonground[in_box], self.cluster_tolerance,
                self.cluster_min_points, self.cluster_max_points, self.merge_dist)
            if fg is None:
                continue
            results.append(fg)
            n_hit += 1

        self.publish(cloud.header, results)
        self._n_published += 1
        self._last_skip = 'ok'
        self._last_nbbox = n_bbox
        self._last_nhit = n_hit

        if self.diag_period_frames > 0 and self._frame % self.diag_period_frames == 0:
            self.get_logger().info(
                f"[det] bbox={n_bbox} persone_localizzate={n_hit} "
                f"nonground={len(nonground)} dt_sync={abs(t_bbox - (cloud.header.stamp.sec + cloud.header.stamp.nanosec * 1e-9)):.3f}s")

    # -----------------------------------------------------------------------
    def publish(self, header, results):
        # PoseArray dei centroidi (frame = nuvola, stamp = nuvola) -> il tracker fa
        # la sua TF a odom come prima. Interfaccia identica al vecchio detector.
        pa = PoseArray()
        pa.header = header
        for c, _size, _n in results:
            ps = Pose()
            ps.position.x = float(c[0])
            ps.position.y = float(c[1])
            ps.position.z = float(c[2])
            ps.orientation.w = 1.0
            pa.poses.append(ps)
        self.pub_det.publish(pa)

        ma = MarkerArray()
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for i, (c, size, _n) in enumerate(results):
            mk = Marker()
            mk.header = header
            mk.ns = 'detections'
            mk.id = i
            mk.type = Marker.CUBE
            mk.action = Marker.ADD
            mk.pose.position.x = float(c[0])
            mk.pose.position.y = float(c[1])
            mk.pose.position.z = float(c[2])
            mk.pose.orientation.w = 1.0
            mk.scale.x = max(float(size[0]), 0.05)
            mk.scale.y = max(float(size[1]), 0.05)
            mk.scale.z = max(float(size[2]), 0.05)
            mk.color = ColorRGBA(r=0.10, g=0.90, b=0.20, a=0.55)
            ma.markers.append(mk)
        self.pub_mrk.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = BboxLidarDetector()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()