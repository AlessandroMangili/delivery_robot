#!/usr/bin/env python3
"""
dynamic_tracker.py -- Rilevamento e anticipazione di ostacoli dinamici.

APPROCCIO (chiaro e diretto):
  1. il LiDAR rileva i punti degli ostacoli, in frame mondo 'odom'
  2. i punti vicini sono RAGGRUPPATI in cluster (un cluster = un oggetto):
     le due gambe di un pedone, essendo vicine, finiscono nello stesso cluster
     -> il pedone e' UN oggetto solo (come una bici o un'auto)
  3. il centroide di ogni cluster e' TRACCIATO scansione dopo scansione
  4. dallo spostamento del centroide fra frame si stima la VELOCITA' (vx,vy)
  5. se l'oggetto si muove (|v| > soglia), si proietta un CONO di costo nella
     direzione del moto, lungo ~ velocita' x orizzonte (dove l'oggetto SARA')
  6. una FRECCIA per oggetto mostra la direzione stimata (debug in RViz)

MODALITA' (flag use_camera):
  - use_camera=false : solo LiDAR. Clustering per distanza. Semplice e robusto,
    ma due pedoni molto vicini possono fondersi in un cluster.
  - use_camera=true  : la camera semantica separa i pedoni vicini e conferma che
    un cluster e' davvero una classe dinamica (pedone/veicolo), scartando i
    cluster statici (muri/alberi che il LiDAR vede ma non sono dinamici).

Output:
  /dynamic_cost   (OccupancyGrid, frame odom)  -> layer local costmap di Nav2
  /dynamic_tracks (MarkerArray)                -> frecce direzione (debug)
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import tf2_ros
from tf2_ros import TransformException

try:
    from cv_bridge import CvBridge
    _HAVE_BRIDGE = True
except Exception:
    _HAVE_BRIDGE = False


def transform_to_matrix(t):
    """TransformStamped -> matrice 4x4 (rototraslazione)."""
    q = t.transform.rotation
    tr = t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


class DynamicTracker(Node):
    def __init__(self):
        super().__init__('dynamic_tracker')

        # ---------------- parametri ----------------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('output_topic', '/dynamic_cost')
        self.declare_parameter('world_frame', 'odom')       # FISSO ma continuo
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('odom_topic', '/odom')

        # griglia rolling di uscita
        self.declare_parameter('size_m', 8.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('max_range', 8.0)

        # --- clustering LiDAR ---
        # due punti entro cluster_eps_m appartengono allo stesso oggetto: unisce
        # le due gambe del pedone (~30-40 cm) in UN cluster.
        self.declare_parameter('cluster_eps_m', 0.5)
        self.declare_parameter('cluster_min_pts', 2)     # min punti per cluster valido
        self.declare_parameter('cluster_max_size_m', 1.5)  # scarta cluster enormi (muri)

        # --- tracking ---
        self.declare_parameter('gate_dist_m', 0.7)       # associazione track<->cluster
        self.declare_parameter('vel_ema', 0.4)           # smoothing velocita' (0..1)
        self.declare_parameter('pos_ema', 0.5)           # smoothing posizione
        self.declare_parameter('min_obs', 3)             # frame min prima di fidarsi
        self.declare_parameter('track_timeout_s', 0.6)   # eta' max track non visto
        self.declare_parameter('min_speed', 0.15)        # [m/s] sotto = fermo, no cono

        # --- proiezione cono ---
        self.declare_parameter('horizon_s', 2.0)         # [s] quanto proietto avanti
        self.declare_parameter('cone_halfwidth_m', 0.4)  # semi-larghezza base
        self.declare_parameter('cone_spread', 1.5)       # quanto si allarga in punta
        self.declare_parameter('max_cost', 100)
        self.declare_parameter('mark_obstacle', True)    # marca anche l'oggetto stesso

        # --- gate movimento robot (rotazioni rapide sporcano il tracking) ---
        self.declare_parameter('gate_ang_vel', 1.0)      # [rad/s]

        # --- camera (opzionale) ---
        self.declare_parameter('use_camera', False)
        self.declare_parameter('seg_topic', '/semantic/segmentation')
        self.declare_parameter('info_topic', '/camera/camera_info')
        self.declare_parameter('camera_optical_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('dynamic_classes', [11, 12, 13, 14, 15, 16, 17, 18])
        self.declare_parameter('cam_assoc_m', 0.8)       # tolleranza assoc LiDAR<->camera

        gp = self.get_parameter
        self.scan_topic = gp('scan_topic').value
        self.out_topic = gp('output_topic').value
        self.world_frame = gp('world_frame').value
        self.robot_frame = gp('robot_frame').value
        self.odom_topic = gp('odom_topic').value
        self.size_m = float(gp('size_m').value)
        self.res = float(gp('resolution').value)
        self.rate_hz = float(gp('rate_hz').value)
        self.max_range = float(gp('max_range').value)
        self.cluster_eps = float(gp('cluster_eps_m').value)
        self.cluster_min_pts = int(gp('cluster_min_pts').value)
        self.cluster_max_size = float(gp('cluster_max_size_m').value)
        self.gate_dist = float(gp('gate_dist_m').value)
        self.vel_ema = float(gp('vel_ema').value)
        self.pos_ema = float(gp('pos_ema').value)
        self.min_obs = int(gp('min_obs').value)
        self.track_timeout = float(gp('track_timeout_s').value)
        self.min_speed = float(gp('min_speed').value)
        self.horizon_s = float(gp('horizon_s').value)
        self.cone_halfwidth = float(gp('cone_halfwidth_m').value)
        self.cone_spread = float(gp('cone_spread').value)
        self.max_cost = int(gp('max_cost').value)
        self.mark_obstacle = bool(gp('mark_obstacle').value)
        self.gate_ang_vel = float(gp('gate_ang_vel').value)
        self.use_camera = bool(gp('use_camera').value)
        self.seg_topic = gp('seg_topic').value
        self.info_topic = gp('info_topic').value
        self.cam_frame = gp('camera_optical_frame').value
        self.dyn_classes = list(gp('dynamic_classes').value)
        self.cam_assoc = float(gp('cam_assoc_m').value)

        self.n = int(round(self.size_m / self.res))
        self.dyn_lut = np.zeros(256, dtype=bool)
        for c in self.dyn_classes:
            self.dyn_lut[int(c)] = True

        # ---------------- stato ----------------
        self.last_scan = None
        self.tracks = []          # dict(id,x,y,vx,vy,n,t)
        self.next_id = 0
        self.robot_ang = 0.0
        self.arrows = []
        # camera
        self.K = None
        self.last_seg = None
        self.bridge = CvBridge() if (_HAVE_BRIDGE and self.use_camera) else None

        # ---------------- TF / IO ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos)
        if self.use_camera:
            self.create_subscription(CameraInfo, self.info_topic, self.info_cb, qos)
            self.create_subscription(Image, self.seg_topic, self.seg_cb, qos)

        self.pub = self.create_publisher(OccupancyGrid, self.out_topic, 1)
        self.markers_pub = self.create_publisher(MarkerArray, '/dynamic_tracks', 1)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.update)

        self.get_logger().info(
            f'dynamic_tracker avviato | scan={self.scan_topic} use_camera={self.use_camera} | '
            f'griglia {self.n}x{self.n}@{self.res}m in {self.world_frame} | '
            f'cluster_eps={self.cluster_eps}m')

    # -------------------------------------------------------------------------
    def scan_cb(self, msg): self.last_scan = msg
    def odom_cb(self, msg): self.robot_ang = abs(float(msg.twist.twist.angular.z))
    def info_cb(self, msg): self.K = np.array(msg.k).reshape(3, 3)
    def seg_cb(self, msg):  self.last_seg = msg

    def lookup(self, frame, stamp):
        try:
            return self.tf_buffer.lookup_transform(
                self.world_frame, frame, stamp, timeout=Duration(seconds=0.05))
        except Exception:
            try:
                return self.tf_buffer.lookup_transform(
                    self.world_frame, frame, rclpy.time.Time())
            except Exception:
                return None

    # -------------------------------------------------------------------------
    def update(self):
        if self.last_scan is None:
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9

        # posa robot -> origine finestra rolling (centrata sul robot, quantizzata)
        T_base = self.lookup(self.robot_frame, rclpy.time.Time())
        if T_base is None:
            return
        rx = T_base.transform.translation.x
        ry = T_base.transform.translation.y
        ox = np.floor((rx - self.size_m / 2.0) / self.res) * self.res
        oy = np.floor((ry - self.size_m / 2.0) / self.res) * self.res

        # 1) punti LiDAR in frame mondo
        pts = self.scan_points_world(self.last_scan, rx, ry)
        if pts is None:
            return

        # 2) clustering: raggruppa punti vicini in oggetti (unisce le gambe)
        clusters = self.cluster_points(pts)

        # 2b) FILTRO DINAMICO/STATICO con la camera semantica.
        # Solo i cluster confermati come classe dinamica (pedone/veicolo) passano;
        # gli statici (muri/alberi) sono scartati -> non entrano nel layer.
        if self.use_camera:
            if self.K is not None and self.last_seg is not None:
                clusters = self.camera_filter(clusters)
            else:
                # camera richiesta ma non ancora disponibile: non generare nulla
                # (meglio nessun cono che falsi coni su statici)
                clusters = []
                self.get_logger().warn(
                    'use_camera=true ma segmentazione/camera_info non ancora ricevute',
                    throttle_duration_sec=3.0)

        # centroidi dei cluster = detection
        dets = [c['centroid'] for c in clusters]

        # 3-4) tracking dei centroidi -> velocita'
        self.track(dets, now_s)

        # 5-6) proietta coni + frecce
        cost = self.build_cost(ox, oy)
        self.publish_cost(cost, ox, oy)
        self.publish_arrows()

    # -------------------------------------------------------------------------
    def scan_points_world(self, scan, rx, ry):
        """Converte lo scan in punti (x,y) nel frame mondo, entro max_range."""
        T = self.lookup(scan.header.frame_id, rclpy.time.Time())
        if T is None:
            return None
        M = transform_to_matrix(T)
        sx, sy = M[0, 3], M[1, 3]
        yaw = np.arctan2(M[1, 0], M[0, 0])

        ang = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        r = np.asarray(scan.ranges, dtype=np.float32)
        good = np.isfinite(r) & (r >= scan.range_min) & (r <= scan.range_max)
        ang = ang[good]; r = r[good]
        if r.size == 0:
            return np.empty((0, 2))
        # punti nel frame sensore -> mondo
        px = r * np.cos(ang); py = r * np.sin(ang)
        wx = sx + px * np.cos(yaw) - py * np.sin(yaw)
        wy = sy + px * np.sin(yaw) + py * np.cos(yaw)
        d = np.hypot(wx - rx, wy - ry)
        keep = d <= self.max_range
        return np.stack([wx[keep], wy[keep]], axis=1)

    # -------------------------------------------------------------------------
    def cluster_points(self, pts):
        """Clustering per distanza (DBSCAN-like leggero): punti entro cluster_eps
        nello stesso cluster. Unisce le due gambe di un pedone in un oggetto.
        Implementazione a griglia per efficienza (no librerie esterne)."""
        clusters = []
        if len(pts) == 0:
            return clusters

        # griglia hash a celle di lato eps: punti nella stessa cella o adiacenti
        # sono vicini. Uso union-find sui punti.
        eps = self.cluster_eps
        cell = {}
        keys = np.floor(pts / eps).astype(int)
        for i, (kx, ky) in enumerate(keys):
            cell.setdefault((kx, ky), []).append(i)

        parent = list(range(len(pts)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        eps2 = eps * eps
        for i, (kx, ky) in enumerate(keys):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in cell.get((kx + dx, ky + dy), []):
                        if j <= i:
                            continue
                        d2 = np.sum((pts[i] - pts[j]) ** 2)
                        if d2 <= eps2:
                            union(i, j)

        groups = {}
        for i in range(len(pts)):
            groups.setdefault(find(i), []).append(i)

        for idxs in groups.values():
            if len(idxs) < self.cluster_min_pts:
                continue
            cpts = pts[idxs]
            size = np.max(np.ptp(cpts, axis=0)) if len(cpts) > 1 else 0.0
            if size > self.cluster_max_size:      # scarta muri/oggetti grandi
                continue
            centroid = cpts.mean(axis=0)
            clusters.append({'centroid': (float(centroid[0]), float(centroid[1])),
                             'pts': cpts, 'size': float(size)})
        return clusters

    # -------------------------------------------------------------------------
    def camera_filter(self, clusters):
        """FILTRO DINAMICO/STATICO tramite camera semantica.
        Tiene SOLO i cluster LiDAR che coincidono con un pedone/veicolo visto
        dalla camera (classe dinamica). Tutti gli altri (muri, alberi, cordoli:
        classi statiche) sono SCARTATI -> non entrano nel layer dinamico.
        Rigoroso: se la conferma camera non e' disponibile, il cluster e' scartato
        (in caso di dubbio, meglio non generare un falso cono su uno statico)."""
        if not clusters:
            return []
        T_cam = self.lookup(self.cam_frame, self.last_seg.header.stamp)
        if T_cam is None:
            return []      # niente TF camera -> non posso confermare -> scarto tutto
        try:
            seg = self.bridge.imgmsg_to_cv2(self.last_seg, desired_encoding='mono8')
        except Exception:
            return []
        h, w = seg.shape
        M = transform_to_matrix(T_cam)
        origin = M[:3, 3]; R = M[:3, :3]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # proietta a terra i pixel di classe DINAMICA
        r0 = int(h * 0.4)
        vs = np.arange(r0, h, 4); us = np.arange(0, w, 4)
        uu, vv = np.meshgrid(us, vs); uu = uu.ravel(); vv = vv.ravel()
        is_dyn = self.dyn_lut[seg[vv, uu]]
        if not is_dyn.any():
            return []      # camera non vede NESSUN dinamico -> scarto tutti i cluster
        uu = uu[is_dyn]; vv = vv[is_dyn]
        dir_opt = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], axis=1)
        dir_w = dir_opt @ R.T
        dz = dir_w[:, 2]
        val = dz < -1e-6
        t = np.full(uu.shape, -1.0); t[val] = -origin[2] / dz[val]
        ok = val & (t > 0)
        gp = origin[None, :] + t[:, None] * dir_w
        cam_pts = gp[ok][:, :2]
        if len(cam_pts) == 0:
            return []

        # tieni SOLO i cluster con un pixel-dinamico-camera vicino (conferma classe)
        kept = []
        for c in clusters:
            cx0, cy0 = c['centroid']
            d = np.hypot(cam_pts[:, 0] - cx0, cam_pts[:, 1] - cy0)
            if np.min(d) <= self.cam_assoc:
                kept.append(c)
        return kept

    # -------------------------------------------------------------------------
    def track(self, dets, now_s):
        """Associa le detection ai track (nearest neighbor), aggiorna posizione
        e velocita' con smoothing, crea/rimuove track."""
        used = set()
        for tr in self.tracks:
            best = None; bd = self.gate_dist
            for j, (dx, dy) in enumerate(dets):
                if j in used:
                    continue
                d = np.hypot(dx - tr['x'], dy - tr['y'])
                if d < bd:
                    bd = d; best = j
            if best is not None:
                dx, dy = dets[best]
                dt = max(1e-2, now_s - tr['t'])
                # smoothing posizione, poi velocita' dalla posizione filtrata
                xs = self.pos_ema * dx + (1 - self.pos_ema) * tr['x']
                ys = self.pos_ema * dy + (1 - self.pos_ema) * tr['y']
                vx_i = (xs - tr['x']) / dt
                vy_i = (ys - tr['y']) / dt
                tr['vx'] = self.vel_ema * vx_i + (1 - self.vel_ema) * tr['vx']
                tr['vy'] = self.vel_ema * vy_i + (1 - self.vel_ema) * tr['vy']
                tr['x'] = xs; tr['y'] = ys; tr['t'] = now_s
                tr['n'] = min(tr['n'] + 1, 999)
                used.add(best)

        for j, (dx, dy) in enumerate(dets):
            if j in used:
                continue
            self.tracks.append(dict(id=self.next_id, x=dx, y=dy,
                                    vx=0.0, vy=0.0, n=1, t=now_s))
            self.next_id += 1

        self.tracks = [tr for tr in self.tracks
                       if now_s - tr['t'] <= self.track_timeout]

    # -------------------------------------------------------------------------
    def build_cost(self, ox, oy):
        """Proietta un cono per ogni track confermato e in movimento."""
        cost = np.zeros((self.n, self.n), dtype=np.uint8)
        self.arrows = []
        spinning = self.robot_ang > self.gate_ang_vel

        for tr in self.tracks:
            speed = np.hypot(tr['vx'], tr['vy'])
            if tr['n'] < self.min_obs:
                continue
            # marca comunque l'oggetto (nucleo) se richiesto
            if self.mark_obstacle:
                self._stamp(cost, tr['x'], tr['y'], ox, oy, self.cone_halfwidth, self.max_cost)
            if spinning or speed < self.min_speed:
                continue
            # cono nella direzione del moto
            self._paint_cone(cost, tr, ox, oy, speed)
            self.arrows.append((tr['x'], tr['y'], tr['vx'], tr['vy'], speed))
        return cost

    def _stamp(self, cost, wx, wy, ox, oy, radius_m, val):
        """Marca un disco di costo attorno a (wx,wy)."""
        cxi = int((wx - ox) / self.res); cyi = int((wy - oy) / self.res)
        rr = int(radius_m / self.res)
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if dx*dx + dy*dy <= rr*rr:
                    gx, gy = cxi + dx, cyi + dy
                    if 0 <= gx < self.n and 0 <= gy < self.n and cost[gy, gx] < val:
                        cost[gy, gx] = val

    def _paint_cone(self, cost, tr, ox, oy, speed):
        """Cono dal centroide nella direzione (vx,vy), lungo speed*horizon,
        che si allarga e decade con la distanza."""
        ux, uy = tr['vx'] / speed, tr['vy'] / speed
        length = min(speed * self.horizon_s, self.size_m)
        n_steps = max(1, int(length / self.res))
        px = (tr['x'] - ox) / self.res
        py = (tr['y'] - oy) / self.res
        half0 = self.cone_halfwidth / self.res
        nx, ny = -uy, ux
        for s in range(n_steps):
            frac = s / max(1, n_steps - 1)
            cx = px + ux * s
            cy = py + uy * s
            half = half0 * (1.0 + self.cone_spread * frac)
            val = int(self.max_cost * (1.0 - 0.5 * frac))
            wv = -half
            while wv <= half + 1e-6:
                gx = int(round(cx + nx * wv))
                gy = int(round(cy + ny * wv))
                if 0 <= gx < self.n and 0 <= gy < self.n and cost[gy, gx] < val:
                    cost[gy, gx] = val
                wv += 1.0

    # -------------------------------------------------------------------------
    def publish_arrows(self):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for k, (wx, wy, vx, vy, speed) in enumerate(self.arrows):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'dyn'; m.id = k
            m.type = Marker.ARROW; m.action = Marker.ADD
            p0 = Point(x=float(wx), y=float(wy), z=0.1)
            p1 = Point(x=float(wx + vx), y=float(wy + vy), z=0.1)  # 1 s di moto
            m.points = [p0, p1]
            m.scale.x = 0.08; m.scale.y = 0.18; m.scale.z = 0.0
            m.color.r = 1.0; m.color.g = 0.9; m.color.b = 0.0; m.color.a = 1.0
            arr.markers.append(m)
        self.markers_pub.publish(arr)

    def publish_cost(self, cost, ox, oy):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.info.resolution = self.res
        msg.info.width = self.n
        msg.info.height = self.n
        msg.info.origin.position.x = float(ox)
        msg.info.origin.position.y = float(oy)
        msg.info.origin.orientation.w = 1.0
        msg.data = cost.astype(np.int8).flatten(order='C').tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = DynamicTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()