#!/usr/bin/env python3
"""
dynamic_vo_camera.py -- Velocity Obstacle anticipatorio basato sulla CAMERA semantica.

Perche' camera e non LiDAR: la camera semantica SA cos'e' un pedone (classe
'person'), mentre il LiDAR planare vede solo punti a una distanza e deve INFERIRE
il movimento (rumoroso, genera falsi coni sugli statici). Con la segmentazione,
i pixel 'person' sono gia' isolati -> niente falsi positivi da muri/alberi.

Pipeline:
  /semantic/segmentation (Image mono8, classi Cityscapes) + /camera/camera_info
    -> seleziona i pixel delle classi DINAMICHE (person/rider/veicoli)
    -> proiezione IPM a terra (z=0) nel frame 'odom'  [stessa matematica del
       semantic_costmap_node, cosi' e' coerente]
    -> connected-components -> centroide-mondo di ogni pedone
    -> tracking (nearest-neighbor + EMA velocita' su finestra) -> vettore velocita'
    -> proiezione CONO VO nella direzione del moto, lungo velocita' x orizzonte
    -> /dynamic_cost (OccupancyGrid, frame odom)  ->  local costmap di Nav2

NOTA REALE: l'input e' un topic di segmentazione. Oggi e' ground-truth da Gazebo;
nel reale sara' l'output di un modello di segmentazione. Il nodo NON cambia: e'
agnostico alla fonte. I filtri (area minima, conferma su piu' frame, EMA) sono
pensati per reggere anche il rumore di un modello reale, non solo il GT pulito.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import tf2_ros


def transform_to_matrix(t):
    """TransformStamped -> matrice 4x4 (identica al semantic_costmap_node)."""
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


class DynamicVOCamera(Node):
    def __init__(self):
        super().__init__('dynamic_vo_camera')

        # ---------------- parametri ----------------
        # I/O e frame (allineati al semantic_costmap_node)
        self.declare_parameter('seg_topic', '/semantic/segmentation')
        self.declare_parameter('info_topic', '/camera/camera_info')
        self.declare_parameter('output_topic', '/dynamic_cost')
        self.declare_parameter('world_frame', 'odom')            # FISSO ma continuo
        self.declare_parameter('camera_optical_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('odom_topic', '/odom')

        # classi dinamiche da TRACCIARE (Cityscapes train id).
        # DEFAULT: solo person=11 e rider=12. I VEICOLI (13-18) sono esclusi di
        # default perche' nello scenario sono spesso FERMI: un veicolo fermo, visto
        # mentre il robot si muove, puo' generare detection/track spuri con velocita'
        # sbagliata (coni fantasma). Aggiungili solo se hai veicoli in movimento.
        self.declare_parameter('dynamic_classes', [11, 12])

        # griglia rolling di uscita
        self.declare_parameter('size_m', 6.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('rate_hz', 10.0)

        # IPM (come semantic_costmap_node)
        self.declare_parameter('pixel_stride', 3)
        self.declare_parameter('max_range', 6.0)
        self.declare_parameter('min_row_frac', 0.45)

        # detection: dimensione minima blob (in celle) per essere un pedone
        self.declare_parameter('min_blob_cells', 4)

        # tracking
        self.declare_parameter('gate_dist', 0.6)      # [m] associazione track<->det
        self.declare_parameter('vel_ema', 0.5)        # smoothing velocita'
        self.declare_parameter('min_obs', 3)          # frame min prima di proiettare
        self.declare_parameter('track_timeout', 0.6)  # [s] eta' max track

        # proiezione cono VO
        self.declare_parameter('min_speed', 0.20)     # [m/s] sotto = fermo
        self.declare_parameter('horizon_s', 2.0)      # [s] quanto proietto nel futuro
        self.declare_parameter('cone_halfwidth_m', 0.35)
        self.declare_parameter('max_cost', 100)
        self.declare_parameter('cost_ema', 0.4)       # stabilita' temporale costo

        # gate sul movimento del robot (rotazioni rapide sporcano la proiezione)
        self.declare_parameter('gate_ang_vel', 0.8)

        gp = self.get_parameter
        self.seg_topic = gp('seg_topic').value
        self.info_topic = gp('info_topic').value
        self.out_topic = gp('output_topic').value
        self.world_frame = gp('world_frame').value
        self.cam_frame = gp('camera_optical_frame').value
        self.base_frame = gp('base_frame').value
        self.odom_topic = gp('odom_topic').value
        self.dyn_classes = list(gp('dynamic_classes').value)
        self.size_m = float(gp('size_m').value)
        self.res = float(gp('resolution').value)
        self.rate_hz = float(gp('rate_hz').value)
        self.stride = int(gp('pixel_stride').value)
        self.max_range = float(gp('max_range').value)
        self.min_row_frac = float(gp('min_row_frac').value)
        self.min_blob_cells = int(gp('min_blob_cells').value)
        self.gate_dist = float(gp('gate_dist').value)
        self.vel_ema = float(gp('vel_ema').value)
        self.min_obs = int(gp('min_obs').value)
        self.track_timeout = float(gp('track_timeout').value)
        self.min_speed = float(gp('min_speed').value)
        self.horizon_s = float(gp('horizon_s').value)
        self.cone_halfwidth_m = float(gp('cone_halfwidth_m').value)
        self.max_cost = int(gp('max_cost').value)
        self.cost_ema = float(gp('cost_ema').value)
        self.gate_ang_vel = float(gp('gate_ang_vel').value)

        self.n = int(round(self.size_m / self.res))

        # LUT classi dinamiche
        self.dyn_lut = np.zeros(256, dtype=bool)
        for c in self.dyn_classes:
            self.dyn_lut[int(c)] = True

        # ---------------- stato ----------------
        self.K = None
        self.last_seg = None
        self.bridge = CvBridge()
        self.tracks = []
        self.next_id = 0
        self.robot_ang = 0.0
        self.cost_prev = None
        self.prev_origin = None

        # ---------------- TF / IO ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS sensor-like: BEST_EFFORT per combaciare con i publisher di immagini
        # (la segmentazione e la camera_info sono tipicamente BEST_EFFORT).
        qos_sensor = QoSProfile(depth=1,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(CameraInfo, self.info_topic, self.info_cb, qos_sensor)
        self.create_subscription(Image, self.seg_topic, self.seg_cb, qos_sensor)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos_sensor)
        self.pub = self.create_publisher(OccupancyGrid, self.out_topic, 1)
        # frecce di debug: direzione stimata del moto di ogni pedone tracciato
        self.markers_pub = self.create_publisher(MarkerArray, '/dynamic_tracks', 1)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.update)

        self.get_logger().info(
            f'dynamic_vo_camera avviato | seg={self.seg_topic} -> {self.out_topic} | '
            f'griglia {self.n}x{self.n}@{self.res}m in {self.world_frame} | '
            f'classi dinamiche={self.dyn_classes}')

    # -------------------------------------------------------------------------
    def info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)

    def seg_cb(self, msg: Image):
        self.last_seg = msg

    def odom_cb(self, msg: Odometry):
        self.robot_ang = abs(float(msg.twist.twist.angular.z))

    def lookup(self, frame, stamp):
        try:
            return self.tf_buffer.lookup_transform(
                self.world_frame, frame, stamp, timeout=Duration(seconds=0.1))
        except Exception:
            try:
                return self.tf_buffer.lookup_transform(
                    self.world_frame, frame, rclpy.time.Time())
            except Exception:
                return None

    # -------------------------------------------------------------------------
    def update(self):
        if self.K is None or self.last_seg is None:
            return
        msg = self.last_seg
        now_s = self.get_clock().now().nanoseconds * 1e-9

        seg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        h, w = seg.shape

        T_cam = self.lookup(self.cam_frame, msg.header.stamp)
        T_base = self.lookup(self.base_frame, msg.header.stamp)
        if T_cam is None or T_base is None:
            return
        M = transform_to_matrix(T_cam)          # camera_optical -> odom
        origin = M[:3, 3]
        R = M[:3, :3]

        bx = T_base.transform.translation.x
        by = T_base.transform.translation.y
        # origine finestra rolling, quantizzata (celle-mondo coerenti tra frame)
        ox = np.floor((bx - self.size_m / 2.0) / self.res) * self.res
        oy = np.floor((by - self.size_m / 2.0) / self.res) * self.res

        # --- campiona i pixel (solo parte bassa = suolo) ---
        r0 = int(h * self.min_row_frac)
        vs = np.arange(r0, h, self.stride)
        us = np.arange(0, w, self.stride)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.ravel(); vv = vv.ravel()

        # tieni solo i pixel di classe DINAMICA
        classes = seg[vv, uu]
        is_dyn = self.dyn_lut[classes]
        if not is_dyn.any():
            # nessun pedone visto: aggiorna comunque i track (timeout) e pubblica
            cost = self.build_cost([], ox, oy, now_s)
            self.publish_cost(cost, ox, oy)
            return
        uu = uu[is_dyn]; vv = vv[is_dyn]

        # --- IPM: proietta i pixel dinamici a terra (z=0), come semantic_costmap ---
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        dir_opt = np.stack([(uu - cx) / fx, (vv - cy) / fy,
                            np.ones_like(uu, dtype=float)], axis=1)
        dir_world = dir_opt @ R.T
        dz = dir_world[:, 2]
        valid = dz < -1e-6
        t = np.full(uu.shape, -1.0)
        t[valid] = -origin[2] / dz[valid]
        ok = valid & (t > 0)
        pts = origin[None, :] + t[:, None] * dir_world
        X, Y = pts[:, 0], pts[:, 1]
        dist = np.hypot(X - bx, Y - by)
        ok = ok & (dist < self.max_range)

        # --- rasterizza i punti dinamici in una griglia e trova i blob ---
        ci = ((X - ox) / self.res).astype(int)
        cj = ((Y - oy) / self.res).astype(int)
        inside = ok & (ci >= 0) & (ci < self.n) & (cj >= 0) & (cj < self.n)
        grid = np.zeros((self.n, self.n), dtype=np.uint8)
        grid[cj[inside], ci[inside]] = 1     # righe=y(j), colonne=x(i)

        dets = self.detect_blobs(grid, ox, oy)
        cost = self.build_cost(dets, ox, oy, now_s)
        self.publish_cost(cost, ox, oy)

    # -------------------------------------------------------------------------
    def detect_blobs(self, grid, ox, oy):
        """Connected-components sui pixel dinamici proiettati -> centroidi mondo."""
        dets = []
        if not grid.any():
            return dets
        m = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n_lbl, lbl, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n_lbl):
            if stats[i, cv2.CC_STAT_AREA] < self.min_blob_cells:
                continue
            cx_cell, cy_cell = cents[i]
            wx = ox + cx_cell * self.res
            wy = oy + cy_cell * self.res
            dets.append((wx, wy))
        return dets

    # -------------------------------------------------------------------------
    def build_cost(self, dets, ox, oy, now_s):
        """Tracking + proiezione cono VO -> griglia di costo."""
        cost = np.zeros((self.n, self.n), dtype=np.uint8)

        # gate sul robot che ruota veloce (la proiezione IPM diventa inaffidabile)
        robot_spinning = self.robot_ang > self.gate_ang_vel

        # associazione det <-> track (nearest neighbor)
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
                vx_i = (dx - tr['x']) / dt
                vy_i = (dy - tr['y']) / dt
                tr['vx'] = self.vel_ema * vx_i + (1 - self.vel_ema) * tr['vx']
                tr['vy'] = self.vel_ema * vy_i + (1 - self.vel_ema) * tr['vy']
                tr['x'] = dx; tr['y'] = dy; tr['t'] = now_s
                tr['n'] = min(tr['n'] + 1, 999)
                used.add(best)

        # nuove detection -> nuovi track
        for j, (dx, dy) in enumerate(dets):
            if j in used:
                continue
            self.tracks.append(dict(id=self.next_id, x=dx, y=dy,
                                    vx=0.0, vy=0.0, n=1, t=now_s))
            self.next_id += 1

        # scarta track vecchi
        self.tracks = [tr for tr in self.tracks
                       if now_s - tr['t'] <= self.track_timeout]

        # proietta il cono VO per i track confermati e in movimento
        if not robot_spinning:
            for tr in self.tracks:
                speed = np.hypot(tr['vx'], tr['vy'])
                if tr['n'] < self.min_obs or speed < self.min_speed:
                    continue
                self.paint_cone(cost, tr, ox, oy, speed)

        # pubblica le frecce di debug (direzione del moto di ogni pedone)
        self.publish_markers()

        # diagnostica: cosa vede/traccia
        if self.tracks:
            info = [f"id{tr['id']}:pos({tr['x']:.1f},{tr['y']:.1f}) "
                    f"v({tr['vx']:.2f},{tr['vy']:.2f}) |v|={np.hypot(tr['vx'],tr['vy']):.2f} n{tr['n']}"
                    for tr in self.tracks]
            self.get_logger().info(f"[VO] {len(dets)} det | tracks: {' | '.join(info)}",
                                   throttle_duration_sec=1.0)

        # smoothing temporale del costo (stabilita')
        cf = cost.astype(np.float32)
        if self.cost_ema > 0.0 and self.cost_prev is not None and self.prev_origin is not None:
            pox, poy = self.prev_origin
            sx = int(round((pox - ox) / self.res))
            sy = int(round((poy - oy) / self.res))
            cprev_al = self.shift_grid(self.cost_prev, sx, sy, fill=0.0)
            cf = self.cost_ema * cf + (1.0 - self.cost_ema) * cprev_al
        self.cost_prev = cf.copy()
        self.prev_origin = (ox, oy)
        return np.clip(cf, 0, self.max_cost).astype(np.uint8)

    def publish_markers(self):
        """Frecce in RViz: per ogni pedone tracciato, una freccia dalla sua
        posizione nella direzione del moto, lunga proporzionalmente alla velocita'.
        Strumento di debug: vedi a occhio se il tracking stima la direzione giusta."""
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for tr in self.tracks:
            speed = np.hypot(tr['vx'], tr['vy'])
            if tr['n'] < self.min_obs or speed < self.min_speed:
                continue
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'dynamic_tracks'
            m.id = int(tr['id'])
            m.type = Marker.ARROW
            m.action = Marker.ADD
            # freccia lunga 1 secondo di moto (dove sara' il pedone tra 1s)
            p0 = Point(x=float(tr['x']), y=float(tr['y']), z=0.1)
            p1 = Point(x=float(tr['x'] + tr['vx']), y=float(tr['y'] + tr['vy']), z=0.1)
            m.points = [p0, p1]
            m.scale.x = 0.08   # diametro fusto
            m.scale.y = 0.16   # diametro punta
            m.scale.z = 0.0
            m.color.r = 1.0; m.color.g = 0.9; m.color.b = 0.0; m.color.a = 1.0
            arr.markers.append(m)
        self.markers_pub.publish(arr)

    # -------------------------------------------------------------------------
    def paint_cone(self, cost, tr, ox, oy, speed):
        """Cono di costo dal pedone nella direzione del moto, lungo speed*horizon,
        che si allarga e decade con la distanza (piu' lontano = piu' incerto)."""
        ux, uy = tr['vx'] / speed, tr['vy'] / speed
        length = min(speed * self.horizon_s, self.size_m)
        n_steps = max(1, int(length / self.res))
        px = (tr['x'] - ox) / self.res
        py = (tr['y'] - oy) / self.res
        half0 = self.cone_halfwidth_m / self.res
        nx, ny = -uy, ux
        for s in range(n_steps):
            frac = s / max(1, n_steps - 1)
            cx = px + ux * s
            cy = py + uy * s
            half = half0 * (1.0 + 1.5 * frac)
            val = int(self.max_cost * (1.0 - 0.6 * frac))
            for wv in np.arange(-half, half + 1e-6, 1.0):
                gx = int(round(cx + nx * wv))
                gy = int(round(cy + ny * wv))
                if 0 <= gx < self.n and 0 <= gy < self.n:
                    if cost[gy, gx] < val:
                        cost[gy, gx] = val

    # -------------------------------------------------------------------------
    @staticmethod
    def shift_grid(grid, sx, sy, fill=0.0):
        out = np.full_like(grid, fill)
        n = grid.shape[0]
        def span(s):
            if s >= 0:
                return slice(s, n), slice(0, n - s)
            else:
                return slice(0, n + s), slice(-s, n)
        dy, syy = span(sy)
        dx, sxx = span(sx)
        out[dy, dx] = grid[syy, sxx]
        return out

    # -------------------------------------------------------------------------
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
    node = DynamicVOCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()