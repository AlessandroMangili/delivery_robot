#!/usr/bin/env python3
"""
dynamic_esdf_layer.py -- Rilevamento e anticipazione di ostacoli dinamici (SOLO LiDAR).

  LiDAR -> POSIZIONE, VELOCITA', 360 gradi: clustering (unisce le due gambe in UN
           oggetto), tracking del centroide, stima velocita', proiezione del cono
           di costo anticipatorio.
  Discriminazione statico/dinamico: PER MOVIMENTO (velocita' sopra 'min_speed').
  Un cluster fermo (muro, aiuola) non genera ne' nucleo ne' cono.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import tf2_ros


def transform_to_matrix(t):
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

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('output_topic', '/dynamic_cost')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('size_m', 8.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('cluster_eps_m', 0.5)
        self.declare_parameter('cluster_min_pts', 2)
        self.declare_parameter('cluster_max_size_m', 1.5)
        self.declare_parameter('gate_dist_m', 0.7)
        self.declare_parameter('vel_ema', 0.4)
        self.declare_parameter('pos_ema', 0.5)
        self.declare_parameter('min_obs', 3)
        self.declare_parameter('track_timeout_s', 0.6)
        self.declare_parameter('min_speed', 0.30)
        self.declare_parameter('horizon_s', 3.0)
        self.declare_parameter('cone_min_length', 4.0)   # [m] lunghezza minima anche a bassa velocita'
        self.declare_parameter('cone_halfwidth_m', 0.4)
        self.declare_parameter('cone_spread', 1.5)
        self.declare_parameter('max_cost', 100)
        self.declare_parameter('nucleus_radius_m', 0.20)  # raggio NUCLEO (corpo pedone), piccolo: solo il pedone e' letale
        self.declare_parameter('cone_base_cost', 99)      # 99 -> costmap 251: quasi letale MA scavalcabile. 100 = letale/intrappola
        self.declare_parameter('cone_tip_cost', 90)       # costo alla PUNTA: alto = reagisce in anticipo
        self.declare_parameter('cone_lateral_falloff', 0.35)  # 0..1: quanto cala il costo dal centro ai bordi (per USCIRE)
        self.declare_parameter('mark_obstacle', True)
        self.declare_parameter('gate_ang_vel', 1.0)

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
        self.cone_min_length = float(gp('cone_min_length').value)
        self.cone_halfwidth = float(gp('cone_halfwidth_m').value)
        self.cone_spread = float(gp('cone_spread').value)
        self.max_cost = int(gp('max_cost').value)
        self.nucleus_radius = float(gp('nucleus_radius_m').value)
        self.cone_base_cost = int(gp('cone_base_cost').value)
        self.cone_tip_cost = int(gp('cone_tip_cost').value)
        self.cone_lateral_falloff = float(gp('cone_lateral_falloff').value)
        self.mark_obstacle = bool(gp('mark_obstacle').value)
        self.gate_ang_vel = float(gp('gate_ang_vel').value)

        self.n = int(round(self.size_m / self.res))

        self.last_scan = None
        self.tracks = []
        self.next_id = 0
        self.robot_ang = 0.0
        self.arrows = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos)

        # TRANSIENT_LOCAL: il layer della costmap chiede durability transient_local;
        # un publisher volatile e' incompatibile e ROS2 blocca il flusso.
        # transient_local qui e' compatibile con qualsiasi subscriber.
        qos_map = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(OccupancyGrid, self.out_topic, qos_map)
        self.markers_pub = self.create_publisher(MarkerArray, '/dynamic_tracks', 1)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.update)

        self.get_logger().info(
            f'dynamic_esdf_layer avviato (solo LiDAR) | '
            f'griglia {self.n}x{self.n}@{self.res}m frame={self.world_frame}')

    def scan_cb(self, msg): self.last_scan = msg
    def odom_cb(self, msg): self.robot_ang = abs(float(msg.twist.twist.angular.z))

    def lookup(self, frame, stamp=None):
        # stamp=None -> ultima TF disponibile (per origine griglia).
        # stamp=header.stamp -> TF all'ISTANTE dell'osservazione: indispensabile
        # per lo scan, altrimenti in rotazione i punti fermi atterrano ruotati
        # frame dopo frame e sembrano muoversi (velocita' spuria -> falso dinamico).
        try:
            t = rclpy.time.Time() if stamp is None else rclpy.time.Time.from_msg(stamp)
            return self.tf_buffer.lookup_transform(self.world_frame, frame, t)
        except Exception:
            return None

    def update(self):
        if self.last_scan is None:
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9
        T_base = self.lookup(self.robot_frame)
        if T_base is None:
            return
        rx = T_base.transform.translation.x
        ry = T_base.transform.translation.y
        ox = np.floor((rx - self.size_m / 2.0) / self.res) * self.res
        oy = np.floor((ry - self.size_m / 2.0) / self.res) * self.res

        pts = self.scan_points_world(self.last_scan, rx, ry)
        if pts is None:
            return
        clusters = self.cluster_points(pts)
        dets = [c['centroid'] for c in clusters]
        self.track(dets, now_s)

        cost = self.build_cost(ox, oy)
        self.publish_cost(cost, ox, oy)
        self.publish_arrows()

    def scan_points_world(self, scan, rx, ry):
        # TF al timestamp dello scan (non l'ultima): allinea i punti al momento
        # in cui il LiDAR li ha misurati. Se non e' ancora nel buffer, T=None e
        # si salta questo frame (il tracker gira a 10Hz, saltarne uno e' innocuo).
        T = self.lookup(scan.header.frame_id, stamp=scan.header.stamp)
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
        px = r * np.cos(ang); py = r * np.sin(ang)
        wx = sx + px * np.cos(yaw) - py * np.sin(yaw)
        wy = sy + px * np.sin(yaw) + py * np.cos(yaw)
        d = np.hypot(wx - rx, wy - ry)
        keep = d <= self.max_range
        return np.stack([wx[keep], wy[keep]], axis=1)

    def cluster_points(self, pts):
        clusters = []
        if len(pts) == 0:
            return clusters
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
                        if np.sum((pts[i] - pts[j]) ** 2) <= eps2:
                            union(i, j)
        groups = {}
        for i in range(len(pts)):
            groups.setdefault(find(i), []).append(i)
        for idxs in groups.values():
            if len(idxs) < self.cluster_min_pts:
                continue
            cpts = pts[idxs]
            size = np.max(np.ptp(cpts, axis=0)) if len(cpts) > 1 else 0.0
            if size > self.cluster_max_size:
                continue
            centroid = cpts.mean(axis=0)
            clusters.append({'centroid': (float(centroid[0]), float(centroid[1])),
                             'size': float(size)})
        return clusters

    def track(self, dets, now_s):
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
            self.tracks.append(dict(id=self.next_id, x=dx, y=dy, vx=0.0, vy=0.0,
                                    n=1, t=now_s))
            self.next_id += 1
        self.tracks = [tr for tr in self.tracks
                       if now_s - tr['t'] <= self.track_timeout]

    def build_cost(self, ox, oy):
        cost = np.zeros((self.n, self.n), dtype=np.uint8)
        self.arrows = []
        spinning = self.robot_ang > self.gate_ang_vel
        for tr in self.tracks:
            if tr['n'] < self.min_obs:
                continue
            speed = np.hypot(tr['vx'], tr['vy'])
            # DISCRIMINAZIONE PER MOVIMENTO: un track fermo (statico) non produce
            # nulla, ne' nucleo ne' cono. Il gate velocita' viene PRIMA del timbro
            # del nucleo -> muri e aiuole (velocita' ~ 0) non vengono marcati.
            if speed < self.min_speed:
                continue
            if self.mark_obstacle:
                self._stamp(cost, tr['x'], tr['y'], ox, oy, self.nucleus_radius, self.max_cost)
            # in rotazione rapida la DIREZIONE della velocita' e' inaffidabile:
            # marca il nucleo (posizione attuale, valida) ma non proiettare il cono.
            if spinning:
                continue
            self._paint_cone(cost, tr, ox, oy, speed)
            self.arrows.append((tr['x'], tr['y'], tr['vx'], tr['vy']))
        return cost

    def _stamp(self, cost, wx, wy, ox, oy, radius_m, val):
        cxi = int((wx - ox) / self.res); cyi = int((wy - oy) / self.res)
        rr = int(radius_m / self.res)
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if dx*dx + dy*dy <= rr*rr:
                    gx, gy = cxi + dx, cyi + dy
                    if 0 <= gx < self.n and 0 <= gy < self.n and cost[gy, gx] < val:
                        cost[gy, gx] = val

    def _paint_cone(self, cost, tr, ox, oy, speed):
        ux, uy = tr['vx'] / speed, tr['vy'] / speed
        # lunghezza minima: anche un dinamico lento lascia un cono usabile,
        # abbastanza lungo da far reagire il controller PRIMA del nucleo.
        length = min(max(self.cone_min_length, speed * self.horizon_s), self.size_m)
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
            # costo longitudinale: base (sul pedone) -> punta. Tenuto <100 apposta:
            # nella costmap resta SOTTO il letale, quindi il controller lo evita
            # fortissimo ma se ci finisce dentro trova ancora traiettorie valide per uscire.
            base_val = self.cone_base_cost - (self.cone_base_cost - self.cone_tip_cost) * frac
            wv = -half
            while wv <= half + 1e-6:
                gx = int(round(cx + nx * wv))
                gy = int(round(cy + ny * wv))
                # gradiente LATERALE: massimo al centro, cala verso i bordi -> chi e'
                # dentro il cono ha un verso "in discesa" per uscire di lato al piu' presto.
                lat = abs(wv) / half if half > 1e-6 else 0.0
                val = int(base_val * (1.0 - self.cone_lateral_falloff * lat))
                if 0 <= gx < self.n and 0 <= gy < self.n and cost[gy, gx] < val:
                    cost[gy, gx] = val
                wv += 1.0

    def publish_arrows(self):
        arr = MarkerArray()
        clear = Marker(); clear.header.frame_id = self.world_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for k, (wx, wy, vx, vy) in enumerate(self.arrows):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'dyn'; m.id = k
            m.type = Marker.ARROW; m.action = Marker.ADD
            p0 = Point(x=float(wx), y=float(wy), z=0.1)
            p1 = Point(x=float(wx + vx), y=float(wy + vy), z=0.1)
            m.points = [p0, p1]
            m.scale.x = 0.08; m.scale.y = 0.18; m.scale.z = 0.0
            m.color.r = 0.1; m.color.g = 1.0; m.color.b = 0.1; m.color.a = 1.0
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