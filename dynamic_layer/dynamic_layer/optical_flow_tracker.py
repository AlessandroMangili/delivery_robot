#!/usr/bin/env python3
"""
optical_flow_tracker.py -- Stima del moto degli ostacoli SENZA tracking a
                           oggetti, via flusso ottico sulla griglia di
                           occupazione da LiDAR 2D.

BASELINE a CAMPO, alternativa al tracker a oggetti (dynamic_tracker.py).

IDEA
----
Non si identificano oggetti, non si mantengono tracce, non si associano
detection tra frame. Si costruisce una griglia di occupazione da ogni scansione
LiDAR, la si tratta come un'IMMAGINE, e si calcola il FLUSSO OTTICO (Farneback,
denso) tra il frame di adesso e quello di prima. Il flusso da', per ogni cella
occupata, il vettore di quanto il suo contenuto si e' spostato -> velocita' e
direzione per cella, direttamente in m/s dopo la conversione di scala.

DIFFERENZA RISPETTO ALLA DIFFERENZA-ESDF (paper 2D-EDSDF)
--------------------------------------------------------
La differenza-ESDF dice SE una cella si e' mossa (scalare con segno). Il flusso
ottico dice DI QUANTO e IN CHE DIREZIONE (vettore). E' il gradino che serve per
avere la velocita', non solo il rilevamento del movimento.

PERCHE' FUNZIONA CON LIDAR 2D
-----------------------------
Non serve densita': serve solo che l'ostacolo lasci una traccia coerente sulla
griglia tra due frame. Un muro fermo ha flusso nullo (non c'e' centroide che
slitta: non c'e' centroide affatto). Una persona che cammina trascina un
gruppo di celle occupate, e il flusso le segue.

COMPATIBILITA'
--------------
Pubblica TrackArray su /dynamic_tracks_state, ESATTAMENTE come dynamic_tracker.
Il critic spazio-temporale C++ e il cono di costo non cambiano: si sostituisce
un nodo con l'altro e si confrontano (A/B per la tesi). Pubblica anche il cono
di costo su /dynamic_cost, cosi' e' un rimpiazzo completo.

LIMITE ONESTO
-------------
La velocita' per cella e' piu' RUMOROSA di quella di un Kalman su traccia,
perche' guarda due frame invece di integrare nel tempo. Un filtro EMA sui blob
recupera parte della stabilita', ma resta un metodo a bassa latenza e alto
rumore rispetto al tracker a oggetti. E' il compromesso caratteristico dei
metodi a campo.

RIFERIMENTI
-----------
- G. Farneback, "Two-Frame Motion Estimation Based on Polynomial Expansion",
  SCIA 2003 -- l'algoritmo di flusso ottico denso usato qui (cv2.calcOpticalFlowFarneback).
- C. Coue, C. Pradalier, C. Laugier et al., "Bayesian Occupancy Filtering for
  Multitarget Tracking: An Automotive Application", IJRR 2006 -- il metodo a
  campo di riferimento: occupazione + velocita' per cella, filtrate nel tempo.
  Il flusso ottico ne e' la versione a due frame, non filtrata.
- R. Danescu, F. Oniga, S. Nedevschi, "Modeling and Tracking the Driving
  Environment with a Particle-Based Occupancy Grid", IEEE T-ITS 2011.
- H. Zhang et al., 2D-EDSDF (il paper allegato): rilevamento del moto per cella
  via differenza di campo; qui se ne estende l'idea dal rilevamento alla stima.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from dynamic_tracker_msgs.msg import Track, TrackArray

import tf2_ros


def transform_to_matrix(t):
    q = t.transform.rotation
    tr = t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


class OpticalFlowTracker(Node):

    def __init__(self):
        super().__init__('optical_flow_tracker')

        # ---------------- I/O ----------------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('output_topic', '/dynamic_cost')
        self.declare_parameter('tracks_topic', '/dynamic_tracks_state')
        self.declare_parameter('markers_topic', '/dynamic_tracks')
        self.declare_parameter('flow_markers_topic', '/optical_flow/vectors')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('robot_frame', 'base_link')

        # ---------------- griglia ----------------
        self.declare_parameter('size_m', 8.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('rate_hz', 10.0)

        # ---------------- flusso ottico (Farneback) ----------------
        self.declare_parameter('flow_pyr_scale', 0.5)
        self.declare_parameter('flow_levels', 3)
        self.declare_parameter('flow_winsize', 15)
        self.declare_parameter('flow_iterations', 3)
        self.declare_parameter('flow_poly_n', 5)
        self.declare_parameter('flow_poly_sigma', 1.2)
        # una cella e' "in moto" se il suo flusso supera questa velocita'
        self.declare_parameter('min_speed', 0.30)          # m/s
        # blob: celle in moto vicine tra loro = stesso ostacolo (solo per il cono)
        self.declare_parameter('blob_eps_m', 0.4)
        self.declare_parameter('blob_min_cells', 4)
        # lisciamento temporale della velocita' dei blob (recupera stabilita')
        self.declare_parameter('vel_ema', 0.4)
        # conferma temporale: un blob genera cono solo dopo N frame di moto
        # coerente. Filtra il flusso apparente dei muri ai bordi del FOV.
        self.declare_parameter('promote_frames', 3)

        # ---------------- cono di costo (identico al tracker a oggetti) ----------------
        self.declare_parameter('enable_cone', True)
        self.declare_parameter('mark_obstacle', True)
        self.declare_parameter('horizon_s', 3.0)
        self.declare_parameter('cone_min_length', 4.0)
        self.declare_parameter('cone_halfwidth_m', 0.4)
        self.declare_parameter('cone_spread', 1.5)
        self.declare_parameter('max_cost', 100)
        self.declare_parameter('nucleus_radius_m', 0.20)
        self.declare_parameter('cone_base_cost', 99)
        self.declare_parameter('cone_tip_cost', 90)
        self.declare_parameter('cone_lateral_falloff', 0.35)
        self.declare_parameter('cone_sector_deg', 180.0)
        self.declare_parameter('relevance_horizon_s', 10.0)
        self.declare_parameter('relevance_radius_m', 2.0)
        self.declare_parameter('relevance_always_dist_m', 0.8)

        gp = self.get_parameter
        self.scan_topic = gp('scan_topic').value
        self.odom_topic = gp('odom_topic').value
        self.world_frame = gp('world_frame').value
        self.robot_frame = gp('robot_frame').value

        self.size_m = float(gp('size_m').value)
        self.res = float(gp('resolution').value)
        self.n = int(self.size_m / self.res)
        self.max_range = float(gp('max_range').value)
        self.rate_hz = float(gp('rate_hz').value)

        self.flow_params = dict(
            pyr_scale=float(gp('flow_pyr_scale').value),
            levels=int(gp('flow_levels').value),
            winsize=int(gp('flow_winsize').value),
            iterations=int(gp('flow_iterations').value),
            poly_n=int(gp('flow_poly_n').value),
            poly_sigma=float(gp('flow_poly_sigma').value),
            flags=0)
        self.min_speed = float(gp('min_speed').value)
        self.blob_eps = float(gp('blob_eps_m').value)
        self.blob_min_cells = int(gp('blob_min_cells').value)
        self.vel_ema = float(gp('vel_ema').value)
        self.promote_frames = int(gp('promote_frames').value)

        self.enable_cone = bool(gp('enable_cone').value)
        self.mark_obstacle = bool(gp('mark_obstacle').value)
        self.horizon_s = float(gp('horizon_s').value)
        self.cone_min_length = float(gp('cone_min_length').value)
        self.cone_halfwidth = float(gp('cone_halfwidth_m').value)
        self.cone_spread = float(gp('cone_spread').value)
        self.max_cost = int(gp('max_cost').value)
        self.nucleus_radius = float(gp('nucleus_radius_m').value)
        self.cone_base_cost = int(gp('cone_base_cost').value)
        self.cone_tip_cost = int(gp('cone_tip_cost').value)
        self.cone_lateral_falloff = float(gp('cone_lateral_falloff').value)
        self.cone_sector = float(gp('cone_sector_deg').value)
        self.relevance_horizon = float(gp('relevance_horizon_s').value)
        self.relevance_radius = float(gp('relevance_radius_m').value)
        self.relevance_always = float(gp('relevance_always_dist_m').value)

        # ---------------- stato ----------------
        self.last_scan = None
        self.prev_grid = None          # griglia di occupazione del frame precedente
        self.prev_origin = None        # (ox, oy) del frame precedente
        self.prev_t = None
        self.robot_xy = (0.0, 0.0)
        self.robot_yaw = 0.0
        self.robot_v_world = (0.0, 0.0)
        self.robot_v_body = (0.0, 0.0)
        self.blobs = []                # blob del frame precedente, per l'EMA
        self.arrows = []
        self.diag = dict(n_occ=0, n_moving=0, n_blobs=0)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos)

        qos_grid = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              history=HistoryPolicy.KEEP_LAST)
        self.cost_pub = self.create_publisher(OccupancyGrid, gp('output_topic').value, qos_grid)
        self.tracks_pub = self.create_publisher(TrackArray, gp('tracks_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, gp('markers_topic').value, 10)
        self.flow_pub = self.create_publisher(MarkerArray, gp('flow_markers_topic').value, 10)

        self.create_timer(1.0 / self.rate_hz, self.update)
        self.create_timer(5.0, self.log_diag)

        # cv2 e' obbligatorio per il flusso denso
        try:
            import cv2  # noqa
            self._have_cv2 = True
        except Exception:
            self._have_cv2 = False
            self.get_logger().error(
                'OpenCV (cv2) non disponibile: il flusso ottico denso lo richiede. '
                'Installa python3-opencv.')

        self.get_logger().info(
            'Optical-flow tracker (baseline a campo) avviato. '
            'Pubblica gli stessi TrackArray del tracker a oggetti: il critic non cambia.')

    def scan_cb(self, msg):
        self.last_scan = msg

    def odom_cb(self, msg):
        self.robot_v_body = (float(msg.twist.twist.linear.x),
                             float(msg.twist.twist.linear.y))

    def lookup(self, target, source, stamp=None):
        try:
            t = rclpy.time.Time() if stamp is None else rclpy.time.Time.from_msg(stamp)
            return self.tf_buffer.lookup_transform(target, source, t)
        except Exception:
            if stamp is not None:
                try:
                    return self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
                except Exception:
                    return None
            return None

    # ---------------------------------------------------- costruzione griglia
    def build_occupancy(self, scan, ox, oy):
        """Griglia binaria di occupazione dallo scan, in coordinate mondo,
        con origine (ox, oy). Ogni raggio valido accende la cella del suo
        punto finale."""
        grid = np.zeros((self.n, self.n), dtype=np.uint8)
        ang = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        r = np.asarray(scan.ranges, dtype=np.float32)
        good = np.isfinite(r) & (r >= scan.range_min) & (r <= scan.range_max) \
            & (r <= self.max_range)
        ang, r = ang[good], r[good]
        if r.size == 0:
            return grid
        xl = r * np.cos(ang)
        yl = r * np.sin(ang)
        # laser -> mondo
        Tw = self.lookup(self.world_frame, scan.header.frame_id, scan.header.stamp)
        if Tw is None:
            return grid
        M = transform_to_matrix(Tw)
        P = np.stack([xl, yl, np.zeros_like(xl), np.ones_like(xl)], axis=1)
        Pw = (M @ P.T).T[:, :2]
        gx = ((Pw[:, 0] - ox) / self.res).astype(int)
        gy = ((Pw[:, 1] - oy) / self.res).astype(int)
        inside = (gx >= 0) & (gx < self.n) & (gy >= 0) & (gy < self.n)
        grid[gy[inside], gx[inside]] = 255
        return grid

    # ---------------------------------------------------- flusso ottico
    def compute_flow(self, prev, cur):
        """Flusso ottico denso di Farneback tra due griglie. Ritorna un campo
        HxWx2 in celle/frame (dx, dy). Un leggero blur rende le celle sparse del
        LiDAR piu' 'dense' per il flusso, senza spostarne il baricentro."""
        import cv2
        p = cv2.GaussianBlur(prev, (5, 5), 0)
        c = cv2.GaussianBlur(cur, (5, 5), 0)
        flow = cv2.calcOpticalFlowFarneback(p, c, None, **self.flow_params)
        return flow

    # ---------------------------------------------------- blob delle celle in moto
    def cluster_moving(self, moving_cells, vel_cells):
        """Raggruppa le celle in moto in blob (union-find su griglia). Per ogni
        blob restituisce centroide-mondo e velocita' media, in m/s.
        Serve SOLO a disegnare il cono e a pubblicare i Track: la stima per
        cella resta disponibile a monte."""
        if len(moving_cells) == 0:
            return []
        eps = max(1, int(self.blob_eps / self.res))
        cellset = {(int(x), int(y)): i for i, (x, y) in enumerate(moving_cells)}
        parent = list(range(len(moving_cells)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (x, y), i in cellset.items():
            for dx in range(-eps, eps + 1):
                for dy in range(-eps, eps + 1):
                    j = cellset.get((x + dx, y + dy))
                    if j is not None and j > i:
                        union(i, j)

        groups = {}
        for i in range(len(moving_cells)):
            groups.setdefault(find(i), []).append(i)

        blobs = []
        for idxs in groups.values():
            if len(idxs) < self.blob_min_cells:
                continue
            cells = np.array([moving_cells[i] for i in idxs], dtype=float)
            vels = np.array([vel_cells[i] for i in idxs], dtype=float)
            cx = float(cells[:, 0].mean())
            cy = float(cells[:, 1].mean())
            vx = float(np.median(vels[:, 0]))     # mediana: robusta agli outlier del flusso
            vy = float(np.median(vels[:, 1]))
            blobs.append(dict(gx=cx, gy=cy, vx=vx, vy=vy, n=len(idxs)))
        return blobs

    def match_and_smooth(self, blobs, ox, oy):
        """Associa i blob a quelli del frame prima (nearest) e liscia la
        velocita' con una EMA. Aggiunge una CONFERMA TEMPORALE: un blob genera
        cono solo dopo essersi mosso in modo coerente per promote_frames frame
        consecutivi. Un muro, che ai bordi del FOV da' flusso incoerente frame
        per frame, non accumula conferme e non genera mai cono. E' l'isteresi
        del tracker a oggetti, applicata ai blob del campo."""
        for b in blobs:
            b['wx'] = ox + b['gx'] * self.res
            b['wy'] = oy + b['gy'] * self.res
            b['mov_count'] = 0
            best, bd = None, self.blob_eps * 3.0
            for pb in self.blobs:
                d = np.hypot(b['wx'] - pb['wx'], b['wy'] - pb['wy'])
                if d < bd:
                    best, bd = pb, d
            if best is not None:
                a = self.vel_ema
                b['vx'] = a * b['vx'] + (1 - a) * best['vx']
                b['vy'] = a * b['vy'] + (1 - a) * best['vy']
                # coerenza: la velocita' di adesso punta come quella di prima?
                sp = np.hypot(b['vx'], b['vy'])
                psp = np.hypot(best['vx'], best['vy'])
                if sp > self.min_speed and psp > 1e-6:
                    cosang = (b['vx'] * best['vx'] + b['vy'] * best['vy']) / (sp * psp)
                    if cosang > 0.5:              # entro ~60 gradi: moto coerente
                        b['mov_count'] = best.get('mov_count', 0) + 1
        self.blobs = blobs
        return blobs

    # ---------------------------------------------------- ciclo
    def update(self):
        if self.last_scan is None or not self._have_cv2:
            return
        scan = self.last_scan
        now_s = self.get_clock().now().nanoseconds * 1e-9

        Tr = self.lookup(self.world_frame, self.robot_frame)
        if Tr is None:
            return
        M = transform_to_matrix(Tr)
        rx, ry = M[0, 3], M[1, 3]
        self.robot_xy = (rx, ry)
        self.robot_yaw = np.arctan2(M[1, 0], M[0, 0])
        bx, by = self.robot_v_body
        self.robot_v_world = (bx * np.cos(self.robot_yaw) - by * np.sin(self.robot_yaw),
                              bx * np.sin(self.robot_yaw) + by * np.cos(self.robot_yaw))
        # origine ANCORATA alla griglia: cosi' tra due frame lo spostamento
        # dell'origine e' un multiplo INTERO della cella, e np.roll compensa il
        # moto del robot in modo ESATTO. Senza questo, il residuo sub-cella
        # diventa flusso apparente su TUTTO -- muri compresi -> falsi coni.
        ox = np.floor((rx - self.size_m / 2.0) / self.res) * self.res
        oy = np.floor((ry - self.size_m / 2.0) / self.res) * self.res

        grid = self.build_occupancy(scan, ox, oy)
        self.diag['n_occ'] = int((grid > 0).sum())

        blobs = []
        if self.prev_grid is not None and self.prev_origin is not None:
            # riallineo la griglia precedente alla nuova origine: il robot si e'
            # mosso tra i due frame, e senza questo il flusso misurerebbe ANCHE
            # il moto del robot. Traslo la griglia vecchia di (dorigin/res) celle.
            dox = int(round((self.prev_origin[0] - ox) / self.res))
            doy = int(round((self.prev_origin[1] - oy) / self.res))
            prev_aligned = np.roll(self.prev_grid, (doy, dox), axis=(0, 1))

            flow = self.compute_flow(prev_aligned, grid)
            dt = max(1e-2, now_s - self.prev_t) if self.prev_t else 1.0 / self.rate_hz
            # da celle/frame a m/s
            vscale = self.res / dt

            ys, xs = np.where(grid > 0)
            moving_cells, vel_cells = [], []
            for x, y in zip(xs, ys):
                fx_, fy_ = flow[y, x, 0], flow[y, x, 1]
                vx = fx_ * vscale
                vy = fy_ * vscale
                if np.hypot(vx, vy) >= self.min_speed:
                    moving_cells.append((x, y))
                    vel_cells.append((vx, vy))
            self.diag['n_moving'] = len(moving_cells)

            blobs = self.cluster_moving(moving_cells, vel_cells)
            blobs = self.match_and_smooth(blobs, ox, oy)
            self.diag['n_blobs'] = len(blobs)

        self.prev_grid = grid
        self.prev_origin = (ox, oy)
        self.prev_t = now_s

        cost = self.build_cost(blobs, ox, oy)
        self.publish_cost(cost, ox, oy)
        self.publish_tracks(blobs)
        self.publish_markers(blobs)

    # ---------------------------------------------------- rilevanza (identica al tracker)
    def is_relevant(self, wx, wy, vx, vy):
        px = wx - self.robot_xy[0]
        py = wy - self.robot_xy[1]
        if self.cone_sector < 360.0:
            rel = np.arctan2(py, px) - self.robot_yaw
            rel = (rel + np.pi) % (2 * np.pi) - np.pi
            if abs(rel) > np.radians(self.cone_sector) / 2.0:
                return False
        d0 = np.hypot(px, py)
        if d0 <= self.relevance_always:
            return True
        rvx = vx - self.robot_v_world[0]
        rvy = vy - self.robot_v_world[1]
        vv = rvx * rvx + rvy * rvy
        if vv < 1e-6:
            return False
        t_cpa = -(px * rvx + py * rvy) / vv
        if t_cpa <= 0.0 or t_cpa > self.relevance_horizon:
            return False
        return np.hypot(px + rvx * t_cpa, py + rvy * t_cpa) <= self.relevance_radius

    # ---------------------------------------------------- cono (identico)
    def build_cost(self, blobs, ox, oy):
        cost = np.zeros((self.n, self.n), dtype=np.uint8)
        self.arrows = []
        for b in blobs:
            wx = ox + b['gx'] * self.res
            wy = oy + b['gy'] * self.res
            speed = np.hypot(b['vx'], b['vy'])
            if self.mark_obstacle:
                self._stamp(cost, wx, wy, ox, oy, self.nucleus_radius, self.max_cost)
            if not self.enable_cone or speed < self.min_speed:
                continue
            if b.get('mov_count', 0) < self.promote_frames:
                continue                    # non ancora confermato: niente cono (anti-muro)
            if not self.is_relevant(wx, wy, b['vx'], b['vy']):
                continue
            self._paint_cone(cost, wx, wy, b['vx'], b['vy'], ox, oy, speed)
            self.arrows.append((wx, wy, b['vx'], b['vy']))
        return cost

    def _stamp(self, cost, wx, wy, ox, oy, radius_m, val):
        cxi = int((wx - ox) / self.res); cyi = int((wy - oy) / self.res)
        rr = int(radius_m / self.res)
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if dx * dx + dy * dy <= rr * rr:
                    gx, gy = cxi + dx, cyi + dy
                    if 0 <= gx < self.n and 0 <= gy < self.n and cost[gy, gx] < val:
                        cost[gy, gx] = val

    def _paint_cone(self, cost, wx, wy, vx, vy, ox, oy, speed):
        ux, uy = vx / speed, vy / speed
        length = min(max(self.cone_min_length, speed * self.horizon_s), self.size_m)
        n_steps = max(1, int(length / self.res))
        px = (wx - ox) / self.res
        py = (wy - oy) / self.res
        half0 = self.cone_halfwidth / self.res
        nx, ny = -uy, ux
        for s in range(n_steps):
            frac = s / max(1, n_steps - 1)
            cx = px + ux * s
            cy = py + uy * s
            half = half0 * (1.0 + self.cone_spread * frac)
            base_val = self.cone_base_cost - (self.cone_base_cost - self.cone_tip_cost) * frac
            wv = -half
            while wv <= half + 1e-6:
                gx = int(round(cx + nx * wv))
                gy = int(round(cy + ny * wv))
                lat = abs(wv) / half if half > 1e-6 else 0.0
                val = int(base_val * (1.0 - self.cone_lateral_falloff * lat))
                if 0 <= gx < self.n and 0 <= gy < self.n and cost[gy, gx] < val:
                    cost[gy, gx] = val
                wv += 1.0

    # ---------------------------------------------------- output
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
        self.cost_pub.publish(msg)

    def publish_tracks(self, blobs):
        msg = TrackArray()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for i, b in enumerate(blobs):
            if np.hypot(b['vx'], b['vy']) < self.min_speed:
                continue
            if b.get('mov_count', 0) < self.promote_frames:
                continue                    # solo blob confermati vanno al critic
            t = Track()
            t.id = i
            t.x = float(b['wx']); t.y = float(b['wy'])
            t.vx = float(b['vx']); t.vy = float(b['vy'])
            t.pos_std = float(self.res)
            # incertezza di velocita': il flusso e' rumoroso -> la dichiaro alta,
            # cosi' l'alone del critic cresce di conseguenza (onesto)
            t.vel_std = float(max(0.2, 0.3 * np.hypot(b['vx'], b['vy'])))
            msg.tracks.append(t)
        self.tracks_pub.publish(msg)

    def publish_markers(self, blobs):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        now = self.get_clock().now().to_msg()
        for i, (wx, wy, vx, vy) in enumerate(self.arrows):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = now
            m.ns = 'flow_tracks'
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.06; m.scale.y = 0.12; m.scale.z = 0.12
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.6, 1.0, 0.9
            m.points = [Point(x=wx, y=wy, z=0.1),
                        Point(x=wx + vx, y=wy + vy, z=0.1)]
            m.lifetime.sec = 0
            m.lifetime.nanosec = 300000000
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def log_diag(self):
        d = self.diag
        self.get_logger().info(
            f"[flusso] celle_occupate={d['n_occ']} in_moto={d['n_moving']} "
            f"blob={d['n_blobs']}"
            + ('' if self._have_cv2 else '  [cv2 MANCANTE: nessun flusso]'))


def main():
    rclpy.init()
    node = OpticalFlowTracker()
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