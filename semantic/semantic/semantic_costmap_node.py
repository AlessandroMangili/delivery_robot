#!/usr/bin/env python3
"""
Proiezione semantica -> costmap del marciapiede.
Mappa PERSISTENTE (frame 'map') con:
  - fusione a confidenza (peso = vicinanza x non-occlusione; gate stabilita'/plasticita');
  - GESTIONE DINAMICI: gli oggetti di classi dinamiche (persona, veicoli...) vengono
    tracciati frame-per-frame e ne viene stimata la VELOCITA' nel mondo. Se si muovono
    (sopra soglia) i loro pixel sono ESCLUSI dalla mappa statica; se sono fermi/
    parcheggiati entrano come ostacoli. Un dinamico non ancora confermato fermo viene
    escluso in via cautelativa (evita scie di chi e' in movimento).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
import numpy as np
import cv2
from cv_bridge import CvBridge
import tf2_ros
from rclpy.duration import Duration
from std_srvs.srv import Trigger
import os


CLASS_COST = {
    1: 0, 9: 70, 8: 80, 0: 90,
    2: 100, 3: 100, 4: 100, 5: 100, 7: 100,
    11: 100, 12: 100, 13: 100, 14: 100, 15: 100, 16: 100, 17: 100, 18: 100,
}
DEFAULT_BLOCKERS = [2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
DEFAULT_DYNAMIC = [11, 12, 13, 14, 15, 16, 17, 18]   # persona, rider, veicoli


def transform_to_matrix(t):
    q = t.transform.rotation
    tr = t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


class Tracker:
    """Tracking NN in coordinate mondo + stima velocita' + classificazione moving/static."""
    def __init__(self, gate=1.0, v_thresh=0.3, min_obs=3, timeout=1.0, beta=0.6):
        self.gate = gate; self.v_thresh = v_thresh; self.min_obs = min_obs
        self.timeout = timeout; self.beta = beta
        self.tracks = []; self.next_id = 0

    def update(self, dets, t):
        """dets: lista (x,y) nel mondo. Ritorna lista di track allineata a dets."""
        assign = [None] * len(dets)
        used = set()
        for tr in self.tracks:
            best = None; bd = self.gate
            for i, (x, y) in enumerate(dets):
                if i in used:
                    continue
                d = np.hypot(x - tr['x'], y - tr['y'])
                if d < bd:
                    bd = d; best = i
            if best is not None:
                x, y = dets[best]; dt = max(1e-3, t - tr['t'])
                vx = (x - tr['x']) / dt; vy = (y - tr['y']) / dt
                tr['vx'] = self.beta * tr['vx'] + (1 - self.beta) * vx
                tr['vy'] = self.beta * tr['vy'] + (1 - self.beta) * vy
                tr['x'] = x; tr['y'] = y; tr['t'] = t; tr['n'] += 1
                used.add(best); assign[best] = tr
        for i, (x, y) in enumerate(dets):
            if i in used:
                continue
            tr = dict(id=self.next_id, x=x, y=y, vx=0.0, vy=0.0, n=1, t=t)
            self.next_id += 1; self.tracks.append(tr); assign[i] = tr
        self.tracks = [tr for tr in self.tracks if t - tr['t'] <= self.timeout]
        return assign

    def confirmed_static(self, tr):
        return tr['n'] >= self.min_obs and np.hypot(tr['vx'], tr['vy']) <= self.v_thresh


class SemanticCostmapNode(Node):
    def __init__(self):
        super().__init__('semantic_costmap_node')

        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('camera_optical_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('initial_size_m', 10.0)   # m, griglia iniziale (poi cresce da sola)
        self.declare_parameter('grow_margin_m', 3.0)     # m, margine aggiunto a ogni espansione
        self.declare_parameter('pixel_stride', 4)
        self.declare_parameter('max_range', 5.0)
        self.declare_parameter('min_row_frac', 0.62)
        # anti-spalmamento: scarta i raggi troppo orizzontali (vicini all'orizzonte)
        # che proiettano lontano e impreciso. min |dz| = elevazione minima.
        self.declare_parameter('min_ray_downness', 0.08)   # ~4.5 gradi
        # scrivo in mappa solo entro questa distanza (dove l'IPM e' accurato)
        self.declare_parameter('map_write_max_range', 4.0)
        self.declare_parameter('occ_sector_deg', 2.0)
        self.declare_parameter('occ_margin', 0.15)
        self.declare_parameter('occ_weak', 0.10)
        self.declare_parameter('w_dist_min', 0.03)
        self.declare_parameter('gate_ratio', 0.5)
        self.declare_parameter('blocker_classes', DEFAULT_BLOCKERS)
        # --- dinamici ---
        self.declare_parameter('dynamic_classes', DEFAULT_DYNAMIC)
        self.declare_parameter('dyn_v_thresh', 0.3)   # m/s sopra cui = in movimento
        self.declare_parameter('dyn_gate', 1.0)       # m, associazione tracce
        self.declare_parameter('dyn_min_obs', 3)      # frame prima di fidarsi della velocita'
        self.declare_parameter('dyn_timeout', 1.0)    # s prima di scartare una traccia
        self.declare_parameter('dyn_vel_beta', 0.6)   # smoothing velocita'

        gp = self.get_parameter
        self.target = gp('target_frame').value
        self.cam_frame = gp('camera_optical_frame').value
        self.base_frame = gp('base_frame').value
        self.res = gp('resolution').value
        self.initial_size_m = gp('initial_size_m').value
        self.grow_margin = gp('grow_margin_m').value
        self.stride = gp('pixel_stride').value
        self.max_range = gp('max_range').value
        self.min_row_frac = gp('min_row_frac').value
        self.min_ray_downness = float(gp('min_ray_downness').value)
        self.map_write_max_range = float(gp('map_write_max_range').value)
        self.occ_dth = np.deg2rad(gp('occ_sector_deg').value)
        self.occ_nsec = int(np.ceil(2 * np.pi / self.occ_dth))
        self.occ_margin = gp('occ_margin').value
        self.occ_weak = gp('occ_weak').value
        self.w_dist_min = gp('w_dist_min').value
        self.gate_ratio = gp('gate_ratio').value
        blockers = gp('blocker_classes').value
        dynamic = gp('dynamic_classes').value

        self.block_lut = np.zeros(256, dtype=bool)
        for c in blockers:
            self.block_lut[int(c)] = True
        self.dyn_lut = np.zeros(256, dtype=bool)
        for c in dynamic:
            self.dyn_lut[int(c)] = True

        self.tracker = Tracker(gate=gp('dyn_gate').value, v_thresh=gp('dyn_v_thresh').value,
                               min_obs=gp('dyn_min_obs').value, timeout=gp('dyn_timeout').value,
                               beta=gp('dyn_vel_beta').value)

        self.gnx = int(self.initial_size_m / self.res)   # colonne (x), cresce da sola
        self.gny = int(self.initial_size_m / self.res)   # righe (y)
        self.gox = -self.initial_size_m / 2.0
        self.goy = -self.initial_size_m / 2.0
        self.grid = np.full((self.gny, self.gnx), -1.0, dtype=np.float32)
        self.conf = np.zeros((self.gny, self.gnx), dtype=np.float32)

        self.bridge = CvBridge()
        self.K = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 1)
        seg_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/semantic/segmentation', self.seg_cb, seg_qos)

        # --- PUBBLICAZIONE MAPPA (due canali) ---
        # 1) /semantic_costmap : FULL MAP latched (transient_local) a bassa
        #    frequenza. Confinement e SSRL la ricevono COMPLETA (per il global
        #    planner) e aggiornata ogni full_map_period_s. Il latching fa si' che
        #    un subscriber che si connette dopo riceva subito l'ultima mappa intera.
        latched_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(OccupancyGrid, '/semantic_costmap', latched_qos)
        # marker del raggio/area mappabile (giallo)
        self.pub_range = self.create_publisher(Marker, '/semantic_map_range', 1)

        # ogni quanto ripubblicare la FULL map (bassa freq -> non rallenta)
        self.declare_parameter('full_map_period_s', 1.5)
        self.full_map_period = float(self.get_parameter('full_map_period_s').value)
        self._last_full_pub = 0.0
        self._map_dirty = True   # la full map va ripubblicata (qualcosa e' cambiato)
        self._significant_change = 0   # n. celle cambiate in modo rilevante
        # soglia: sopra questo n. di celle "rilevanti", pubblica SUBITO la full map
        self.declare_parameter('significant_change_cells', 15)
        self.significant_thresh = int(self.get_parameter('significant_change_cells').value)

        self.cost_lut = np.full(256, -2, dtype=np.int16)
        for cls, c in CLASS_COST.items():
            self.cost_lut[cls] = c

        # --- persistenza mappa (salva/carica grid + confidenza) ---
        # Si indicano solo i NOMI dei file; vengono risolti nella cartella maps/
        # del SORGENTE del pacchetto 'semantic' (persiste tra le colcon build).
        self.declare_parameter('map_load_name', '')   # nome file da caricare all'avvio (vuoto = no)
        self.declare_parameter('map_save_name', '')    # nome file usato dal servizio ~/save_map (vuoto = chiede un nome)
        self.maps_dir = self.resolve_maps_dir()
        load_name = self.get_parameter('map_load_name').value
        save_name = self.get_parameter('map_save_name').value
        self.map_save_path = os.path.join(self.maps_dir, save_name) if save_name else ''
        load_path = os.path.join(self.maps_dir, load_name) if load_name else ''
        if load_path:
            if not load_path.endswith('.npz'):
                load_path += '.npz'
            if os.path.isfile(load_path):
                self.load_map(load_path)
            else:
                self.get_logger().warn(f'Mappa da caricare non trovata: {load_path}')
        self.create_service(Trigger, '~/save_map', self.save_map_cb)
        self.get_logger().info(f'Cartella mappe: {self.maps_dir}')
        save_hint = self.map_save_path if self.map_save_path else f'{self.maps_dir}/semantic_map.npz'
        self.get_logger().info(
            'SALVATAGGIO MANUALE (niente autosave). Per salvare la mappa: '
            f'ros2 service call /semantic_costmap_node/save_map std_srvs/srv/Trigger  ->  {save_hint}')

        self.get_logger().info(
            f'Semantic costmap (confidenza + dinamici) pronto. frame={self.target}.')

    def resolve_maps_dir(self):
        """Trova (o crea) la cartella maps/ nel SORGENTE del pacchetto semantic.
        Cerca una cartella 'semantic' OVUNQUE sotto <ws>/src (anche in sottocartelle),
        riconoscendola dal package.xml. Se non la trova, ripiega sulla share installata."""
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('semantic')  # .../install/.../share/semantic
            # risali fino a trovare la radice del workspace (quella che contiene 'src')
            ws = share
            for _ in range(8):
                ws = os.path.dirname(ws)
                src = os.path.join(ws, 'src')
                if os.path.isdir(src):
                    # cammina sotto src/ cercando una cartella 'semantic' con package.xml
                    for root, dirs, files in os.walk(src):
                        if (os.path.basename(root) == 'semantic'
                                and 'package.xml' in files):
                            d = os.path.join(root, 'maps')
                            os.makedirs(d, exist_ok=True)
                            return d
                    break
            d = os.path.join(share, 'maps')          # ripiego: share installata
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            d = os.path.join(os.path.expanduser('~'), '.semantic_maps')
            os.makedirs(d, exist_ok=True)
            return d

    def load_map(self, path):
        try:
            d = np.load(path)
            if float(d['res']) != self.res:
                self.get_logger().warn(
                    'Mappa salvata con risoluzione diversa: ignorata.')
                return
            # con l'auto-grow adotto direttamente la griglia salvata (qualsiasi dimensione)
            self.grid = d['grid'].astype(np.float32)
            self.conf = d['conf'].astype(np.float32)
            self.gny, self.gnx = self.grid.shape
            self.gox = float(d['gox']); self.goy = float(d['goy'])
            self.get_logger().info(
                f'Mappa semantica caricata da {path} ({self.gnx}x{self.gny}).')
        except Exception as e:
            self.get_logger().warn(f'Caricamento mappa fallito: {e}')

    def save_map(self, path):
        try:
            if not path.endswith('.npz'):
                path = path + '.npz'
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            np.savez_compressed(path, grid=self.grid, conf=self.conf,
                                gnx=self.gnx, gny=self.gny, res=self.res,
                                gox=self.gox, goy=self.goy)
            self.get_logger().info(f'Mappa semantica salvata in {path}.')
            return path
        except Exception as e:
            self.get_logger().error(f'Salvataggio mappa fallito: {e}')
            return None

    def save_map_cb(self, request, response):
        path = self.map_save_path or os.path.join(self.maps_dir, 'semantic_map')
        saved = self.save_map(path)
        response.success = saved is not None
        response.message = (f'Mappa salvata in {saved}' if saved else 'Salvataggio fallito')
        return response

    def info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)

    def lookup(self, frame):
        try:
            return self.tf_buffer.lookup_transform(
                self.target, frame, rclpy.time.Time(), timeout=Duration(seconds=0.2))
        except Exception as e:
            self.get_logger().warn(f'TF non disponibile {frame}->{self.target}: {e}',
                                   throttle_duration_sec=2.0)
            return None

    def ensure_capacity(self, X, Y):
        """Espande la griglia (auto-grow) se le coordinate mondo X,Y cadono fuori.
        Copia i dati esistenti nella nuova griglia, aggiorna origine e dimensioni."""
        gi_min = int(np.floor((X.min() - self.gox) / self.res))
        gi_max = int(np.floor((X.max() - self.gox) / self.res))
        gj_min = int(np.floor((Y.min() - self.goy) / self.res))
        gj_max = int(np.floor((Y.max() - self.goy) / self.res))
        m = int(self.grow_margin / self.res)
        pl = (m - gi_min) if gi_min < 0 else 0
        pb = (m - gj_min) if gj_min < 0 else 0
        pr = (gi_max - (self.gnx - 1) + m) if gi_max >= self.gnx else 0
        pt = (gj_max - (self.gny - 1) + m) if gj_max >= self.gny else 0
        if not (pl or pr or pb or pt):
            return
        new_gnx = self.gnx + pl + pr
        new_gny = self.gny + pb + pt
        new_grid = np.full((new_gny, new_gnx), -1.0, dtype=np.float32)
        new_conf = np.zeros((new_gny, new_gnx), dtype=np.float32)
        new_grid[pb:pb + self.gny, pl:pl + self.gnx] = self.grid
        new_conf[pb:pb + self.gny, pl:pl + self.gnx] = self.conf
        self.grid = new_grid; self.conf = new_conf
        self.gnx = new_gnx; self.gny = new_gny
        self.gox -= pl * self.res; self.goy -= pb * self.res
        self.get_logger().info(
            f'Griglia espansa -> {self.gnx}x{self.gny} celle, origine '
            f'({self.gox:.1f},{self.goy:.1f}).', throttle_duration_sec=2.0)

    def seg_cb(self, msg: Image):
        if self.K is None:
            return
        seg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        h, w = seg.shape

        T_cam = self.lookup(self.cam_frame)
        T_base = self.lookup(self.base_frame)
        if T_cam is None or T_base is None:
            return
        M = transform_to_matrix(T_cam)
        origin = M[:3, 3]; R = M[:3, :3]
        bx, by = T_base.transform.translation.x, T_base.transform.translation.y

        r0 = int(h * self.min_row_frac)
        vs = np.arange(r0, h, self.stride)
        us = np.arange(0, w, self.stride)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.ravel(); vv = vv.ravel()

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        dir_opt = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, dtype=float)], axis=1)
        dir_world = dir_opt @ R.T
        dz = dir_world[:, 2]

        valid = dz < -1e-6
        # ANTI-SPALMAMENTO: scarta i raggi troppo orizzontali (vicini all'orizzonte),
        # che proiettano lontano e con enorme imprecisione (un errore di 0.5 gradi a
        # 2 gradi sposta il punto di metri). Tengo solo raggi ben inclinati in giu'.
        valid = valid & (dz < -self.min_ray_downness)
        t = np.full(uu.shape, -1.0)
        t[valid] = -origin[2] / dz[valid]
        ok = valid & (t > 0)
        pts = origin[None, :] + t[:, None] * dir_world
        X, Y = pts[:, 0], pts[:, 1]
        dist = np.hypot(X - bx, Y - by)
        # scrivo in mappa solo entro map_write_max_range (dove l'IPM e' accurato):
        # i pixel piu' lontani proiettano imprecisi e, con drift, si spalmano.
        ok = ok & (dist < self.map_write_max_range)

        # registro il range di distanza REALE dei punti scritti in mappa, per
        # disegnare il raggio giallo esatto (non stimato).
        if np.any(ok):
            dok = dist[ok]
            self._range_near_est = float(np.percentile(dok, 5))
            self._range_far_est = float(np.percentile(dok, 95))

        classes = seg[vv, uu]
        costs = self.cost_lut[classes]
        ok = ok & (costs >= 0)

        # ---------- DINAMICI: traccia, stima velocita', escludi quelli in movimento ----------
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dyn_pix = self.dyn_lut[classes] & ok
        if dyn_pix.any():
            full_dyn = self.dyn_lut[seg].astype(np.uint8)
            _, lbl = cv2.connectedComponents(full_dyn)
            comp_s = lbl[vv, uu]                       # id componente per pixel campionato
            dets = []; det_cids = []
            for cid in np.unique(comp_s[dyn_pix]):
                if cid == 0:
                    continue
                m = (comp_s == cid) & ok
                if not m.any():
                    continue
                idx = np.where(m)[0]
                base_i = idx[np.argmax(vv[idx])]       # pixel piu' basso = base a terra
                dets.append((float(X[base_i]), float(Y[base_i]))); det_cids.append(cid)
            assign = self.tracker.update(dets, stamp_sec)
            exclude_cids = [cid for cid, tr in zip(det_cids, assign)
                            if not self.tracker.confirmed_static(tr)]
            if exclude_cids:
                excl = np.isin(comp_s, exclude_cids) & dyn_pix
                ok = ok & (~excl)

        # ---------- OCCLUSIONE -> peso ----------
        camx, camy = origin[0], origin[1]
        ang = np.arctan2(Y - camy, X - camx)
        rad = np.hypot(X - camx, Y - camy)
        sec = np.clip(((ang + np.pi) / self.occ_dth).astype(int), 0, self.occ_nsec - 1)
        is_block = self.block_lut[classes] & ok
        occ_r = np.full(self.occ_nsec, np.inf, dtype=np.float64)
        if is_block.any():
            np.minimum.at(occ_r, sec[is_block], rad[is_block])
        occluded = rad > (occ_r[sec] + self.occ_margin)

        w_dist = np.clip(1.0 - dist / self.max_range, self.w_dist_min, 1.0)
        w_occ = np.where(occluded, self.occ_weak, 1.0)
        wpix = w_dist * w_occ

        # AUTO-GROW: se le osservazioni valide cadono fuori griglia, espandi
        okx = ok & np.isfinite(X) & np.isfinite(Y)
        if okx.any():
            self.ensure_capacity(X[okx], Y[okx])

        gi = ((X - self.gox) / self.res).astype(int)
        gj = ((Y - self.goy) / self.res).astype(int)
        inside = ok & (gi >= 0) & (gi < self.gnx) & (gj >= 0) & (gj < self.gny)

        gi_u = gi[inside]; gj_u = gj[inside]
        co_u = costs[inside].astype(np.float32)
        w_u = wpix[inside].astype(np.float32)
        if gi_u.size == 0:
            self.publish_map(bx, by, msg.header.stamp)
            return

        flat = gj_u * self.gnx + gi_u
        N = self.gnx * self.gny
        order = np.argsort(w_u)
        flat_s = flat[order]
        frame_cost = np.full(N, -1.0, dtype=np.float32)
        frame_w = np.full(N, -1.0, dtype=np.float32)
        frame_cost[flat_s] = co_u[order]
        frame_w[flat_s] = w_u[order]
        seen = frame_w >= 0.0

        V = self.grid.ravel(); C = self.conf.ravel()
        V_before = V.copy()   # per rilevare cambiamenti rilevanti (pubblica subito)
        accept = seen & (frame_w >= self.gate_ratio * C)
        unknown = accept & (V < 0)
        knownup = accept & (V >= 0)
        V[unknown] = frame_cost[unknown]
        fa = frame_w[knownup] / (frame_w[knownup] + C[knownup])
        V[knownup] = (1.0 - fa) * V[knownup] + fa * frame_cost[knownup]
        C[accept] = np.maximum(C[accept], frame_w[accept])
        if accept.any():
            self._map_dirty = True   # la mappa e' cambiata -> ripubblica la full
            # conta i cambiamenti RILEVANTI: celle che passano da libero a ostacolo
            # o viceversa (non piccoli aggiustamenti di costo). Un cambiamento
            # rilevante (nuovo ostacolo!) merita pubblicazione IMMEDIATA.
            old_vals = V_before[accept]
            new_vals = V[accept]
            # "rilevante" = attraversa la soglia ostacolo (es. 50): appare/sparisce
            # qualcosa di solido dove prima non c'era.
            crossed = ((old_vals < 50) & (new_vals >= 50)) | \
                      ((old_vals >= 50) & (new_vals < 50)) | (old_vals < 0)
            self._significant_change = int(np.count_nonzero(crossed))

        self.grid = V.reshape(self.gny, self.gnx)
        self.conf = C.reshape(self.gny, self.gnx)
        self.publish_map(bx, by, msg.header.stamp)

    def _grid_to_msg(self, sub, ox, oy, width, height, stamp):
        """Converte una porzione di griglia in OccupancyGrid. Usa tobytes (112x
        piu' veloce di .tolist() sulle mappe grandi)."""
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target
        msg.info.resolution = self.res
        msg.info.width = width; msg.info.height = height
        msg.info.origin.position.x = float(ox)
        msg.info.origin.position.y = float(oy)
        msg.info.origin.orientation.w = 1.0
        data = np.where(sub < 0, -1, np.clip(np.round(sub), 0, 100)).astype(np.int8)
        import array
        msg.data = array.array('b', np.ascontiguousarray(data).tobytes())
        return msg

    def publish_map(self, bx, by, stamp):
        now_s = self.get_clock().now().nanoseconds * 1e-9

        # --- FULL MAP latched (per confinement/SSRL/global planner) ---
        # Pubblico la full map completa quando:
        #  (a) e' passato full_map_period_s dall'ultima (aggiornamento periodico), OPPURE
        #  (b) c'e' stato un CAMBIAMENTO RILEVANTE (nuovo ostacolo!) -> pubblico
        #      SUBITO, anche se ho appena pubblicato: un ostacolo nuovo non puo'
        #      aspettare 1.5s, il planner deve saperlo ora.
        periodic_due = (now_s - self._last_full_pub) >= self.full_map_period
        urgent = self._significant_change >= self.significant_thresh
        if self._map_dirty and (periodic_due or urgent):
            full = self._grid_to_msg(self.grid, self.gox, self.goy,
                                     self.gnx, self.gny, stamp)
            self.pub.publish(full)
            if urgent:
                self.get_logger().info(
                    f'Cambiamento rilevante ({self._significant_change} celle) '
                    f'-> full map pubblicata SUBITO', throttle_duration_sec=1.0)
            self._last_full_pub = now_s
            self._map_dirty = False
            self._significant_change = 0

        # --- MARKER: raggio/area mappabile (giallo) ---
        self.publish_range_marker(bx, by, stamp)

    def publish_range_marker(self, bx, by, stamp):
        """Area mappabile REALE: un anello nel settore frontale, tra il limite
        VICINO (i raggi piu' inclinati, bordo basso immagine) e quello LONTANO
        (map_write_max_range). Rispecchia dove il robot scrive davvero in mappa."""
        T = self.lookup(self.base_frame)
        if T is None:
            return
        q = T.transform.rotation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # limiti REALI misurati dai punti effettivamente scritti in mappa
        # (percentili 5-95 delle distanze), non stime fisse.
        r_far = getattr(self, '_range_far_est', self.map_write_max_range)
        r_near = getattr(self, '_range_near_est', 0.3)
        # apertura angolare = FOV orizzontale reale (ricavato da K se disponibile)
        if self.K is not None:
            fx = self.K[0, 0]; cx = self.K[0, 2]
            half_fov = np.arctan(cx / fx)   # meta' FOV orizzontale
        else:
            half_fov = np.deg2rad(30.0)

        from geometry_msgs.msg import Point
        m = Marker()
        m.header.frame_id = self.target
        m.header.stamp = stamp
        m.ns = 'map_range'; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.04
        m.color.r = 1.0; m.color.g = 0.9; m.color.b = 0.0; m.color.a = 0.7
        pts = []
        angs = np.linspace(-half_fov, half_fov, 24)
        # arco lontano
        for a in angs:
            ang = yaw + a
            pts.append(Point(x=float(bx + r_far * np.cos(ang)),
                             y=float(by + r_far * np.sin(ang)), z=0.05))
        # arco vicino (tornando indietro) -> chiude l'anello
        for a in reversed(angs):
            ang = yaw + a
            pts.append(Point(x=float(bx + r_near * np.cos(ang)),
                             y=float(by + r_near * np.sin(ang)), z=0.05))
        pts.append(pts[0])   # chiudi
        m.points = pts
        self.pub_range.publish(m)


def main():
    rclpy.init()
    node = SemanticCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Nodo in chiusura')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()