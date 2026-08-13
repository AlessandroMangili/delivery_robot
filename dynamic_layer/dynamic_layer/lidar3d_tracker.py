#!/usr/bin/env python3
"""
lidar3d_tracker.py — Core del tracker AB3DMOT (Step 2 della pipeline LiDAR 3D).

Consuma le detection dello Step 1 e produce TRACCE con identità, posizione e
velocità. Riferimento: AB3DMOT (Kalman a velocità costante + associazione),
con le buone pratiche di SimpleTrack (CV + gating).

    /detections (PoseArray, frame sensore base_scan)
        -> TF base_scan -> odom   (frame fisso: nel frame del sensore che ruota
                                    l'ego-moto fingerebbe una velocità)
        -> predizione Kalman CV di tutte le tracce
        -> associazione (Hungarian) con gate di Mahalanobis + gate euclideo
        -> correzione Kalman delle tracce associate
        -> nascita/morte tracce (min_hits per confermare, max_age per il coasting)
        -> GATE DI VELOCITÀ: pubblica solo le tracce in moto (uccide alberi,
           panchine, pali: sono statici, v ~ 0)

Uscita (Step 2, per verifica):
    /tracks/markers   visualization_msgs/MarkerArray   (cilindro + freccia velocità
                                                         + id, colore stabile per id)

Lo Step 4 aggancerà TrackArray(x,y,vx,vy,pos_std,vel_std) su /dynamic_tracks_state
per il critic; qui restiamo sui marker per verificare tracking e gate.

Dipendenze: numpy. Associazione greedy (niente scipy nel tracker).
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image, CameraInfo

import tf2_ros
from tf2_ros import TransformException


# ===========================================================================
# NUCLEO DI TRACKING — indipendente da ROS (testabile in isolamento)
# ===========================================================================
class KalmanCV:
    """Filtro di Kalman a velocità costante. Stato [x, y, vx, vy].
    Misura la sola posizione [x, y]; la velocità emerge dalla sequenza."""

    def __init__(self, xy, accel_std, meas_std, init_pos_std, init_vel_std):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([init_pos_std**2, init_pos_std**2,
                          init_vel_std**2, init_vel_std**2]).astype(float)
        self.q = float(accel_std) ** 2      # densità rumore di accelerazione
        self.r = float(meas_std) ** 2       # varianza misura di posizione
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        # rumore di processo a accelerazione bianca discreta (modello CV canonico)
        Q = self.q * np.array([
            [dt**4 / 4, 0,          dt**3 / 2, 0],
            [0,          dt**4 / 4, 0,          dt**3 / 2],
            [dt**3 / 2, 0,          dt**2,      0],
            [0,          dt**3 / 2, 0,          dt**2]], dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.r * np.eye(2)
        return y, S

    def update(self, z):
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self):
        return self.x[:2].copy()

    @property
    def vel(self):
        return self.x[2:].copy()

    @property
    def speed(self):
        return float(math.hypot(self.x[2], self.x[3]))

    def pos_std(self):
        return float(math.sqrt(0.5 * (self.P[0, 0] + self.P[1, 1])))

    def vel_std(self):
        return float(math.sqrt(0.5 * (self.P[2, 2] + self.P[3, 3])))


class Track:
    def __init__(self, tid, xy, params):
        self.id = tid
        self.kf = KalmanCV(xy, params['accel_std'], params['meas_std'],
                           params['init_pos_std'], params['init_vel_std'])
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = False
        self.published_dynamic = False   # confermato dinamico?
        self.mov_count = 0               # frame in moto COERENTE consecutivi
        self.still_count = 0             # frame lenti consecutivi
        # --- stato semantico (dalla camera; persiste anche fuori dal cono) ---
        self.sem_class = -1              # classe Cityscapes dinamica (-1 = ignota)
        self.sem_static_count = 0        # letture consecutive su classe statica
        self.sem_dynamic_count = 0       # letture consecutive su classe dinamica


class MultiObjectTracker:
    """AB3DMOT: predizione CV -> associazione con gate -> update -> ciclo di vita."""

    def __init__(self, params):
        self.p = params
        self.tracks = []
        self.next_id = 0

    def step(self, dets, dt):
        """dets: ndarray (M,2) nel frame di tracking. Ritorna la lista di tracce."""
        dt = float(min(max(dt, 1e-3), 1.0))

        # 1. predizione di tutte le tracce
        for t in self.tracks:
            t.kf.predict(dt)
            t.age += 1
            t.time_since_update += 1

        # 2. associazione
        matches, un_tracks, un_dets = self._associate(dets)

        # 3. correzione delle tracce associate
        for ti, di in matches:
            self.tracks[ti].kf.update(dets[di])
            self.tracks[ti].hits += 1
            self.tracks[ti].time_since_update = 0
            if self.tracks[ti].hits >= self.p['min_hits']:
                self.tracks[ti].confirmed = True

        # 4. nascita: nuova traccia per ogni detection non associata
        for di in un_dets:
            self.tracks.append(Track(self.next_id, dets[di], self.p))
            self.next_id += 1

        # 5. morte: rimuovi le tracce senza update da troppo tempo
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.p['max_age']]

        # 6. conferma dinamico (life-cycle): moto COERENTE per promote_frames frame
        #    consecutivi + gate SNR (velocità significativa vs la sua incertezza).
        #    SimpleTrack indica la terminazione precoce / le tracce spurie come causa
        #    #1 degli ID-switch: un bordo di muro che balla a media zero NON accumula
        #    frame consecutivi di moto vero e ha SNR basso -> non viene mai promosso.
        for t in self.tracks:
            speed = t.kf.speed
            sv = t.kf.vel_std()
            moving = ((speed >= self.p['min_speed']) and
                      (self.p['snr_min'] <= 0.0 or speed >= self.p['snr_min'] * sv))
            if moving:
                t.mov_count += 1
                t.still_count = 0
            else:
                t.still_count += 1
                t.mov_count = 0
            # promozione: traccia confermata (Kalman stabilizzato) + moto coerente
            if ((not t.published_dynamic) and t.confirmed and
                    t.mov_count >= self.p['promote_frames']):
                t.published_dynamic = True
            # retrocessione con isteresi: ferma per demote_frames -> torna statica
            if t.published_dynamic and t.still_count >= self.p['demote_frames']:
                t.published_dynamic = False

        return self.tracks

    def _associate(self, dets):
        """Associazione GREEDY (SimpleTrack Sec. 4.3.2: le metriche a distanza come
        Mahalanobis vanno con il greedy, non con l'ungherese, che gli outlier possono
        sviare). Si assegnano prima le coppie a costo minore, saltando quelle già prese."""
        T, M = len(self.tracks), len(dets)
        if T == 0 or M == 0:
            return [], list(range(T)), list(range(M))
        gate_maha = self.p['gating_mahalanobis']
        gate_dist = self.p['max_assoc_dist']
        # coppie ammissibili (dentro entrambi i gate) con il loro costo di Mahalanobis^2
        pairs = []
        for i, t in enumerate(self.tracks):
            for j in range(M):
                y, S = t.kf.innovation(dets[j])
                if math.hypot(y[0], y[1]) > gate_dist:   # gate euclideo di sicurezza
                    continue
                try:
                    m2 = float(y @ np.linalg.inv(S) @ y)  # distanza di Mahalanobis^2
                except np.linalg.LinAlgError:
                    continue
                if m2 > gate_maha:                        # gate chi-quadro (2 gdl)
                    continue
                pairs.append((m2, i, j))
        # greedy: costo crescente, assegna solo se traccia e detection sono libere
        pairs.sort(key=lambda p: p[0])
        matched_t, matched_d, matches = set(), set(), []
        for _cost, i, j in pairs:
            if i in matched_t or j in matched_d:
                continue
            matches.append((i, j))
            matched_t.add(i)
            matched_d.add(j)
        un_tracks = [i for i in range(T) if i not in matched_t]
        un_dets = [j for j in range(M) if j not in matched_d]
        return matches, un_tracks, un_dets


# ===========================================================================
# Utilità geometriche
# ===========================================================================
def quat_to_rot(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1 - (yy + zz), xy - wz,       xz + wy],
        [xy + wz,       1 - (xx + zz), yz - wx],
        [xz - wy,       yz + wx,       1 - (xx + yy)]])


PALETTE = [(0.20, 0.80, 0.95), (0.95, 0.60, 0.20), (0.60, 0.85, 0.25),
           (0.90, 0.35, 0.75), (0.95, 0.85, 0.25), (0.45, 0.55, 0.95),
           (0.30, 0.90, 0.55), (0.95, 0.45, 0.35)]


def color_for(tid, a=0.9):
    r, g, b = PALETTE[tid % len(PALETTE)]
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


CITYSCAPES_NAMES = {11: 'person', 12: 'rider', 13: 'car', 14: 'truck',
                    15: 'bus', 16: 'train', 17: 'moto', 18: 'bike'}


def project_to_pixel(K, p_cam, w, h):
    """Proietta un punto 3D (frame ottico camera) nel pixel (u,v). None se dietro
    la camera o fuori inquadratura. Funzione pura -> testabile senza ROS."""
    Xc, Yc, Zc = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if Zc <= 0.05:
        return None
    u = int(round(K[0, 0] * Xc / Zc + K[0, 2]))
    v = int(round(K[1, 1] * Yc / Zc + K[1, 2]))
    if 0 <= u < w and 0 <= v < h:
        return (u, v)
    return None


def patch_majority(seg, u, v, r):
    """Classe di maggioranza in una finestra (2r+1) attorno a (u,v). Funzione pura."""
    h, w = seg.shape
    patch = seg[max(0, v - r):min(h, v + r + 1),
                max(0, u - r):min(w, u + r + 1)].ravel()
    if patch.size == 0:
        return None
    vals, counts = np.unique(patch, return_counts=True)
    return int(vals[np.argmax(counts)])


def cell_is_static(smap, res, ox, oy, w, h, xm, ym, thresh, radius):
    """True se (xm,ym) [frame map] cade su/vicino a una cella occupata >= thresh.
    Funzione pura (numpy) -> testabile senza ROS. La finestra di 'radius' celle
    assorbe il jitter che potrebbe spostare la traccia appena fuori dal muro."""
    gi = int((xm - ox) / res)
    gj = int((ym - oy) / res)
    i0, i1 = max(0, gi - radius), min(w, gi + radius + 1)
    j0, j1 = max(0, gj - radius), min(h, gj + radius + 1)
    if i0 >= i1 or j0 >= j1:
        return False
    return bool((smap[j0:j1, i0:i1] >= thresh).any())


# ===========================================================================
# Nodo ROS2
# ===========================================================================
class Lidar3DTracker(Node):
    def __init__(self):
        super().__init__('lidar3d_tracker')

        d = self.declare_parameter
        d('input_topic', '/detections')
        d('tracking_frame', 'odom')
        d('markers_topic', '/tracks/markers')

        # Kalman
        d('accel_std', 0.8)          # densità rumore di accelerazione [m/s^2]
                                     # (tarato: statico fermo resta sotto ~0.12 m/s,
                                     #  ben sotto il gate 0.20; vedi test)
        d('meas_std', 0.10)          # rumore misura di posizione [m]
        d('init_pos_std', 0.30)
        d('init_vel_std', 2.0)

        # associazione
        d('gating_mahalanobis', 9.21)  # chi^2, 2 gdl, ~99%
        d('max_assoc_dist', 1.5)       # gate euclideo di sicurezza [m]

        # ciclo di vita
        d('min_hits', 3)             # frame per confermare una traccia
        d('max_age', 10)             # frame di coasting prima di ucciderla (10 = ~1 s a 10 Hz)
        d('coast_publish_frames', 3) # per quanti frame di coasting una traccia resta
                                     # PUBBLICATA (fantasma corto). Oltre, resta viva
                                     # internamente per ri-associarsi ma non è mostrata.

        # gate di velocità (isteresi) — uccide gli statici
        d('min_speed', 0.30)         # [m/s] soglia di "in movimento"
        d('snr_min', 1.0)            # velocità >= snr_min * incertezza -> direzione affidabile
        d('promote_frames', 4)       # frame di moto coerente consecutivi per promuovere
        d('demote_frames', 8)        # frame lenti consecutivi per retrocedere

        # cross-check con la mappa semantica statica (/semantic_costmap)
        # una traccia su una cella occupata (struttura mappata) è un falso
        # dinamico: i dinamici NON scrivono in mappa, quindi occupata = statico.
        d('use_static_map_gate', True)
        d('static_map_topic', '/semantic_costmap')
        d('static_cost_thresh', 99)          # 100 = struttura dura; 99 la cattura
        d('static_gate_radius_cells', 2)     # finestra di celle attorno alla traccia

        # gate semantico camera sulle tracce (/semantic/segmentation, ID Cityscapes)
        # proietta la traccia nell'immagine: classe statica -> scarta; dinamica ->
        # etichetta (la classe persiste con la traccia anche fuori dal cono camera).
        d('use_semantic_gate', True)
        d('seg_topic', '/semantic/segmentation')
        d('camera_info_topic', '/camera/camera_info')
        d('camera_frame', 'camera_rgb_optical_frame')
        d('probe_height', 0.9)               # [m] quota a cui proiettare la traccia (mezzo busto)
        d('probe_patch', 2)                  # semi-finestra pixel per il voto di maggioranza
        d('sem_static_hits', 2)              # letture statiche consecutive per scartare
        d('static_seg_classes', [2, 3, 4, 5, 7, 8])          # building/wall/fence/pole/sign/veg
        d('dynamic_seg_classes', [11, 12, 13, 14, 15, 16, 17, 18])

        # visualizzazione
        d('marker_z', 0.6)
        d('arrow_time_scale', 1.0)   # lunghezza freccia = v * questo [s]
        d('publish_static', False)   # se true, mostra in grigio le tracce ferme (debug)
        d('diag_period_frames', 20)

        g = lambda k: self.get_parameter(k).value
        self.input_topic = g('input_topic')
        self.tracking_frame = g('tracking_frame')
        self.markers_topic = g('markers_topic')
        self.marker_z = float(g('marker_z'))
        self.arrow_time_scale = float(g('arrow_time_scale'))
        self.publish_static = bool(g('publish_static'))
        self.diag_period_frames = int(g('diag_period_frames'))
        self.coast_publish_frames = int(g('coast_publish_frames'))

        self.use_static_map_gate = bool(g('use_static_map_gate'))
        self.static_cost_thresh = int(g('static_cost_thresh'))
        self.static_gate_radius = int(g('static_gate_radius_cells'))

        self.use_semantic_gate = bool(g('use_semantic_gate'))
        self.camera_frame = g('camera_frame')
        self.probe_height = float(g('probe_height'))
        self.probe_patch = int(g('probe_patch'))
        self.sem_static_hits = int(g('sem_static_hits'))
        self.static_set = set(int(c) for c in g('static_seg_classes'))
        self.dynamic_set = set(int(c) for c in g('dynamic_seg_classes'))

        self.params = dict(
            accel_std=float(g('accel_std')),
            meas_std=float(g('meas_std')),
            init_pos_std=float(g('init_pos_std')),
            init_vel_std=float(g('init_vel_std')),
            gating_mahalanobis=float(g('gating_mahalanobis')),
            max_assoc_dist=float(g('max_assoc_dist')),
            min_hits=int(g('min_hits')),
            max_age=int(g('max_age')),
            min_speed=float(g('min_speed')),
            snr_min=float(g('snr_min')),
            promote_frames=int(g('promote_frames')),
            demote_frames=int(g('demote_frames')),
        )
        self.mot = MultiObjectTracker(self.params)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        # spin_thread=True: il listener aggiorna il buffer su un thread proprio,
        # così una lookup con timeout nella callback non va in deadlock.
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE)
        # Gruppi di callback: on_dets (tracking) su un gruppo suo (mai concorrente
        # con se stesso -> stato del tracker al sicuro); camera e mappa su un gruppo
        # separato -> girano su un ALTRO thread e non serializzano con on_dets.
        # Con MultiThreadedExecutor, le callback camera non affamano più il feed TF.
        self.cb_track = MutuallyExclusiveCallbackGroup()
        self.cb_side = MutuallyExclusiveCallbackGroup()

        self.pub_mrk = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.sub = self.create_subscription(
            PoseArray, self.input_topic, self.on_dets, qos,
            callback_group=self.cb_track)

        # mappa semantica statica (latched -> TRANSIENT_LOCAL per ricevere l'ultima)
        self.smap = None
        if self.use_static_map_gate:
            map_qos = QoSProfile(depth=1,
                                 reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST)
            self.sub_map = self.create_subscription(
                OccupancyGrid, g('static_map_topic'), self.on_static_map, map_qos,
                callback_group=self.cb_side)

        # segmentazione camera + intrinseci (per il gate semantico sulle tracce)
        self.seg = None
        self.seg_stamp = None
        self.K = None
        if self.use_semantic_gate:
            seg_qos = QoSProfile(depth=1,
                                 reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST)
            self.sub_seg = self.create_subscription(
                Image, g('seg_topic'), self.on_seg, seg_qos,
                callback_group=self.cb_side)
            self.sub_cam = self.create_subscription(
                CameraInfo, g('camera_info_topic'), self.on_caminfo, 1,
                callback_group=self.cb_side)

        self._frame = 0
        self.last_t = None
        self._n_suppressed = 0
        self._n_sem_suppressed = 0
        self.get_logger().info(
            f"lidar3d_tracker avviato: {self.input_topic} -> tracce in "
            f"'{self.tracking_frame}'")

    # -----------------------------------------------------------------------
    def on_seg(self, msg):
        """Segmentazione mono8 (ID Cityscapes per pixel). Decodifica senza cv_bridge."""
        step = msg.step if msg.step else msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            self.seg = arr.reshape(msg.height, step)[:, :msg.width]
        except ValueError:
            return
        self.seg_stamp = msg.header.stamp

    def on_caminfo(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)

    def _lookup_cam(self):
        """TF tracking_frame -> camera_frame, NON bloccante (timeout 0, ultima TF).
        Il gate camera legge solo la classe a un pixel, non stima velocità: un
        piccolo sfasamento temporale lo assorbe il patch di maggioranza. Timeout 0
        = ritorna subito se la TF non c'è, così NON blocca on_dets né affama il
        thread del listener TF (era questo a far slittare la lookup delle detection)."""
        try:
            return self.tf_buffer.lookup_transform(
                self.camera_frame, self.tracking_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.0))
        except TransformException:
            return None

    def _semantic_class(self, xt, yt, tf_cam):
        """Classe Cityscapes letta proiettando la traccia (xt,yt,probe_height)
        nella segmentazione. None se dietro la camera / fuori inquadratura."""
        if self.seg is None or self.K is None or tf_cam is None:
            return None
        p_cam = self._apply_tf(tf_cam, np.array([[xt, yt, self.probe_height]]))[0]
        h, w = self.seg.shape
        px = project_to_pixel(self.K, p_cam, w, h)
        if px is None:
            return None
        return patch_majority(self.seg, px[0], px[1], self.probe_patch)

    def _update_semantics(self, tracks):
        """Aggiorna lo stato semantico delle tracce dalla camera. La classe letta
        persiste: fuori dal cono non si tocca nulla (nessuna lettura -> nessun cambio)."""
        if not self.use_semantic_gate:
            return
        tf_cam = self._lookup_cam()
        if tf_cam is None:
            return
        for t in tracks:
            if not t.confirmed:
                continue
            cls = self._semantic_class(t.kf.pos[0], t.kf.pos[1], tf_cam)
            if cls is None:
                continue                      # fuori cono/dietro -> mantiene lo stato
            if cls in self.static_set:
                t.sem_static_count += 1
                t.sem_dynamic_count = 0
            elif cls in self.dynamic_set:
                t.sem_dynamic_count += 1
                t.sem_static_count = 0
                t.sem_class = cls             # etichetta persistente
            # altre classi (strada/marciapiede/terreno/cielo): nessuna decisione

    def _sem_is_static(self, t):
        return (t.sem_static_count >= self.sem_static_hits and
                t.sem_dynamic_count == 0)

    # -----------------------------------------------------------------------
    def _lookup(self, src, stamp):
        """TF src -> tracking_frame ALLO STAMP ESATTO del cloud.
        NIENTE fallback all'ultima TF: userebbe una posa del robot diversa da
        quella di cattura e in movimento farebbe 'muovere' in odom gli oggetti
        statici (ego-moto non compensato). Se la TF manca, si salta il frame:
        il tracker fa coasting e nessuna velocità viene falsata."""
        try:
            return self.tf_buffer.lookup_transform(
                self.tracking_frame, src,
                rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=0.1))
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self.tracking_frame} <- {src} @stamp non disponibile, "
                f"salto il frame ({e})",
                throttle_duration_sec=2.0)
            return None

    @staticmethod
    def _apply_tf(tf, pts):
        q = tf.transform.rotation
        tr = tf.transform.translation
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        return (R @ pts.T).T + np.array([tr.x, tr.y, tr.z])

    def on_static_map(self, msg):
        """Memorizza l'ultima mappa semantica (costo per cella, frame map)."""
        self.smap = np.array(msg.data, dtype=np.int16).reshape(
            msg.info.height, msg.info.width)
        self.smap_res = msg.info.resolution
        self.smap_ox = msg.info.origin.position.x
        self.smap_oy = msg.info.origin.position.y
        self.smap_h, self.smap_w = self.smap.shape

    def _lookup_map(self, stamp):
        """TF tracking_frame -> map per il cross-check. Qui il fallback all'ultima
        TF è innocuo: la mappa è statica, un piccolo sfasamento non falsa la velocità
        (che è già stata stimata) — serve solo la posizione approssimata in map."""
        if self.tracking_frame == 'map':
            return None
        for tp in (rclpy.time.Time.from_msg(stamp), rclpy.time.Time()):
            try:
                return self.tf_buffer.lookup_transform(
                    'map', self.tracking_frame, tp, timeout=Duration(seconds=0.1))
            except TransformException:
                continue
        return None

    def _on_static_structure(self, xt, yt, tf_map):
        """True se la traccia (xt,yt in tracking_frame) sta su struttura mappata."""
        if self.smap is None:
            return False
        if tf_map is not None:
            pm = self._apply_tf(tf_map, np.array([[xt, yt, 0.0]]))[0]
            xm, ym = pm[0], pm[1]
        elif self.tracking_frame == 'map':
            xm, ym = xt, yt
        else:
            return False
        return cell_is_static(self.smap, self.smap_res, self.smap_ox, self.smap_oy,
                              self.smap_w, self.smap_h, xm, ym,
                              self.static_cost_thresh, self.static_gate_radius)

    def on_dets(self, msg):
        self._frame += 1
        stamp = msg.header.stamp
        t_now = stamp.sec + stamp.nanosec * 1e-9
        dt = 0.1 if self.last_t is None else (t_now - self.last_t)
        self.last_t = t_now

        tf = self._lookup(msg.header.frame_id, stamp)
        if tf is None:
            return

        if len(msg.poses) == 0:
            dets = np.zeros((0, 2))
        else:
            pts = np.array([[p.position.x, p.position.y, p.position.z]
                            for p in msg.poses], dtype=float)
            dets = self._apply_tf(tf, pts)[:, :2]

        tracks = self.mot.step(dets, dt)
        self._update_semantics(tracks)
        self.publish(stamp, tracks)

        if self.diag_period_frames > 0 and self._frame % self.diag_period_frames == 0:
            n_conf = sum(1 for t in tracks if t.confirmed)
            n_dyn = sum(1 for t in tracks if t.confirmed and t.published_dynamic)
            self.get_logger().info(
                f"[trk] dets={len(dets)} tracce={len(tracks)} "
                f"confermate={n_conf} dinamiche={n_dyn} "
                f"soppresse_mappa={self._n_suppressed} "
                f"soppresse_sem={self._n_sem_suppressed}")

    # -----------------------------------------------------------------------
    def publish(self, stamp, tracks):
        tf_map = self._lookup_map(stamp) if self.use_static_map_gate else None
        n_suppressed = 0
        n_sem = 0
        ma = MarkerArray()
        for t in tracks:
            if not t.confirmed:
                continue
            dynamic = t.published_dynamic
            # cross-check con la mappa statica: se la traccia dinamica sta su una
            # struttura mappata, è un falso positivo -> non pubblicarla come dinamica
            if dynamic and self.use_static_map_gate:
                px0, py0 = t.kf.pos
                if self._on_static_structure(px0, py0, tf_map):
                    dynamic = False
                    n_suppressed += 1
            # gate semantico camera: se la traccia è vista come classe STATICA
            # (building/wall/vegetation...), è un falso positivo -> scarta
            if dynamic and self.use_semantic_gate and self._sem_is_static(t):
                dynamic = False
                n_sem += 1
            # coasting a finestra corta: una traccia in coasting da troppi frame non
            # viene più mostrata (niente fantasma lungo dopo una svolta), ma resta
            # viva internamente fino a max_age per ri-associarsi se il pedone riappare.
            if dynamic and t.time_since_update > self.coast_publish_frames:
                dynamic = False
            if (not dynamic) and (not self.publish_static):
                continue
            px, py = t.kf.pos
            vx, vy = t.kf.vel
            col = color_for(t.id) if dynamic else ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.4)
            base = t.id * 3

            cyl = self._mk(stamp, 'tracks', base, Marker.CYLINDER,
                           px, py, self.marker_z, col)
            cyl.scale.x = 0.4
            cyl.scale.y = 0.4
            cyl.scale.z = 1.0
            ma.markers.append(cyl)

            if dynamic:
                arr = self._mk(stamp, 'tracks', base + 1, Marker.ARROW,
                               0.0, 0.0, 0.0, col)
                arr.scale.x = 0.06   # diametro asta
                arr.scale.y = 0.12   # diametro testa
                arr.scale.z = 0.18   # lunghezza testa
                arr.points = [
                    Point(x=float(px), y=float(py), z=float(self.marker_z)),
                    Point(x=float(px + vx * self.arrow_time_scale),
                          y=float(py + vy * self.arrow_time_scale),
                          z=float(self.marker_z)),
                ]
                ma.markers.append(arr)

            label = CITYSCAPES_NAMES.get(t.sem_class, '?')
            txt = self._mk(stamp, 'tracks', base + 2, Marker.TEXT_VIEW_FACING,
                           px, py, self.marker_z + 1.0,
                           ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9))
            txt.scale.z = 0.3
            txt.text = f"{t.id} {label} v={t.kf.speed:.2f}"
            ma.markers.append(txt)

        self.pub_mrk.publish(ma)
        self._n_suppressed = n_suppressed
        self._n_sem_suppressed = n_sem

    def _mk(self, stamp, ns, mid, mtype, x, y, z, color):
        mk = Marker()
        mk.header.frame_id = self.tracking_frame
        mk.header.stamp = stamp
        mk.ns = ns
        mk.id = int(mid)
        mk.type = mtype
        mk.action = Marker.ADD
        mk.pose.position.x = float(x)
        mk.pose.position.y = float(y)
        mk.pose.position.z = float(z)
        mk.pose.orientation.w = 1.0
        mk.color = color
        mk.frame_locked = True
        mk.lifetime = Duration(seconds=0.4).to_msg()
        return mk


def main(args=None):
    rclpy.init(args=args)
    node = Lidar3DTracker()
    # MultiThreadedExecutor: le callback camera (gruppo cb_side) girano su un thread
    # diverso da on_dets (cb_track) e non affamano il feed TF -> niente slittamento.
    executor = MultiThreadedExecutor(num_threads=4)
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