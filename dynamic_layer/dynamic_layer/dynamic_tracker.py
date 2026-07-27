#!/usr/bin/env python3
"""
dynamic_tracker.py -- Rilevamento e anticipazione di pedoni con CAMERA + DEPTH.

APPROCCIO (solo camera)
-----------------------
La maschera semantica "persona" dice QUALI pixel sono un pedone; la depth dice
DOVE (distanza); il centroide immagine dà la direzione, molto piu' preciso del
LiDAR. Retroproiezione con gli intrinseci -> posizione 3D nel mondo. Un muro non
e' mai etichettato "persona", quindi non genera mai una traccia: il problema del
centroide che slitta lungo la parete non esiste, perche' non si clusterizza piu'
geometricamente il LiDAR.

Questo tracker copre solo il campo visivo della camera (~72 gradi). E' una
scelta deliberata: nel nostro scenario i pedoni che contano arrivano da davanti,
e il ramo LiDAR laterale (versione precedente) introduceva cluster fantasma --
il centroide che slitta su muri e ombre di occlusione mentre il robot si muove.
Per la copertura laterale a 360 gradi esiste un tracker separato basato sul
LiDAR (vedi lidar_semantic_tracker.py).

PIPELINE
--------
1. componenti connesse sulla maschera "persona" -> regioni grezze (le gambe
   possono essere due regioni: si fondono dopo, nel mondo);
2. ogni regione -> distanza (mediana depth, o i pochi pixel validi se rada) +
   direzione (centroide immagine) -> posizione 3D;
3. FUSIONE NEL MONDO: due detection piu' vicine di una soglia in metri e a
   profondita' simile sono la stessa persona (le sue gambe) -> una detection;
4. Kalman a velocita' costante per traccia; promozione a dinamico a isteresi;
5. cono di costo predittivo, con direzione dallo SPOSTAMENTO reale su finestra
   (non dalla velocita' istantanea, che per una traccia nuova punta a caso).

RIFERIMENTI
-----------
- Munaro & Menegatti, "Fast RGB-D people tracking for service robots", 2014
  (sub-clustering su profondita' + rilevatore + Kalman/UKF): il nostro impianto,
  con la segmentazione semantica al posto del rilevatore di teste.
- Linder et al., "Deep 3D perception of people and their mobility aids", RAS 2019
  (people detector RGB-D open source, tracker congiunto classe+posizione+velocita').
"""

import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from dynamic_tracker_msgs.msg import Track, TrackArray

from cv_bridge import CvBridge
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


class SemanticDynamicTracker(Node):

    def __init__(self):
        super().__init__('dynamic_tracker')

        # ---------------- I/O e frame ----------------
        self.declare_parameter('seg_topic', '/semantic/segmentation')
        self.declare_parameter('depth_topic', '/camera/depth_image')
        # affidabilita' della sottoscrizione depth: 'best_effort' (default sensori)
        # o 'reliable'. Se il ponte Gazebo pubblica RELIABLE e non arriva nulla,
        # prova a mettere 'reliable' qui.
        self.declare_parameter('depth_reliability', 'best_effort')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('output_topic', '/dynamic_cost')
        self.declare_parameter('tracks_topic', '/dynamic_tracks_state')
        self.declare_parameter('markers_topic', '/dynamic_tracks')
        self.declare_parameter('debug_points_topic', '/dynamic_tracker/semantic_points')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('robot_frame', 'base_link')

        # ---------------- fusione semantica ----------------
        self.declare_parameter('dynamic_classes', [11, 12, 17, 18])
        # quanti punti del cluster devono essere inquadrati per poter GIUDICARE.
        # Sotto questa soglia lo stato e' IGNOTO e si ricade sulla geometria.
        # true  = pubblica solo tracce confermate dalla camera (nessun fallback)
        # false = pubblica anche le IGNOTE (copertura 360), mai le RESPINTE
        self.declare_parameter('max_seg_age_s', 0.5)

        # ---------------- griglia rolling e cono ----------------
        self.declare_parameter('size_m', 8.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('rate_hz', 10.0)
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
        self.declare_parameter('gate_ang_vel', 1.0)
        # --- filtro di RILEVANZA (tempo al massimo avvicinamento) ---
        # Un dinamico merita un cono solo se, mantenendo le velocita' attuali,
        # arrivera' VICINO al robot ENTRO l'orizzonte. Cosi' cade tutto cio' che
        # sta dietro o di lato e non incrocia: muri, aiuole, auto parcheggiate,
        # e il pedone che segue il robot senza raggiungerlo.
        # Rilevanza: si disegna il cono solo per i dinamici nel SETTORE FRONTALE.
        # Il robot non ha retromarcia (vx_min=0), quindi cio' che sta dietro non
        # e' un pericolo di collisione. Non si usa piu' il criterio CPA (punto di
        # massimo avvicinamento): scartava chi ci precede nella stessa direzione,
        # che invece vogliamo anticipare.
        self.declare_parameter('relevance_enabled', True)
        # Settore ANGOLARE frontale. 180 = semipiano anteriore (davanti + lati);
        # 72 = solo cio' che la camera inquadra; 360 = disattivato (tutto).
        self.declare_parameter('cone_sector_deg', 180.0)
        self.declare_parameter('min_region_px', 40)      # pixel minimi per una persona
        self.declare_parameter('min_valid_depth_px', 10) # pixel depth per una stima "piena"
        # FUSIONE NEL MONDO: due detection piu' vicine di questa distanza (in
        # metri) e a profondita' simile sono la stessa persona -- tipicamente le
        # sue due gambe. Invariante alla distanza, a differenza della chiusura in
        # pixel. 0.4 m unisce le gambe senza fondere due persone affiancate.
        self.declare_parameter('person_merge_dist_m', 0.4)
        self.declare_parameter('person_merge_depth_m', 0.6)
        self.declare_parameter('depth_min_m', 0.3)
        self.declare_parameter('depth_max_m', 8.0)
        # Direzione del cono dallo spostamento reale su una finestra di N frame,
        # non dalla velocita' istantanea (che per una traccia nuova puo' puntare
        # all'indietro). Il cono si disegna solo se lo spostamento netto supera
        # cone_min_disp: cosi' non parte mai storto per poi girarsi.
        self.declare_parameter('cone_dir_window', 6)      # frame nella finestra
        self.declare_parameter('cone_min_disp_m', 0.15)   # spostamento minimo per disegnare
        # Campo visivo della camera: il ramo LiDAR tiene solo cio' che sta FUORI
        # da questo cono (piu' un margine), perche' il davanti e' compito della
        # camera. Evita cluster fantasma nell'ombra dietro un pedone.

        # ---------------- clustering ----------------

        # ---------------- tracking ----------------
        self.declare_parameter('gate_dist_m', 0.7)
        self.declare_parameter('min_obs', 3)
        self.declare_parameter('track_timeout_s', 0.6)
        self.declare_parameter('min_speed', 0.40)
        # i laterali (LiDAR, senza classe) richiedono N volte la soglia frontale
        # per essere promossi: piu' cauti, cosi' i muri che ballano non passano
        self.declare_parameter('promote_frames', 4)
        self.declare_parameter('demote_frames', 8)

        # ---------------- Kalman ----------------
        self.declare_parameter('kf_sigma_a', 0.6)
        self.declare_parameter('kf_sigma_z', 0.12)
        self.declare_parameter('kf_v_init', 1.0)
        self.declare_parameter('kf_min_snr', 1.0)

        gp = self.get_parameter
        self.seg_topic = gp('seg_topic').value
        self.caminfo_topic = gp('camera_info_topic').value
        self.world_frame = gp('world_frame').value
        self.robot_frame = gp('robot_frame').value

        self.dynamic_classes = [int(c) for c in gp('dynamic_classes').value]
        self.max_seg_age = float(gp('max_seg_age_s').value)

        self.size_m = float(gp('size_m').value)
        self.res = float(gp('resolution').value)
        self.n = int(self.size_m / self.res)
        self.rate_hz = float(gp('rate_hz').value)
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
        self.gate_ang_vel = float(gp('gate_ang_vel').value)
        self.relevance_enabled = bool(gp('relevance_enabled').value)
        self.person_merge_dist = float(gp('person_merge_dist_m').value)
        self.person_merge_depth = float(gp('person_merge_depth_m').value)
        self.cone_sector = float(gp('cone_sector_deg').value)
        self.min_region_px = int(gp('min_region_px').value)
        self.min_valid_depth_px = int(gp('min_valid_depth_px').value)
        self.depth_min = float(gp('depth_min_m').value)
        self.depth_max = float(gp('depth_max_m').value)
        self.cone_dir_window = int(gp('cone_dir_window').value)
        self.cone_min_disp = float(gp('cone_min_disp_m').value)

        self.gate_dist = float(gp('gate_dist_m').value)
        self.min_obs = int(gp('min_obs').value)
        self.track_timeout = float(gp('track_timeout_s').value)
        self.min_speed = float(gp('min_speed').value)
        self.promote_frames = int(gp('promote_frames').value)
        self.demote_frames = int(gp('demote_frames').value)

        self.kf_sigma_a = float(gp('kf_sigma_a').value)
        self.kf_sigma_z = float(gp('kf_sigma_z').value)
        self.kf_v_init = float(gp('kf_v_init').value)
        self.kf_min_snr = float(gp('kf_min_snr').value)

        # ---------------- stato ----------------
        self.bridge = CvBridge()
        self.seg_img = None
        self.seg_stamp = None
        self.depth_img = None
        self.depth_stamp = None
        self._depth_logged = False
        self.K = None
        self.cam_w = None
        self.cam_h = None
        self.cam_frame = None
        self.tracks = []
        self.next_id = 0
        self.robot_ang = 0.0
        self.robot_xy = (0.0, 0.0)
        self.robot_yaw = 0.0
        self.arrows = []
        self.dyn_pts_w = np.empty((0, 2))
        self.diag = dict(n_scan=0, n_fov=0, n_dyn=0, motivo='ok',
                         conf=0, rej=0, unk=0)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, self.seg_topic, self.seg_cb, qos)
        depth_rel = (ReliabilityPolicy.RELIABLE
                     if str(gp('depth_reliability').value) == 'reliable'
                     else ReliabilityPolicy.BEST_EFFORT)
        qos_depth = QoSProfile(depth=1, reliability=depth_rel,
                               history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, gp('depth_topic').value, self.depth_cb, qos_depth)
        self.get_logger().info(
            f"Sottoscritto depth su '{gp('depth_topic').value}' "
            f"({str(gp('depth_reliability').value)}). Se resta 'nessuna depth', "
            f"il topic o la QoS non combaciano: ros2 topic info <topic> -v")
        self.create_subscription(CameraInfo, self.caminfo_topic, self.caminfo_cb, 10)

        qos_grid = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL,
                              history=HistoryPolicy.KEEP_LAST)
        self.cost_pub = self.create_publisher(OccupancyGrid, gp('output_topic').value, qos_grid)
        self.tracks_pub = self.create_publisher(TrackArray, gp('tracks_topic').value, 10)
        self.marker_pub = self.create_publisher(MarkerArray, gp('markers_topic').value, 10)
        self.dbg_pub = self.create_publisher(MarkerArray, gp('debug_points_topic').value, 10)

        self.create_timer(1.0 / self.rate_hz, self.update)
        self.create_timer(5.0, self.log_diag)

        self.get_logger().info(
            f'Tracker semantico avviato.  classi dinamiche={self.dynamic_classes}  '
            'solo camera (depth + segmentazione).')

    # ------------------------------------------------------------------ IO

    def seg_cb(self, msg):
        try:
            self.seg_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.seg_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().warn(f'segmentazione non leggibile: {e}', once=True)

    def depth_cb(self, msg):
        try:
            self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_stamp = msg.header.stamp
            # diagnostica una tantum: encoding e range effettivo dei valori.
            # In Gazebo la depth e' 32FC1 in metri, ma spesso lo sfondo e' inf
            # o 0, e certe versioni la pubblicano in un frame diverso.
            if not self._depth_logged:
                self._depth_logged = True
                d = np.asarray(self.depth_img, dtype=np.float32)
                finite = d[np.isfinite(d)]
                self.get_logger().info(
                    f'Depth ricevuta: encoding={msg.encoding} shape={self.depth_img.shape} '
                    f'dtype={self.depth_img.dtype} | valori finiti: '
                    f'{finite.size}/{d.size}'
                    + (f' range [{finite.min():.2f}, {finite.max():.2f}]' if finite.size else ' (NESSUNO)'))
        except Exception as e:
            self.get_logger().warn(f'depth non leggibile: {e}', once=True)

    def caminfo_cb(self, msg):
        k = np.array(msg.k, dtype=float).reshape(3, 3)
        if k[0, 0] <= 0.0:
            return
        self.K = k
        self.cam_w, self.cam_h = int(msg.width), int(msg.height)
        self.cam_frame = msg.header.frame_id
        fov = 2.0 * np.degrees(np.arctan(self.cam_w / 2.0 / k[0, 0]))
        self.get_logger().info(
            f'Intrinseci: {self.cam_w}x{self.cam_h} fx={k[0,0]:.1f} '
            f'frame={self.cam_frame} -> campo visivo {fov:.0f} gradi '
            f'({fov/360*100:.0f}% dello scan). Fuori da qui la classe e\' IGNOTA '
            f'e si ricade sulla geometria.', once=True)

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

    # ------------------------------------------------------- fusione semantica
    def _kf_predict(self, tr, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                      [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = self.kf_sigma_a ** 2
        dt2 = dt * dt; dt3 = dt2 * dt; dt4 = dt3 * dt
        Q = q * np.array([[dt4 / 4, 0, dt3 / 2, 0], [0, dt4 / 4, 0, dt3 / 2],
                          [dt3 / 2, 0, dt2, 0], [0, dt3 / 2, 0, dt2]], dtype=float)
        tr['s'] = F @ tr['s']
        tr['P'] = F @ tr['P'] @ F.T + Q

    def _kf_update(self, tr, z):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = (self.kf_sigma_z ** 2) * np.eye(2)
        y = z - H @ tr['s']
        S = H @ tr['P'] @ H.T + R
        K = tr['P'] @ H.T @ np.linalg.inv(S)
        tr['s'] = tr['s'] + K @ y
        tr['P'] = (np.eye(4) - K @ H) @ tr['P']

    def track(self, dets, now_s):
        for tr in self.tracks:
            dt = max(1e-3, min(now_s - tr['t'], 1.0))
            self._kf_predict(tr, dt)
            tr['t'] = now_s

        used = set()
        for tr in self.tracks:
            best, bestd = -1, self.gate_dist
            for j, d in enumerate(dets):
                if j in used:
                    continue
                dist = np.hypot(d[0] - tr['s'][0], d[1] - tr['s'][1])
                if dist < bestd:
                    best, bestd = j, dist
            if best >= 0:
                self._kf_update(tr, np.array(dets[best], dtype=float))
                tr['t_seen'] = now_s
                tr['n'] = min(tr['n'] + 1, 999)
                used.add(best)

        for j, d in enumerate(dets):
            if j in used:
                continue
            s = np.array([d[0], d[1], 0.0, 0.0], dtype=float)
            P = np.diag([self.kf_sigma_z ** 2, self.kf_sigma_z ** 2,
                         self.kf_v_init ** 2, self.kf_v_init ** 2])
            self.tracks.append(dict(id=self.next_id, s=s, P=P, n=1,
                                    t=now_s, t_seen=now_s, is_dynamic=False,
                                    mov_count=0, still_count=0))
            self.next_id += 1

        self.tracks = [t for t in self.tracks
                       if now_s - t['t_seen'] <= self.track_timeout]

        for tr in self.tracks:
            tr['x'], tr['y'] = float(tr['s'][0]), float(tr['s'][1])
            tr['vx'], tr['vy'] = float(tr['s'][2]), float(tr['s'][3])
            tr['sp'] = float(np.sqrt(max(tr['P'][0, 0] + tr['P'][1, 1], 0.0)))
            tr['sv'] = float(np.sqrt(max(tr['P'][2, 2] + tr['P'][3, 3], 0.0)))

            # storico posizioni per la direzione del cono (finestra scorrevole)
            hist = tr.setdefault('pos_hist', deque(maxlen=self.cone_dir_window))
            hist.append((tr['x'], tr['y']))

            # promozione a dinamico: tutte le tracce sono persone (la camera le
            # ha classificate), quindi conta solo che si muovano in modo stabile.
            speed = np.hypot(tr['vx'], tr['vy'])
            moving = (speed >= self.min_speed) and \
                     (self.kf_min_snr <= 0.0 or speed >= self.kf_min_snr * tr['sv'])
            if moving:
                tr['mov_count'] += 1
                tr['still_count'] = 0
            else:
                tr['still_count'] += 1
                tr['mov_count'] = 0

            if (not tr['is_dynamic']) and tr['n'] >= self.min_obs \
                    and tr['mov_count'] >= self.promote_frames:
                tr['is_dynamic'] = True
            if tr['is_dynamic'] and tr['still_count'] >= self.demote_frames:
                tr['is_dynamic'] = False

    def is_relevant(self, tr):
        """Vale la pena disegnare il cono per questo dinamico?

        Regola unica: e' nel SETTORE FRONTALE del robot (cone_sector, di
        default 180 = davanti + lati). Chiunque sia davanti puo' essere sulla
        rotta -- che si avvicini o che vada nella nostra stessa direzione, se
        cammina piu' lento lo raggiungiamo -- quindi merita l'anticipazione.

        Si scarta SOLO cio' che sta DIETRO: il robot non ha retromarcia
        (vx_min=0), non puo' collidere all'indietro, e un pedone che arriva da
        dietro si scansa da solo.

        NB: qui NON si usa piu' il criterio del punto di massimo avvicinamento
        (CPA). Scartava chi non incrociava la nostra traiettoria, ma cosi'
        toglieva il cono a un pedone che ci precede nella stessa direzione --
        proprio quello che vogliamo anticipare per non tamponarlo. Un cono in
        piu' su chi si allontana e' spreco innocuo; un cono in meno su chi ci
        precede e' un rischio.
        """
        if not self.relevance_enabled:
            return True
        px = tr['x'] - self.robot_xy[0]
        py = tr['y'] - self.robot_xy[1]
        if self.cone_sector >= 360.0:
            return True
        rel = np.arctan2(py, px) - self.robot_yaw
        rel = (rel + np.pi) % (2 * np.pi) - np.pi          # in [-pi, pi]
        return abs(rel) <= np.radians(self.cone_sector) / 2.0

    # ------------------------------------------------------------- ciclo
    @staticmethod
    def _stamp_s(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    @staticmethod
    def _connected_components(mask, min_px):
        """Etichettatura delle componenti connesse a 4-vicini, senza dipendenze.

        Ritorna (labels, n). Usa cv2 se disponibile (piu' veloce), altrimenti
        ricade su un flood-fill iterativo in numpy.
        """
        try:
            import cv2
            n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
            return labels, n - 1        # cv2 conta lo sfondo come label 0
        except Exception:
            pass

        labels = np.zeros(mask.shape, dtype=np.int32)
        cur = 0
        H, W = mask.shape
        stack = []
        for i0 in range(H):
            for j0 in range(W):
                if not mask[i0, j0] or labels[i0, j0]:
                    continue
                cur += 1
                stack.append((i0, j0))
                labels[i0, j0] = cur
                while stack:
                    i, j = stack.pop()
                    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < H and 0 <= nj < W and mask[ni, nj] and not labels[ni, nj]:
                            labels[ni, nj] = cur
                            stack.append((ni, nj))
        return labels, cur


    def detect_people_camera(self, now_s):
        """Rileva le persone dalla coppia (segmentazione, depth).

        Pipeline pulita, in tre passi:
          1. componenti connesse sulla maschera "persona" -> regioni grezze
             (le gambe possono essere due regioni separate: va bene, le fondiamo
             dopo, nel mondo);
          2. per ogni regione: distanza dalla depth (mediana dei pixel validi,
             o i pochi che ci sono se la depth e' rada) + direzione dal centroide
             immagine -> posizione 3D nel mondo;
          3. FUSIONE NEL MONDO: due detection piu' vicine di person_merge_dist_m
             sono la stessa persona (tipicamente le due gambe) e si fondono.
             Questo e' invariante alla distanza -- lavora in metri, non in pixel
             -- ed e' l'unico posto dove si decide "una persona o due".

        Ritorna la lista di (x, y) nel mondo, una per persona.
        """
        dets = []
        self.diag['n_people'] = 0
        self.diag['n_valid_depth'] = 0
        self.diag['n_sparse_depth'] = 0
        self.diag['n_regions'] = 0
        self.diag['n_small'] = 0
        self.diag['n_nodepth'] = 0

        if self.seg_img is None:
            self.diag['motivo'] = 'nessuna segmentazione'
            return dets
        if self.depth_img is None:
            self.diag['motivo'] = 'nessuna depth'
            return dets
        if self.K is None or self.cam_frame is None:
            self.diag['motivo'] = 'nessun camera_info'
            return dets

        # segmentazione e depth ragionevolmente sincronizzate
        if self.seg_stamp is not None and self.depth_stamp is not None:
            if abs(self._stamp_s(self.seg_stamp) - self._stamp_s(self.depth_stamp)) > self.max_seg_age:
                self.diag['motivo'] = 'seg/depth desync'
                return dets
        self.diag['motivo'] = 'ok'

        seg = self.seg_img
        depth = self.depth_img
        H, W = seg.shape[:2]

        # allineo la depth alla risoluzione della segmentazione (nearest, niente
        # interpolazione sui bordi di profondita' = niente distanze fantasma)
        if depth.shape[:2] != (H, W):
            iy = np.linspace(0, depth.shape[0] - 1, H).astype(int)
            ix = np.linspace(0, depth.shape[1] - 1, W).astype(int)
            depth = depth[np.ix_(iy, ix)]
        depth_m = (depth.astype(np.float32) * 0.001) if depth.dtype == np.uint16 \
            else depth.astype(np.float32)

        dyn_mask = np.isin(seg, self.dynamic_classes)
        self.diag['n_dyn_px'] = int(dyn_mask.sum())
        if not dyn_mask.any():
            self.diag['motivo'] = 'nessun pixel dinamico'
            return dets

        # PASSO 1: componenti connesse grezze (nessuna chiusura in pixel)
        labels, n_lab = self._connected_components(dyn_mask, self.min_region_px)
        self.diag['n_regions'] = int(n_lab)

        Tw = self.lookup(self.world_frame, self.cam_frame)
        if Tw is None:
            self.diag['motivo'] = f'manca TF {self.world_frame}<-{self.cam_frame}'
            return dets
        M = transform_to_matrix(Tw)

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        sx = W / float(self.cam_w) if self.cam_w else 1.0
        sy = H / float(self.cam_h) if self.cam_h else 1.0

        # PASSO 2: ogni regione -> una posizione 3D nel mondo
        raw = []                             # (x, y, Z) nel mondo, prima della fusione
        for lab in range(1, n_lab + 1):
            ys, xs = np.where(labels == lab)
            if len(xs) < self.min_region_px:
                self.diag['n_small'] += 1
                continue
            self.diag['n_people'] += 1

            d = depth_m[ys, xs]
            valid = np.isfinite(d) & (d > self.depth_min) & (d < self.depth_max)
            n_valid = int(np.count_nonzero(valid))
            # depth rada: si tiene la persona con qualunque pixel valido; solo a
            # zero pixel non sappiamo la distanza e la saltiamo
            if n_valid == 0:
                self.diag['n_nodepth'] += 1
                continue
            if n_valid < self.min_valid_depth_px:
                self.diag['n_sparse_depth'] += 1
            self.diag['n_valid_depth'] += 1

            Z = float(np.median(d[valid]))
            u = float(np.mean(xs)) / sx
            v = float(np.mean(ys)) / sy
            Xo = (u - cx) * Z / fx
            Yo = (v - cy) * Z / fy
            pw = M @ np.array([Xo, Yo, Z, 1.0])
            raw.append((float(pw[0]), float(pw[1]), Z))

        # PASSO 3: fusione nel mondo. Due detection piu' vicine di
        # person_merge_dist_m (e a distanza-camera simile) sono la stessa
        # persona: le sue due gambe. Le fondo nel loro punto medio. Invariante
        # alla distanza, a differenza della vecchia chiusura in pixel.
        dets = self._merge_world(raw)
        self.dyn_pts_w = np.array([(x, y) for x, y in dets]) if dets else np.empty((0, 2))
        return dets

    def _merge_world(self, raw):
        """Fonde detection vicine nel mondo (union-find su soglia di distanza).
        raw: lista di (x, y, Z). Ritorna lista di (x, y) fusi."""
        n = len(raw)
        if n == 0:
            return []
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(raw[i][0] - raw[j][0], raw[i][1] - raw[j][1])
                dz = abs(raw[i][2] - raw[j][2])
                # stessa persona se vicine nel piano E a profondita' simile
                # (due persone alla stessa distanza ma affiancate distano > soglia;
                #  due gambe distano ~0.2-0.3 m e hanno la stessa Z)
                if d < self.person_merge_dist and dz < self.person_merge_depth:
                    parent[find(i)] = find(j)

        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        out = []
        for idxs in groups.values():
            xs = [raw[k][0] for k in idxs]
            ys = [raw[k][1] for k in idxs]
            out.append((float(np.mean(xs)), float(np.mean(ys))))
        return out

    def update(self):
        now_s = self.get_clock().now().nanoseconds * 1e-9

        # ------ RILEVAMENTO: SOLO CAMERA ------
        # La maschera semantica trova le persone, la depth le localizza. Niente
        # ramo LiDAR: nel nostro scenario i pedoni che contano arrivano da
        # davanti (nel campo visivo della camera), e il LiDAR laterale
        # introduceva solo cluster fantasma (centroide che slitta su muri e
        # ombre di occlusione). Le detection sono tutte persone confermate.
        dets = self.detect_people_camera(now_s)
        self.track(dets, now_s)

        Tr = self.lookup(self.world_frame, self.robot_frame)
        if Tr is not None:
            M = transform_to_matrix(Tr)
            rx, ry = M[0, 3], M[1, 3]
            self.robot_xy = (rx, ry)
            self.robot_yaw = np.arctan2(M[1, 0], M[0, 0])
            ox, oy = rx - self.size_m / 2.0, ry - self.size_m / 2.0
            cost = self.build_cost(ox, oy)
            self.publish_cost(cost, ox, oy)

        self.publish_tracks()
        self.publish_markers()
        self.publish_debug_points()

    # ------------------------------------------------------------- cono
    def build_cost(self, ox, oy):
        cost = np.zeros((self.n, self.n), dtype=np.uint8)
        self.arrows = []
        spinning = self.robot_ang > self.gate_ang_vel
        for tr in self.tracks:
            if tr['n'] < self.min_obs or not tr.get('is_dynamic', False):
                continue
            # il nucleo si marca comunque (e' un ostacolo fisico dove si trova)
            if self.mark_obstacle:
                self._stamp(cost, tr['x'], tr['y'], ox, oy,
                            self.nucleus_radius, self.max_cost)
            if not self.enable_cone or spinning or not self.is_relevant(tr):
                continue
            # DIREZIONE DEL CONO dallo spostamento reale su una finestra, non
            # dalla velocita' istantanea del Kalman: una traccia appena nata ha
            # velocita' rumorosa che puo' puntare all'indietro. Il cono si
            # disegna SOLO quando c'e' spostamento netto sufficiente a dare una
            # direzione affidabile -- cosi' non parte mai storto per poi girarsi.
            direction = self._cone_direction(tr)
            if direction is None:
                continue
            speed = np.hypot(tr['vx'], tr['vy'])
            self._paint_cone(cost, tr, ox, oy, speed, direction)
            self.arrows.append((tr['x'], tr['y'], direction[0] * speed,
                                direction[1] * speed))
        return cost

    def _cone_direction(self, tr):
        """Direzione del cono dallo spostamento netto sulla finestra di storia.
        Ritorna (ux, uy) normalizzato, oppure None se lo spostamento e' troppo
        piccolo perche' la direzione sia affidabile (traccia appena nata o
        quasi ferma). Robusto al backwards-flip della velocita' istantanea."""
        hist = tr.get('pos_hist')
        if hist is None or len(hist) < 2:
            return None
        x0, y0 = hist[0]
        x1, y1 = hist[-1]
        dx, dy = x1 - x0, y1 - y0
        disp = np.hypot(dx, dy)
        if disp < self.cone_min_disp:
            return None                      # non si e' mosso abbastanza: niente cono
        return (dx / disp, dy / disp)

    def _stamp(self, cost, wx, wy, ox, oy, radius_m, val):
        cxi = int((wx - ox) / self.res); cyi = int((wy - oy) / self.res)
        rr = int(radius_m / self.res)
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if dx * dx + dy * dy <= rr * rr:
                    gx, gy = cxi + dx, cyi + dy
                    if 0 <= gx < self.n and 0 <= gy < self.n and cost[gy, gx] < val:
                        cost[gy, gx] = val

    def _paint_cone(self, cost, tr, ox, oy, speed, direction):
        ux, uy = direction
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

    # ------------------------------------------------------------- output
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

    def publish_tracks(self):
        msg = TrackArray()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for tr in self.tracks:
            if not tr.get('is_dynamic', False):
                continue
            t = Track()
            t.id = int(tr['id'])
            t.x = float(tr['x']); t.y = float(tr['y'])
            t.vx = float(tr['vx']); t.vy = float(tr['vy'])
            t.pos_std = float(tr.get('sp', 0.0))
            t.vel_std = float(tr.get('sv', 0.0))
            msg.tracks.append(t)
        self.tracks_pub.publish(msg)

    def publish_markers(self):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        now = self.get_clock().now().to_msg()
        for tr in self.tracks:
            if not tr.get('is_dynamic', False):
                continue
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = now
            m.ns = 'tracks'
            m.id = int(tr['id'])
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.06; m.scale.y = 0.12; m.scale.z = 0.12
            m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.2, 0.9  # verde = persona
            m.points = [Point(x=tr['x'], y=tr['y'], z=0.1),
                        Point(x=tr['x'] + tr['vx'], y=tr['y'] + tr['vy'], z=0.1)]
            m.lifetime.sec = 0
            m.lifetime.nanosec = 300000000
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def publish_debug_points(self):
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = self.world_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'semantic_points'
        m.id = 0
        m.type = Marker.POINTS
        m.action = Marker.ADD
        m.scale.x = 0.08; m.scale.y = 0.08
        m.color.a = 1.0
        m.lifetime.sec = 0
        m.lifetime.nanosec = 500000000
        for p in self.dyn_pts_w:
            m.points.append(Point(x=float(p[0]), y=float(p[1]), z=0.05))
            m.colors.append(ColorRGBA(r=0.1, g=1.0, b=0.2, a=1.0))
        arr.markers.append(m)
        self.dbg_pub.publish(arr)

    def log_diag(self):
        d = self.diag
        n_acc = sum(1 for t in self.tracks if t.get('is_dynamic'))
        self.get_logger().info(
            f"pixel_dinamici={d.get('n_dyn_px',0)} regioni={d.get('n_regions',0)} "
            f"persone={d.get('n_people',0)} con_depth={d.get('n_valid_depth',0)} "
            f"(di cui rada={d.get('n_sparse_depth',0)}; scartate: piccole={d.get('n_small',0)} "
            f"senza_depth={d.get('n_nodepth',0)}) | "
            f"LATERALI(lidar)={d.get('unk',0)} | "
            f"tracce={len(self.tracks)} usate={n_acc} | {d['motivo']}")


def main():
    rclpy.init()
    node = SemanticDynamicTracker()
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