#!/usr/bin/env python3
"""
dynamic_tracker.py -- Rilevamento e anticipazione di ostacoli dinamici con
                      FUSIONE SEMANTICA camera + LiDAR 2D.

IDEA
----
La camera dice CHE COSA, il LiDAR dice DOVE. Ogni punto dello scan viene
proiettato nell'immagine di segmentazione e ne eredita la classe. Un muro non e'
mai etichettato "persona", quindi non genera mai una traccia: il problema del
centroide che slitta lungo la parete sparisce a monte invece di essere filtrato
a valle.

IL PUNTO DELICATO: IL CAMPO VISIVO
----------------------------------
Il LiDAR vede 360 gradi, la camera molto meno (con 320x240 e fx=222 sono ~72
gradi, cioe' il 20% dello scan). Buttare tutto cio' che la camera non inquadra
significa perdere l'80% dell'informazione e far morire le tracce appena il
pedone esce dall'inquadratura. Per questo la classificazione ha TRE stati e non
due:

  CONFERMATO  il cluster e' inquadrato e i suoi punti cadono su una classe
              dinamica -> e' una persona/bici, si traccia.
  RESPINTO    il cluster e' inquadrato ma i suoi punti cadono su edificio,
              vegetazione, marciapiede... -> NON e' un dinamico, si scarta.
              *** E' QUI CHE MUOIONO I MURI. ***
  IGNOTO      il cluster non e' inquadrato (fuori campo visivo, dietro il
              robot): non possiamo sapere. Si ricade sulla geometria come
              faceva il tracker vecchio.

Cosi' si tiene la copertura a 360 gradi e si eliminano i falsi positivi che la
camera PUO' vedere: strettamente meglio del tracker puramente geometrico, senza
rinunciare a nulla.

MEMORIA SEMANTICA
-----------------
Una traccia confermata dalla camera resta "persona" anche quando esce dal campo
visivo: il punteggio 'sem_score' sale su conferma, scende su rifiuto, e resta
fermo quando non c'e' informazione. Senza questo, un pedone superato dal robot
verrebbe dimenticato nell'istante in cui esce dall'inquadratura.

RIFERIMENTI
-----------
- Jia, Hermans, Leibe, "Self-Supervised Person Detection in 2D Range Data using
  a Calibrated Camera", ICRA 2021 (proiezione a frustum).
- "Robot Object Detection and Tracking Based on Image-Point Cloud Instance
  Matching", Sensors 2026 (maschere proiettate + clustering + Kalman con gating).
- "Semantic Fusion Algorithm of 2D LiDAR and Camera Based on Contour and
  Inverse Projection" (caso specifico del LiDAR 2D).
- "Human Detection from a Mobile Robot Using Fusion of Laser and Vision
  Information", Sensors 2013 (cluster laser proiettati sull'immagine).
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan, Image, CameraInfo
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from dynamic_tracker_msgs.msg import Track, TrackArray

from cv_bridge import CvBridge
import tf2_ros

# stati della classificazione semantica di un cluster
SEM_UNKNOWN = 0
SEM_CONFIRMED = 1
SEM_REJECTED = -1


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
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('seg_topic', '/semantic/segmentation')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        # affidabilita' della sottoscrizione depth: 'best_effort' (default sensori)
        # o 'reliable'. Se il ponte Gazebo pubblica RELIABLE e non arriva nulla,
        # prova a mettere 'reliable' qui.
        self.declare_parameter('depth_reliability', 'best_effort')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('output_topic', '/dynamic_cost')
        self.declare_parameter('tracks_topic', '/dynamic_tracks_state')
        self.declare_parameter('markers_topic', '/dynamic_tracks')
        self.declare_parameter('debug_points_topic', '/dynamic_tracker/semantic_points')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('robot_frame', 'base_link')

        # ---------------- fusione semantica ----------------
        self.declare_parameter('dynamic_classes', [11, 12, 17, 18])
        self.declare_parameter('pixel_dilation', 2)
        self.declare_parameter('min_semantic_frac', 0.4)
        # quanti punti del cluster devono essere inquadrati per poter GIUDICARE.
        # Sotto questa soglia lo stato e' IGNOTO e si ricade sulla geometria.
        self.declare_parameter('min_fov_points', 2)
        # true  = pubblica solo tracce confermate dalla camera (nessun fallback)
        # false = pubblica anche le IGNOTE (copertura 360), mai le RESPINTE
        self.declare_parameter('require_semantic', False)
        self.declare_parameter('sem_score_max', 5)
        self.declare_parameter('max_seg_age_s', 0.5)

        # ---------------- griglia rolling e cono ----------------
        self.declare_parameter('size_m', 8.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('max_range', 8.0)
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
        self.declare_parameter('relevance_enabled', True)
        self.declare_parameter('relevance_horizon_s', 10.0)
        self.declare_parameter('relevance_radius_m', 2.0)
        self.declare_parameter('relevance_always_dist_m', 0.8)
        # Settore ANGOLARE davanti al robot entro cui un dinamico merita un cono.
        # Il robot non va mai in retromarcia (vx_min=0), quindi non puo' collidere
        # con cio' che ha dietro: se un pedone lo raggiunge da dietro, sara' lui a
        # scansarsi. 180 = semipiano anteriore (davanti + laterale).
        # 72 = solo cio' che la camera inquadra davvero.
        self.declare_parameter('cone_sector_deg', 180.0)
        # --- MODALITA' DI RILEVAMENTO ---
        # 'camera' = camera primaria: la maschera semantica trova le persone,
        #            la depth le localizza. Direzione 8x piu' precisa, niente
        #            centroide-tra-le-gambe. Solo entro il campo visivo.
        # 'lidar'  = il vecchio front-end LiDAR con gating semantico (360 gradi).
        # 'fusion' = camera per le persone nel FOV, LiDAR per il resto/dietro.
        self.declare_parameter('detection_mode', 'camera')
        self.declare_parameter('min_region_px', 40)      # pixel minimi per una persona
        self.declare_parameter('min_valid_depth_px', 10) # pixel con depth valida minimi
        # CHIUSURA morfologica per riunire le gambe di una stessa persona quando
        # il busto non e' segmentato (il gap tra le gambe e' ~8-12 px a distanza
        # media). 0 = disattivata. 11 = sicuro senza unire persone diverse.
        self.declare_parameter('leg_merge_px', 11)
        # SALVAGUARDIA: se la chiusura unisce due persone vicine, la regione
        # risultante e' piu' larga del limite e viene rispezzata per colonna.
        self.declare_parameter('split_wide', True)
        self.declare_parameter('max_person_width_m', 0.8)
        self.declare_parameter('depth_min_m', 0.3)
        self.declare_parameter('depth_max_m', 8.0)
        # Direzione del cono lisciata a parte dalla velocita' del Kalman: il
        # centroide di un pedone salta tra le due gambe e fa oscillare il cono.
        self.declare_parameter('cone_dir_ema', 0.3)

        # ---------------- clustering ----------------
        self.declare_parameter('cluster_eps_m', 0.5)
        self.declare_parameter('cluster_min_pts', 2)
        self.declare_parameter('cluster_max_size_m', 1.5)

        # ---------------- tracking ----------------
        self.declare_parameter('gate_dist_m', 0.7)
        self.declare_parameter('min_obs', 3)
        self.declare_parameter('track_timeout_s', 0.6)
        self.declare_parameter('min_speed', 0.40)
        self.declare_parameter('promote_frames', 4)
        self.declare_parameter('demote_frames', 8)

        # ---------------- Kalman ----------------
        self.declare_parameter('kf_sigma_a', 0.6)
        self.declare_parameter('kf_sigma_z', 0.12)
        self.declare_parameter('kf_v_init', 1.0)
        self.declare_parameter('kf_min_snr', 1.0)

        gp = self.get_parameter
        self.scan_topic = gp('scan_topic').value
        self.seg_topic = gp('seg_topic').value
        self.caminfo_topic = gp('camera_info_topic').value
        self.odom_topic = gp('odom_topic').value
        self.world_frame = gp('world_frame').value
        self.robot_frame = gp('robot_frame').value

        self.dynamic_classes = [int(c) for c in gp('dynamic_classes').value]
        self.pixel_dilation = int(gp('pixel_dilation').value)
        self.min_semantic_frac = float(gp('min_semantic_frac').value)
        self.min_fov_points = int(gp('min_fov_points').value)
        self.require_semantic = bool(gp('require_semantic').value)
        self.sem_score_max = int(gp('sem_score_max').value)
        self.max_seg_age = float(gp('max_seg_age_s').value)

        self.size_m = float(gp('size_m').value)
        self.res = float(gp('resolution').value)
        self.n = int(self.size_m / self.res)
        self.max_range = float(gp('max_range').value)
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
        self.relevance_horizon = float(gp('relevance_horizon_s').value)
        self.relevance_radius = float(gp('relevance_radius_m').value)
        self.relevance_always = float(gp('relevance_always_dist_m').value)
        self.cone_sector = float(gp('cone_sector_deg').value)
        self.detection_mode = str(gp('detection_mode').value)
        self.min_region_px = int(gp('min_region_px').value)
        self.min_valid_depth_px = int(gp('min_valid_depth_px').value)
        self.leg_merge_px = int(gp('leg_merge_px').value)
        self.split_wide = bool(gp('split_wide').value)
        self.max_person_width_m = float(gp('max_person_width_m').value)
        self.depth_min = float(gp('depth_min_m').value)
        self.depth_max = float(gp('depth_max_m').value)
        self.cone_dir_ema = float(gp('cone_dir_ema').value)

        self.cluster_eps = float(gp('cluster_eps_m').value)
        self.cluster_min_pts = int(gp('cluster_min_pts').value)
        self.cluster_max_size = float(gp('cluster_max_size_m').value)

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
        self.last_scan = None
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
        self.robot_v_body = (0.0, 0.0)   # velocita' lineare nel frame del robot
        self.robot_v_world = (0.0, 0.0)  # ...ruotata nel frame mondo
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
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos)
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
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos)
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
            f'min_semantic_frac={self.min_semantic_frac}  '
            f'require_semantic={self.require_semantic}')

    # ------------------------------------------------------------------ IO
    def scan_cb(self, msg):
        self.last_scan = msg

    def odom_cb(self, msg):
        self.robot_ang = abs(float(msg.twist.twist.angular.z))
        self.robot_v_body = (float(msg.twist.twist.linear.x),
                             float(msg.twist.twist.linear.y))

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
    def scan_points_laser(self, scan):
        ang = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        r = np.asarray(scan.ranges, dtype=np.float32)
        good = np.isfinite(r) & (r >= scan.range_min) & (r <= scan.range_max) \
            & (r <= self.max_range)
        ang, r = ang[good], r[good]
        if r.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1).astype(np.float32)

    def label_points(self, pts_laser, scan):
        """Per ogni punto: (in_fov, is_dyn).

        in_fov=False significa "non classificabile": fuori inquadratura, dietro
        la camera, oppure mancano segmentazione/intrinseci/TF. In quel caso NON
        si conclude nulla: sara' il cluster a finire in stato IGNOTO.
        """
        n = len(pts_laser)
        in_fov = np.zeros(n, dtype=bool)
        is_dyn = np.zeros(n, dtype=bool)
        if n == 0:
            return in_fov, is_dyn

        if self.seg_img is None:
            self.diag['motivo'] = 'nessuna segmentazione'
            return in_fov, is_dyn
        if self.K is None or self.cam_frame is None:
            self.diag['motivo'] = 'nessun camera_info'
            return in_fov, is_dyn
        if self.seg_stamp is not None:
            t_scan = rclpy.time.Time.from_msg(scan.header.stamp).nanoseconds * 1e-9
            t_seg = rclpy.time.Time.from_msg(self.seg_stamp).nanoseconds * 1e-9
            if abs(t_scan - t_seg) > self.max_seg_age:
                self.diag['motivo'] = f'segmentazione vecchia di {abs(t_scan-t_seg):.2f}s'
                return in_fov, is_dyn

        T = self.lookup(self.cam_frame, scan.header.frame_id, scan.header.stamp)
        if T is None:
            self.diag['motivo'] = f'manca TF {self.cam_frame}<-{scan.header.frame_id}'
            return in_fov, is_dyn
        self.diag['motivo'] = 'ok'

        M = transform_to_matrix(T)
        P = np.concatenate([pts_laser, np.zeros((n, 1), dtype=np.float32),
                            np.ones((n, 1), dtype=np.float32)], axis=1)
        Pc = (M @ P.T).T[:, :3]
        z = Pc[:, 2]
        front = z > 1e-3
        if not np.any(front):
            return in_fov, is_dyn

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        u = np.full(n, -1.0)
        v = np.full(n, -1.0)
        u[front] = fx * Pc[front, 0] / z[front] + cx
        v[front] = fy * Pc[front, 1] / z[front] + cy

        H, W = self.seg_img.shape[:2]
        if self.cam_w and self.cam_h and (W != self.cam_w or H != self.cam_h):
            u *= W / float(self.cam_w)
            v *= H / float(self.cam_h)

        ui = np.round(u).astype(int)
        vi = np.round(v).astype(int)
        inside = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        in_fov[:] = inside

        d = max(0, self.pixel_dilation)
        for i in np.where(inside)[0]:
            y0, y1 = max(0, vi[i] - d), min(H, vi[i] + d + 1)
            x0, x1 = max(0, ui[i] - d), min(W, ui[i] + d + 1)
            patch = self.seg_img[y0:y1, x0:x1]
            is_dyn[i] = bool(np.isin(patch, self.dynamic_classes).any())

        self.diag['n_fov'] = int(np.count_nonzero(inside))
        self.diag['n_dyn'] = int(np.count_nonzero(is_dyn))
        return in_fov, is_dyn

    # ------------------------------------------------------------- clustering
    def cluster_points(self, pts, in_fov, is_dyn):
        clusters = []
        if len(pts) == 0:
            return clusters
        eps = self.cluster_eps
        keys = np.floor(pts / eps).astype(int)
        cell = {}
        for i, (kx, ky) in enumerate(keys):
            cell.setdefault((kx, ky), []).append(i)

        parent = list(range(len(pts)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for (kx, ky), idxs in cell.items():
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    other = cell.get((kx + dx, ky + dy))
                    if not other:
                        continue
                    for i in idxs:
                        for j in other:
                            if i < j and np.hypot(*(pts[i] - pts[j])) <= eps:
                                union(i, j)

        groups = {}
        for i in range(len(pts)):
            groups.setdefault(find(i), []).append(i)

        for idxs in groups.values():
            if len(idxs) < self.cluster_min_pts:
                continue
            cpts = pts[idxs]
            size = float(np.max(np.ptp(cpts, axis=0))) if len(cpts) > 1 else 0.0
            if size > self.cluster_max_size:
                continue

            n_fov = int(np.count_nonzero(in_fov[idxs]))
            n_dyn = int(np.count_nonzero(is_dyn[idxs]))
            # TRE STATI, non due. Vedi il commento in testa al file.
            if n_fov < self.min_fov_points:
                stato = SEM_UNKNOWN          # non inquadrato: non possiamo sapere
            elif n_dyn / float(n_fov) >= self.min_semantic_frac:
                stato = SEM_CONFIRMED        # inquadrato e riconosciuto dinamico
            else:
                stato = SEM_REJECTED         # inquadrato e NON dinamico -> muro

            c = cpts.mean(axis=0)
            clusters.append({'centroid': (float(c[0]), float(c[1])),
                             'size': size, 'stato': stato,
                             'idxs': idxs})
        return clusters

    # ------------------------------------------------------------- Kalman
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

    def track(self, dets, stati, now_s):
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
                # MEMORIA SEMANTICA: sale su conferma, scende su rifiuto, resta
                # ferma quando non c'e' informazione. Cosi' una persona gia'
                # riconosciuta non viene dimenticata appena esce dall'inquadratura.
                st = stati[best]
                if st == SEM_CONFIRMED:
                    tr['sem'] = min(tr['sem'] + 1, self.sem_score_max)
                elif st == SEM_REJECTED:
                    tr['sem'] = max(tr['sem'] - 1, -self.sem_score_max)
                used.add(best)

        for j, d in enumerate(dets):
            if j in used:
                continue
            s = np.array([d[0], d[1], 0.0, 0.0], dtype=float)
            P = np.diag([self.kf_sigma_z ** 2, self.kf_sigma_z ** 2,
                         self.kf_v_init ** 2, self.kf_v_init ** 2])
            sem0 = 1 if stati[j] == SEM_CONFIRMED else (-1 if stati[j] == SEM_REJECTED else 0)
            self.tracks.append(dict(id=self.next_id, s=s, P=P, n=1,
                                    t=now_s, t_seen=now_s, is_dynamic=False,
                                    mov_count=0, still_count=0, sem=sem0))
            self.next_id += 1

        self.tracks = [t for t in self.tracks
                       if now_s - t['t_seen'] <= self.track_timeout]

        for tr in self.tracks:
            tr['x'], tr['y'] = float(tr['s'][0]), float(tr['s'][1])
            tr['vx'], tr['vy'] = float(tr['s'][2]), float(tr['s'][3])
            tr['sp'] = float(np.sqrt(max(tr['P'][0, 0] + tr['P'][1, 1], 0.0)))
            tr['sv'] = float(np.sqrt(max(tr['P'][2, 2] + tr['P'][3, 3], 0.0)))

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
        """Il dinamico incrocera' il robot? Criterio del punto di massimo
        avvicinamento (CPA), calcolato in coordinate RELATIVE.

            t* = -(p_rel . v_rel) / |v_rel|^2      istante di minima distanza
            d* = |p_rel + v_rel * t*|              quella distanza

        t* <= 0  -> si stanno gia' allontanando: irrilevante.
        t* oltre l'orizzonte -> troppo in la' nel tempo: irrilevante.
        d* oltre il raggio   -> passera' comunque distante: irrilevante.

        Serve a non disegnare coni per cio' che sta DIETRO o DI LATO senza
        incrociare. NON toglie i coni da cio' che sta davvero sulla rotta: se un
        muro e' davanti e il robot ci va contro, resta rilevante -- ed e' giusto.
        """
        if not self.relevance_enabled:
            return True
        px = tr['x'] - self.robot_xy[0]
        py = tr['y'] - self.robot_xy[1]

        # SETTORE FRONTALE: il robot non ha retromarcia (vx_min=0), quindi non
        # puo' collidere con cio' che ha alle spalle. Un pedone che lo raggiunge
        # da dietro si scansera' da solo: non e' un problema di navigazione.
        if self.cone_sector < 360.0:
            rel = np.arctan2(py, px) - self.robot_yaw
            rel = (rel + np.pi) % (2 * np.pi) - np.pi      # riporta in [-pi, pi]
            if abs(rel) > np.radians(self.cone_sector) / 2.0:
                return False

        d0 = np.hypot(px, py)
        if d0 <= self.relevance_always:
            return True                       # gia' addosso: sempre rilevante
        vx = tr['vx'] - self.robot_v_world[0]
        vy = tr['vy'] - self.robot_v_world[1]
        vv = vx * vx + vy * vy
        if vv < 1e-6:
            return False                      # moto relativo nullo: non si avvicina
        t_cpa = -(px * vx + py * vy) / vv
        if t_cpa <= 0.0 or t_cpa > self.relevance_horizon:
            return False
        return np.hypot(px + vx * t_cpa, py + vy * t_cpa) <= self.relevance_radius

    def accepted(self, tr):
        """Una traccia va usata? Mai se la camera l'ha respinta (muro).
        Sempre se l'ha confermata. Se non sa, dipende da require_semantic."""
        if tr['sem'] < 0:
            return False
        if tr['sem'] > 0:
            return True
        return not self.require_semantic

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

        Ritorna (dets_world, dirs_img) dove:
          dets_world : lista di (x, y) nel frame mondo, una per persona;
          dirs_img   : lista di angoli-immagine (rad) per la direzione, se serve.
        Aggiorna anche self.diag e self.dyn_pts_w per la visualizzazione.
        """
        dets, dbg_pts = [], []
        self.diag['n_people'] = 0
        self.diag['n_valid_depth'] = 0

        if self.seg_img is None:
            self.diag['motivo'] = 'nessuna segmentazione'
            return dets
        if self.depth_img is None:
            self.diag['motivo'] = 'nessuna depth'
            return dets
        if self.K is None or self.cam_frame is None:
            self.diag['motivo'] = 'nessun camera_info'
            return dets

        # segmentazione e depth devono essere ragionevolmente sincronizzate
        if self.seg_stamp is not None and self.depth_stamp is not None:
            ts = self._stamp_s(self.seg_stamp)
            td = self._stamp_s(self.depth_stamp)
            if abs(ts - td) > self.max_seg_age:
                self.diag['motivo'] = f'seg/depth desync {abs(ts-td):.2f}s'
                return dets
        self.diag['motivo'] = 'ok'

        seg = self.seg_img
        depth = self.depth_img
        H, W = seg.shape[:2]

        # la depth puo' avere risoluzione diversa dalla segmentazione: la riscalo
        # agli indici della seg con nearest-neighbour (nessuna interpolazione sui
        # bordi di profondita', che creerebbe distanze fantasma)
        if depth.shape[:2] != (H, W):
            ys = (np.linspace(0, depth.shape[0] - 1, H)).astype(int)
            xs = (np.linspace(0, depth.shape[1] - 1, W)).astype(int)
            depth = depth[np.ix_(ys, xs)]

        # depth in metri: 16UC1 e' in millimetri, 32FC1 e' gia' in metri
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 0.001
        else:
            depth_m = depth.astype(np.float32)

        # maschera dei pixel di classe dinamica
        dyn_mask = np.isin(seg, self.dynamic_classes)
        self.diag['n_dyn_px'] = int(dyn_mask.sum())
        if not dyn_mask.any():
            self.diag['motivo'] = 'nessun pixel di classe dinamica nella segmentazione'
            return dets

        # CHIUSURA MORFOLOGICA: quando il busto non e' segmentato, le due gambe
        # restano due regioni separate e il tracker vedrebbe DUE persone. Una
        # chiusura (dilata + erode) col kernel largo quanto il gap tra le gambe
        # le ricongiunge, senza spostare i bordi esterni. Kernel dispari.
        if self.leg_merge_px > 0:
            dyn_mask = self._close_mask(dyn_mask, self.leg_merge_px)

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        sx = W / float(self.cam_w) if self.cam_w else 1.0
        sy = H / float(self.cam_h) if self.cam_h else 1.0

        # connected components sulla maschera: una regione per persona
        labels, n_lab = self._connected_components(dyn_mask, self.min_region_px)
        self.diag['n_regions'] = int(n_lab)

        Tw = self.lookup(self.world_frame, self.cam_frame)   # ottico -> mondo
        if Tw is None:
            self.diag['motivo'] = f'manca TF {self.world_frame}<-{self.cam_frame}'
            return dets
        M = transform_to_matrix(Tw)

        n_small = n_nodepth = 0
        for lab in range(1, n_lab + 1):
            ys, xs = np.where(labels == lab)
            if len(xs) < self.min_region_px:
                n_small += 1
                continue                       # regione troppo piccola: rumore

            # SALVAGUARDIA: se la chiusura ha unito due persone vicine, la
            # regione e' piu' larga di una persona plausibile a quella distanza.
            # In quel caso la si spezza in sotto-regioni per colonna.
            sub_regions = self._split_if_wide(xs, ys, depth_m)
            for sxs, sys in sub_regions:
                if len(sxs) < self.min_region_px:
                    n_small += 1
                    continue
                self.diag['n_people'] += 1

                # profondita' valida: scarto zeri, inf e fuori range
                d = depth_m[sys, sxs]
                valid = np.isfinite(d) & (d > self.depth_min) & (d < self.depth_max)
                if np.count_nonzero(valid) < self.min_valid_depth_px:
                    n_nodepth += 1
                    continue
                self.diag['n_valid_depth'] += 1

                Z = float(np.median(d[valid]))
                u = float(np.mean(sxs)) / sx
                v = float(np.mean(sys)) / sy
                Xo = (u - cx) * Z / fx
                Yo = (v - cy) * Z / fy
                pw = M @ np.array([Xo, Yo, Z, 1.0])
                dets.append((float(pw[0]), float(pw[1])))
                dbg_pts.append((float(pw[0]), float(pw[1])))

        self.diag['n_small'] = n_small
        self.diag['n_nodepth'] = n_nodepth
        self.dyn_pts_w = np.array(dbg_pts) if dbg_pts else np.empty((0, 2))
        return dets

    @staticmethod
    def _close_mask(mask, k):
        """Chiusura morfologica con kernel k x k. Riunisce le gambe di una
        stessa persona quando il busto non e' segmentato. Usa cv2 se c'e',
        altrimenti una dilata+erode manuale con vicinato quadrato."""
        k = int(k) | 1                       # forzo dispari
        try:
            import cv2
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                    ker).astype(bool)
        except Exception:
            pass
        r = k // 2
        m = mask.copy()
        # dilata
        dil = np.zeros_like(m)
        H, W = m.shape
        ys, xs = np.where(m)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny = np.clip(ys + dy, 0, H - 1)
                nx = np.clip(xs + dx, 0, W - 1)
                dil[ny, nx] = True
        # erode
        ero = dil.copy()
        ys, xs = np.where(~dil)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ny = np.clip(ys + dy, 0, H - 1)
                nx = np.clip(xs + dx, 0, W - 1)
                ero[ny, nx] = False
        return ero

    def _split_if_wide(self, xs, ys, depth_m):
        """Se una regione e' piu' larga di una persona plausibile a quella
        distanza, la spezza per colonna: e' il caso di due persone vicine unite
        dalla chiusura. Ritorna una lista di (xs, ys). Se non serve, ne ritorna
        una sola: la regione intera."""
        if not self.split_wide:
            return [(xs, ys)]
        d = depth_m[ys, xs]
        valid = np.isfinite(d) & (d > self.depth_min) & (d < self.depth_max)
        if np.count_nonzero(valid) < self.min_valid_depth_px:
            return [(xs, ys)]
        Z = float(np.median(d[valid]))
        # larghezza reale della regione in metri, a quella distanza
        fx = self.K[0, 0]
        width_px = xs.max() - xs.min() + 1
        width_m = width_px * Z / fx
        if width_m <= self.max_person_width_m:
            return [(xs, ys)]                # larghezza plausibile: una persona
        # troppo larga: quante persone ci stanno, e taglio in colonne uguali
        n = int(np.ceil(width_m / self.max_person_width_m))
        edges = np.linspace(xs.min(), xs.max() + 1, n + 1)
        out = []
        for i in range(n):
            sel = (xs >= edges[i]) & (xs < edges[i + 1])
            if sel.any():
                out.append((xs[sel], ys[sel]))
        return out if out else [(xs, ys)]

    def update(self):
        now_s = self.get_clock().now().nanoseconds * 1e-9

        # ------ RILEVAMENTO: sceglie il front-end secondo detection_mode ------
        dets, stati = [], []

        # A) ramo CAMERA: la maschera semantica trova le persone, la depth le
        #    localizza. Le detection sono per costruzione CONFERMATE (la camera
        #    ha gia' detto "persona"): non esistono muri qui, non serve il
        #    meccanismo a tre stati.
        if self.detection_mode in ('camera', 'fusion'):
            cam_dets = self.detect_people_camera(now_s)
            dets += cam_dets
            stati += [SEM_CONFIRMED] * len(cam_dets)

        # B) ramo LiDAR: copre cio' che la camera non inquadra. In 'camera' e'
        #    spento; in 'fusion' aggiunge SOLO gli IGNOTI (fuori FOV), perche' i
        #    dinamici nel FOV li ha gia' dati la camera in modo piu' preciso.
        if self.detection_mode in ('lidar', 'fusion') and self.last_scan is not None:
            scan = self.last_scan
            pts_l = self.scan_points_laser(scan)
            self.diag['n_scan'] = len(pts_l)
            if len(pts_l) > 0:
                in_fov, is_dyn = self.label_points(pts_l, scan)
                Tw = self.lookup(self.world_frame, scan.header.frame_id, scan.header.stamp)
                if Tw is not None:
                    Mw = transform_to_matrix(Tw)
                    P = np.concatenate([pts_l, np.zeros((len(pts_l), 1)),
                                        np.ones((len(pts_l), 1))], axis=1)
                    pts_w = (Mw @ P.T).T[:, :2]
                    clusters = self.cluster_points(pts_w, in_fov, is_dyn)
                    self.diag['conf'] = sum(1 for c in clusters if c['stato'] == SEM_CONFIRMED)
                    self.diag['rej'] = sum(1 for c in clusters if c['stato'] == SEM_REJECTED)
                    self.diag['unk'] = sum(1 for c in clusters if c['stato'] == SEM_UNKNOWN)
                    if self.detection_mode == 'lidar':
                        keep = [c for c in clusters if c['stato'] != SEM_REJECTED]
                    else:
                        # in fusion il LiDAR aggiunge solo cio' che la camera non vede
                        keep = [c for c in clusters if c['stato'] == SEM_UNKNOWN]
                    dets += [c['centroid'] for c in keep]
                    stati += [c['stato'] for c in keep]
                    if self.detection_mode == 'lidar':
                        self.dyn_pts_w = pts_w[is_dyn] if len(pts_w) else np.empty((0, 2))

        self.track(dets, stati, now_s)

        Tr = self.lookup(self.world_frame, self.robot_frame)
        if Tr is not None:
            M = transform_to_matrix(Tr)
            rx, ry = M[0, 3], M[1, 3]
            self.robot_xy = (rx, ry)
            # la velocita' di /odom e' nel frame del robot: la ruoto nel mondo,
            # serve al criterio di rilevanza che ragiona su velocita' RELATIVE
            yaw = np.arctan2(M[1, 0], M[0, 0])
            self.robot_yaw = yaw
            bx, by = self.robot_v_body
            self.robot_v_world = (bx * np.cos(yaw) - by * np.sin(yaw),
                                  bx * np.sin(yaw) + by * np.cos(yaw))
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
            if not self.accepted(tr):
                continue
            # il nucleo si marca comunque (e' un ostacolo fisico dove si trova),
            # ma il CONO lo disegniamo solo se incrocera' davvero il robot
            rilevante = self.is_relevant(tr)
            if self.mark_obstacle:
                self._stamp(cost, tr['x'], tr['y'], ox, oy,
                            self.nucleus_radius, self.max_cost)
            if not self.enable_cone or spinning or not rilevante:
                continue
            speed = np.hypot(tr['vx'], tr['vy'])
            if self.kf_min_snr > 0.0 and speed < self.kf_min_snr * tr.get('sv', 0.0):
                continue
            self._update_cone_dir(tr, speed)
            self._paint_cone(cost, tr, ox, oy, speed)
            self.arrows.append((tr['x'], tr['y'], tr['vx'], tr['vy']))
        return cost

    def _update_cone_dir(self, tr, speed):
        """Direzione del cono lisciata con una media esponenziale.

        Il centroide di un pedone salta tra le due gambe a ogni passo, e la
        direzione della velocita' del Kalman balla di conseguenza: il cono
        oscilla come se la persona barcollasse. Qui si liscia SOLO la direzione,
        lasciando intatte posizione e velocita' stimate -- che devono restare
        reattive per il critic spazio-temporale."""
        if speed < 1e-6:
            return
        nx, ny = tr['vx'] / speed, tr['vy'] / speed
        old = tr.get('dir')
        if old is None:
            tr['dir'] = (nx, ny)
            return
        a = self.cone_dir_ema
        dx = a * nx + (1.0 - a) * old[0]
        dy = a * ny + (1.0 - a) * old[1]
        m = np.hypot(dx, dy)
        tr['dir'] = (dx / m, dy / m) if m > 1e-6 else (nx, ny)

    def _stamp(self, cost, wx, wy, ox, oy, radius_m, val):
        cxi = int((wx - ox) / self.res); cyi = int((wy - oy) / self.res)
        rr = int(radius_m / self.res)
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if dx * dx + dy * dy <= rr * rr:
                    gx, gy = cxi + dx, cyi + dy
                    if 0 <= gx < self.n and 0 <= gy < self.n and cost[gy, gx] < val:
                        cost[gy, gx] = val

    def _paint_cone(self, cost, tr, ox, oy, speed):
        ux, uy = tr.get('dir', (tr['vx'] / speed, tr['vy'] / speed))
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
            if not tr.get('is_dynamic', False) or not self.accepted(tr):
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
            if not tr.get('is_dynamic', False) or not self.accepted(tr):
                continue
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = now
            m.ns = 'tracks'
            m.id = int(tr['id'])
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.06; m.scale.y = 0.12; m.scale.z = 0.12
            # verde = confermata dalla camera, giallo = solo geometrica
            if tr['sem'] > 0:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.9, 0.3, 0.9
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.8, 0.1, 0.9
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
        n_acc = sum(1 for t in self.tracks if t.get('is_dynamic') and self.accepted(t))
        n_sem = sum(1 for t in self.tracks if t['sem'] > 0)
        if self.detection_mode in ('camera', 'fusion'):
            # diagnostica dello STADIO in cui si perdono le persone
            self.get_logger().info(
                f"[camera] pixel_dinamici={d.get('n_dyn_px',0)} "
                f"regioni={d.get('n_regions',0)} persone={d.get('n_people',0)} "
                f"con_depth_valida={d.get('n_valid_depth',0)} "
                f"(scartate: piccole={d.get('n_small',0)} senza_depth={d.get('n_nodepth',0)}) | "
                f"tracce={len(self.tracks)} usate={n_acc} | {d['motivo']}")
        else:
            self.get_logger().info(
                f"[lidar] scan={d['n_scan']} inquadrati={d['n_fov']} "
                f"su_classe_dinamica={d['n_dyn']} | "
                f"cluster: conf={d['conf']} RESPINTI={d['rej']} ignoti={d['unk']} | "
                f"tracce={len(self.tracks)} (sem+={n_sem}) usate={n_acc} | {d['motivo']}")


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