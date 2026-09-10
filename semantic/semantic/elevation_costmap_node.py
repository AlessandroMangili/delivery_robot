#!/usr/bin/env python3

import os
import array
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid, Odometry
from cv_bridge import CvBridge
import tf2_ros
from std_srvs.srv import Trigger

DEFAULT_DYNAMIC = [11, 12, 13, 14, 15, 16, 17, 18]   # classi dinamiche Cityscapes


# ============================================================================
#  FUNZIONI PURE (testabili in isolamento, senza ROS)
# ============================================================================

def transform_to_matrix(t):
    """geometry_msgs TransformStamped -> 4x4 omogenea."""
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


def transform_points(pts, M):
    """Applica la 4x4 M a punti Nx3 (p' = R p + t). Ritorna Nx3."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:3, :3].T + M[:3, 3][None, :]


def project_to_pixel(pts_cam, K, width, height):
    """Punti nel frame OTTICO camera -> (u,v) interi + maschera valida
    (davanti alla camera E dentro l'immagine). Identica al nodo semantico:
    e' cosi' che ogni punto sa su quale pixel (e quindi classe) cade."""
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    in_front = z > 1e-6
    zz = np.where(in_front, z, 1.0)
    u = np.round(fx * (x / zz) + cx).astype(np.int64)
    v = np.round(fy * (y / zz) + cy).astype(np.int64)
    valid = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = np.where(valid, u, 0)
    v = np.where(valid, v, 0)
    return u, v, valid


def sensor_variance(ranges, var_base, var_dist):
    """Modello di rumore di MISURA dell'altezza per punto [m^2].
    Cresce con la distanza: un punto lontano ha altezza meno affidabile
    (raggio radente, footprint del beam piu' grande). Modello quadratico:
        var_meas = var_base + var_dist * range^2
    range = distanza 3D dal sensore [m]. Fedele allo spirito del modello
    sensoriale di Fankhauser (var che sale con la distanza)."""
    ranges = np.asarray(ranges, dtype=np.float64)
    return var_base + var_dist * ranges * ranges


def aggregate_frame(flat_idx, heights, var_meas_pts, ncells):
    """Fonde i punti dello STESSO frame che cadono nella stessa cella, per
    inverse-variance weighting (stima ML della misura di cella per questo frame):
        w = 1/var_meas ; h_meas = sum(w*h)/sum(w) ; var_meas = 1/sum(w)
    flat_idx: indice piatto di cella per punto (int, gia' dentro griglia).
    Ritorna (seen, h_meas, var_meas) di lunghezza ncells."""
    w = 1.0 / np.maximum(var_meas_pts, 1e-9)
    sum_w = np.zeros(ncells, dtype=np.float64)
    sum_wh = np.zeros(ncells, dtype=np.float64)
    np.add.at(sum_w, flat_idx, w)
    np.add.at(sum_wh, flat_idx, w * heights)
    seen = sum_w > 0.0
    safe_w = np.where(seen, sum_w, 1.0)
    h_meas = np.where(seen, sum_wh / safe_w, 0.0)
    var_meas = np.where(seen, 1.0 / safe_w, np.inf)
    return seen, h_meas, var_meas


def kalman_fuse(h, var, h_meas, var_meas):
    """Aggiornamento di misura (fusione di due gaussiane). In-place safe:
    ritorna (h_new, var_new). Applicare solo su celle gia' inizializzate."""
    K = var / (var + var_meas)
    h_new = h + K * (h_meas - h)
    var_new = (1.0 - K) * var
    return h_new, var_new


def median_smooth_masked(h, known, ksize=3):
    """Smoothing MEDIANO di h sulle celle note, PRIMA di derivare slope/rough/step.
    Perche' serve: slope/roughness/step sono DERIVATE spaziali di h, e le derivate
    AMPLIFICANO il rumore. Anche 1-2 cm di rumore residuo in h (normale dopo la
    fusione, e rimescolato ad ogni ri-osservazione per il micro-offset di posa)
    diventano roughness/step visibili -> aloni grigi sul terreno piatto.
    Il filtro MEDIANO e' edge-preserving: uccide il rumore isolato cella-a-cella
    ma PRESERVA i salti reali (un cordolo resta un cordolo). Le celle ignote sono
    riempite temporaneamente con la media locale nota, per non creare falsi bordi."""
    if ksize < 3:
        return h
    h = h.astype(np.float32)
    knownf = known.astype(np.float32)
    h0 = np.where(known, h, 0.0).astype(np.float32)
    cnt = cv2.boxFilter(knownf, -1, (5, 5), normalize=False, borderType=cv2.BORDER_REPLICATE)
    s = cv2.boxFilter(h0, -1, (5, 5), normalize=False, borderType=cv2.BORDER_REPLICATE)
    filled = np.where(known, h, s / np.where(cnt > 0, cnt, 1.0)).astype(np.float32)
    sm = cv2.medianBlur(filled, int(ksize))
    return np.where(known, sm, h).astype(np.float32)


def derive_slope_roughness(h, known, res, rough_win, min_known):
    """Da campo altezza h (float32) + maschera known -> (slope, roughness, valid).
    slope     = |grad h| [m/m] (adimensionale; tan della pendenza).
    roughness = std locale di h in finestra rough_win x rough_win [m].
    valid     = celle con abbastanza vicini noti da fidarsi.
    Le celle ignote vengono riempite con la media locale nota (box filter mascherato)
    per non introdurre gradienti finti ai bordi dell'ignoto."""
    k = int(rough_win)
    knownf = known.astype(np.float32)
    h0 = np.where(known, h, 0.0).astype(np.float32)

    # box filter NON normalizzato = somme locali; conto dei noti nella finestra
    cnt = cv2.boxFilter(knownf, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)
    s = cv2.boxFilter(h0, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)
    ss = cv2.boxFilter(h0 * h0, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)

    valid = cnt >= float(min_known)
    safe_cnt = np.where(cnt > 0, cnt, 1.0)
    mean = s / safe_cnt                       # altezza locale liscia (definita dove cnt>0)
    var_local = np.maximum(ss / safe_cnt - mean * mean, 0.0)
    roughness = np.sqrt(var_local)

    # slope come gradiente spaziale della MEDIA liscia (robusto all'ignoto)
    gy, gx = np.gradient(mean, res)           # d/dy, d/dx
    slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    slope = np.where(valid, slope, 0.0).astype(np.float32)
    roughness = np.where(valid, roughness, 0.0).astype(np.float32)
    return slope, roughness, valid


def elevation_cost(slope, roughness, step, valid,
                   slope_crit, rough_crit, step_crit,
                   slope_dead=0.0, rough_dead=0.0, step_dead=0.0):
    """f(M): combina i pericoli geometrici in un costo 0..100.
      - scoscese : slope alto        -> cost_slope
      - accidentato: roughness alta  -> cost_rough
      - buche/gradini: step alto     -> cost_step
    Ognuno satura al proprio critico; si combina con MAX (il pericolo peggiore
    domina) e si satura a 100. Celle non valide -> -1 (ignoto: il robot le esplora,
    coerente con include_unknown_as_edge:false della confinement)."""
    # DEAD-ZONE: sotto la soglia di rumore il contributo e' ESATTAMENTE 0 (non
    # 'poco'). La normalizzazione lineare da 0 dava un filo di costo anche a 1.5cm
    # di roughness (rumore) -> grigio. Con la dead-zone, sotto *_dead = 0 netto;
    # sopra sale linearmente da *_dead a *_crit. Uccide il residuo che sopravvive
    # al mediano, senza toccare i pericoli veri (che stanno sopra *_dead).
    def _band(x, dead, crit):
        return np.clip((x - dead) / max(crit - dead, 1e-6), 0.0, 1.0)
    cs = _band(slope, slope_dead, slope_crit)
    cr = _band(roughness, rough_dead, rough_crit)
    ck = _band(step, step_dead, step_crit)
    c = np.maximum(np.maximum(cs, cr), ck) * 100.0
    cost = np.where(valid, np.clip(np.round(c), 0, 100), -1).astype(np.int16)
    return cost


def neighbor_step(h, known, res):
    """Massimo salto di altezza verso i 4-vicini [m] (gradino/bordo marciapiede,
    o buco). Celle ignote non contano nel confronto."""
    step = np.zeros_like(h, dtype=np.float32)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.roll(np.roll(h, dy, axis=0), dx, axis=1)
        sk = np.roll(np.roll(known, dy, axis=0), dx, axis=1)
        both = known & sk
        d = np.where(both, np.abs(h - sh), 0.0).astype(np.float32)
        step = np.maximum(step, d)
    return step


# ============================================================================
#  NODO
# ============================================================================

class ElevationCostmapNode(Node):
    def __init__(self):
        super().__init__('elevation_costmap_node')

        # --- Frames ---
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('lidar_frame', 'base_scan')

        # --- Grid ---
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('initial_size_m', 10.0)
        self.declare_parameter('grow_margin_m', 3.0)

        # --- LiDAR ---
        self.declare_parameter('lidar_topic', '/scan/points')
        self.declare_parameter('lidar_max_age_s', 0.3)
        self.declare_parameter('lidar_min_range', 0.4)
        self.declare_parameter('map_write_max_range', 5.0)
        # banda di quota (rel. al robot) per accettare un punto come TERRENO:
        self.declare_parameter('elev_min_rel', -1.0)   # sotto = buco molto profondo (scartato)
        self.declare_parameter('elev_max_rel', 1.0)    # sopra = overhead (chiome/insegne)

        # --- Camera + segmentazione (co-registrazione col layer semantico) ---
        # I punti LiDAR sono proiettati nel FOV camera e SCARTATI se cadono su un
        # pixel di classe DINAMICA: cosi' i pedoni non lasciano scie nella mappa
        # di elevazione. Copertura = FOV camera (non piu' 360 LiDAR).
        self.declare_parameter('camera_optical_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('seg_topic', '/semantic/segmentation')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('seg_max_age_s', 0.3)
        self.declare_parameter('dynamic_classes', DEFAULT_DYNAMIC)
        self.declare_parameter('dyn_dilate_px', 6)     # alone attorno ai dinamici (bordi seg.)

        # --- Modello di misura Kalman ---
        self.declare_parameter('sensor_var_base', 0.0009)   # (0.03 m)^2 rumore base
        self.declare_parameter('sensor_var_dist', 0.0004)   # coeff. * range^2
        self.declare_parameter('var_init', 1.0)             # varianza cella mai vista
        self.declare_parameter('var_max', 1.0)              # tetto varianza (inflazione)
        self.declare_parameter('process_var_rate', 0.005)   # [m^2/s] inflazione nel tempo (SIM)
        self.declare_parameter('maha_thresh', 3.0)          # gate outlier (rumore residuo)
        self.declare_parameter('min_var_for_valid', 0.25)   # var oltre = cella non affidabile

        # --- Derivazione costo (f(M)) ---
        self.declare_parameter('rough_win', 5)              # finestra roughness [celle]
        self.declare_parameter('rough_min_known', 6)        # min vicini noti per validita'
        self.declare_parameter('slope_crit', 0.35)          # [m/m] ~19 deg -> costo max
        self.declare_parameter('rough_crit', 0.06)          # [m] std locale -> costo max
        self.declare_parameter('step_crit', 0.06)           # [m] salto -> costo max (cordolo)
        # smoothing mediano di h prima di derivare (5 = pulisce bene il rumore 1cm;
        # 3 = piu' leggero; 0/1 = off). Con ksize 5 la soglia step affidabile e' ~5cm.
        self.declare_parameter('median_ksize', 5)
        # dead-zone / soglia minima: sotto = rumore (0), sopra sale verso *_crit
        self.declare_parameter('slope_dead', 0.08)          # [m/m] sotto = piatto
        self.declare_parameter('rough_dead', 0.025)         # [m] sotto = liscio (rumore)
        self.declare_parameter('step_dead', 0.02)           # [m] step_min: sotto = rumore

        # --- Publishing ---
        self.declare_parameter('publish_period_s', 1.0)
        self.declare_parameter('out_topic', '/traversability_cost')

        # --- Gate di localizzazione (stesso schema del semantico) ---
        self.declare_parameter('use_loc_quality_gate', True)
        self.declare_parameter('loc_pose_topic', '/odometry/global')
        self.declare_parameter('loc_std_suspend_m', 1.0)
        self.declare_parameter('loc_std_resume_m', 0.5)
        self.declare_parameter('loc_cov_ema', 0.4)

        # --- Persistenza ---
        self.declare_parameter('map_load_name', '')
        self.declare_parameter('map_save_name', '')
        self.declare_parameter('autosave', False)

        gp = self.get_parameter
        self.target = gp('target_frame').value
        self.base_frame = gp('base_frame').value
        self.lidar_frame = gp('lidar_frame').value
        self.res = float(gp('resolution').value)
        self.initial_size_m = float(gp('initial_size_m').value)
        self.grow_margin = float(gp('grow_margin_m').value)

        self.lidar_max_age = float(gp('lidar_max_age_s').value)
        self.lidar_min_range = float(gp('lidar_min_range').value)
        self.map_write_max_range = float(gp('map_write_max_range').value)
        self.elev_min_rel = float(gp('elev_min_rel').value)
        self.elev_max_rel = float(gp('elev_max_rel').value)

        self.cam_frame = gp('camera_optical_frame').value
        self.seg_max_age = float(gp('seg_max_age_s').value)
        self.dyn_dilate_px = int(gp('dyn_dilate_px').value)
        self.dyn_lut = np.zeros(256, dtype=bool)
        for c in gp('dynamic_classes').value:
            self.dyn_lut[int(c)] = True

        self.sensor_var_base = float(gp('sensor_var_base').value)
        self.sensor_var_dist = float(gp('sensor_var_dist').value)
        self.var_init = float(gp('var_init').value)
        self.var_max = float(gp('var_max').value)
        self.process_var_rate = float(gp('process_var_rate').value)
        self.maha_thresh = float(gp('maha_thresh').value)
        self.min_var_for_valid = float(gp('min_var_for_valid').value)

        self.rough_win = int(gp('rough_win').value)
        self.rough_min_known = int(gp('rough_min_known').value)
        self.slope_crit = float(gp('slope_crit').value)
        self.rough_crit = float(gp('rough_crit').value)
        self.step_crit = float(gp('step_crit').value)
        self.median_ksize = int(gp('median_ksize').value)
        self.slope_dead = float(gp('slope_dead').value)
        self.rough_dead = float(gp('rough_dead').value)
        self.step_dead = float(gp('step_dead').value)

        self.publish_period = float(gp('publish_period_s').value)

        self.use_loc_gate = bool(gp('use_loc_quality_gate').value)
        self.loc_std_suspend = float(gp('loc_std_suspend_m').value)
        self.loc_std_resume = float(gp('loc_std_resume_m').value)
        self.loc_cov_ema = float(gp('loc_cov_ema').value)
        self.loc_std_smooth = 0.0
        self.loc_ok = True

        # --- Stato griglia (frame map, auto-grow) ---
        self.gnx = int(self.initial_size_m / self.res)
        self.gny = int(self.initial_size_m / self.res)
        self.gox = -self.initial_size_m / 2.0
        self.goy = -self.initial_size_m / 2.0
        self.h = np.zeros((self.gny, self.gnx), dtype=np.float32)          # altezza stimata
        self.var = np.full((self.gny, self.gnx), self.var_init, np.float32)  # varianza stima (corrente)
        self.var_min = np.full((self.gny, self.gnx), np.inf, np.float32)   # varianza MINIMA storica (validita')
        self.n = np.zeros((self.gny, self.gnx), dtype=np.uint16)          # conteggio osservazioni

        # --- Cache nuvola LiDAR (PULITA una volta sola, come nel semantico) ---
        self.cloud = None
        self.cloud_t = -1e9

        # --- Cache segmentazione + intrinseci camera (per escludere i dinamici) ---
        self.bridge = CvBridge()
        self.K = None
        self.seg = None
        self.seg_t = -1e9

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        cloud_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PointCloud2, gp('lidar_topic').value,
                                 self.cloud_cb, cloud_qos)

        self.create_subscription(CameraInfo, gp('camera_info_topic').value,
                                 self.info_cb, 1)
        seg_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, gp('seg_topic').value, self.seg_cb, seg_qos)

        if self.use_loc_gate:
            self.create_subscription(Odometry, gp('loc_pose_topic').value,
                                     self.loc_cb, 10)

        latched_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(OccupancyGrid, gp('out_topic').value, latched_qos)

        self._last_pred_t = None

        # --- Persistenza ---
        self.maps_dir = self.resolve_maps_dir()
        load_name = gp('map_load_name').value
        save_name = gp('map_save_name').value
        self.map_save_path = os.path.join(self.maps_dir, save_name) if save_name else ''
        if load_name:
            p = os.path.join(self.maps_dir, load_name)
            if not p.endswith('.npz'):
                p += '.npz'
            if os.path.isfile(p):
                self.load_map(p)
            else:
                self.get_logger().warn(f'Elevation map to load not found: {p}')
        self.autosave = bool(gp('autosave').value)
        if self.autosave and not save_name:
            self.get_logger().warn('autosave:true ma map_save_name vuoto -> autosave DISABILITATO.')
            self.autosave = False
        self.create_service(Trigger, '~/save_map', self.save_map_cb)

        # timer di pubblicazione (deriva costo e pubblica, indipendente dal LiDAR)
        self.create_timer(self.publish_period, self.publish_timer)

        self.get_logger().info(
            f'Elevation 2.5D (Kalman-per-cella, LiDAR-primary) pronto. frame={self.target}. '
            f'res={self.res} m. Uscita: {gp("out_topic").value}.')
        
        save_hint = self.map_save_path or os.path.join(self.maps_dir, 'elevation_map')
        if self.autosave:
            self.get_logger().info(f'AUTOSAVE alla chiusura (Ctrl-C) -> {self.map_save_path}')
        self.get_logger().info(
            'Salvataggio su richiesta: '
            f'ros2 service call /elevation_costmap_node/save_map std_srvs/srv/Trigger  ->  {save_hint}')

    # ---------------- persistenza ----------------
    def resolve_maps_dir(self):
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('semantic')
            ws = share
            for _ in range(8):
                ws = os.path.dirname(ws)
                src = os.path.join(ws, 'src')
                if os.path.isdir(src):
                    for root, dirs, files in os.walk(src):
                        if os.path.basename(root) == 'semantic' and 'package.xml' in files:
                            d = os.path.join(root, 'maps')
                            os.makedirs(d, exist_ok=True)
                            return d
                    break
            d = os.path.join(share, 'maps'); os.makedirs(d, exist_ok=True); return d
        except Exception:
            d = os.path.join(os.path.expanduser('~'), '.semantic_maps')
            os.makedirs(d, exist_ok=True); return d

    def load_map(self, path):
        try:
            d = np.load(path)
            if float(d['res']) != self.res:
                self.get_logger().warn('Elevation map con risoluzione diversa: ignorata.')
                return
            self.h = d['h'].astype(np.float32)
            self.var = d['var'].astype(np.float32)
            self.n = d['n'].astype(np.uint16)
            # retrocompat: mappe vecchie senza var_min -> ricostruisci dalla var corrente
            if 'var_min' in d.files:
                self.var_min = d['var_min'].astype(np.float32)
            else:
                self.var_min = self.var.copy()
            self.gny, self.gnx = self.h.shape
            self.gox = float(d['gox']); self.goy = float(d['goy'])
            self.get_logger().info(f'Elevation map caricata da {path} ({self.gnx}x{self.gny}).')
        except Exception as e:
            self.get_logger().warn(f'Load elevation fallito: {e}')

    def save_map(self, path):
        try:
            if not path.endswith('.npz'):
                path += '.npz'
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            np.savez_compressed(path, h=self.h, var=self.var, var_min=self.var_min, n=self.n,
                                gnx=self.gnx, gny=self.gny, res=self.res,
                                gox=self.gox, goy=self.goy)
            self.get_logger().info(f'Elevation map salvata in {path}.')
            return path
        except Exception as e:
            self.get_logger().error(f'Save elevation fallito: {e}')
            return None

    def save_map_cb(self, request, response):
        path = self.map_save_path or os.path.join(self.maps_dir, 'elevation_map')
        saved = self.save_map(path)
        response.success = saved is not None
        response.message = f'Salvata in {saved}' if saved else 'Salvataggio fallito'
        return response

    # ---------------- callbacks ----------------
    def loc_cb(self, msg: Odometry):
        cov = msg.pose.covariance
        var_x = max(0.0, float(cov[0])); var_y = max(0.0, float(cov[7]))
        std_xy = float(np.sqrt(max(var_x, var_y)))
        a = self.loc_cov_ema
        self.loc_std_smooth = a * std_xy + (1.0 - a) * self.loc_std_smooth
        if self.loc_ok and self.loc_std_smooth > self.loc_std_suspend:
            self.loc_ok = False
            self.get_logger().warn(
                f'LOCALIZZAZIONE PERSA (std liscia {self.loc_std_smooth:.2f}m > '
                f'{self.loc_std_suspend}m) -> MAPPATURA ELEVAZIONE SOSPESA.')
        elif (not self.loc_ok) and self.loc_std_smooth < self.loc_std_resume:
            self.loc_ok = True
            self.get_logger().info(
                f'Localizzazione RECUPERATA (std liscia {self.loc_std_smooth:.2f}m) -> RIPRESA.')

    def cloud_cb(self, msg: PointCloud2):
        """Nuvola LiDAR (frame base_scan): pulita e messa in cache.
        Scarta NaN/inf (skip_nans NON toglie gli inf del gpu_lidar) e il rumore
        entro lidar_min_range (come il 'blind' di FAST-LIO)."""
        try:
            rec = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception:
            return
        n = int(rec.shape[0])
        if n == 0:
            return
        pts = np.empty((n, 3), dtype=np.float64)
        pts[:, 0] = rec['x']; pts[:, 1] = rec['y']; pts[:, 2] = rec['z']
        finite = np.isfinite(pts).all(axis=1)
        r2 = pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2
        keep = finite & (r2 >= self.lidar_min_range ** 2)
        pts = pts[keep]
        if pts.shape[0] == 0:
            return
        self.cloud = pts
        self.cloud_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.integrate()   # integra appena arriva una nuvola fresca

    def info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)

    def seg_cb(self, msg: Image):
        """Immagine di segmentazione (Cityscapes ID, mono8): messa in cache.
        Serve solo a sapere quali pixel sono DINAMICI, per escludere i punti
        LiDAR che ci cadono sopra."""
        try:
            self.seg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception:
            return
        self.seg_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def lookup(self, frame):
        try:
            return self.tf_buffer.lookup_transform(
                self.target, frame, rclpy.time.Time(), timeout=Duration(seconds=0.2))
        except Exception as e:
            self.get_logger().warn(f'TF non disponibile {frame}->{self.target}: {e}',
                                   throttle_duration_sec=2.0)
            return None

    # ---------------- integrazione (misura Kalman) ----------------
    def integrate(self):
        if self.cloud is None:
            return
        if self.use_loc_gate and not self.loc_ok:
            return

        # serve la camera: intrinseci + segmentazione fresca (per i dinamici)
        if self.K is None or self.seg is None:
            return
        if (self.cloud_t - self.seg_t) > self.seg_max_age:
            self.get_logger().warn('Segmentazione troppo vecchia in questo frame.',
                                   throttle_duration_sec=10.0)
            return

        T_scan = self.lookup(self.lidar_frame)
        T_base = self.lookup(self.base_frame)
        T_cam = self.lookup(self.cam_frame)
        if T_scan is None or T_base is None or T_cam is None:
            return

        M_scan2map = transform_to_matrix(T_scan)
        M_cam = transform_to_matrix(T_cam)                # map <- camera ottica
        M_scan2cam = np.linalg.inv(M_cam) @ M_scan2map    # base_scan -> camera ottica

        Pm = transform_points(self.cloud, M_scan2map)     # punti in map (posizione)
        Pc = transform_points(self.cloud, M_scan2cam)     # punti in camera (per la classe)
        X = Pm[:, 0]; Y = Pm[:, 1]; Z = Pm[:, 2]

        bx = T_base.transform.translation.x
        by = T_base.transform.translation.y
        bz = T_base.transform.translation.z

        # range 3D dal sensore (per il modello di rumore) -- gia' nel frame base_scan
        rng = np.sqrt(self.cloud[:, 0]**2 + self.cloud[:, 1]**2 + self.cloud[:, 2]**2)

        # --- proiezione nel FOV camera: (u,v) + maschera dentro-immagine ---
        seg = self.seg
        hgt, wid = seg.shape
        uu, vv, in_img = project_to_pixel(Pc, self.K, wid, hgt)

        # --- ESCLUSIONE DINAMICI: scarta i punti che cadono su pixel dinamici ---
        # (pedoni, veicoli...). Dilatazione = alone attorno ai dinamici, cosi' i
        # bordi mal segmentati non lasciano frangie. E' cio' che elimina la scia.
        full_dyn = self.dyn_lut[seg].astype(np.uint8)
        if self.dyn_dilate_px > 0 and full_dyn.any():
            k = 2 * self.dyn_dilate_px + 1
            full_dyn = cv2.dilate(full_dyn, np.ones((k, k), np.uint8), iterations=1)
        not_dyn = ~(full_dyn[vv, uu].astype(bool))

        # banda di quota TERRENO (rel. al robot) + distanza + dentro FOV + non dinamico
        rel = Z - bz
        dist = np.hypot(X - bx, Y - by)
        ok = (np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
              & in_img & not_dyn
              & (rel >= self.elev_min_rel) & (rel <= self.elev_max_rel)
              & (dist < self.map_write_max_range))
        if not np.any(ok):
            return

        Xo = X[ok]; Yo = Y[ok]; Zo = Z[ok]; rngo = rng[ok]

        self.ensure_capacity(Xo, Yo)

        gi = ((Xo - self.gox) / self.res).astype(np.int64)
        gj = ((Yo - self.goy) / self.res).astype(np.int64)
        inside = (gi >= 0) & (gi < self.gnx) & (gj >= 0) & (gj < self.gny)
        if not np.any(inside):
            return
        gi = gi[inside]; gj = gj[inside]; Zi = Zo[inside]; rngi = rngo[inside]

        flat = gj * self.gnx + gi
        N = self.gnx * self.gny

        # --- passo di PREDIZIONE (inflazione varianza nel tempo) ---
        # SIM: sostituto della propagazione di covarianza di posa di Fankhauser.
        if self._last_pred_t is not None:
            dt = max(0.0, self.cloud_t - self._last_pred_t)
            if dt > 0.0:
                np.minimum(self.var + self.process_var_rate * dt, self.var_max, out=self.var)
        self._last_pred_t = self.cloud_t

        # --- misura di frame per cella (inverse-variance) ---
        var_pts = sensor_variance(rngi, self.sensor_var_base, self.sensor_var_dist)
        seen, h_meas, var_meas = aggregate_frame(flat, Zi, var_pts, N)

        Hs = self.h.ravel(); Vs = self.var.ravel(); Ns = self.n.ravel()
        Vmin = self.var_min.ravel()

        first = seen & (Ns == 0)
        prev = seen & (Ns > 0)

        # gate di Mahalanobis: misure troppo lontane dalla stima = outlier
        # (rumore residuo) -> non fuse, non corrompono il terreno.
        maha = np.zeros(N, dtype=np.float64)
        denom = np.sqrt(Vs + var_meas)
        good = prev & (denom > 0)
        maha[good] = np.abs(h_meas[good] - Hs[good]) / denom[good]
        gated = prev & (maha > self.maha_thresh)
        fuse = prev & (~gated)

        # prima osservazione: init diretto
        Hs[first] = h_meas[first]
        Vs[first] = var_meas[first]

        # fusione Kalman sulle celle gia' viste e coerenti
        if np.any(fuse):
            hf, vf = kalman_fuse(Hs[fuse], Vs[fuse], h_meas[fuse], var_meas[fuse])
            Hs[fuse] = hf; Vs[fuse] = vf

        # varianza MINIMA storica: la migliore confidenza mai raggiunta dalla cella.
        # E' cio' che decide la VALIDITA' (visibilita'), disaccoppiata dalla var
        # CORRENTE che l'inflazione fa ricrescere. Una cella mappata bene una volta
        # resta valida: non svanisce col tempo mentre il robot si allontana.
        touched = first | fuse
        if np.any(touched):
            Vmin[touched] = np.minimum(Vmin[touched], Vs[touched])

        # conteggio (i gated non incrementano: non li consideriamo osservazioni valide)
        Ns[first] = 1
        Ns[fuse] = np.minimum(Ns[fuse] + 1, 65535)

        self.h = Hs.reshape(self.gny, self.gnx)
        self.var = Vs.reshape(self.gny, self.gnx)
        self.var_min = Vmin.reshape(self.gny, self.gnx)
        self.n = Ns.reshape(self.gny, self.gnx)

    def ensure_capacity(self, X, Y):
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
        ngx = self.gnx + pl + pr; ngy = self.gny + pb + pt
        nh = np.zeros((ngy, ngx), np.float32)
        nv = np.full((ngy, ngx), self.var_init, np.float32)
        nvm = np.full((ngy, ngx), np.inf, np.float32)
        nn = np.zeros((ngy, ngx), np.uint16)
        nh[pb:pb + self.gny, pl:pl + self.gnx] = self.h
        nv[pb:pb + self.gny, pl:pl + self.gnx] = self.var
        nvm[pb:pb + self.gny, pl:pl + self.gnx] = self.var_min
        nn[pb:pb + self.gny, pl:pl + self.gnx] = self.n
        self.h = nh; self.var = nv; self.var_min = nvm; self.n = nn
        self.gnx = ngx; self.gny = ngy
        self.gox -= pl * self.res; self.goy -= pb * self.res
        self.get_logger().info(
            f'Grid elevazione espansa -> {self.gnx}x{self.gny}, origine '
            f'({self.gox:.1f},{self.goy:.1f}).', throttle_duration_sec=2.0)

    # ---------------- pubblicazione (deriva costo) ----------------
    def publish_timer(self):
        # VALIDITA' su var_min (migliore confidenza storica), NON su var corrente:
        # una cella mappata bene una volta resta pubblicata anche se l'inflazione
        # ha fatto ricrescere la sua var. Cosi' le zone gia' viste non svaniscono.
        known = (self.n > 0) & (self.var_min < self.min_var_for_valid)
        if not np.any(known):
            return
        # SMOOTHING MEDIANO di h PRIMA di derivare: uccide il rumore residuo
        # cella-a-cella (causa degli aloni) preservando i bordi veri (cordoli).
        h_sm = median_smooth_masked(self.h, known, self.median_ksize)
        slope, rough, valid = derive_slope_roughness(
            h_sm, known, self.res, self.rough_win, self.rough_min_known)
        step = neighbor_step(h_sm, known, self.res)
        valid = valid & known
        cost = elevation_cost(slope, rough, step, valid,
                              self.slope_crit, self.rough_crit, self.step_crit,
                              self.slope_dead, self.rough_dead, self.step_dead)

        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.target
        msg.info.resolution = self.res
        msg.info.width = self.gnx; msg.info.height = self.gny
        msg.info.origin.position.x = float(self.gox)
        msg.info.origin.position.y = float(self.goy)
        msg.info.origin.orientation.w = 1.0
        data = np.clip(cost, -1, 100).astype(np.int8)
        msg.data = array.array('b', np.ascontiguousarray(data).tobytes())
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ElevationCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, 'autosave', False) and node.map_save_path:
            node.save_map(node.map_save_path)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()