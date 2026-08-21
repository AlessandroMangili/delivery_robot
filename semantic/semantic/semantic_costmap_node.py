import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
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
    1: 0, 9: 70, 0: 90,
    8: 100, 2: 100, 3: 100, 4: 100, 5: 100, 7: 100,
    11: 100, 12: 100, 13: 100, 14: 100, 15: 100, 16: 100, 17: 100, 18: 100,
}
DEFAULT_BLOCKERS = [2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
DEFAULT_DYNAMIC = [11, 12, 13, 14, 15, 16, 17, 18]


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


# ===== POSIZIONAMENTO LiDAR-PRIMARY (funzioni pure, testate in isolamento) =====
def transform_points(pts, M):
    """Applica la 4x4 M a punti Nx3 (p' = R p + t). Ritorna Nx3."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:3, :3].T + M[:3, 3][None, :]


def project_to_pixel(pts_cam, K, width, height):
    """Punti nel frame OTTICO camera -> (u,v) interi + maschera valida (davanti+dentro)."""
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


def position_points(cloud_scan, M_scan2map, M_scan2cam, K, width, height,
                    bz, max_obs_h, bx, by, map_write_max_range):
    """cloud_scan: Nx3 nel frame del LiDAR (base_scan).
    Ritorna uu, vv, X, Y, Xz, Yz, ok, dist (allineati ai punti LiDAR).
    La POSIZIONE viene dal LiDAR (map), la CLASSE si leggera' dal pixel (uu,vv)."""
    Pm = transform_points(cloud_scan, M_scan2map)   # punti in map (posizione)
    Pc = transform_points(cloud_scan, M_scan2cam)   # punti in camera ottica (per la classe)
    uu, vv, in_img = project_to_pixel(Pc, K, width, height)

    X = Pm[:, 0]; Y = Pm[:, 1]; Zm = Pm[:, 2]
    # overhead: scarta cio' che sta piu' in alto di max_obs_h sul robot
    # (tettoie/insegne/chiome: il robot ci passa sotto). z REALE dal LiDAR.
    overhead = (Zm - bz) > max_obs_h
    dist = np.hypot(X - bx, Y - by)
    ok = in_img & (~overhead) & (dist < map_write_max_range)

    Xz = np.where(np.isfinite(X), X, 0.0)
    Yz = np.where(np.isfinite(Y), Y, 0.0)
    return uu, vv, X, Y, Xz, Yz, ok, dist


class Tracker:
    """Nearest-neighbour world tracking + velocity estimate + moving/static classification."""
    def __init__(self, gate=1.0, v_thresh=0.3, min_obs=3, timeout=1.0, beta=0.6):
        self.gate = gate; self.v_thresh = v_thresh; self.min_obs = min_obs
        self.timeout = timeout; self.beta = beta
        self.tracks = []; self.next_id = 0

    def update(self, dets, t):
        """dets: list of (x,y) in world frame. Returns a track list aligned to dets."""
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
        self.declare_parameter('use_loc_quality_gate', True)
        self.declare_parameter('loc_pose_topic', '/amcl_pose')
        self.declare_parameter('loc_cov_source', 'pose_with_cov')
        self.declare_parameter('loc_std_suspend_m', 1.0)
        self.declare_parameter('loc_std_resume_m', 0.5)
        self.declare_parameter('loc_cov_ema', 0.4)
        self.declare_parameter('camera_optical_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('initial_size_m', 10.0)
        self.declare_parameter('grow_margin_m', 3.0)
        self.declare_parameter('max_range', 5.0)
        self.declare_parameter('map_write_max_range', 4.0)
        self.declare_parameter('occ_sector_deg', 2.0)
        self.declare_parameter('occ_margin', 0.15)
        self.declare_parameter('w_dist_min', 0.03)
        self.declare_parameter('conf_decay', 0.99)
        self.declare_parameter('conf_max', 8.0)
        self.declare_parameter('conf_init', 2.0)
        self.declare_parameter('confirm_gain', 1.0)
        self.declare_parameter('deny_penalty', 2.0)
        self.declare_parameter('new_hits', 2)
        self.declare_parameter('change_hits', 2)
        self.declare_parameter('max_obstacle_height', 0.20)
        self.declare_parameter('blocker_classes', DEFAULT_BLOCKERS)
        self.declare_parameter('dynamic_classes', DEFAULT_DYNAMIC)
        self.declare_parameter('dyn_v_thresh', 0.3)
        self.declare_parameter('dyn_gate', 1.0)
        self.declare_parameter('dyn_min_obs', 3)
        self.declare_parameter('dyn_timeout', 1.0)
        self.declare_parameter('dyn_vel_beta', 0.6)
        self.declare_parameter('map_static_dynamics', False)
        self.declare_parameter('dyn_dilate_px', 6)

        # --- LiDAR come sorgente di GEOMETRIA (LiDAR-primary) ---
        self.declare_parameter('lidar_topic', '/scan/points')
        self.declare_parameter('lidar_frame', 'base_scan')
        self.declare_parameter('lidar_max_age_s', 0.3)
        self.declare_parameter('lidar_min_range', 0.4) 

        gp = self.get_parameter
        self.target = gp('target_frame').value
        self.cam_frame = gp('camera_optical_frame').value
        self.base_frame = gp('base_frame').value
        self.res = gp('resolution').value
        self.initial_size_m = gp('initial_size_m').value
        self.grow_margin = gp('grow_margin_m').value
        self.max_range = gp('max_range').value
        self.map_write_max_range = float(gp('map_write_max_range').value)
        self.conf_decay = float(gp('conf_decay').value)
        self.conf_max = float(gp('conf_max').value)
        self.conf_init = float(gp('conf_init').value)
        self.confirm_gain = float(gp('confirm_gain').value)
        self.deny_penalty = float(gp('deny_penalty').value)
        self.new_hits = int(gp('new_hits').value)
        self.change_hits = int(gp('change_hits').value)
        self.get_logger().info(
            f'A cell enters the map after {self.new_hits} CONSECUTIVE observations '
            f'(the counter that keeps noise out of the map)')
        self.occ_dth = np.deg2rad(gp('occ_sector_deg').value)
        self.occ_nsec = int(np.ceil(2 * np.pi / self.occ_dth))
        self.occ_margin = gp('occ_margin').value
        self.w_dist_min = gp('w_dist_min').value
        self.map_static_dynamics = bool(gp('map_static_dynamics').value)
        self.dyn_dilate_px = int(gp('dyn_dilate_px').value)
        self.use_loc_gate = bool(gp('use_loc_quality_gate').value)
        self.loc_pose_topic = gp('loc_pose_topic').value
        self.loc_cov_source = gp('loc_cov_source').value
        self.loc_std_suspend = float(gp('loc_std_suspend_m').value)
        self.loc_std_resume = float(gp('loc_std_resume_m').value)
        self.loc_cov_ema = float(gp('loc_cov_ema').value)
        self.loc_std_smooth = 0.0
        self.loc_ok = True
        if self.use_loc_gate:
            from geometry_msgs.msg import PoseWithCovarianceStamped
            from nav_msgs.msg import Odometry as _Odom
            if self.loc_cov_source == 'odometry':
                self.create_subscription(_Odom, self.loc_pose_topic,
                                         self.loc_cb_odom, 10)
            else:
                self.create_subscription(PoseWithCovarianceStamped, self.loc_pose_topic,
                                         self.loc_cb_pwc, 10)
        self.max_obs_h = float(gp('max_obstacle_height').value)

        # --- LiDAR: cache della nuvola (sorgente di geometria) ---
        self.lidar_frame = gp('lidar_frame').value
        self.lidar_max_age = float(gp('lidar_max_age_s').value)
        self.lidar_min_range = float(gp('lidar_min_range').value)
        self.cloud = None            # Nx3 nel frame base_scan
        self.cloud_t = -1e9
        cloud_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PointCloud2, gp('lidar_topic').value,
                                 self.cloud_cb, cloud_qos)

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

        self.gnx = int(self.initial_size_m / self.res)
        self.gny = int(self.initial_size_m / self.res)
        self.gox = -self.initial_size_m / 2.0
        self.goy = -self.initial_size_m / 2.0
        self.grid = np.full((self.gny, self.gnx), -1.0, dtype=np.float32)
        self.conf = np.zeros((self.gny, self.gnx), dtype=np.float32)
        self.cls = np.full((self.gny, self.gnx), 255, dtype=np.uint8)
        self.cand = np.full((self.gny, self.gnx), 255, dtype=np.uint8)
        self.cand_n = np.zeros((self.gny, self.gnx), dtype=np.uint8)

        self.bridge = CvBridge()
        self.K = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 1)    # Change with real topic
        seg_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/semantic/segmentation', self.seg_cb, seg_qos) # Change with real topic

        latched_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(OccupancyGrid, '/semantic_costmap', latched_qos)
        self.pub_range = self.create_publisher(Marker, '/semantic_map_range', 1)

        self.declare_parameter('full_map_period_s', 1.5)
        self.full_map_period = float(self.get_parameter('full_map_period_s').value)
        self._last_full_pub = 0.0
        self._map_dirty = True
        self._last_decay_t = None
        self._significant_change = 0
        self.declare_parameter('significant_change_cells', 15)
        self.significant_thresh = int(self.get_parameter('significant_change_cells').value)

        self.cost_lut = np.full(256, -2, dtype=np.int16)
        for cls, c in CLASS_COST.items():
            self.cost_lut[cls] = c

        self.declare_parameter('map_load_name', '')
        self.declare_parameter('map_save_name', '')
        self.declare_parameter('autosave', False)
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
                self.get_logger().warn(f'Map to load not found: {load_path}')
        self.create_service(Trigger, '~/save_map', self.save_map_cb)

        self.create_timer(self.full_map_period, self._full_map_timer)

        self.autosave = bool(self.get_parameter('autosave').value)
        if self.autosave and not save_name:
            self.get_logger().warn(
                'autosave: true but map_save_name is EMPTY -> autosave DISABLED. '
                'Set a name in map_save_name.')
            self.autosave = False

        self.get_logger().info(f'Maps folder: {self.maps_dir}')
        save_hint = self.map_save_path if self.map_save_path else f'{self.maps_dir}/semantic_map.npz'
        if self.autosave:
            self.get_logger().info(
                f'AUTOSAVE on node SHUTDOWN (Ctrl-C) -> {save_hint}')
        self.get_logger().info(
            'Save on demand: '
            f'ros2 service call /semantic_costmap_node/save_map std_srvs/srv/Trigger  ->  {save_hint}')

        self.get_logger().info(
            f'Semantic costmap (LiDAR-primary) ready. frame={self.target}.')

    def resolve_maps_dir(self):
        """Find (or create) the maps/ folder in the semantic package SOURCE tree."""
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('semantic')
            ws = share
            for _ in range(8):
                ws = os.path.dirname(ws)
                src = os.path.join(ws, 'src')
                if os.path.isdir(src):
                    for root, dirs, files in os.walk(src):
                        if (os.path.basename(root) == 'semantic'
                                and 'package.xml' in files):
                            d = os.path.join(root, 'maps')
                            os.makedirs(d, exist_ok=True)
                            return d
                    break
            d = os.path.join(share, 'maps')
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
                    'Saved map has a different resolution: ignored.')
                return
            self.grid = d['grid'].astype(np.float32)
            self.conf = d['conf'].astype(np.float32)
            self.gny, self.gnx = self.grid.shape
            self.gox = float(d['gox']); self.goy = float(d['goy'])

            if 'cls' in d.files and d['cls'].shape == self.grid.shape:
                self.cls = d['cls'].astype(np.uint8)
            else:
                self.cls = np.full(self.grid.shape, 255, dtype=np.uint8)
                for c, cost in CLASS_COST.items():
                    self.cls[np.isclose(self.grid, cost)] = c
            self.cand = np.full(self.grid.shape, 255, dtype=np.uint8)
            self.cand_n = np.zeros(self.grid.shape, dtype=np.uint8)
            self.get_logger().info(
                f'Semantic map loaded from {path} ({self.gnx}x{self.gny}).')
        except Exception as e:
            self.get_logger().warn(f'Map load failed: {e}')

    def save_map(self, path):
        try:
            if not path.endswith('.npz'):
                path = path + '.npz'
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            np.savez_compressed(path, grid=self.grid, conf=self.conf,
                                cls=self.cls,
                                gnx=self.gnx, gny=self.gny, res=self.res,
                                gox=self.gox, goy=self.goy)
            self.get_logger().info(f'Semantic map saved to {path}.')
            return path
        except Exception as e:
            self.get_logger().error(f'Map save failed: {e}')
            return None

    def save_map_cb(self, request, response):
        path = self.map_save_path or os.path.join(self.maps_dir, 'semantic_map')
        saved = self.save_map(path)
        response.success = saved is not None
        response.message = (f'Map saved to {saved}' if saved else 'Save failed')
        return response

    def info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)

    def lookup(self, frame):
        try:
            return self.tf_buffer.lookup_transform(
                self.target, frame, rclpy.time.Time(), timeout=Duration(seconds=0.2))
        except Exception as e:
            self.get_logger().warn(f'TF unavailable {frame}->{self.target}: {e}',
                                   throttle_duration_sec=2.0)
            return None

    def ensure_capacity(self, X, Y):
        """Grow the grid (auto-grow) if world X,Y fall outside it."""
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
        new_cls = np.full((new_gny, new_gnx), 255, dtype=np.uint8)
        new_cand = np.full((new_gny, new_gnx), 255, dtype=np.uint8)
        new_cand_n = np.zeros((new_gny, new_gnx), dtype=np.uint8)
        new_grid[pb:pb + self.gny, pl:pl + self.gnx] = self.grid
        new_conf[pb:pb + self.gny, pl:pl + self.gnx] = self.conf
        new_cls[pb:pb + self.gny, pl:pl + self.gnx] = self.cls
        new_cand[pb:pb + self.gny, pl:pl + self.gnx] = self.cand
        new_cand_n[pb:pb + self.gny, pl:pl + self.gnx] = self.cand_n
        self.grid = new_grid; self.conf = new_conf
        self.cls = new_cls; self.cand = new_cand; self.cand_n = new_cand_n
        self.gnx = new_gnx; self.gny = new_gny
        self.gox -= pl * self.res; self.goy -= pb * self.res
        self.get_logger().info(
            f'Grid expanded -> {self.gnx}x{self.gny} cells, origin '
            f'({self.gox:.1f},{self.goy:.1f}).', throttle_duration_sec=2.0)

    def loc_cb_pwc(self, msg):
        """Pose covariance from AMCL (PoseWithCovarianceStamped)."""
        self._update_loc_quality(msg.pose.covariance)

    def loc_cb_odom(self, msg):
        """Pose covariance from EKF (Odometry). Same 6x6 layout."""
        self._update_loc_quality(msg.pose.covariance)

    def _update_loc_quality(self, cov):
        """ROBUST gate: EMA on pose std + HYSTERESIS."""
        var_x = max(0.0, float(cov[0]))
        var_y = max(0.0, float(cov[7]))
        std_xy = float(np.sqrt(max(var_x, var_y)))
        a = self.loc_cov_ema
        self.loc_std_smooth = a * std_xy + (1.0 - a) * self.loc_std_smooth
        if self.loc_ok and self.loc_std_smooth > self.loc_std_suspend:
            self.loc_ok = False
            self.get_logger().warn(
                f'LOCALIZATION LOST (smoothed std {self.loc_std_smooth:.2f}m > '
                f'{self.loc_std_suspend}m) -> MAPPING SUSPENDED. The existing map '
                f'stays published; recalibrate localization to resume.')
        elif (not self.loc_ok) and self.loc_std_smooth < self.loc_std_resume:
            self.loc_ok = True
            self.get_logger().info(
                f'Localization RECOVERED (smoothed std {self.loc_std_smooth:.2f}m < '
                f'{self.loc_std_resume}m) -> mapping RESUMED.')

    def cloud_cb(self, msg: PointCloud2):
        """Nuvola LiDAR (frame base_scan): messa in cache, usata dal seg_cb per
        posizionare la semantica (LiDAR-primary)."""
        try:
            rec = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception:
            return
        n = int(rec.shape[0])
        if n == 0:
            return
        pts = np.empty((n, 3), dtype=np.float64)
        pts[:, 0] = rec['x']; pts[:, 1] = rec['y']; pts[:, 2] = rec['z']

    def seg_cb(self, msg: Image):
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.K is None:
            return
        if self.use_loc_gate and not self.loc_ok:
            T_base = self.lookup(self.base_frame)
            if T_base is not None:
                self.publish_map(T_base.transform.translation.x,
                                 T_base.transform.translation.y, msg.header.stamp)
            return
        seg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        h, w = seg.shape

        T_cam = self.lookup(self.cam_frame)
        T_base = self.lookup(self.base_frame)
        if T_cam is None or T_base is None:
            return
        M = transform_to_matrix(T_cam)
        origin = M[:3, 3]                      # posizione camera in map (serve alle occlusioni)
        bx, by = T_base.transform.translation.x, T_base.transform.translation.y
        bz = T_base.transform.translation.z
        qb = T_base.transform.rotation
        yaw_b = np.arctan2(2.0 * (qb.w * qb.z + qb.x * qb.y),
                           1.0 - 2.0 * (qb.y * qb.y + qb.z * qb.z))

        # ===== POSIZIONAMENTO LiDAR-PRIMARY =====
        # La posizione di ogni punto viene MISURATA dal LiDAR 3D (geometria accurata,
        # z reale, occlusioni native: il raggio non attraversa i muri). La classe
        # viene letta dal pixel di segmentazione su cui il punto si proietta.
        T_scan = self.lookup(self.lidar_frame)          # map <- base_scan
        if self.cloud is None or T_scan is None:
            if self.cloud is None:
                self.get_logger().warn(
                    'Nessuna nuvola LiDAR ricevuta: non si mappa nulla. '
                    'Controlla il topic del LiDAR.', throttle_duration_sec=10.0)
            self.publish_map(bx, by, msg.header.stamp)
            return
        if (stamp_sec - self.cloud_t) > self.lidar_max_age:
            self.get_logger().warn('Nuvola LiDAR troppo vecchia in questo frame.',
                                   throttle_duration_sec=10.0)
            self.publish_map(bx, by, msg.header.stamp)
            return

        M_scan2map = transform_to_matrix(T_scan)
        M_scan2cam = np.linalg.inv(M) @ M_scan2map      # base_scan -> camera ottica
        uu, vv, X, Y, Xz, Yz, ok, dist = position_points(
            self.cloud, M_scan2map, M_scan2cam, self.K, w, h,
            bz, self.max_obs_h, bx, by, self.map_write_max_range)

        classes = seg[vv, uu]
        costs = self.cost_lut[classes]
        ok = ok & (costs >= 0)

        # --- esclusione dinamici (per classe) + occlusioni + pesi ---
        full_dyn = self.dyn_lut[seg].astype(np.uint8)
        if full_dyn.any():
            if self.dyn_dilate_px > 0:
                k = 2 * self.dyn_dilate_px + 1
                kern = np.ones((k, k), np.uint8)
                dyn_mask = cv2.dilate(full_dyn, kern, iterations=1)
            else:
                dyn_mask = full_dyn
            in_dyn = dyn_mask[vv, uu].astype(bool)

            if not self.map_static_dynamics:
                ok = ok & (~in_dyn)
            else:
                _, lbl = cv2.connectedComponents(full_dyn)
                comp_s = lbl[vv, uu]
                dyn_pix = self.dyn_lut[classes] & ok
                dets = []; det_cids = []
                for cid in np.unique(comp_s[dyn_pix]):
                    if cid == 0:
                        continue
                    m = (comp_s == cid) & ok
                    if not m.any():
                        continue
                    idx = np.where(m)[0]
                    base_i = idx[np.argmax(vv[idx])]
                    dets.append((float(X[base_i]), float(Y[base_i]))); det_cids.append(cid)
                assign = self.tracker.update(dets, stamp_sec)
                keep = np.zeros_like(in_dyn)
                for cid, tr in zip(det_cids, assign):
                    if self.tracker.confirmed_static(tr):
                        keep |= (comp_s == cid)
                ok = ok & (~(in_dyn & ~keep))

        camx, camy = origin[0], origin[1]
        ang = np.arctan2(Yz - camy, Xz - camx)
        rad = np.hypot(Xz - camx, Yz - camy)
        sec = np.clip(((ang + np.pi) / self.occ_dth).astype(int), 0, self.occ_nsec - 1)
        is_block = self.block_lut[classes] & ok
        occ_r = np.full(self.occ_nsec, np.inf, dtype=np.float64)
        if is_block.any():
            np.minimum.at(occ_r, sec[is_block], rad[is_block])
        occluded = rad > (occ_r[sec] + self.occ_margin)

        ok = ok & (~occluded)
        w_dist = np.clip(1.0 - dist / self.max_range, self.w_dist_min, 1.0)
        wpix = w_dist

        okx = ok & np.isfinite(X) & np.isfinite(Y)
        if okx.any():
            self.ensure_capacity(X[okx], Y[okx])

        gi = ((Xz - self.gox) / self.res).astype(int)
        gj = ((Yz - self.goy) / self.res).astype(int)
        inside = ok & (gi >= 0) & (gi < self.gnx) & (gj >= 0) & (gj < self.gny)

        if np.any(inside):
            dxr = X[inside] - bx; dyr = Y[inside] - by
            cy_, sy_ = np.cos(yaw_b), np.sin(yaw_b)
            fwd = dxr * cy_ + dyr * sy_
            lat = -dxr * sy_ + dyr * cy_
            f_lo, f_hi = float(fwd.min()), float(fwd.max())
            if f_hi - f_lo > 0.2:
                band = 0.25 * (f_hi - f_lo)
                near = fwd <= (f_lo + band)
                far = fwd >= (f_hi - band)
                if np.any(near) and np.any(far):
                    self._hull = (
                        f_lo, float(lat[near].min()), float(lat[near].max()),
                        f_hi, float(lat[far].min()), float(lat[far].max()),
                        float(bx), float(by), float(yaw_b))

        gi_u = gi[inside]; gj_u = gj[inside]
        co_u = costs[inside].astype(np.float32)
        cl_u = classes[inside].astype(np.uint8)
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

        frame_cls = np.full(N, 255, dtype=np.uint8)
        frame_cls[flat_s] = cl_u[order]

        V = self.grid.ravel(); C = self.conf.ravel()
        CL = self.cls.ravel(); CD = self.cand.ravel(); CN = self.cand_n.ravel()
        V_before = V.copy()
        w = np.where(seen, frame_w, 0.0).astype(np.float32)

        known = seen & (CL != 255)
        agree = known & (frame_cls == CL)
        deny = known & (frame_cls != CL)
        fresh = seen & (CL == 255)

        C[agree] = np.minimum(C[agree] + self.confirm_gain * w[agree], self.conf_max)
        CD[agree] = 255
        CN[agree] = 0

        C[deny] = C[deny] - self.deny_penalty * w[deny]
        same_cand = deny & (CD == frame_cls)
        new_cand = deny & (CD != frame_cls)
        CN[same_cand] = np.minimum(CN[same_cand] + 1, 255)
        CD[new_cand] = frame_cls[new_cand]
        CN[new_cand] = 1

        CD[fresh & (CD != frame_cls)] = frame_cls[fresh & (CD != frame_cls)]
        CN[fresh & (CD != frame_cls)] = 0
        CN[fresh] = np.minimum(CN[fresh] + 1, 255)

        commit = ((fresh & (CN >= self.new_hits)) |
                  (deny & (C <= 0.0) & (CN >= self.change_hits)))
        if commit.any():
            CL[commit] = frame_cls[commit]
            V[commit] = self.cost_lut[frame_cls[commit]].astype(np.float32)
            C[commit] = self.conf_init
            CD[commit] = 255
            CN[commit] = 0

        np.clip(C, 0.0, self.conf_max, out=C)
        accept = commit

        if accept.any():
            self._map_dirty = True
            self._dirty_since_save = True
            old_vals = V_before[accept]; new_vals = V[accept]
            crossed = ((old_vals < 50) & (new_vals >= 50)) | \
                      ((old_vals >= 50) & (new_vals < 50)) | (old_vals < 0)
            self._significant_change = int(np.count_nonzero(crossed))

        if self.conf_decay < 1.0:
            if self._last_decay_t is not None:
                dt = stamp_sec - self._last_decay_t
                if dt > 0.0:
                    C[~seen] *= self.conf_decay ** dt
            self._last_decay_t = stamp_sec

        self.grid = V.reshape(self.gny, self.gnx)
        self.conf = C.reshape(self.gny, self.gnx)
        self.cls = CL.reshape(self.gny, self.gnx)
        self.cand = CD.reshape(self.gny, self.gnx)
        self.cand_n = CN.reshape(self.gny, self.gnx)
        self.publish_map(bx, by, msg.header.stamp)

    def _full_map_timer(self):
        """Republish the full map periodically, INDEPENDENTLY of camera and TF."""
        if self.grid is None:
            return
        stamp = self.get_clock().now().to_msg()
        full = self._grid_to_msg(self.grid, self.gox, self.goy,
                                 self.gnx, self.gny, stamp)
        self.pub.publish(full)
        self._last_full_pub = self.get_clock().now().nanoseconds * 1e-9
        self._map_dirty = False

    def _grid_to_msg(self, sub, ox, oy, width, height, stamp):
        """Convert a grid slice into an OccupancyGrid."""
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

        periodic_due = (now_s - self._last_full_pub) >= self.full_map_period
        urgent = self._significant_change >= self.significant_thresh
        if urgent:
            full = self._grid_to_msg(self.grid, self.gox, self.goy,
                                     self.gnx, self.gny, stamp)
            self.pub.publish(full)
            self.get_logger().info(
                f'Significant change ({self._significant_change} cells) '
                f'-> full map published IMMEDIATELY', throttle_duration_sec=1.0)
            self._last_full_pub = now_s
            self._map_dirty = False
            self._significant_change = 0

        self.publish_range_marker(bx, by, stamp)

    def publish_range_marker(self, bx, by, stamp):
        """Outline of the area actually written to the map."""
        h = getattr(self, '_hull', None)
        if h is None or len(h) != 9:
            return
        f_lo, ln_min, ln_max, f_hi, lf_min, lf_max, hx, hy, hyaw = h

        from geometry_msgs.msg import Point
        c, sn = np.cos(hyaw), np.sin(hyaw)

        def P(f, l):
            return Point(x=float(hx + f * c - l * sn),
                         y=float(hy + f * sn + l * c), z=0.05)

        m = Marker()
        m.header.frame_id = self.target
        m.header.stamp = stamp
        m.ns = 'map_range'; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r = 1.0; m.color.g = 0.9; m.color.b = 0.0; m.color.a = 0.9
        m.points = [P(f_lo, ln_max), P(f_hi, lf_max),
                    P(f_hi, lf_min), P(f_lo, ln_min), P(f_lo, ln_max)]
        self.pub_range.publish(m)
        
    def cloud_cb(self, msg: PointCloud2):
        """Nuvola LiDAR (frame base_scan): messa in cache, usata dal seg_cb per
        posizionare la semantica (LiDAR-primary). Qui la nuvola viene PULITA una
        volta sola (unico punto d'ingresso): si scartano i punti non finiti
        (NaN/inf dei raggi senza ritorno del gpu_lidar) e quelli entro
        lidar_min_range (rumore a corto raggio, come il 'blind' di FAST-LIO)."""
        try:
            rec = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception:
            return
        n = int(rec.shape[0])
        if n == 0:
            return
        pts = np.empty((n, 3), dtype=np.float64)
        pts[:, 0] = rec['x']; pts[:, 1] = rec['y']; pts[:, 2] = rec['z']

        # scarta NaN/inf (skip_nans toglie i NaN ma NON gli inf del gpu_lidar)
        finite = np.isfinite(pts).all(axis=1)
        # scarta il rumore a corto raggio (range 3D nel frame base_scan)
        r2 = pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2
        keep = finite & (r2 >= self.lidar_min_range ** 2)
        pts = pts[keep]
        if pts.shape[0] == 0:
            return

        self.cloud = pts
        self.cloud_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def main():
    rclpy.init()
    node = SemanticCostmapNode()
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