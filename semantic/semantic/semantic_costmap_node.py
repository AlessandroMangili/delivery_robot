import array
import os
import signal
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from semantic.map_jump_gate import MapJumpGate, lookup_via_odom

CLASS_COST = {
    1: 0, 9: 70, 0: 90,
    8: 100, 2: 100, 3: 100, 4: 100, 5: 100, 7: 100,
    11: 100, 12: 100, 13: 100, 14: 100, 15: 100, 16: 100, 17: 100, 18: 100,
}
DEFAULT_BLOCKERS = [2, 3, 4, 5, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]
DEFAULT_DYNAMIC = [11, 12, 13, 14, 15, 16, 17, 18]


N_CLASSES = 19


def cost_vector(class_cost, n_classes=N_CLASSES):
    c = np.full(n_classes, np.nan, dtype=np.float32)
    for k, v in class_cost.items():
        if 0 <= int(k) < n_classes:
            c[int(k)] = float(v)
    return c


def fuse(alpha, flat, cls, w, alpha_max, valid):
    cls = np.asarray(cls)
    ok = (cls >= 0) & (cls < alpha.shape[1]) & (w > 0)
    ok[ok] = valid[cls[ok]]
    flat = flat[ok]
    cls = cls[ok].astype(np.int64)
    w = w[ok].astype(np.float32)
    if flat.size == 0:
        return np.empty(0, dtype=np.int64)
    np.add.at(alpha, (flat, cls), w)
    cells = np.unique(flat)
    s = alpha[cells].sum(axis=1)
    over = s > alpha_max
    if np.any(over):
        alpha[cells[over]] *= (alpha_max / s[over])[:, None]
    return cells


def stacked_classes(gi, gj, Z, cls, gnx, gny, targets, low_min, low_max, high_max, tol, min_votes, n_classes=N_CLASSES):
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if gi.size == 0 or targets.size == 0:
        return empty
    flat = gj.astype(np.int64) * gnx + gi
    order = np.argsort(flat, kind='stable')
    fs = flat[order]
    first = np.r_[True, fs[1:] != fs[:-1]]
    ucell = fs[first]
    zmin = np.minimum.reduceat(Z[order], np.nonzero(first)[0])
    ucy, ucx = ucell // gnx, ucell % gnx
    ground = zmin.copy()
    for dy in range(-tol, tol + 1):
        for dx in range(-tol, tol + 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = ucy + dy, ucx + dx
            okn = (ny >= 0) & (ny < gny) & (nx >= 0) & (nx < gnx)
            key = ny * gnx + nx
            pos = np.searchsorted(ucell, key)
            pos = np.clip(pos, 0, ucell.size - 1)
            hit = okn & (ucell[pos] == key)
            ground[hit] = np.minimum(ground[hit], zmin[pos[hit]])
    hrel = Z - ground[np.searchsorted(ucell, flat)]
    low = (hrel >= low_min) & (hrel <= low_max)
    struct = (hrel >= low_min) & (hrel <= high_max)
    if not np.any(low):
        return empty
    tgt = np.unique(targets)
    flagged = np.zeros(tgt.size, dtype=bool)
    lcell = np.unique(flat[low])
    ly, lx = lcell // gnx, lcell % gnx
    sflat = flat[struct]; sy, sx = sflat // gnx, sflat % gnx; scls = cls[struct].astype(np.int64)
    keys, kcls = [], []
    for dy in range(-tol, tol + 1):
        for dx in range(-tol, tol + 1):
            ny, nx = ly + dy, lx + dx
            okn = (ny >= 0) & (ny < gny) & (nx >= 0) & (nx < gnx)
            key = ny[okn] * gnx + nx[okn]
            pos = np.clip(np.searchsorted(tgt, key), 0, tgt.size - 1)
            flagged[pos[tgt[pos] == key]] = True
            ny, nx = sy + dy, sx + dx
            okn = (ny >= 0) & (ny < gny) & (nx >= 0) & (nx < gnx)
            keys.append(ny[okn] * gnx + nx[okn]); kcls.append(scls[okn])
    ftgt = tgt[flagged]
    if ftgt.size == 0:
        return empty
    keys = np.concatenate(keys); kcls = np.concatenate(kcls)
    pos = np.clip(np.searchsorted(ftgt, keys), 0, ftgt.size - 1)
    m = ftgt[pos] == keys
    if not np.any(m):
        return empty
    pair = pos[m].astype(np.int64) * n_classes + kcls[m]
    up, cnt = np.unique(pair, return_counts=True)
    cell_i, cl = up // n_classes, up % n_classes
    o = np.lexsort((cnt, cell_i))
    cell_i, cl, cnt = cell_i[o], cl[o], cnt[o]
    lastm = np.r_[cell_i[1:] != cell_i[:-1], True]
    tot = np.bincount(cell_i, weights=cnt.astype(np.float64), minlength=ftgt.size)
    best_cell, best_cl = cell_i[lastm], cl[lastm]
    keep = tot[best_cell] >= min_votes
    return ftgt[best_cell[keep]], best_cl[keep]


def free_ground_cells(O, P, gox, goy, res, gnx, gny, bz, low_h, ground_tol, end_margin, max_range, stride, target):
    P = P[::max(int(stride), 1)]
    P = P[np.all(np.isfinite(P), axis=1)]
    P = P[np.abs(P[:, 2] - bz) < ground_tol]
    D = P - O
    L = np.hypot(D[:, 0], D[:, 1])
    dz = D[:, 2]
    keep = (L > 0.5) & (L < max_range) & (dz < -0.05)
    D, L, dz = D[keep], L[keep], dz[keep]
    if L.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    a = np.maximum(L * (1.0 - low_h / (-dz)), 0.0)
    b = L - end_margin
    ok = (b - a) > res
    a, b, D, L = a[ok], b[ok], D[ok], L[ok]
    if L.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    n = np.floor((b - a) / res).astype(np.int64) + 1
    ray = np.repeat(np.arange(n.size), n)
    start = np.repeat(np.cumsum(n) - n, n)
    dist = a[ray] + (np.arange(ray.size) - start) * res
    f = dist / L[ray]
    gi = np.floor((O[0] + f * D[ray, 0] - gox) / res).astype(np.int64)
    gj = np.floor((O[1] + f * D[ray, 1] - goy) / res).astype(np.int64)
    okc = (gi >= 0) & (gi < gnx) & (gj >= 0) & (gj < gny)
    flat = gj[okc] * gnx + gi[okc]
    ray = ray[okc]
    sel = target[flat]
    flat, ray = flat[sel], ray[sel]
    key = np.unique(flat * n.size + ray)
    return np.unique(key // n.size, return_counts=True)


def expected_cost(alpha, costs, known, lethal_classes, p_lethal):
    usable = np.isfinite(costs)
    a = alpha[:, usable]
    s = a.sum(axis=1)
    safe = np.where(s > 0, s, 1.0)
    exp_cost = (a @ costs[usable]) / safe
    lethal_mask = np.isin(np.flatnonzero(usable), lethal_classes)
    p_obst = a[:, lethal_mask].sum(axis=1) / safe
    out = np.where(p_obst >= p_lethal, 100.0, np.minimum(exp_cost, 99.0))
    return np.where(known & (s > 0), np.round(out), -1.0).astype(np.float32)


def majority_class(alpha):
    s = alpha.sum(axis=1)
    return np.where(s > 0, np.argmax(alpha, axis=1), 255).astype(np.uint8)


def grid_to_i8(v):
    return np.where(v < 0, -1, np.clip(np.round(v), 0, 100)).astype(np.int8)


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


def transform_points(pts, M):
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:3, :3].T + M[:3, 3][None, :]


def project_to_pixel(pts_cam, K, width, height):
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
    Pm = transform_points(cloud_scan, M_scan2map)
    Pc = transform_points(cloud_scan, M_scan2cam)
    uu, vv, in_img = project_to_pixel(Pc, K, width, height)

    X = Pm[:, 0]; Y = Pm[:, 1]; Zm = Pm[:, 2]
    overhead = (Zm - bz) > max_obs_h
    dist = np.hypot(X - bx, Y - by)
    ok = in_img & (~overhead) & (dist < map_write_max_range)
    high = in_img & overhead & (dist < map_write_max_range) & np.isfinite(Zm)

    Xz = np.where(np.isfinite(X), X, 0.0)
    Yz = np.where(np.isfinite(Y), Y, 0.0)
    return uu, vv, X, Y, Xz, Yz, ok, dist, Zm, high, in_img


class SemanticCostmapNode(Node):
    def __init__(self):
        super().__init__('semantic_costmap_node')

        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('camera_optical_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('camera_info_topic', '/camera_info')
        self.declare_parameter('seg_topic', '/semantic/segmentation')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('initial_size_m', 10.0)
        self.declare_parameter('grow_margin_m', 3.0)
        self.declare_parameter('max_range', 5.0)
        self.declare_parameter('map_write_max_range', 4.0)
        self.declare_parameter('occ_sector_deg', 2.0)
        self.declare_parameter('occ_margin', 0.15)
        self.declare_parameter('w_dist_min', 0.03)
        self.declare_parameter('alpha_min', 2.0)
        self.declare_parameter('alpha_max', 10.0)
        self.declare_parameter('p_lethal', 0.5)
        self.declare_parameter('tf_timeout_s', 0.1)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('map_jump_gate_m', 0.25)
        self.declare_parameter('map_jump_gate_deg', 2.0)
        self.declare_parameter('map_jump_hold_s', 1.0)
        self.declare_parameter('max_obstacle_height', 0.20)
        self.declare_parameter('stacked_vote', False)
        self.declare_parameter('ray_clear_enable', False)
        self.declare_parameter('ray_clear_every_n', 5)
        self.declare_parameter('ray_clear_stride', 4)
        self.declare_parameter('ray_clear_min_h', 0.15)
        self.declare_parameter('ray_clear_end_margin', 0.3)
        self.declare_parameter('ray_clear_min_rays', 3)
        self.declare_parameter('ray_clear_step', 1.0)
        self.declare_parameter('ray_clear_protect_s', 10.0)
        self.declare_parameter('ray_clear_protect_cells', 1)
        self.declare_parameter('ost_write_range_m', 0.0)
        self.declare_parameter('footprint_clear', False)
        self.declare_parameter('footprint_clear_half_length', 0.40)
        self.declare_parameter('footprint_clear_half_width', 0.28)
        self.declare_parameter('ost_classes', [2, 3, 4, 5])
        self.declare_parameter('ray_clear_low_h', 0.12)
        self.declare_parameter('ray_clear_ground_tol', 0.25)
        self.declare_parameter('stacked_low_min', 0.15)
        self.declare_parameter('stacked_high_max', 2.0)
        self.declare_parameter('stacked_tol_cells', 1)
        self.declare_parameter('stacked_min_votes', 3)
        self.declare_parameter('blocker_classes', DEFAULT_BLOCKERS)
        self.declare_parameter('dynamic_classes', DEFAULT_DYNAMIC)
        self.declare_parameter('dyn_dilate_px', 6)

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
        self.alpha_min = float(gp('alpha_min').value)
        self.alpha_max = float(gp('alpha_max').value)
        self.p_lethal = float(gp('p_lethal').value)
        self.tf_timeout = float(gp('tf_timeout_s').value)
        self.odom_frame = gp('odom_frame').value
        self.jump_gate = (MapJumpGate(gp('map_jump_gate_m').value, gp('map_jump_gate_deg').value,
                                      gp('map_jump_hold_s').value)
                          if self.target != self.odom_frame else None)
        self.cost_vec = cost_vector(CLASS_COST)
        self.valid_cls = np.isfinite(self.cost_vec)
        self.get_logger().info(
            f'Vote fusion: a cell is known from alpha {self.alpha_min}, cap {self.alpha_max}, '
            f'lethal from P(obstacle) {self.p_lethal}')
        self.occ_dth = np.deg2rad(gp('occ_sector_deg').value)
        self.occ_nsec = int(np.ceil(2 * np.pi / self.occ_dth))
        self.occ_margin = gp('occ_margin').value
        self.w_dist_min = gp('w_dist_min').value
        self.dyn_dilate_px = int(gp('dyn_dilate_px').value)
        self.max_obs_h = float(gp('max_obstacle_height').value)
        self.rc_on = bool(gp('ray_clear_enable').value)
        self.rc_every = max(int(gp('ray_clear_every_n').value), 1)
        self.rc_stride = int(gp('ray_clear_stride').value)
        self.rc_min_h = float(gp('ray_clear_min_h').value)
        self.rc_margin = float(gp('ray_clear_end_margin').value)
        self.rc_min_rays = int(gp('ray_clear_min_rays').value)
        self.rc_step = float(gp('ray_clear_step').value)
        self.rc_protect = float(gp('ray_clear_protect_s').value)
        self.rc_dil = int(gp('ray_clear_protect_cells').value)
        self.ost_range = float(gp('ost_write_range_m').value)
        self.fp_clear = bool(gp('footprint_clear').value)
        self.fp_hl = float(gp('footprint_clear_half_length').value)
        self.fp_hw = float(gp('footprint_clear_half_width').value)
        self.fp_stats = [0, 0]
        self.fp_last = time.monotonic()
        self.ost_classes = [int(c) for c in gp('ost_classes').value]
        self.rc_low_h = float(gp('ray_clear_low_h').value)
        self.rc_ground_tol = float(gp('ray_clear_ground_tol').value)
        self.rc_n = 0
        self.rc_stats = [0, 0, 0, 0.0]
        self.rc_last = time.monotonic()
        self.stacked_on = bool(gp('stacked_vote').value)
        self.stacked_low_min = float(gp('stacked_low_min').value)
        self.stacked_high_max = float(gp('stacked_high_max').value)
        self.stacked_tol = int(gp('stacked_tol_cells').value)
        self.stacked_min_votes = int(gp('stacked_min_votes').value)
        self._st_frames = 0
        self._st_cells = 0
        self._st_changed = 0
        self._st_time = 0.0
        self._st_last = time.monotonic()

        self.lidar_frame = gp('lidar_frame').value
        self.lidar_max_age = float(gp('lidar_max_age_s').value)
        self.lidar_min_range = float(gp('lidar_min_range').value)
        self.cloud = None
        self.cloud_stamp = None
        self.cloud_t = -1e9
        self.declare_parameter('update', True)
        self.declare_parameter('frozen_full_map_period_s', 5.0)
        self.update = bool(self.get_parameter('update').value)
        if self.update:
            cloud_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                   history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(PointCloud2, gp('lidar_topic').value,
                                     self.cloud_cb, cloud_qos)

        blockers = gp('blocker_classes').value
        dynamic = gp('dynamic_classes').value

        self.lethal_classes = [int(c) for c in blockers]
        self.block_lut = np.zeros(256, dtype=bool)
        for c in blockers:
            self.block_lut[int(c)] = True
        self.dyn_lut = np.zeros(256, dtype=bool)
        for c in dynamic:
            self.dyn_lut[int(c)] = True

        self.gnx = int(self.initial_size_m / self.res)
        self.gny = int(self.initial_size_m / self.res)
        self.gox = -self.initial_size_m / 2.0
        self.goy = -self.initial_size_m / 2.0
        self.grid = np.full((self.gny, self.gnx), -1.0, dtype=np.float32)
        self.grid_i8 = np.full((self.gny, self.gnx), -1, dtype=np.int8)
        self.alpha = np.zeros((self.gny, self.gnx, N_CLASSES), dtype=np.float32)
        self.known = np.zeros((self.gny, self.gnx), dtype=bool)
        self.t_hit = np.full((self.gny, self.gnx), -np.inf, dtype=np.float64)

        self.bridge = CvBridge()
        self.K = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, None, spin_thread=True)

        if self.update:
            self.create_subscription(CameraInfo, gp('camera_info_topic').value, self.info_cb, 1)
            seg_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(Image, gp('seg_topic').value, self.seg_cb, seg_qos)

        latched_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(OccupancyGrid, '/semantic_costmap', latched_qos)
        self.pub_range = self.create_publisher(Marker, '/semantic_map_range', 1)

        self.declare_parameter('full_map_period_s', 1.5)
        self.full_map_period = float(self.get_parameter('full_map_period_s').value)
        self._significant_change = 0
        self.declare_parameter('significant_change_cells', 15)
        self.significant_thresh = int(self.get_parameter('significant_change_cells').value)

        self.cost_lut = np.full(256, -2, dtype=np.int16)
        for cls, c in CLASS_COST.items():
            self.cost_lut[cls] = c

        self.declare_parameter('map_name', '')
        self.maps_dir = self.resolve_maps_dir()
        map_name = self.get_parameter('map_name').value
        load_name = save_name = f'semantic_{map_name}' if map_name else ''
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

        self.create_timer(self.full_map_period if self.update
                          else float(self.get_parameter('frozen_full_map_period_s').value), self._full_map_timer)
        if not self.update:
            self.get_logger().info('Frozen semantic map: publishing the loaded map only, no updates.')

        self.autosave = bool(map_name) and self.update

        self.get_logger().info(f'Maps folder: {self.maps_dir}')
        save_hint = self.map_save_path if self.map_save_path else f'{self.maps_dir}/semantic_map.npz'
        if self.autosave:
            self.get_logger().info(
                f'The map is saved on Ctrl-C to {save_hint}')
        self.get_logger().info(
            'Save on request: '
            f'ros2 service call /semantic_costmap_node/save_map std_srvs/srv/Trigger  ->  {save_hint}')

        self.get_logger().info(
            f'Semantic map ready: frame {self.target}.')

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
            grid = d['grid'].astype(np.float32)
            gny, gnx = grid.shape
            if 'alpha' in d.files and d['alpha'].shape == (gny, gnx, N_CLASSES):
                alpha = d['alpha'].astype(np.float32)
                known = d['known'].astype(bool)
            else:
                alpha = np.zeros((gny, gnx, N_CLASSES), dtype=np.float32)
                cls = d['cls'].astype(np.int64) if 'cls' in d.files else None
                if cls is None:
                    cls = np.full(grid.shape, 255, dtype=np.int64)
                    for c, cost in CLASS_COST.items():
                        cls[np.isclose(grid, cost)] = c
                old = cls < N_CLASSES
                alpha[old, cls[old]] = self.alpha_min
                known = old
            self.alpha = alpha
            self.known = known
            self.gny, self.gnx = gny, gnx
            self.gox = float(d['gox']); self.goy = float(d['goy'])
            self.grid = expected_cost(self.alpha.reshape(-1, N_CLASSES), self.cost_vec,
                                      self.known.reshape(-1), self.lethal_classes,
                                      self.p_lethal).reshape(gny, gnx)
            self.grid_i8 = grid_to_i8(self.grid)
            self.t_hit = np.full((gny, gnx), -np.inf, dtype=np.float64)
            self.get_logger().info(
                f'Semantic map loaded from {path} ({self.gnx}x{self.gny}).')
        except Exception as e:
            self.get_logger().warn(f'Map load failed: {e}')

    def save_map(self, path):
        try:
            if not path.endswith('.npz'):
                path = path + '.npz'
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = path[:-4] + '.tmp.npz'
            with open(tmp, 'wb') as f:
                np.savez_compressed(f, grid=self.grid, alpha=self.alpha, known=self.known,
                                    cls=majority_class(self.alpha.reshape(-1, N_CLASSES)).reshape(
                                        self.gny, self.gnx),
                                    gnx=self.gnx, gny=self.gny, res=self.res,
                                    gox=self.gox, goy=self.goy)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
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

    def lookup(self, frame, stamp):
        try:
            if self.target != self.odom_frame:
                return lookup_via_odom(self.tf_buffer, self.target, self.odom_frame, frame, stamp, self.tf_timeout)
            return self.tf_buffer.lookup_transform(
                self.target, frame, rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=self.tf_timeout))
        except Exception as e:
            self.get_logger().warn(f'Transform {frame} -> {self.target} not available: {e}',
                                   throttle_duration_sec=2.0)
            return None

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
        new_gnx = self.gnx + pl + pr
        new_gny = self.gny + pb + pt
        new_grid = np.full((new_gny, new_gnx), -1.0, dtype=np.float32)
        new_alpha = np.zeros((new_gny, new_gnx, N_CLASSES), dtype=np.float32)
        new_known = np.zeros((new_gny, new_gnx), dtype=bool)
        new_grid[pb:pb + self.gny, pl:pl + self.gnx] = self.grid
        new_alpha[pb:pb + self.gny, pl:pl + self.gnx] = self.alpha
        new_known[pb:pb + self.gny, pl:pl + self.gnx] = self.known
        new_i8 = np.full((new_gny, new_gnx), -1, dtype=np.int8)
        new_i8[pb:pb + self.gny, pl:pl + self.gnx] = self.grid_i8
        self.grid_i8 = new_i8
        new_th = np.full((new_gny, new_gnx), -np.inf, dtype=np.float64)
        new_th[pb:pb + self.gny, pl:pl + self.gnx] = self.t_hit
        self.t_hit = new_th
        self.grid = new_grid
        self.alpha = new_alpha
        self.known = new_known
        self.gnx = new_gnx; self.gny = new_gny
        self.gox -= pl * self.res; self.goy -= pb * self.res
        self.get_logger().info(
            f'Grid expanded -> {self.gnx}x{self.gny} cells, origin '
            f'({self.gox:.1f},{self.goy:.1f}).', throttle_duration_sec=2.0)

    def seg_cb(self, msg: Image):
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.K is None:
            return
        seg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        h, w = seg.shape

        T_cam = self.lookup(self.cam_frame, msg.header.stamp)
        T_base = self.lookup(self.base_frame, msg.header.stamp)
        if T_cam is None or T_base is None:
            return
        M = transform_to_matrix(T_cam)
        origin = M[:3, 3]
        bx, by = T_base.transform.translation.x, T_base.transform.translation.y
        bz = T_base.transform.translation.z
        if not self.map_jump_ok(msg.header.stamp, stamp_sec, bx, by, bz):
            self.publish_map(bx, by, msg.header.stamp)
            return
        qb = T_base.transform.rotation
        yaw_b = np.arctan2(2.0 * (qb.w * qb.z + qb.x * qb.y),
                           1.0 - 2.0 * (qb.y * qb.y + qb.z * qb.z))
        self._fov_pose = (float(origin[0]), float(origin[1]), float(yaw_b), float(bx), float(by))

        T_scan = self.lookup(self.lidar_frame, self.cloud_stamp) if self.cloud_stamp is not None else None
        if self.cloud is None or T_scan is None:
            if self.cloud is None:
                self.get_logger().warn(
                    'No LiDAR cloud received: nothing is mapped. '
                    'Check the LiDAR topic.', throttle_duration_sec=10.0)
            self.publish_map(bx, by, msg.header.stamp)
            return
        if (stamp_sec - self.cloud_t) > self.lidar_max_age:
            self.get_logger().warn('LiDAR cloud too old for this frame.',
                                   throttle_duration_sec=10.0)
            self.publish_map(bx, by, msg.header.stamp)
            return

        M_scan2map = transform_to_matrix(T_scan)
        M_scan2cam = np.linalg.inv(M) @ M_scan2map
        uu, vv, X, Y, Xz, Yz, ok, dist, Zm, high, in_img = position_points(
            self.cloud, M_scan2map, M_scan2cam, self.K, w, h,
            bz, self.max_obs_h, bx, by, self.map_write_max_range)

        classes = seg[vv, uu]
        costs = self.cost_lut[classes]
        ok = ok & (costs >= 0)
        high = high & (costs >= 0)
        if self.ost_range > self.map_write_max_range:
            far = (in_img & np.isin(classes, self.ost_classes) & (dist < self.ost_range)
                   & ((Zm - bz) <= self.stacked_high_max) & np.isfinite(Zm))
            ok = ok | far
            high = high & ~far

        full_dyn = self.dyn_lut[seg].astype(np.uint8)
        if full_dyn.any():
            if self.dyn_dilate_px > 0:
                k = 2 * self.dyn_dilate_px + 1
                kern = np.ones((k, k), np.uint8)
                dyn_mask = cv2.dilate(full_dyn, kern, iterations=1)
            else:
                dyn_mask = full_dyn
            in_dyn = dyn_mask[vv, uu].astype(bool)
            ok = ok & (~in_dyn)
            high = high & (~in_dyn)

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
        high = high & (~occluded)
        w_dist = np.clip(1.0 - dist / self.max_range, self.w_dist_min, 1.0)
        wpix = w_dist

        okx = ok & np.isfinite(X) & np.isfinite(Y)
        if okx.any():
            self.ensure_capacity(X[okx], Y[okx])

        gi = ((Xz - self.gox) / self.res).astype(int)
        gj = ((Yz - self.goy) / self.res).astype(int)
        inside = ok & (gi >= 0) & (gi < self.gnx) & (gj >= 0) & (gj < self.gny)

        gi_u = gi[inside]; gj_u = gj[inside]
        cl_u = classes[inside].astype(np.uint8)
        w_u = wpix[inside].astype(np.float32)
        if gi_u.size == 0:
            self.publish_map(bx, by, msg.header.stamp)
            return

        flat = gj_u * self.gnx + gi_u
        rank = np.empty(w_u.size, dtype=np.int64)
        rank[np.argsort(w_u)] = np.arange(w_u.size)
        order = np.lexsort((rank, flat))
        flat_s = flat[order]
        last = np.r_[flat_s[1:] != flat_s[:-1], True]
        cells = flat_s[last]
        cell_cls = cl_u[order][last]
        cell_w = w_u[order][last]
        if self.stacked_on:
            t_st = time.perf_counter()
            cand = (ok | high) & (gi >= 0) & (gi < self.gnx) & (gj >= 0) & (gj < self.gny)
            sc_cells, sc_cls = stacked_classes(
                gi[cand], gj[cand], Zm[cand], classes[cand], self.gnx, self.gny, cells,
                self.stacked_low_min, self.max_obs_h, self.stacked_high_max,
                self.stacked_tol, self.stacked_min_votes)
            if sc_cells.size:
                pos = np.searchsorted(cells, sc_cells)
                self._st_changed += int(np.count_nonzero(cell_cls[pos] != sc_cls))
                cell_cls[pos] = sc_cls.astype(cell_cls.dtype)
            self._st_frames += 1
            self._st_cells += int(sc_cells.size)
            self._st_time += time.perf_counter() - t_st
            if time.monotonic() - self._st_last > 30.0:
                nf = max(self._st_frames, 1)
                self.get_logger().info(
                    f'stacked vote: frames {self._st_frames}, voted cells {self._st_cells}, '
                    f'class changed {self._st_changed}, {1000.0 * self._st_time / nf:.1f} ms per frame')
                self._st_last = time.monotonic()

        A = self.alpha.reshape(-1, N_CLASSES)
        KN = self.known.reshape(-1)
        V = self.grid.reshape(-1)
        fuse(A, cells, cell_cls, cell_w, self.alpha_max, self.valid_cls)
        KN[cells] |= A[cells].sum(axis=1) >= self.alpha_min
        old = V[cells].copy()
        new = expected_cost(A[cells], self.cost_vec, KN[cells], self.lethal_classes, self.p_lethal)
        V[cells] = new
        self.grid_i8.reshape(-1)[cells] = grid_to_i8(new)
        changed = old != new
        if np.any(changed):
            crossed = (((old < 50) & (new >= 50)) | ((old >= 50) & (new < 50))
                       | ((old < 0) & (new >= 0)))
            self._significant_change = int(np.count_nonzero(crossed & changed))
        if self.rc_on:
            self.ray_clear(M_scan2map, X, Y, Zm, bz, stamp_sec)
        if self.fp_clear:
            self.footprint_clear(bx, by, yaw_b)
        self.publish_map(bx, by, msg.header.stamp)

    def ray_clear(self, M_scan2map, X, Y, Zm, bz, stamp_sec):
        fin = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Zm)
        hit = fin & ((Zm - bz) >= self.rc_min_h) & ((Zm - bz) <= self.stacked_high_max)
        if np.any(hit):
            hi = np.floor((X[hit] - self.gox) / self.res).astype(np.int64)
            hj = np.floor((Y[hit] - self.goy) / self.res).astype(np.int64)
            hk = (hi >= 0) & (hi < self.gnx) & (hj >= 0) & (hj < self.gny)
            hi, hj = hi[hk], hj[hk]
            d = self.rc_dil
            for di in range(-d, d + 1):
                for dj in range(-d, d + 1):
                    ii, jj = hi + di, hj + dj
                    kk = (ii >= 0) & (ii < self.gnx) & (jj >= 0) & (jj < self.gny)
                    self.t_hit.reshape(-1)[jj[kk] * self.gnx + ii[kk]] = stamp_sec
        self.rc_n += 1
        if self.rc_n % self.rc_every:
            return
        t0 = time.perf_counter()
        O = M_scan2map[:3, 3]
        V = self.grid.reshape(-1)
        target = (V >= 100.0) & (self.t_hit.reshape(-1) < stamp_sec - self.rc_protect)
        self.rc_stats[0] += 1
        if not np.any(target):
            self.rc_stats[3] += time.perf_counter() - t0
            return
        P = np.stack([X, Y, Zm], axis=1)
        cells, counts = free_ground_cells(O, P, self.gox, self.goy, self.res, self.gnx, self.gny, bz,
                                          self.rc_low_h, self.rc_ground_tol, self.rc_margin,
                                          self.map_write_max_range, self.rc_stride, target)
        c = cells[counts >= self.rc_min_rays]
        if c.size:
            A = self.alpha.reshape(-1, N_CLASSES)
            KN = self.known.reshape(-1)
            sub = A[c]
            sub[:, self.lethal_classes] = np.maximum(sub[:, self.lethal_classes] - self.rc_step, 0.0)
            A[c] = sub
            gone = A[c].sum(axis=1) < self.alpha_min
            KN[c[gone]] = False
            new = expected_cost(A[c], self.cost_vec, KN[c], self.lethal_classes, self.p_lethal)
            V[c] = new
            self.grid_i8.reshape(-1)[c] = grid_to_i8(new)
            self.rc_stats[1] += int(c.size)
            self.rc_stats[2] += int(np.count_nonzero(new < 100.0))
        self.rc_stats[3] += time.perf_counter() - t0
        if time.monotonic() - self.rc_last > 30.0:
            n = max(self.rc_stats[0], 1)
            self.get_logger().info(
                f'ray clearing: runs {self.rc_stats[0]}, lethal cells touched {self.rc_stats[1]}, '
                f'back to non-lethal {self.rc_stats[2]}, {1000.0 * self.rc_stats[3] / n:.1f} ms per run')
            self.rc_last = time.monotonic()

    def footprint_clear(self, bx, by, yaw):
        n = int(np.ceil(max(self.fp_hl, self.fp_hw) / self.res)) + 1
        oi = int(np.floor((bx - self.gox) / self.res))
        oj = int(np.floor((by - self.goy) / self.res))
        ii, jj = np.meshgrid(np.arange(oi - n, oi + n + 1), np.arange(oj - n, oj + n + 1))
        ii = ii.ravel()
        jj = jj.ravel()
        cx = self.gox + (ii + 0.5) * self.res - bx
        cy = self.goy + (jj + 0.5) * self.res - by
        u = cx * np.cos(yaw) + cy * np.sin(yaw)
        v = -cx * np.sin(yaw) + cy * np.cos(yaw)
        k = ((np.abs(u) <= self.fp_hl) & (np.abs(v) <= self.fp_hw)
             & (ii >= 0) & (ii < self.gnx) & (jj >= 0) & (jj < self.gny))
        c = jj[k] * self.gnx + ii[k]
        if c.size == 0:
            return
        V = self.grid.reshape(-1)
        c = c[V[c] >= 100.0]
        if c.size == 0:
            return
        A = self.alpha.reshape(-1, N_CLASSES)
        KN = self.known.reshape(-1)
        sub = A[c]
        sub[:, self.lethal_classes] = 0.0
        A[c] = sub
        gone = A[c].sum(axis=1) < self.alpha_min
        KN[c[gone]] = False
        new = expected_cost(A[c], self.cost_vec, KN[c], self.lethal_classes, self.p_lethal)
        V[c] = new
        self.grid_i8.reshape(-1)[c] = grid_to_i8(new)
        self.fp_stats[0] += int(c.size)
        self.fp_stats[1] += int(np.count_nonzero(new < 100.0))
        if time.monotonic() - self.fp_last > 30.0:
            self.get_logger().info(
                f'footprint clear: lethal cells under the robot {self.fp_stats[0]}, now free {self.fp_stats[1]}')
            self.fp_last = time.monotonic()

    def map_jump_ok(self, stamp, t, bx, by, bz):
        if self.jump_gate is None:
            return True
        T_mo = self.lookup(self.odom_frame, stamp)
        if T_mo is None:
            return False
        ok = self.jump_gate.update(transform_to_matrix(T_mo), (bx, by, bz), t)
        if not ok:
            self.get_logger().warn(
                f'Localization jump ({self.jump_gate.last_jump_m:.2f} m, '
                f'{self.jump_gate.last_jump_deg:.1f} deg): map writing paused '
                f'for {self.jump_gate.hold_s:.1f} s ({self.jump_gate.n_jumps} jumps so far).',
                throttle_duration_sec=1.0)
        return ok

    def _full_map_timer(self):
        if self.grid is None:
            return
        stamp = self.get_clock().now().to_msg()
        self.pub.publish(self._full_msg(stamp))

    def _full_msg(self, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target
        msg.info.resolution = self.res
        msg.info.width = self.gnx; msg.info.height = self.gny
        msg.info.origin.position.x = float(self.gox)
        msg.info.origin.position.y = float(self.goy)
        msg.info.origin.orientation.w = 1.0
        msg.data = array.array('b', np.ascontiguousarray(self.grid_i8).tobytes())
        return msg

    def publish_map(self, bx, by, stamp):
        if self._significant_change >= self.significant_thresh:
            self.pub.publish(self._full_msg(stamp))
            self.get_logger().info(
                f'Significant change ({self._significant_change} cells): full map published now',
                throttle_duration_sec=1.0)
            self._significant_change = 0

        self.publish_range_marker(bx, by, stamp)

    def publish_range_marker(self, bx, by, stamp):
        fp = getattr(self, '_fov_pose', None)
        if fp is None or self.K is None:
            return
        cx, cy, yaw, px, py = fp
        half = float(np.arctan2(self.K[0, 2], self.K[0, 0]))
        R = self.map_write_max_range
        pts = [Point(x=cx, y=cy, z=0.05)]
        for a in np.linspace(-half, half, 25):
            dx, dy = np.cos(yaw + a), np.sin(yaw + a)
            ox, oy = cx - px, cy - py
            b = dx * ox + dy * oy
            t = -b + np.sqrt(max(b * b - (ox * ox + oy * oy - R * R), 0.0))
            pts.append(Point(x=float(cx + t * dx), y=float(cy + t * dy), z=0.05))
        pts.append(Point(x=cx, y=cy, z=0.05))
        m = Marker()
        m.header.frame_id = self.target
        m.header.stamp = stamp
        m.ns = 'map_range'; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r = 1.0; m.color.g = 0.9; m.color.b = 0.0; m.color.a = 0.9
        m.points = pts
        self.pub_range.publish(m)

    def cloud_cb(self, msg: PointCloud2):
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
        self.cloud_stamp = msg.header.stamp


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    rclpy.init()
    node = SemanticCostmapNode()
    signal.signal(signal.SIGHUP, _interrupt)
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
