#!/usr/bin/env python3
"""
Proiezione semantica -> costmap del marciapiede.
Mappa PERSISTENTE nel frame target (es. 'map') con FUSIONE robusta:

  - inerzia (media esponenziale): ogni cella tiene una stima che si aggiorna
    verso le nuove osservazioni invece di sovrascrivere -> osservazioni multiple
    convergono al valore vero e gli errori isolati vengono lavati via.
  - peso per distanza: un pixel proiettato vicino e' molto piu' affidabile di uno
    lontano (l'IPM si stira vicino all'orizzonte). Le osservazioni vicine pesano
    molto, quelle lontane quasi nulla.
  - soglia di commit: una cella MAI vista viene "creata" solo se l'osservazione e'
    abbastanza vicina. Cosi' le bave lontane non scrivono mai celle nuove, e quando
    il robot ci ripassa vicino le celle sbagliate vengono corrette.

Risultato: mappi vicino e accurato, e la mappa lunga si costruisce nel tempo
mentre ti muovi, autocorreggendosi al ripasso.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
import numpy as np
from cv_bridge import CvBridge
import tf2_ros
from rclpy.duration import Duration


CLASS_COST = {
    1: 0,     # sidewalk -> preferito
    9: 70,    # terrain (erba/aiuola)
    8: 80,    # vegetation
    0: 90,    # road
    2: 100,   # building
    3: 100,   # wall
    4: 100,   # fence
    5: 100,   # pole
    7: 100,   # traffic sign
    11: 100,  # person (poi: sottosistema dinamico)
    12: 100,  # rider
    13: 100,  # car
    14: 100, 15: 100, 16: 100, 17: 100, 18: 100,
}


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


class SemanticCostmapNode(Node):
    def __init__(self):
        super().__init__('semantic_costmap_node')

        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('camera_optical_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('size_m', 8.0)
        self.declare_parameter('global_size_m', 80.0)
        self.declare_parameter('pixel_stride', 4)
        self.declare_parameter('max_range', 5.0)
        self.declare_parameter('min_row_frac', 0.62)
        self.declare_parameter('publish_window', True)
        self.declare_parameter('fuse_alpha_near', 0.45)
        self.declare_parameter('fuse_alpha_far', 0.03)
        self.declare_parameter('fuse_commit_w', 0.20)

        self.target = self.get_parameter('target_frame').value
        self.cam_frame = self.get_parameter('camera_optical_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.res = self.get_parameter('resolution').value
        self.size_m = self.get_parameter('size_m').value
        self.global_size_m = self.get_parameter('global_size_m').value
        self.stride = self.get_parameter('pixel_stride').value
        self.max_range = self.get_parameter('max_range').value
        self.min_row_frac = self.get_parameter('min_row_frac').value
        self.publish_window = self.get_parameter('publish_window').value
        self.a_near = self.get_parameter('fuse_alpha_near').value
        self.a_far = self.get_parameter('fuse_alpha_far').value
        self.w_commit = self.get_parameter('fuse_commit_w').value

        self.gn = int(self.global_size_m / self.res)
        self.gox = -self.global_size_m / 2.0
        self.goy = -self.global_size_m / 2.0
        self.global_grid = np.full((self.gn, self.gn), -1.0, dtype=np.float32)

        self.win = int(self.size_m / self.res)
        self.bridge = CvBridge()
        self.K = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 1)
        seg_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/semantic/segmentation', self.seg_cb, seg_qos)
        self.pub = self.create_publisher(OccupancyGrid, '/semantic_costmap', 1)

        self.cost_lut = np.full(256, -2, dtype=np.int16)
        for cls, c in CLASS_COST.items():
            self.cost_lut[cls] = c

        self.get_logger().info(
            f'Semantic costmap (persistente, fusione pesata) pronto. '
            f'frame={self.target}, mappa {self.global_size_m}m.')

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
        origin = M[:3, 3]
        R = M[:3, :3]
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
        t = np.full(uu.shape, -1.0)
        t[valid] = -origin[2] / dz[valid]
        ok = valid & (t > 0)
        pts = origin[None, :] + t[:, None] * dir_world
        X, Y = pts[:, 0], pts[:, 1]

        dist = np.hypot(X - bx, Y - by)
        ok = ok & (dist < self.max_range)

        gi = ((X - self.gox) / self.res).astype(int)
        gj = ((Y - self.goy) / self.res).astype(int)
        inside = ok & (gi >= 0) & (gi < self.gn) & (gj >= 0) & (gj < self.gn)

        classes = seg[vv, uu]
        costs = self.cost_lut[classes]
        use = inside & (costs >= 0)

        gi_u = gi[use]; gj_u = gj[use]
        co_u = costs[use].astype(np.float32)
        di_u = dist[use].astype(np.float32)
        if gi_u.size == 0:
            self.publish_map(bx, by, msg.header.stamp)
            return

        flat = gj_u * self.gn + gi_u
        N = self.gn * self.gn

        frame_cost = np.full(N, -1.0, dtype=np.float32)
        frame_mind = np.full(N, np.inf, dtype=np.float32)
        np.maximum.at(frame_cost, flat, co_u)
        np.minimum.at(frame_mind, flat, di_u)
        seen = frame_mind < np.inf

        near = np.clip(1.0 - frame_mind / self.max_range, 0.0, 1.0)
        wgt = self.a_far + (self.a_near - self.a_far) * near

        g = self.global_grid.ravel()
        known = g >= 0.0

        first = seen & (~known) & (wgt >= self.w_commit)
        g[first] = frame_cost[first]

        upd = seen & known
        g[upd] = wgt[upd] * frame_cost[upd] + (1.0 - wgt[upd]) * g[upd]

        self.global_grid = g.reshape(self.gn, self.gn)
        self.publish_map(bx, by, msg.header.stamp)

    def publish_map(self, bx, by, stamp):
        if self.publish_window:
            ci = int((bx - self.gox) / self.res)
            cj = int((by - self.goy) / self.res)
            half = self.win // 2
            i0 = max(0, ci - half); i1 = min(self.gn, ci + half)
            j0 = max(0, cj - half); j1 = min(self.gn, cj + half)
            sub = self.global_grid[j0:j1, i0:i1]
            ox = self.gox + i0 * self.res
            oy = self.goy + j0 * self.res
            width = i1 - i0; height = j1 - j0
        else:
            sub = self.global_grid
            ox, oy = self.gox, self.goy
            width = self.gn; height = self.gn

        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target
        msg.info.resolution = self.res
        msg.info.width = width
        msg.info.height = height
        msg.info.origin.position.x = float(ox)
        msg.info.origin.position.y = float(oy)
        msg.info.origin.orientation.w = 1.0
        data = np.where(sub < 0, -1, np.clip(np.round(sub), 0, 100)).astype(np.int8)
        msg.data = data.ravel(order='C').tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SemanticCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()