#!/usr/bin/env python3
"""
Proiezione semantica -> costmap del marciapiede CON MAPPA PERSISTENTE.

Differenza chiave rispetto alla versione "finestra mobile": qui esiste UNA
griglia globale fissa nel frame 'odom'. Ogni frame proietta a terra i pixel
(IPM, piano z=0) e AGGIORNA le celle corrispondenti della griglia globale; non
si ridisegna da zero. La mappa quindi RESTA FERMA nel mondo e si accumula man
mano che il robot esplora. Per la pubblicazione si ritaglia una finestra attorno
al robot (oppure si pubblica tutta la mappa, vedi 'publish_window').

Fusione per cella: si tiene il costo MASSIMO osservato (conservativo: un
ostacolo visto una volta resta). Con 'decay' opzionale le celle non riviste
sbiadiscono lentamente, utile quando aggiungerai gli ostacoli dinamici.
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


# costo per classe Cityscapes (0..100, -1 = sconosciuto)
CLASS_COST = {
    1: 0,     # sidewalk -> preferito
    9: 70,    # terrain (erba/aiuola)
    8: 80,    # vegetation
    0: 90,    # road -> molto costoso
    2: 100,   # building
    3: 100,   # wall
    4: 100,   # fence
    5: 100,   # pole
    7: 100,   # traffic sign
    11: 100,  # person (in seguito: sottosistema dinamico)
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

        self.declare_parameter('target_frame', 'odom')
        self.declare_parameter('camera_optical_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('size_m', 8.0)          # lato finestra pubblicata
        self.declare_parameter('global_size_m', 60.0)  # lato della mappa globale persistente
        self.declare_parameter('pixel_stride', 4)
        self.declare_parameter('max_range', 6.0)
        self.declare_parameter('min_row_frac', 0.45)
        self.declare_parameter('publish_window', True) # True: ritaglia attorno al robot; False: tutta la mappa
        self.declare_parameter('decay', 0)             # 0 = nessun decay; >0 = quanto sbiadire le celle non riviste

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
        self.decay = int(self.get_parameter('decay').value)

        # ----- mappa globale persistente, fissa nel frame target -----
        self.gn = int(self.global_size_m / self.res)        # celle per lato (globale)
        # origine della griglia globale: angolo in basso-sx, in metri nel frame target.
        # centrata sull'origine del mondo; quantizzata al passo cella.
        self.gox = -self.global_size_m / 2.0
        self.goy = -self.global_size_m / 2.0
        self.global_grid = np.full((self.gn, self.gn), -1, dtype=np.int16)  # -1 = sconosciuto

        self.win = int(self.size_m / self.res)              # celle per lato (finestra)
        self.bridge = CvBridge()
        self.K = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 1)
        seg_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/semantic/segmentation', self.seg_cb, seg_qos)
        self.pub = self.create_publisher(OccupancyGrid, '/semantic_costmap', 1)

        self.cost_lut = np.full(256, -2, dtype=np.int16)    # -2 = ignora
        for cls, c in CLASS_COST.items():
            self.cost_lut[cls] = c

        self.get_logger().info(
            f'Semantic costmap (persistente) pronto. '
            f'mappa globale {self.global_size_m}m ({self.gn}x{self.gn} celle).')

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

        # --- campiona pixel (parte bassa = suolo) ---
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

        # --- indici nella GRIGLIA GLOBALE FISSA (non ricentrata sul robot) ---
        gi = ((X - self.gox) / self.res).astype(int)
        gj = ((Y - self.goy) / self.res).astype(int)
        inside = ok & (gi >= 0) & (gi < self.gn) & (gj >= 0) & (gj < self.gn)

        classes = seg[vv, uu]
        costs = self.cost_lut[classes]
        use = inside & (costs >= -1)            # -2 = ignora

        gi_u = gi[use]; gj_u = gj[use]; co_u = costs[use].astype(np.int16)

        # opzionale: sbiadisci tutta la mappa prima di aggiornare (per i dinamici)
        if self.decay > 0:
            known = self.global_grid >= 0
            self.global_grid[known] = np.maximum(0, self.global_grid[known] - self.decay)

        # --- FUSIONE: tieni il costo massimo per cella (ostacolo vince) ---
        flat = gj_u * self.gn + gi_u
        g = self.global_grid.ravel()
        # ordina cosi' i costi alti sovrascrivono per ultimi, poi np.maximum col valore esistente
        order = np.argsort(co_u)
        flat_s = flat[order]; co_s = co_u[order]
        np.maximum.at(g, flat_s, co_s)          # accumulo: max(esistente, nuovo)
        # le celle ancora -1 (sconosciute) toccate ora prendono il nuovo valore
        first_seen = (g[flat_s] < 0)
        if first_seen.any():
            g[flat_s[first_seen]] = co_s[first_seen]
        self.global_grid = g.reshape(self.gn, self.gn)

        self.publish_map(bx, by, msg.header.stamp)

    def publish_map(self, bx, by, stamp):
        if self.publish_window:
            # ritaglia una finestra attorno al robot, allineata alle celle globali
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
        data = np.clip(sub, -1, 100).astype(np.int8)
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