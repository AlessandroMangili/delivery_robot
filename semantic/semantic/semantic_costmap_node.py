#!/usr/bin/env python3
"""
Nodo di proiezione semantica -> costmap del marciapiede.

Idea: il TurtleBot3 non ha profondita'. Assumiamo terreno piano (z=0 nel frame
target, es. 'odom'). Per ogni pixel classificato calcoliamo il raggio ottico,
lo intersechiamo col piano del suolo e otteniamo il punto (X,Y) a terra
(Inverse Perspective Mapping). A quella cella assegniamo un costo in base alla
classe semantica: marciapiede -> libero, strada/erba -> costoso, ostacoli -> alto.

Pubblica /semantic_costmap (nav_msgs/OccupancyGrid) in una finestra mobile
centrata sul robot, nel frame 'odom'.

NOTA: questa e' la versione base "tieni il marciapiede". La persistenza tra
sessioni (lifelong / decay differenziale) si aggiunge in un secondo momento.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
import numpy as np
from cv_bridge import CvBridge
import tf2_ros
from rclpy.duration import Duration

# costo per classe Cityscapes (0..100, -1 = sconosciuto)
CLASS_COST = {
    1: 0,     # sidewalk -> preferito
    9: 70,    # terrain (erba/aiuola) -> da evitare
    8: 80,    # vegetation
    0: 90,    # road -> molto costoso
    2: 100,   # building
    3: 100,   # wall
    4: 100,   # fence
    5: 100,   # pole
    7: 100,   # traffic sign
    11: 100,  # person  (in seguito gestita dal sottosistema dinamico)
    12: 100,  # rider
    13: 100,  # car
    14: 100, 15: 100, 16: 100, 17: 100, 18: 100,  # truck/bus/train/moto/bici
}
# le classi non elencate (es. sky=10) vengono ignorate


def transform_to_matrix(t):
    """TransformStamped -> matrice 4x4 (numpy)."""
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
        self.declare_parameter('resolution', 0.05)   # m/cella
        self.declare_parameter('size_m', 8.0)         # lato finestra (m)
        self.declare_parameter('pixel_stride', 4)     # sottocampionamento pixel
        self.declare_parameter('max_range', 6.0)      # m, oltre ignoriamo
        self.declare_parameter('min_row_frac', 0.45)  # usa solo parte bassa immagine

        self.target = self.get_parameter('target_frame').value
        self.cam_frame = self.get_parameter('camera_optical_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.res = self.get_parameter('resolution').value
        self.size_m = self.get_parameter('size_m').value
        self.stride = self.get_parameter('pixel_stride').value
        self.max_range = self.get_parameter('max_range').value
        self.min_row_frac = self.get_parameter('min_row_frac').value

        self.ncells = int(self.size_m / self.res)
        self.bridge = CvBridge()
        self.K = None  # intrinseci, da camera_info

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, '/camera/camera_info', self.info_cb, 1)
        self.create_subscription(Image, '/semantic/segmentation', self.seg_cb, 1)
        self.pub = self.create_publisher(OccupancyGrid, '/semantic_costmap', 1)

        # tabella costi -> vettore per lookup veloce (indice = id classe)
        self.cost_lut = np.full(256, -2, dtype=np.int16)  # -2 = ignora
        for cls, c in CLASS_COST.items():
            self.cost_lut[cls] = c

        self.get_logger().info('Semantic costmap node pronto.')

    def info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)
        self.img_w = msg.width
        self.img_h = msg.height

    def lookup(self, frame, stamp):
        try:
            return self.tf_buffer.lookup_transform(
                self.target, frame, stamp, timeout=Duration(seconds=0.1))
        except Exception:
            try:
                return self.tf_buffer.lookup_transform(
                    self.target, frame, rclpy.time.Time())
            except Exception as e:
                self.get_logger().warn(f'TF non disponibile {frame}->{self.target}: {e}')
                return None

    def seg_cb(self, msg: Image):
        if self.K is None:
            return
        seg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        h, w = seg.shape

        T_cam = self.lookup(self.cam_frame, msg.header.stamp)
        T_base = self.lookup(self.base_frame, msg.header.stamp)
        if T_cam is None or T_base is None:
            return
        M = transform_to_matrix(T_cam)          # camera_optical -> odom
        origin = M[:3, 3]
        R = M[:3, :3]

        # robot center per la finestra mobile
        bx, by = T_base.transform.translation.x, T_base.transform.translation.y
        gx0 = bx - self.size_m / 2.0
        gy0 = by - self.size_m / 2.0

        # campiona i pixel (solo parte bassa = suolo)
        r0 = int(h * self.min_row_frac)
        vs = np.arange(r0, h, self.stride)
        us = np.arange(0, w, self.stride)
        uu, vv = np.meshgrid(us, vs)
        uu = uu.ravel(); vv = vv.ravel()

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # direzioni raggio nel frame ottico (z avanti, x destra, y giu')
        dir_opt = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, dtype=float)], axis=1)
        # ruota nel frame target
        dir_world = dir_opt @ R.T
        dz = dir_world[:, 2]

        # intersezione col suolo z=0 : origin_z + t*dz = 0
        valid = dz < -1e-6
        t = np.full(uu.shape, -1.0)
        t[valid] = -origin[2] / dz[valid]
        ok = valid & (t > 0)
        # punti a terra
        pts = origin[None, :] + t[:, None] * dir_world
        X, Y = pts[:, 0], pts[:, 1]

        # distanza dal robot per cutoff
        dist = np.hypot(X - bx, Y - by)
        ok = ok & (dist < self.max_range)

        # cella nella griglia
        ci = ((X - gx0) / self.res).astype(int)
        cj = ((Y - gy0) / self.res).astype(int)
        inside = ok & (ci >= 0) & (ci < self.ncells) & (cj >= 0) & (cj < self.ncells)

        # costo per pixel campionato
        classes = seg[vv, uu]
        costs = self.cost_lut[classes]
        use = inside & (costs >= -1)  # -2 = ignora

        grid = np.full((self.ncells, self.ncells), -1, dtype=np.int16)  # -1 sconosciuto
        ci_u = ci[use]; cj_u = cj[use]; co_u = costs[use]
        # conservativo: per cella tieni il costo massimo (ostacolo vince)
        flat = cj_u * self.ncells + ci_u
        order = np.argsort(co_u)  # cosi' i max sovrascrivono
        flat_sorted = flat[order]; co_sorted = co_u[order]
        g = grid.ravel()
        g[flat_sorted] = co_sorted
        grid = g.reshape(self.ncells, self.ncells)

        self.publish_grid(grid, gx0, gy0, msg.header.stamp)

    def publish_grid(self, grid, ox, oy, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.target
        msg.info.resolution = self.res
        msg.info.width = self.ncells
        msg.info.height = self.ncells
        msg.info.origin.position.x = float(ox)
        msg.info.origin.position.y = float(oy)
        msg.info.origin.orientation.w = 1.0
        # OccupancyGrid vuole int8 in [-1,100]
        data = np.clip(grid, -1, 100).astype(np.int8)
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
