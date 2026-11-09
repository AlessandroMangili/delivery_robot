#!/usr/bin/env python3

import array

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def confinement_field(sem, res, sidewalk_cost_max, conf_max, falloff):
    sem = np.asarray(sem)
    sidewalk = (sem >= 0) & (sem <= sidewalk_cost_max)
    if not sidewalk.any():
        return np.zeros(sem.shape, dtype=np.float32)

    src = sidewalk.astype(np.uint8) * 255
    src[sem < 0] = 255

    if not (src == 0).any():
        return np.zeros(sem.shape, dtype=np.float32)

    dist_m = cv2.distanceTransform(src, cv2.DIST_L2, 5) * float(res)
    conf = float(conf_max) * np.exp(-dist_m / float(falloff))
    conf[~sidewalk] = 0.0
    return conf.astype(np.float32)


def check_grid_compatible(l_res, l_ox, l_oy, l_quat,
                          r_res, r_ox, r_oy, r_quat,
                          tol_cells=0.01):
    if r_res <= 0.0 or l_res <= 0.0:
        return False, f'resolution not positive (layer {l_res}, reference {r_res})'

    if abs(l_res - r_res) > 1e-6:
        return False, (f'different resolution: layer {l_res:.4f} m, '
                       f'reference {r_res:.4f} m')

    for name, q in (('layer', l_quat), ('reference', r_quat)):
        qx, qy, qz, qw = q
        if (abs(qx) > 1e-6 or abs(qy) > 1e-6 or abs(qz) > 1e-6
                or abs(abs(qw) - 1.0) > 1e-6):
            return False, (f'rotated {name} grid: '
                           f'({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})')

    for axis, lo, ro in (('x', l_ox, r_ox), ('y', l_oy, r_oy)):
        cells = (lo - ro) / r_res
        residual = abs(cells - round(cells))
        if residual > tol_cells:
            return False, (f'origins off the cell lattice on {axis}: '
                           f'{residual:.3f} cells ({residual * r_res * 100:.1f} cm)')

    return True, ''


def align_to_ref(layer, l_ox, l_oy, ref_ox, ref_oy, res, ref_h, ref_w, fill=-1):
    out = np.full((ref_h, ref_w), fill, dtype=np.int16)
    dx = int(round((l_ox - ref_ox) / res))
    dy = int(round((l_oy - ref_oy) / res))
    lh, lw = layer.shape
    r0 = max(dy, 0)
    r1 = min(dy + lh, ref_h)
    c0 = max(dx, 0)
    c1 = min(dx + lw, ref_w)
    if r0 >= r1 or c0 >= c1:
        return out
    out[r0:r1, c0:c1] = layer[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
    return out


def fuse_elevation(base, elev, w_elev, soft_ceiling, elev_lethal_thr):
    if elev is None:
        return np.clip(base, 0.0, 100.0)

    room = np.maximum(soft_ceiling - base, 0.0)
    increment = (elev / 100.0) * room * w_elev
    soft = np.minimum(base + increment, soft_ceiling)
    soft = np.maximum(base, soft)
    soft = np.where(elev >= elev_lethal_thr, 100.0, soft)
    return np.clip(soft, 0.0, 100.0)


def quat_of(info):
    o = info.origin.orientation
    return (float(o.x), float(o.y), float(o.z), float(o.w))


class SSRLCombiner(Node):
    def __init__(self):
        super().__init__('ssrl_combiner')
        self.declare_parameter('semantic_topic', '/semantic_costmap')
        self.declare_parameter('elevation_topic', '/elevation_cost')
        self.declare_parameter('out_topic', '/ssrl_costmap')
        self.declare_parameter('confinement_topic', '/confinement_cost')
        self.declare_parameter('use_confinement', True)
        self.declare_parameter('sidewalk_cost_max', 15)
        self.declare_parameter('conf_max', 40.0)
        self.declare_parameter('falloff', 0.4)
        self.declare_parameter('use_elevation', True)
        self.declare_parameter('w_elev', 0.6)
        self.declare_parameter('soft_ceiling', 97)
        self.declare_parameter('elev_lethal_thr', 100)
        self.declare_parameter('grid_tol_cells', 0.01)

        gp = self.get_parameter
        self.semantic_topic = gp('semantic_topic').value
        self.elevation_topic = gp('elevation_topic').value
        out_topic = gp('out_topic').value
        self.use_confinement = bool(gp('use_confinement').value)
        self.sidewalk_cost_max = int(gp('sidewalk_cost_max').value)
        self.conf_max = float(gp('conf_max').value)
        self.falloff = float(gp('falloff').value)
        self.use_elevation = bool(gp('use_elevation').value)
        self.w_elev = float(gp('w_elev').value)
        self.soft_ceiling = float(gp('soft_ceiling').value)
        self.elev_lethal_thr = float(gp('elev_lethal_thr').value)
        self.grid_tol_cells = float(gp('grid_tol_cells').value)

        qos_latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_volatile = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=1)

        self.elev = None
        self.last_sem = None

        self.create_subscription(OccupancyGrid, self.semantic_topic,
                                 self.semantic_cb, qos_latched)
        if self.use_elevation:
            self.create_subscription(OccupancyGrid, self.elevation_topic,
                                     self.elevation_cb, qos_volatile)

        self.pub = self.create_publisher(OccupancyGrid, out_topic, qos_latched)
        self.pub_conf = self.create_publisher(
            OccupancyGrid, gp('confinement_topic').value, qos_latched)

        self.get_logger().info(
            f'Map fusion: {self.semantic_topic} -> {out_topic} | '
            f'confinement {"on" if self.use_confinement else "off"} '
            f'(sidewalk <= {self.sidewalk_cost_max}, max {self.conf_max:.0f}, '
            f'falloff {self.falloff}) | '
            f'elevation {"on" if self.use_elevation else "off"}')

    def elevation_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        arr = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.elev = (msg.info, arr)
        if self.last_sem is not None:
            self.combine_and_publish()

    def semantic_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        sem = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.last_sem = (sem, msg.header, msg.info)
        self.combine_and_publish()

    def _compatible(self, name, linfo, ref_info):
        ok, reason = check_grid_compatible(
            float(linfo.resolution), float(linfo.origin.position.x),
            float(linfo.origin.position.y), quat_of(linfo),
            float(ref_info.resolution), float(ref_info.origin.position.x),
            float(ref_info.origin.position.y), quat_of(ref_info),
            tol_cells=self.grid_tol_cells)
        if not ok:
            self.get_logger().error(
                f'Layer "{name}" does not match the semantic grid: {reason}. '
                f'Layer left out of this fusion.', throttle_duration_sec=5.0)
        return ok

    def _grid_msg(self, arr, header, info):
        msg = OccupancyGrid()
        msg.header = header
        msg.info = info
        msg.data = array.array('b', np.ascontiguousarray(
            np.clip(np.round(arr), -1, 100).astype(np.int8)).tobytes())
        return msg

    def combine_and_publish(self):
        sem, header, info = self.last_sem
        known = sem >= 0
        ref_ox = info.origin.position.x
        ref_oy = info.origin.position.y
        res = info.resolution
        ref_h, ref_w = sem.shape

        base = sem.astype(np.float32).copy()

        if self.use_confinement:
            conf = confinement_field(sem, res, self.sidewalk_cost_max, self.conf_max, self.falloff)
            base = base + conf
            self.pub_conf.publish(self._grid_msg(conf, header, info))

        base = np.clip(base, 0, 100)

        elev = None
        if self.use_elevation and self.elev is not None:
            einfo, earr = self.elev
            if self._compatible(self.elevation_topic, einfo, info):
                elev = align_to_ref(earr, einfo.origin.position.x,
                                    einfo.origin.position.y,
                                    ref_ox, ref_oy, res, ref_h, ref_w, fill=-1)
                elev = np.clip(elev, 0, 100).astype(np.float32)

        total = fuse_elevation(base, elev, self.w_elev, self.soft_ceiling, self.elev_lethal_thr)

        total = np.clip(np.round(total), 0, 100)
        total[~known] = -1
        self.pub.publish(self._grid_msg(total, header, info))


def main():
    rclpy.init()
    node = SSRLCombiner()
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
