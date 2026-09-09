#!/usr/bin/env python3

import numpy as np
import array


def grid_key(info):
    return (info.width, info.height,
            round(info.origin.position.x, 3), round(info.origin.position.y, 3),
            round(info.resolution, 4))


def align_to_ref(layer, l_ox, l_oy, ref_ox, ref_oy, res, ref_h, ref_w, fill=-1):
    """Rimappa 'layer' (origine l_ox,l_oy) sulla griglia di riferimento (semantica).
    Semantica ed elevation sono nodi separati con griglie di dimensione/origine
    diverse ma STESSA risoluzione e orientamento -> basta uno shift intero di celle,
    niente interpolazione. Celle di ref non coperte dal layer = fill (-1 = ignoto)."""
    out = np.full((ref_h, ref_w), fill, dtype=np.int16)
    dx = int(round((l_ox - ref_ox) / res))
    dy = int(round((l_oy - ref_oy) / res))
    lh, lw = layer.shape
    r0 = max(dy, 0); r1 = min(dy + lh, ref_h)
    c0 = max(dx, 0); c1 = min(dx + lw, ref_w)
    if r0 >= r1 or c0 >= c1:
        return out
    out[r0:r1, c0:c1] = layer[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
    return out


def fuse_elevation(base, elev, w_elev, soft_ceiling, elev_lethal_thr,
                   elev_can_be_lethal):
    """Fonde il costo di traversabilita' (elevation) sopra la base
    (semantica + confinement). Funzione PURA -> testabile senza ROS.

    base:  ndarray (H,W) float, 0..100, gia' combinata e clippata
    elev:  ndarray (H,W) float, 0..100, gia' riallineata su base; None = disattiva

    REGOLA 1 (mai demozione): il costo finale non e' MAI inferiore alla base.
      Il soft_ceiling limita QUANTO l'elevation puo' AGGIUNGERE, non e' un tetto
      sulla base. Senza questo, una cella letale (100) della semantica veniva
      schiacciata a soft_ceiling (97) -> in Nav2 perde lo stato LETHAL_OBSTACLE
      e non semina piu' inflation (bug osservato con elevation vuota).

    REGOLA 2 (letale solo se voluto): elev_can_be_lethal=False significa che
      l'elevation contribuisce SOLO costo soft. Rugosita' e pendenza sono
      terreno scomodo, non muri: promuoverle a 100 le rende letali E le fa
      gonfiare dall'inflation di Nav2, che e' l'alone spurio osservato.
    """
    if elev is None:
        return np.clip(base, 0.0, 100.0)

    spazio = np.maximum(soft_ceiling - base, 0.0)
    incremento = (elev / 100.0) * spazio * w_elev
    soft = np.minimum(base + incremento, soft_ceiling)
    # REGOLA 1: l'elevation puo' solo alzare, mai abbassare.
    soft = np.maximum(base, soft)

    if elev_can_be_lethal:
        # REGOLA 2: solo qui l'elevation puo' dichiarare ostacolo fisico.
        soft = np.where(elev >= elev_lethal_thr, 100.0, soft)

    return np.clip(soft, 0.0, 100.0)


# @@@ROS_BOUNDARY@@@  (i test isolati eseguono solo cio' che sta sopra questa riga)

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from nav_msgs.msg import OccupancyGrid


class SSRLCombiner(Node):
    def __init__(self):
        super().__init__('ssrl_combiner')
        self.declare_parameter('semantic_topic', '/semantic_costmap')
        self.declare_parameter('layer_topics', ['/confinement_cost'])
        self.declare_parameter('elevation_topic', '/elevation_cost')
        self.declare_parameter('out_topic', '/ssrl_costmap')
        # --- parametri fusione elevation (traversabilita') ---
        self.declare_parameter('use_elevation', True)       # OFF = semantica+confinement soltanto
        self.declare_parameter('w_elev', 0.6)               # aggressivita' incremento soft
        self.declare_parameter('soft_ceiling', 97)          # tetto di cio' che l'elevation AGGIUNGE
        self.declare_parameter('elev_can_be_lethal', False) # l'elevation puo' dichiarare 100?
        self.declare_parameter('elev_lethal_thr', 92)       # soglia, usata solo se sopra e' True

        self.semantic_topic = self.get_parameter('semantic_topic').value
        self.layer_topics = list(self.get_parameter('layer_topics').value)
        self.elevation_topic = self.get_parameter('elevation_topic').value
        out_topic = self.get_parameter('out_topic').value
        self.use_elevation = bool(self.get_parameter('use_elevation').value)
        self.w_elev = float(self.get_parameter('w_elev').value)
        self.soft_ceiling = float(self.get_parameter('soft_ceiling').value)
        self.elev_can_be_lethal = bool(self.get_parameter('elev_can_be_lethal').value)
        self.elev_lethal_thr = float(self.get_parameter('elev_lethal_thr').value)

        # la semantica e' pubblicata LATCHED (transient_local): sottoscrivo
        # transient_local per ricevere subito l'ultima mappa anche partendo dopo.
        qos_latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        # i layer (confinement, dinamici) sono volatile normali
        qos_vol = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.layers = {}        # topic -> (info, ndarray)  [layer PURI: confinement]
        self.elev = None        # (info, ndarray) elevation, gestita a parte
        self.last_sem = None    # (key, ndarray, header, info) ultima semantica

        self.create_subscription(OccupancyGrid, self.semantic_topic,
                                 self.semantic_cb, qos_latched)
        for tp in self.layer_topics:
            self.create_subscription(
                OccupancyGrid, tp, lambda m, t=tp: self.layer_cb(t, m), qos_vol)
        # elevation: subscription creata SOLO se abilitata. Cosi' use_elevation=false
        # non lascia residui di stato e non spende CPU sul topic.
        if self.use_elevation:
            self.create_subscription(OccupancyGrid, self.elevation_topic,
                                     self.elevation_cb, qos_vol)

        # output LATCHED: il costmap di Nav2 (o chiunque parta dopo) riceve
        # subito l'ultima /ssrl_costmap.
        self.pub = self.create_publisher(OccupancyGrid, out_topic, qos_latched)
        self.get_logger().info(
            f'SSRL combiner: {self.semantic_topic} + {self.layer_topics} -> {out_topic} | '
            f'elevation={"ON" if self.use_elevation else "OFF"} '
            f'(letale={"si" if self.elev_can_be_lethal else "no"})')

    # -------------------------------------------------------------------------
    def layer_cb(self, topic, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        arr = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.layers[topic] = (msg.info, arr)
        if self.last_sem is not None:
            self.combine_and_publish()

    def elevation_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        arr = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.elev = (msg.info, arr)
        if self.last_sem is not None:
            self.combine_and_publish()

    def semantic_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        sem = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.last_sem = (grid_key(msg.info), sem, msg.header, msg.info)
        self.combine_and_publish()

    # -------------------------------------------------------------------------
    def combine_and_publish(self):
        key, sem, header, info = self.last_sem
        known = sem >= 0
        ref_ox = info.origin.position.x
        ref_oy = info.origin.position.y
        res = info.resolution
        ref_h, ref_w = sem.shape

        # --- BASE: semantica + layer PURI (confinement), RIALLINEATI alla semantica ---
        base = sem.astype(np.float32).copy()
        for topic, (linfo, arr) in self.layers.items():
            a = align_to_ref(arr, linfo.origin.position.x, linfo.origin.position.y,
                             ref_ox, ref_oy, res, ref_h, ref_w, fill=0)  # fill 0: non aggiunge
            add = np.clip(a, 0, 100).astype(np.float32)
            base[known] = base[known] + add[known]
        base = np.clip(base, 0, 100)

        # --- ELEVATION: riallineata, poi fusa (funzione pura) ---
        elev = None
        if self.use_elevation and self.elev is not None:
            einfo, earr = self.elev
            elev = align_to_ref(earr, einfo.origin.position.x, einfo.origin.position.y,
                                ref_ox, ref_oy, res, ref_h, ref_w, fill=0)
            elev = np.clip(elev, 0, 100).astype(np.float32)

        total = fuse_elevation(base, elev, self.w_elev, self.soft_ceiling,
                               self.elev_lethal_thr, self.elev_can_be_lethal)

        total = np.clip(np.round(total), 0, 100)
        total[~known] = -1

        out = OccupancyGrid()
        out.header = header
        out.info = info
        out.data = array.array('b', np.ascontiguousarray(
            total.astype(np.int8)).tobytes())
        self.pub.publish(out)


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