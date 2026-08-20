#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from nav_msgs.msg import OccupancyGrid
import numpy as np
import array


def grid_key(info):
    return (info.width, info.height,
            round(info.origin.position.x, 3), round(info.origin.position.y, 3),
            round(info.resolution, 4))


class SSRLCombiner(Node):
    def __init__(self):
        super().__init__('ssrl_combiner')
        self.declare_parameter('semantic_topic', '/semantic_costmap')
        self.declare_parameter('layer_topics', ['/confinement_cost'])
        self.declare_parameter('out_topic', '/ssrl_costmap')

        self.semantic_topic = self.get_parameter('semantic_topic').value
        self.layer_topics = list(self.get_parameter('layer_topics').value)
        out_topic = self.get_parameter('out_topic').value

        # la semantica e' pubblicata LATCHED (transient_local): sottoscrivo
        # transient_local per ricevere subito l'ultima mappa anche partendo dopo.
        qos_latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        # i layer (confinement, dinamici) sono volatile normali
        qos_vol = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.layers = {}        # topic -> (key, ndarray)
        self.last_sem = None    # (key, ndarray, header, info) ultima semantica

        self.create_subscription(OccupancyGrid, self.semantic_topic,
                                 self.semantic_cb, qos_latched)
        for tp in self.layer_topics:
            self.create_subscription(
                OccupancyGrid, tp, lambda m, t=tp: self.layer_cb(t, m), qos_vol)

        # output LATCHED: il costmap di Nav2 (o chiunque parta dopo) riceve
        # subito l'ultima /ssrl_costmap.
        self.pub = self.create_publisher(OccupancyGrid, out_topic, qos_latched)
        self.get_logger().info(
            f'SSRL combiner: {self.semantic_topic} + {self.layer_topics} -> {out_topic}')

    # -------------------------------------------------------------------------
    def layer_cb(self, topic, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        arr = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.layers[topic] = (grid_key(msg.info), arr)
        # RICOMBINA subito: il layer (es. confinement) arriva DOPO la semantica
        # da cui deriva; senza questo, /ssrl_costmap userebbe sempre il layer del
        # ciclo precedente.
        if self.last_sem is not None and self.last_sem[0] == grid_key(msg.info):
            self.combine_and_publish()

    def semantic_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        sem = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.last_sem = (grid_key(msg.info), sem, msg.header, msg.info)
        self.combine_and_publish()

    # -------------------------------------------------------------------------
    def combine_and_publish(self):
        key, sem, header, info = self.last_sem
        total = sem.copy()
        known = sem >= 0
        for topic, (lkey, arr) in self.layers.items():
            if lkey != key or arr.shape != sem.shape:
                continue        # geometria non allineata: salta (si riallinea al giro dopo)
            add = np.clip(arr, 0, 100)
            total[known] = total[known] + add[known]
        total[known] = np.clip(total[known], 0, 100)
        total[~known] = -1

        out = OccupancyGrid()
        out.header = header
        out.info = info
        # tobytes: ~100x piu' veloce di .tolist() sulla full map
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