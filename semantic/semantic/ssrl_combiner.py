#!/usr/bin/env python3
"""
SSRL - COMBINATORE.

Somma la mappa semantica (misura) con i layer di costo (politiche di rischio:
confinamento ora, dinamici in futuro) e pubblica la costmap finale /ssrl_costmap,
quella che andra' data al planner.

Regola: ssrl = semantic + somma(layer), saturato a 100. Le celle ignote (-1)
nella semantica restano ignote. I layer si allineano per geometria identica
(stesso origin/risoluzione/size della semantica, da cui derivano).

Aggiungere un nuovo layer in futuro = aggiungerlo a 'layer_topics'.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
import numpy as np


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

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)

        self.layers = {}   # topic -> (key, ndarray)
        self.create_subscription(OccupancyGrid, self.semantic_topic, self.semantic_cb, qos)
        for tp in self.layer_topics:
            self.create_subscription(
                OccupancyGrid, tp, lambda m, t=tp: self.layer_cb(t, m), qos)

        self.pub = self.create_publisher(OccupancyGrid, out_topic, qos)
        self.get_logger().info(
            f'SSRL combiner: {self.semantic_topic} + {self.layer_topics} -> {out_topic}')

    def layer_cb(self, topic, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        arr = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.layers[topic] = (grid_key(msg.info), arr)

    def semantic_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        sem = np.array(msg.data, dtype=np.int16).reshape(h, w)
        key = grid_key(msg.info)

        total = sem.copy()
        known = sem >= 0
        for topic, (lkey, arr) in self.layers.items():
            if lkey != key or arr.shape != sem.shape:
                continue                      # geometria non allineata: salta (si riallinea al frame dopo)
            add = np.clip(arr, 0, 100)
            total[known] = total[known] + add[known]

        total[known] = np.clip(total[known], 0, 100)
        total[~known] = -1

        out = OccupancyGrid()
        out.header = msg.header
        out.info = msg.info
        out.data = total.astype(np.int8).ravel(order='C').tolist()
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