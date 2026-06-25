#!/usr/bin/env python3
"""
SSRL - Layer di CONFINAMENTO.

Legge /semantic_costmap (mappa di categorie) e produce un campo di costo che
penalizza in modo MORBIDO l'avvicinarsi al BORDO del marciapiede, spingendo il
robot a stare verso il centro. E' una "politica di rischio" separata dalla
mappa-misura: si somma sopra (vedi ssrl_combiner).

Metodo: sulle celle di marciapiede calcola la distanza dal bordo (distance
transform) e applica un decadimento esponenziale continuo:
    conf = conf_max * exp(-dist / falloff)
alto al bordo, ~0 verso il centro. Fuori dal marciapiede il confinamento e' 0
(li' al costo ci pensa gia' la classe: strada, erba, ostacoli).

Pubblica /confinement_cost (OccupancyGrid, stessa geometria dell'ingresso).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2


class ConfinementLayer(Node):
    def __init__(self):
        super().__init__('confinement_layer')

        self.declare_parameter('in_topic', '/semantic_costmap')
        self.declare_parameter('out_topic', '/confinement_cost')
        self.declare_parameter('sidewalk_cost_max', 15)  # celle <= questo = marciapiede
        self.declare_parameter('conf_max', 80.0)         # costo al bordo
        self.declare_parameter('falloff', 0.4)           # m: lunghezza di decadimento
        self.declare_parameter('include_unknown_as_edge', True)  # ignoto conta come "fuori"

        in_topic = self.get_parameter('in_topic').value
        out_topic = self.get_parameter('out_topic').value
        self.sw_max = self.get_parameter('sidewalk_cost_max').value
        self.conf_max = self.get_parameter('conf_max').value
        self.falloff = self.get_parameter('falloff').value
        self.unknown_edge = self.get_parameter('include_unknown_as_edge').value

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(OccupancyGrid, in_topic, self.cb, qos)
        self.pub = self.create_publisher(OccupancyGrid, out_topic, qos)
        self.get_logger().info(f'Confinement layer: {in_topic} -> {out_topic}')

    def cb(self, msg: OccupancyGrid):
        h = msg.info.height; w = msg.info.width
        data = np.array(msg.data, dtype=np.int16).reshape(h, w)

        # marciapiede = celle note con costo basso (<= soglia)
        sidewalk = (data >= 0) & (data <= self.sw_max)
        # tutto il resto = "fuori" (bordo). L'ignoto (-1) conta come fuori se richiesto.
        # distance transform: distanza dal bordo per ogni cella di marciapiede
        src = (sidewalk.astype(np.uint8)) * 255
        if not self.unknown_edge:
            # se l'ignoto NON e' bordo, lo "tappiamo" come marciapiede per il calcolo distanza
            src[(data < 0)] = 255
        dist_px = cv2.distanceTransform(src, cv2.DIST_L2, 5)
        dist_m = dist_px * msg.info.resolution

        conf = self.conf_max * np.exp(-dist_m / self.falloff)
        conf[~sidewalk] = 0.0                      # confinamento solo sul marciapiede

        out = OccupancyGrid()
        out.header = msg.header
        out.info = msg.info
        out.data = np.clip(np.round(conf), 0, 100).astype(np.int8).ravel(order='C').tolist()
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ConfinementLayer()
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