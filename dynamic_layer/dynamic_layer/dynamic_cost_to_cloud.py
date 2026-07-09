#!/usr/bin/env python3
"""
dynamic_cost_to_cloud.py -- Convertitore RAPIDO per test (senza plugin C++).

Si iscrive a /dynamic_cost (OccupancyGrid, in odom) e ripubblica le celle sopra
soglia come PointCloud2 in odom. Poi aggiungi quella PointCloud2 come
observation_source dell'obstacle_layer della local costmap: e' l'obstacle_layer
a marcarla come ostacolo nella costmap rolling (rolling gestito nativamente).

LIMITI (per questo il layer C++ resta la soluzione definitiva):
  - marcatura BINARIA/letale: il gradiente del cono si perde (diventa un muro).
  - possibili SCIE: l'obstacle_layer marca ma pulisce solo via raytrace dello
    scan, quindi vecchie posizioni del cono possono restare qualche istante.

Uso: python3 dynamic_cost_to_cloud.py --ros-args -p use_sim_time:=true
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Header
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


class DynamicCostToCloud(Node):
    def __init__(self):
        super().__init__('dynamic_cost_to_cloud')

        self.declare_parameter('input_topic', '/dynamic_cost')
        self.declare_parameter('output_topic', '/dynamic_obstacles')
        self.declare_parameter('threshold', 1)     # celle >= soglia -> ostacolo
        self.declare_parameter('point_z', 0.1)     # quota punti (entro max_obstacle_height)

        self.in_topic = self.get_parameter('input_topic').value
        self.out_topic = self.get_parameter('output_topic').value
        self.threshold = int(self.get_parameter('threshold').value)
        self.z = float(self.get_parameter('point_z').value)

        # QoS che combacia col publisher del tracker (transient_local + reliable)
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST)

        self.sub = self.create_subscription(OccupancyGrid, self.in_topic, self.cb, qos)
        self.pub = self.create_publisher(PointCloud2, self.out_topic, 1)

        self.get_logger().info(
            f'dynamic_cost_to_cloud avviato | {self.in_topic} -> {self.out_topic} '
            f'| soglia={self.threshold}')

    def cb(self, msg):
        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        ys, xs = np.where(data >= self.threshold)   # (riga, colonna) sopra soglia

        header = Header()
        header.stamp = msg.header.stamp          # stesso istante della griglia
        header.frame_id = msg.header.frame_id    # odom

        if xs.size == 0:
            self.pub.publish(pc2.create_cloud_xyz32(header, []))
            return

        # centro cella -> coordinata mondo (odom)
        wx = ox + (xs + 0.5) * res
        wy = oy + (ys + 0.5) * res
        wz = np.full(xs.shape, self.z)
        points = np.stack([wx, wy, wz], axis=1).astype(np.float32)

        self.pub.publish(pc2.create_cloud_xyz32(header, points.tolist()))


def main():
    rclpy.init()
    node = DynamicCostToCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
