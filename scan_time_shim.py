#!/usr/bin/env python3
"""
scan_time_shim.py — aggiunge il campo per-punto 'time' alle nuvole di Gazebo.

FAST-LIO deskewa lo scan usando il tempo RELATIVO di ogni punto dentro la
scansione, ma il gpu_lidar di Gazebo pubblica una PointCloud2 SENZA quel campo.
Questo nodo lo sintetizza:

    /scan/points (x,y,z,intensity,ring)  ->  /scan/points_timed (x,y,z,intensity,time)

Stima del tempo: un LiDAR rotante a `scan_period` s spazza 2*pi in un giro, quindi
un punto ad azimuth theta è stato acquisito alla frazione (theta+pi)/(2*pi) del
periodo -> time = frazione * scan_period, in [0, scan_period).
È un'APPROSSIMAZIONE (assume rotazione uniforme). Per un robot lento va bene; il
deskew "vero" servira' sul LiDAR fisico, che il campo time lo fornisce gia'.

NB: leggo con read_points (NON read_points_numpy) perche' la nuvola ha datatype
MISTI (x,y,z,intensity=FLOAT32, ring=UINT16) e read_points_numpy pretende campi
omogenei.

Dipendenze: numpy, sensor_msgs_py.
"""
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


# ---------------------------------------------------------------------------
# Funzione PURA (niente ROS) -> testabile in isolamento
# ---------------------------------------------------------------------------
def azimuth_time(x, y, scan_period):
    """Tempo relativo [s] di ogni punto stimato dall'azimuth. Ritorna un array
    float32 in [0, scan_period). x, y: array numpy. Vettorializzata."""
    az = np.arctan2(y, x)                 # [-pi, pi]
    frac = (az + np.pi) / (2.0 * np.pi)   # [0, 1)
    frac = np.clip(frac, 0.0, 0.999999)   # evita esattamente scan_period
    return (frac * float(scan_period)).astype(np.float32)


class ScanTimeShim(Node):
    def __init__(self):
        super().__init__('scan_time_shim')
        self.declare_parameter('input_topic', '/scan/points')
        self.declare_parameter('output_topic', '/scan/points_timed')
        self.declare_parameter('scan_period', 0.1)   # 10 Hz -> 0.1 s
        self.in_topic = self.get_parameter('input_topic').value
        self.out_topic = self.get_parameter('output_topic').value
        self.scan_period = float(self.get_parameter('scan_period').value)

        # QoS Best Effort: le nuvole dal bridge Gazebo sono Best Effort
        qos = QoSProfile(depth=5,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(PointCloud2, self.out_topic, qos)
        self.sub = self.create_subscription(
            PointCloud2, self.in_topic, self.on_cloud, qos)
        self._frame = 0
        self._has_i = None
        self.get_logger().info(
            f"scan_time_shim: {self.in_topic} -> {self.out_topic} "
            f"(scan_period={self.scan_period}s, campo 'time' in secondi)")

    def on_cloud(self, msg):
        self._frame += 1
        if self._has_i is None:
            self._has_i = any(f.name == 'intensity' for f in msg.fields)
        names = ['x', 'y', 'z', 'intensity'] if self._has_i else ['x', 'y', 'z']

        # read_points: gestisce datatype MISTI (la nuvola ha ring=UINT16).
        # Ritorna un array strutturato con un campo per nome richiesto.
        rec = pc2.read_points(msg, field_names=names, skip_nans=False)
        n = int(rec.shape[0])
        if n == 0:
            return
        x = np.asarray(rec['x'], dtype=np.float32)
        y = np.asarray(rec['y'], dtype=np.float32)
        z = np.asarray(rec['z'], dtype=np.float32)
        inten = (np.asarray(rec['intensity'], dtype=np.float32)
                 if self._has_i else np.zeros(n, dtype=np.float32))
        t = azimuth_time(x, y, self.scan_period)

        # struttura di uscita: x,y,z,intensity,time (layout stile Velodyne)
        out = np.zeros(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                 ('intensity', '<f4'), ('time', '<f4')])
        out['x'] = x
        out['y'] = y
        out['z'] = z
        out['intensity'] = inten
        out['time'] = t

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='time', offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        new = PointCloud2()
        new.header = msg.header            # stesso stamp e frame (base_scan)
        new.height = 1
        new.width = n
        new.fields = fields
        new.is_bigendian = False
        new.point_step = 20
        new.row_step = 20 * n
        new.data = out.tobytes()
        new.is_dense = msg.is_dense
        self.pub.publish(new)

        # log diagnostico: conferma che sta ricevendo e quanti punti processa
        if self._frame == 1 or self._frame % 20 == 0:
            self.get_logger().info(
                f"[shim] frame={self._frame} punti={n} "
                f"time in [{float(t.min()):.4f}, {float(t.max()):.4f}] s")


def main(args=None):
    rclpy.init(args=args)
    node = ScanTimeShim()
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
