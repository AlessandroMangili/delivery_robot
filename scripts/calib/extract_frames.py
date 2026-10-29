#!/usr/bin/env python3
"""Estrae frame sincronizzati camera<->LiDAR da una rosbag2 sqlite3 (anche da 180 GB).

Per ogni istante campione prende la nuvola /rslidar_points e, per ciascuna camera,
l'immagine con header.stamp piu' vicino. Scarta i frame con sincronizzazione
peggiore di --max-dt-ms o con il robot in moto veloce (da /odom): durante la
spazzata del LiDAR la posa cambia e l'allineamento diventa inaffidabile.

Uscita in --out: frame_XXXX.npz con
  cloud  (H,W,4) float32 organizzata (x,y,z,intensity), colonne = anelli
  img1, img2  RGB uint8 grezze (NON schiarite)
  dt1_ms, dt2_ms, speed, yaw_rate, t_rel
+ camera1_intrinsics.yaml / camera2_intrinsics.yaml nel formato di calib_pnp.py.
"""
import argparse
import os
import sqlite3

import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from nav_msgs.msg import Odometry


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


class Bag:
    def __init__(self, db):
        self.cur = sqlite3.connect(f'file:{db}?mode=ro', uri=True).cursor()
        self.ids = {n: i for i, n in self.cur.execute('SELECT id, name FROM topics')}
        self.t0, self.t1 = self.cur.execute(
            'SELECT min(timestamp), max(timestamp) FROM messages').fetchone()

    def window(self, topic, ts, half_ns):
        return self.cur.execute(
            'SELECT timestamp, data FROM messages WHERE timestamp BETWEEN ? AND ? '
            'AND topic_id=? ORDER BY timestamp', (ts - half_ns, ts + half_ns, self.ids[topic])).fetchall()


def intrinsics(bag, topic, path):
    rows = bag.window(topic, bag.t0 + 30_000_000_000, 2_000_000_000)
    m = deserialize_message(rows[0][1], CameraInfo)
    yaml.safe_dump({
        'image_width': int(m.width), 'image_height': int(m.height),
        'camera_matrix': {'rows': 3, 'cols': 3, 'data': [float(v) for v in m.k]},
        'distortion_coefficients': {'rows': 1, 'cols': len(m.d), 'data': [float(v) for v in m.d]},
        'distortion_model': m.distortion_model,
    }, open(path, 'w'))


def closest_image(bag, topic, lidar_stamp, bag_ts):
    best = None
    for _, data in bag.window(topic, bag_ts, 200_000_000):
        m = deserialize_message(data, Image)
        dt = abs(stamp_ns(m) - lidar_stamp)
        if best is None or dt < best[0]:
            best = (dt, m)
    return best


def to_rgb(m):
    buf = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)
    enc = m.encoding.lower()
    if enc in ('rgb8', 'bgr8'):
        img = buf[:, :m.width * 3].reshape(m.height, m.width, 3)
        return img[:, :, ::-1].copy() if enc == 'bgr8' else img.copy()
    raise ValueError(f'encoding non gestito: {m.encoding}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('db')
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=float, default=3.0, help='passo di campionamento [s]')
    ap.add_argument('--max-dt-ms', type=float, default=15.0)
    ap.add_argument('--max-speed', type=float, default=1.0, help='[m/s]')
    ap.add_argument('--max-yaw-rate', type=float, default=0.15, help='[rad/s]')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bag = Bag(args.db)
    intrinsics(bag, '/camera1/camera_info', os.path.join(args.out, 'camera1_intrinsics.yaml'))
    intrinsics(bag, '/camera2/camera_info', os.path.join(args.out, 'camera2_intrinsics.yaml'))

    step = int(args.every * 1e9)
    kept = 0
    for ts in range(bag.t0 + step, bag.t1 - step, step):
        lid = bag.window('/rslidar_points', ts, 100_000_000)   # LiDAR ~6 Hz: periodo ~160 ms
        if not lid:
            continue
        lts, ldata = min(lid, key=lambda r: abs(r[0] - ts))
        cloud = deserialize_message(ldata, PointCloud2)
        ls = stamp_ns(cloud)

        # la nuvola entra nel bag ~330 ms DOPO il suo stamp (registrazione satura),
        # le immagini ~6 ms dopo: si cercano attorno allo stamp LiDAR, non al tempo bag
        c1 = closest_image(bag, '/camera1/image_raw', ls, ls)
        c2 = closest_image(bag, '/camera2/image_raw', ls, ls)
        if c1 is None or c2 is None:
            continue
        dt1, dt2 = c1[0] / 1e6, c2[0] / 1e6

        od = bag.window('/odom', ls, 100_000_000)
        if not od:
            continue
        o = deserialize_message(min(od, key=lambda r: abs(r[0] - ls))[1], Odometry)
        speed = float(np.hypot(o.twist.twist.linear.x, o.twist.twist.linear.y))
        yaw_rate = float(abs(o.twist.twist.angular.z))

        t_rel = (lts - bag.t0) / 1e9
        ok = dt2 <= args.max_dt_ms and speed <= args.max_speed and yaw_rate <= args.max_yaw_rate
        print(f't={t_rel:7.1f}s dt1={dt1:6.1f} dt2={dt2:6.1f}ms v={speed:.2f} w={yaw_rate:.2f} '
              f'{"KEEP" if ok else "skip"}')
        if not ok:
            continue

        arr = np.frombuffer(cloud.data, np.float32).reshape(cloud.height, cloud.width, 4)
        np.savez_compressed(os.path.join(args.out, f'frame_{kept:04d}.npz'),
                            cloud=arr, img1=to_rgb(c1[1]), img2=to_rgb(c2[1]),
                            dt1_ms=dt1, dt2_ms=dt2, speed=speed, yaw_rate=yaw_rate, t_rel=t_rel)
        kept += 1
    print(f'\n{kept} frame salvati in {args.out}')


if __name__ == '__main__':
    main()
