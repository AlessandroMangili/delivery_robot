#!/usr/bin/env python3
"""Estrae frame camera<->LiDAR SOLO a robot fermo (da /odom).

Perche': /rslidar_points non ha timestamp per punto e una spazzata dura 100 ms;
a robot in moto non si sa quando e' stato spazzato il settore visto dalla camera
(a 1 m/s fino a ~11 cm di errore). A robot fermo la scena e' statica e il
problema sparisce (restano solo eventuali pedoni in movimento).

Per ogni intervallo di fermo lungo almeno --min-still s si prendono fino a
--per-interval nuvole equispaziate, con l'immagine piu' vicina per stamp.
Uscita compatibile con calib_edges.py (frame_XXXX.npz + *_intrinsics.yaml).
"""
import argparse
import os

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

from extract_frames import Bag, closest_image, intrinsics, stamp_ns, to_rgb


def still_intervals(bag, v_eps, w_eps, min_len_s):
    rows = bag.cur.execute('SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp',
                           (bag.ids['/odom'],)).fetchall()
    t, still = [], []
    for (d,) in rows:
        o = deserialize_message(d, Odometry)
        t.append(stamp_ns(o))
        tw = o.twist.twist
        still.append(np.hypot(tw.linear.x, tw.linear.y) < v_eps and abs(tw.angular.z) < w_eps)
    t, still = np.array(t), np.array(still)
    out, start = [], None
    for i, s in enumerate(still):
        if s and start is None:
            start = i
        if (not s or i == len(still) - 1) and start is not None:
            end = i if s else i - 1
            if (t[end] - t[start]) / 1e9 >= min_len_s:
                out.append((t[start], t[end]))
            start = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('db')
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-still', type=float, default=2.0, help='durata minima fermo [s]')
    ap.add_argument('--per-interval', type=int, default=2)
    ap.add_argument('--margin', type=float, default=0.5, help='[s] scartati ai bordi del fermo')
    ap.add_argument('--v-eps', type=float, default=0.02)
    ap.add_argument('--w-eps', type=float, default=0.01)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bag = Bag(args.db)
    intrinsics(bag, '/camera1/camera_info', os.path.join(args.out, 'camera1_intrinsics.yaml'))
    intrinsics(bag, '/camera2/camera_info', os.path.join(args.out, 'camera2_intrinsics.yaml'))

    ivs = still_intervals(bag, args.v_eps, args.w_eps, args.min_still)
    print(f'{len(ivs)} intervalli di fermo >= {args.min_still}s')
    # lo stamp e' nell'header (serve deserializzare), ma bag time ~ stamp + 0.33 s
    # basta per restringere la ricerca
    lid_times = np.array([r[0] for r in bag.cur.execute(
        'SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp',
        (bag.ids['/rslidar_points'],)).fetchall()])

    kept = 0
    m = int(args.margin * 1e9)
    for a, b in ivs:
        a, b = a + m, b - m
        if b <= a:
            continue
        # nuvole con stamp nell'intervallo: bag time in [a, b + 1.5 s] come prefiltro
        cand = lid_times[(lid_times >= a) & (lid_times <= b + 1_500_000_000)]
        clouds = []
        for ts in cand:
            (data,) = bag.cur.execute('SELECT data FROM messages WHERE topic_id=? AND timestamp=?',
                                      (bag.ids['/rslidar_points'], int(ts))).fetchone()
            c = deserialize_message(data, PointCloud2)
            if a <= stamp_ns(c) <= b:
                clouds.append(c)
        if not clouds:
            continue
        idx = np.unique(np.linspace(0, len(clouds) - 1, args.per_interval).round().astype(int))
        for i in idx:
            c = clouds[i]
            ls = stamp_ns(c)
            c1 = closest_image(bag, '/camera1/image_raw', ls, ls)
            c2 = closest_image(bag, '/camera2/image_raw', ls, ls)
            if c1 is None or c2 is None:
                continue
            arr = np.frombuffer(c.data, np.float32).reshape(c.height, c.width, 4)
            t_rel = (ls - bag.t0) / 1e9
            np.savez_compressed(os.path.join(args.out, f'frame_{kept:04d}.npz'),
                                cloud=arr, img1=to_rgb(c1[1]), img2=to_rgb(c2[1]),
                                dt1_ms=c1[0] / 1e6, dt2_ms=c2[0] / 1e6, speed=0.0, yaw_rate=0.0,
                                t_rel=t_rel)
            print(f'frame_{kept:04d}: t={t_rel:7.1f}s  fermo {(b - a) / 1e9 + 2 * args.margin:5.1f}s  '
                  f'dt1={c1[0] / 1e6:6.1f} dt2={c2[0] / 1e6:6.1f} ms')
            kept += 1
    print(f'\n{kept} frame salvati in {args.out}')


if __name__ == '__main__':
    main()
