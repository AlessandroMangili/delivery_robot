import sys
import os
import glob
import math
import sqlite3
import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, PointCloud2

DT = np.dtype({'names': ['x', 'y', 'z'], 'formats': ['<f4', '<f4', '<f4'], 'offsets': [0, 4, 8], 'itemsize': 22})


def dbs(bag):
    return sorted(glob.glob(os.path.join(bag, '*.db3')), key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))


def messaggi(bag, topic):
    for db in dbs(bag):
        c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
        ids = [i for i, n in c.execute('select id,name from topics') if n == topic]
        if ids:
            yield from c.execute('select timestamp,data from messages where topic_id=? order by timestamp', (ids[0],))


def suolo(m):
    a = np.frombuffer(bytes(m.data), dtype=DT) if m.point_step == 22 else None
    if a is None:
        return None
    x, y, z = a['x'].astype(float), a['y'].astype(float), a['z'].astype(float)
    f = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[f], y[f], z[f]
    r = np.hypot(x, y)
    s = (r > 2.2) & (r < 7.0) & (np.abs(z + 0.535) < 0.25)
    for tol in (0.10, 0.04, 0.03):
        if s.sum() < 400:
            return None
        cf, *_ = np.linalg.lstsq(np.c_[x[s], y[s], np.ones(s.sum())], z[s], rcond=None)
        s = (r > 2.2) & (r < 7.0) & (np.abs(z - (cf[0] * x + cf[1] * y + cf[2])) < tol)
    return math.degrees(math.atan(cf[0])), math.degrees(math.atan(cf[1]))


def main():
    bag = sys.argv[1]
    I = []
    for ts, d in messaggi(bag, '/mavros/imu/data'):
        m = deserialize_message(d, Imu)
        q = m.orientation
        I.append((ts * 1e-9, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
                  math.degrees(math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))),
                  math.degrees(math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))))))
    I = np.array(I)
    yaw = np.unwrap(I[:, 1])
    R = []
    for ts, d in messaggi(bag, '/velodyne_points'):
        g = suolo(deserialize_message(d, PointCloud2))
        if g is None:
            continue
        k = int(np.argmin(np.abs(I[:, 0] - ts * 1e-9)))
        R.append((yaw[k], g[0], g[1], I[k, 2], I[k, 3]))
    R = np.array(R)
    if len(R) == 0:
        print('nessuna nuvola utile nella bag')
        return
    b = (np.mod(R[:, 0], 2 * math.pi) // (math.pi / 6)).astype(int)
    if np.unique(b).size < 12:
        print(f'il robot non ha coperto tutti i 360 gradi (settori di 30 gradi coperti: {np.unique(b).size}/12)')
        return
    M = np.array([R[b == k, 1:].mean(axis=0) for k in range(12)]).mean(axis=0)
    print(f'scansioni {len(R)}, media sui 12 settori di direzione')
    print(f'LiDAR rispetto al telaio: beccheggio {M[0]:+.2f} gradi (suolo che sale davanti), rollio {M[1]:+.2f} gradi (suolo che sale a sinistra)')


if __name__ == '__main__':
    main()
