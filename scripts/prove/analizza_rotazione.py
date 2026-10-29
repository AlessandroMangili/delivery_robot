import sys, glob, sqlite3, os
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag = sys.argv[1]
D = {'/mavros/imu/data': [], '/a200_0000/cmd_vel': [], '/a200_0000/platform/odom': [], '/cmd_vel_safe': []}
for db in sorted(glob.glob(os.path.join(bag, '*.db3'))):
    c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    tp = {i: (n, t) for i, n, t in c.execute('select id,name,type from topics') if n in D}
    for tid, ts, data in c.execute('select topic_id,timestamp,data from messages order by timestamp'):
        if tid not in tp:
            continue
        n, ty = tp[tid]
        m = deserialize_message(data, get_message(ty))
        t = ts * 1e-9
        if n == '/mavros/imu/data':
            D[n].append((t, m.angular_velocity.z))
        elif n == '/a200_0000/cmd_vel':
            D[n].append((t, m.twist.linear.x, m.twist.angular.z))
        elif n == '/cmd_vel_safe':
            D[n].append((t, m.linear.x, m.angular.z))
        else:
            D[n].append((t, m.twist.twist.linear.x, m.twist.twist.angular.z))
imu = np.array(D['/mavros/imu/data']); cmd = np.array(D['/cmd_vel_safe'] if len(D['/cmd_vel_safe']) > 10 else D['/a200_0000/cmd_vel']); odo = np.array(D['/a200_0000/platform/odom'])
ic = np.clip(np.searchsorted(cmd[:, 0], imu[:, 0]) - 1, 0, len(cmd) - 1)
cv, cw = cmd[ic, 1], cmd[ic, 2]
wz = imu[:, 1]
io = np.clip(np.searchsorted(odo[:, 0], imu[:, 0]), 0, len(odo) - 1)
ww = odo[io, 2]
print(f'{"comando":24s} {"s":>5s} {"IMU medio":>10s} {"IMU/cmd":>8s} {"fermo %":>8s} {">2x cmd %":>9s} {"picco":>7s} {"IMU/ruote":>9s}')
for v in sorted(set(np.round(cv, 2))):
    for w in sorted(set(np.round(cw, 2))):
        if abs(w) < 0.05:
            continue
        m = (np.round(cv, 2) == v) & (np.round(cw, 2) == w)
        if m.sum() < 50:
            continue
        g = np.degrees(wz[m]) * np.sign(w)
        cmd_d = np.degrees(abs(w))
        k = np.abs(ww[m]) > 0.02
        rap = np.median(np.abs(wz[m][k]) / np.abs(ww[m][k])) if k.sum() > 20 else np.nan
        print(f'v {v:+.2f} w {w:+.2f} rad/s  {m.sum()/50:5.1f} {g.mean():8.1f}°/s {g.mean()/cmd_d:8.2f} {100*np.mean(np.abs(g) < 3):7.0f}% '
              f'{100*np.mean(g > 2 * cmd_d):8.0f}% {np.abs(g).max():6.0f}° {rap:9.2f}')
print('IMU/cmd = rotazione vera media divisa per quella comandata (1 = giusto). fermo = |rotazione| < 3°/s mentre si comanda.')
print('IMU/ruote = rotazione vera divisa per quella stimata dal driver del Pi: il moltiplicatore giusto è 1.875 / IMU/ruote.')
