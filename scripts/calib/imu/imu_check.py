#!/usr/bin/env python3
"""
imu_check.py — verifica dell'IMU del Pixhawk via MAVROS (robot FERMO).

Ascolta per `durata` secondi e stampa: frequenza dei topic, media/rumore di
accelerometro e giroscopio, modulo della gravita', orientamento EKF, stato FCU.

    python3 imu_check.py          # 10 s
    python3 imu_check.py 30       # 30 s
"""
import sys
import time

import numpy as np
import rclpy
from mavros_msgs.msg import State
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from sensor_msgs.msg import Imu, MagneticField

G = 9.80665


def quat_to_rpy_deg(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([r, p, yw])


def main():
    durata = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    rclpy.init()
    n = Node("imu_check")
    d = {"data": [], "data_raw": [], "mag": []}
    state = [None]
    # nei primi ~30 s dopo la connessione MAVROS scarica gli ~800 parametri del Pixhawk
    # e gli stream rallentano (25-35 Hz invece di 50): misurare dopo "PR: parameters list received"
    qos = QoSProfile(depth=200, reliability=ReliabilityPolicy.BEST_EFFORT)
    for k, typ in (("data", Imu), ("data_raw", Imu), ("mag", MagneticField)):
        n.create_subscription(typ, f"/mavros/imu/{k}",
                              lambda m, k=k: d[k].append((time.monotonic(), m)), qos)
    n.create_subscription(State, "/mavros/state", lambda m: state.__setitem__(0, m), 10)

    print(f"ascolto per {durata:.0f} s, robot fermo...")
    fine = Future()
    n.create_timer(durata, lambda: fine.set_result(True))
    rclpy.spin_until_future_complete(n, fine)

    s = state[0]
    print("\nstato FCU:", "nessun messaggio su /mavros/state (MAVROS e' attivo?)" if s is None
          else f"connected={s.connected} mode={s.mode} armed={s.armed}")
    for k, L in d.items():
        hz = (len(L) - 1) / (L[-1][0] - L[0][0]) if len(L) > 1 else 0.0
        print(f"/mavros/imu/{k:9s} {len(L):5d} msg  {hz:6.1f} Hz")
    if not d["data"]:
        print("\nNessun dato IMU: lo stream non e' stato richiesto? (set_stream_rate)")
        return

    m = [msg for _, msg in d["data"]]
    A = np.array([[x.linear_acceleration.x, x.linear_acceleration.y, x.linear_acceleration.z] for x in m])
    W = np.array([[x.angular_velocity.x, x.angular_velocity.y, x.angular_velocity.z] for x in m])
    g = np.linalg.norm(A.mean(0))
    print(f"\nframe_id: {m[0].header.frame_id}")
    print(f"accelerometro media {A.mean(0).round(3)} m/s^2   rumore {A.std(0).round(4)}")
    print(f"  |g| = {g:.3f} m/s^2  ({100 * (g - G) / G:+.1f}% rispetto a {G})"
          + ("" if abs(g - G) < 0.2 else "   <-- accelerometri da calibrare"))
    tilt = np.degrees(np.arctan2(np.hypot(*A.mean(0)[:2]), A.mean(0)[2]))
    print(f"  inclinazione dalla gravita': {tilt:.1f} deg")
    print(f"giroscopio media {W.mean(0).round(4)} rad/s   rumore {W.std(0).round(4)}")
    r, p, y = quat_to_rpy_deg(m[-1].orientation)
    print(f"orientamento EKF: roll {r:.1f}  pitch {p:.1f}  yaw {y:.1f} deg (yaw indoor = magnetometro, poco affidabile)")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
