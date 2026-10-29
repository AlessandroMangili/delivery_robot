import math, sys, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

DURATA = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
NOM = (-0.5, 0.5)


class Misura(Node):
    def __init__(self):
        super().__init__('controllo_pixhawk')
        self.c = []
        self.create_subscription(Imu, '/mavros/imu/data', self.cb, qos_profile_sensor_data)

    def cb(self, m):
        q, a, g = m.orientation, m.linear_acceleration, m.angular_velocity
        r = math.degrees(math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)))
        p = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))))
        ar = math.degrees(math.atan2(a.y, a.z)); ap = math.degrees(math.atan2(-a.x, math.hypot(a.y, a.z)))
        self.c.append((r - ar, p - ap, math.degrees(g.x), math.degrees(g.y), math.degrees(g.z)))


rclpy.init()
n = Misura()
t0 = time.time()
while time.time() - t0 < DURATA:
    rclpy.spin_once(n, timeout_sec=0.2)
if len(n.c) < 50:
    print('  NO   Pixhawk: /mavros/imu/data non arriva (%d messaggi in %.0f s)' % (len(n.c), DURATA))
    sys.exit(1)
C = np.array(n.c)
gr, gp = np.median(C[:, 0]) - NOM[0], np.median(C[:, 1]) - NOM[1]
gx, gy, gz = np.abs(C[:, 2:5].mean(axis=0))
ok = True
if max(gx, gy, gz) > 0.1:
    print('  NO   Pixhawk: giroscopio fermo a %.2f / %.2f / %.2f gradi/s (normale < 0,05): zero non calibrato.' % (gx, gy, gz))
    print('       Tieni il robot fermo e riavvia il Pixhawk, oppure aspetta ~10 minuti prima dei giri di mappatura.')
    ok = False
if abs(gr) > 0.4 or abs(gp) > 0.4:
    print('  NO   Pixhawk: assetto lontano dagli accelerometri di %+.1f gradi (rollio) e %+.1f gradi (beccheggio), normale entro 0,4.' % (gr, gp))
    print('       L\'elevazione registrata adesso sarebbe inclinata: niente giri di mappatura finche\' non torna nella norma.')
    ok = False
if ok:
    print('  OK   Pixhawk assestato: giroscopio fermo %.2f gradi/s, scarto dall\'accelerometro %+.1f / %+.1f gradi' % (max(gx, gy, gz), gr, gp))
print('       (misura valida solo a robot fermo)')
n.destroy_node(); rclpy.shutdown()
sys.exit(0 if ok else 1)
