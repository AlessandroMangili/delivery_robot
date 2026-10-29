import sys
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

PASSI = [
    ('fermo', 0.0, 0.0, 3.0),
    ('sul posto +0.50', 0.0, 0.50, 5.0), ('fermo', 0.0, 0.0, 3.0),
    ('sul posto -0.50', 0.0, -0.50, 5.0), ('fermo', 0.0, 0.0, 3.0),
    ('sul posto +0.80', 0.0, 0.80, 4.0), ('fermo', 0.0, 0.0, 3.0),
    ('sul posto -0.80', 0.0, -0.80, 4.0), ('fermo', 0.0, 0.0, 3.0),
    ('arco 0.2 m/s +0.50', 0.2, 0.50, 4.0), ('fermo', 0.0, 0.0, 3.0),
    ('arco 0.2 m/s -0.50', 0.2, -0.50, 4.0), ('fermo', 0.0, 0.0, 3.0),
    ('arco 0.3 m/s +0.80', 0.3, 0.80, 3.0), ('fermo', 0.0, 0.0, 3.0),
    ('arco 0.3 m/s -0.80', 0.3, -0.80, 3.0), ('fermo', 0.0, 0.0, 3.0),
]


class Prova(Node):
    def __init__(self):
        super().__init__('prova_rotazione')
        self.pub = self.create_publisher(Twist, 'cmd_vel_safe', 10)

    def manda(self, v, w):
        m = Twist()
        m.linear.x = v
        m.angular.z = w
        self.pub.publish(m)


def main():
    rclpy.init()
    n = Prova()
    try:
        for nome, v, w, durata in PASSI:
            print(f'{time.strftime("%H:%M:%S")} {nome}: v {v} m/s, w {w} rad/s per {durata} s', flush=True)
            t0 = time.time()
            while time.time() - t0 < durata:
                n.manda(v, w)
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(10):
            n.manda(0.0, 0.0)
            time.sleep(0.05)
        print('fine, robot fermo', flush=True)
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
