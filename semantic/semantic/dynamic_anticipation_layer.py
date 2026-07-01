#!/usr/bin/env python3
"""
SSRL - LAYER DI ANTICIPAZIONE DINAMICA (pedoni, bici, oggetti che rotolano...).

CONTRIBUTO risk-aware sui DINAMICI. Per ogni oggetto in movimento proietta:

  (1) NUCLEO DI SICUREZZA attorno alla posizione attuale (costo alto):
      il robot NON gli va addosso. Raggio pieno r_core, poi decadimento.

  (2) SCIA ANTICIPATORIA proiettata IN AVANTI nella direzione del moto,
      lunga in proporzione alla VELOCITA' (oggetti veloci = scia lunga):
      il robot "vede" dove l'oggetto STARA' e lo aggira con anticipo,
      invece di congelarsi davanti a un ostacolo improvviso (anti-freezing).
      La scia si allarga leggermente a cono (incertezza di direzione) e
      decade con la distanza.

Ingressi:
  /dynamic_tracks  (MarkerArray di frecce): posizione + velocita' di ogni
                   oggetto dinamico in movimento (dal semantic_costmap_node).
  /semantic_costmap (OccupancyGrid): SOLO per la geometria (origin/res/size),
                   cosi' la mappa prodotta e' allineata IDENTICA alle altre
                   (requisito del ssrl_combiner).

Uscita:
  /pedestrian_cost (OccupancyGrid, stessa geometria): il campo di costo
                   dinamico, da aggiungere in ssrl_combiner (layer_topics).

NOTA: pubblica sulla callback di /semantic_costmap, cosi' geometria e timing
combaciano sempre con la mappa semantica corrente.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
import numpy as np


class DynamicAnticipationLayer(Node):
    def __init__(self):
        super().__init__('dynamic_anticipation_layer')

        self.declare_parameter('tracks_topic', '/dynamic_tracks')
        self.declare_parameter('geom_topic', '/semantic_costmap')
        self.declare_parameter('out_topic', '/pedestrian_cost')

        # --- nucleo di sicurezza (non andargli addosso) ---
        self.declare_parameter('c_safe', 100.0)      # costo sul corpo dell'oggetto
        self.declare_parameter('r_core', 0.5)        # m: raggio a costo pieno
        self.declare_parameter('core_falloff', 0.4)  # m: decadimento oltre r_core

        # --- scia anticipatoria (dove STARA') ---
        self.declare_parameter('c_ant', 80.0)        # costo max della scia
        self.declare_parameter('horizon_time', 2.0)  # s: quanto lontano anticipare
        self.declare_parameter('l_min', 0.5)         # m: scia minima
        self.declare_parameter('l_max', 4.0)         # m: scia massima (cap)
        self.declare_parameter('w0', 0.4)            # m: mezza-larghezza iniziale
        self.declare_parameter('spread', 0.3)        # allargamento a cono per metro

        # --- limiti di calcolo (efficienza: elabora solo l'intorno di ogni traccia) ---
        self.declare_parameter('max_radius', 5.0)    # m: oltre non calcolo costo

        gp = self.get_parameter
        self.tracks_topic = gp('tracks_topic').value
        self.geom_topic = gp('geom_topic').value
        out_topic = gp('out_topic').value
        self.c_safe = gp('c_safe').value
        self.r_core = gp('r_core').value
        self.core_falloff = gp('core_falloff').value
        self.c_ant = gp('c_ant').value
        self.horizon_time = gp('horizon_time').value
        self.l_min = gp('l_min').value
        self.l_max = gp('l_max').value
        self.w0 = gp('w0').value
        self.spread = gp('spread').value
        self.max_radius = gp('max_radius').value

        # tracce correnti: lista di (px, py, vx, vy)
        self.tracks = []

        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(MarkerArray, self.tracks_topic, self.tracks_cb, qos)
        self.create_subscription(OccupancyGrid, self.geom_topic, self.geom_cb, qos)
        self.pub = self.create_publisher(OccupancyGrid, out_topic, qos)

        self.get_logger().info(
            f'Dynamic anticipation layer: {self.tracks_topic} + {self.geom_topic} '
            f'-> {out_topic}')

    def tracks_cb(self, msg: MarkerArray):
        tracks = []
        for m in msg.markers:
            # solo frecce con 2 punti (posizione -> posizione+velocita')
            if len(m.points) != 2:
                continue
            px, py = m.points[0].x, m.points[0].y
            vx = m.points[1].x - px
            vy = m.points[1].y - py
            tracks.append((px, py, vx, vy))
        self.tracks = tracks

    def geom_cb(self, msg: OccupancyGrid):
        """Su ogni mappa semantica, rigenera il campo dinamico con la stessa geometria."""
        h, w = msg.info.height, msg.info.width
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        cost = np.zeros((h, w), dtype=np.float32)

        if self.tracks:
            # coordinate mondo dei centri cella (una volta sola)
            xs = ox + (np.arange(w) + 0.5) * res
            ys = oy + (np.arange(h) + 0.5) * res
            for (px, py, vx, vy) in self.tracks:
                self._add_track(cost, xs, ys, res, px, py, vx, vy)

        out = OccupancyGrid()
        out.header = msg.header
        out.info = msg.info
        data = np.clip(np.round(cost), 0, 100).astype(np.int8)
        out.data = data.ravel(order='C').tolist()
        self.pub.publish(out)

    def _add_track(self, cost, xs, ys, res, px, py, vx, vy):
        """Aggiunge (in max) il costo di UNA traccia, calcolando solo la finestra
        attorno all'oggetto per efficienza."""
        w = xs.size; h = ys.size
        # finestra di calcolo attorno alla traccia (max_radius + eventuale scia)
        s = float(np.hypot(vx, vy))
        L = float(np.clip(s * self.horizon_time, self.l_min, self.l_max)) if s > 1e-3 else 0.0
        reach = self.max_radius + L
        i0 = max(0, int((px - reach - xs[0]) / res))
        i1 = min(w, int((px + reach - xs[0]) / res) + 1)
        j0 = max(0, int((py - reach - ys[0]) / res))
        j1 = min(h, int((py + reach - ys[0]) / res) + 1)
        if i1 <= i0 or j1 <= j0:
            return

        X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1])
        rx = X - px; ry = Y - py
        r = np.hypot(rx, ry)

        # (1) nucleo di sicurezza omnidirezionale
        core = np.where(r <= self.r_core, self.c_safe,
                        self.c_safe * np.exp(-(r - self.r_core) / self.core_falloff))
        core = np.where(r <= self.max_radius, core, 0.0)

        block = core
        # (2) scia anticipatoria (solo se in movimento)
        if s > 1e-3:
            th = np.arctan2(vy, vx); ct, st = np.cos(th), np.sin(th)
            along = rx * ct + ry * st          # avanti positivo
            lat = -rx * st + ry * ct           # laterale
            width = self.w0 + self.spread * np.clip(along, 0, None)
            along_decay = np.clip(1.0 - along / L, 0.0, 1.0)
            lat_decay = np.exp(-(lat / width) ** 2)
            wake = np.where((along >= 0) & (along <= L),
                            self.c_ant * along_decay * lat_decay, 0.0)
            block = np.maximum(core, wake)

        # combina con quanto gia' presente (max: piu' oggetti non si sommano a caso)
        sub = cost[j0:j1, i0:i1]
        np.maximum(sub, block, out=sub)


def main():
    rclpy.init()
    node = DynamicAnticipationLayer()
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