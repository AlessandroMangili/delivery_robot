#!/usr/bin/env python3
"""
dynamic_esdf_layer.py  --  Layer di costo ANTICIPATORIO per ostacoli dinamici.

Metodo (fedele a Zhong et al., WEVJ 2024, "Dynamic Obstacle Avoidance ...
2D Differential ESDF"): invece di tracciare i pedoni e stimarne la velocita'
(fragile con proiezione monoculare), il MOVIMENTO viene ricavato dalla
DIFFERENZA della mappa di distanza (ESDF) tra due frame consecutivi.

Idea chiave -- perche' distingue statico da dinamico anche col robot in moto:
la ESDF e' costruita nel frame FISSO del mondo ('odom'), NON nel frame del
robot. Una cella della griglia corrisponde sempre allo STESSO punto del mondo:
  - un albero fermo occupa sempre le stesse celle-mondo -> ESDF invariata
    -> dESDF ~ 0  -> STATICO.
  - un pedone che cammina occupa celle-mondo diverse a ogni frame -> dove
    arriva la distanza cala (dESDF < 0), da dove se ne va cresce (dESDF > 0)
    -> DINAMICO.
Il moto del robot e' gia' compensato perche' i punti LiDAR vengono trasformati
in 'odom' con la posa dalla localizzazione prima di entrare nella griglia.

Pipeline:
  /scan (LaserScan, frame base_scan)
    -> punti in frame odom (via TF)
    -> griglia di occupazione ROLLING ancorata al mondo
    -> ESDF con cv2.distanceTransform  (equivalente veloce all'algoritmo BFS
       del paper; distanza euclidea di ogni cella dall'ostacolo piu' vicino)
    -> dESDF = (ESDF_ora - ESDF_prima)/dt, allineata sulle stesse celle-mondo
    -> COSTO ANTICIPATORIO dove dESDF < -soglia  (ostacolo in avvicinamento),
       proporzionale a |dESDF| (piu' veloce l'avvicinamento, piu' alto il costo)
    -> /dynamic_cost (OccupancyGrid, frame odom)  ->  layer della local costmap

NOTA MODULARITA' (per l'estensione semantica futura):
  il gancio `self.semantic_gate(gx, gy)` restituisce ora sempre True (nessun
  filtro). Quando aggiungerai la semantica, qui filtrerai: "considera dinamica
  questa cella solo se la classe semantica in (gx,gy) e' pedone/bici/veicolo".
  Il resto del nodo NON cambia.
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid

import tf2_ros
from tf2_ros import TransformException


class DynamicESDFLayer(Node):
    def __init__(self):
        super().__init__('dynamic_esdf_layer')

        # ---------------- parametri ----------------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('output_topic', '/dynamic_cost')
        self.declare_parameter('world_frame', 'odom')     # frame FISSO (ancoraggio mappa)
        self.declare_parameter('robot_frame', 'base_link')

        self.declare_parameter('size_m', 6.0)             # lato finestra rolling [m]
        self.declare_parameter('resolution', 0.05)        # [m/cella]
        self.declare_parameter('rate_hz', 10.0)           # frequenza di calcolo

        # soglia anti-rumore di localizzazione: sotto questa |dESDF| = statico/rumore
        self.declare_parameter('desdf_thresh', 0.15)      # [m/s] di variazione distanza
        # guadagno costo: costo = gain * (|dESDF| - thresh), saturato a max_cost
        self.declare_parameter('cost_gain', 120.0)
        self.declare_parameter('max_cost', 100)
        # smoothing temporale della ESDF (riduce il jitter frame-a-frame)
        self.declare_parameter('esdf_ema', 0.4)           # 0=nessuno, ->1 molto liscio
        # dilatazione del costo (allarga un po' la zona anticipata, in celle)
        self.declare_parameter('cost_dilate_cells', 2)
        # tempo max senza scan valido prima di azzerare
        self.declare_parameter('scan_timeout', 0.5)

        gp = self.get_parameter
        self.scan_topic = gp('scan_topic').value
        self.out_topic = gp('output_topic').value
        self.world_frame = gp('world_frame').value
        self.robot_frame = gp('robot_frame').value
        self.size_m = float(gp('size_m').value)
        self.res = float(gp('resolution').value)
        self.rate_hz = float(gp('rate_hz').value)
        self.desdf_thresh = float(gp('desdf_thresh').value)
        self.cost_gain = float(gp('cost_gain').value)
        self.max_cost = int(gp('max_cost').value)
        self.esdf_ema = float(gp('esdf_ema').value)
        self.dilate_cells = int(gp('cost_dilate_cells').value)
        self.scan_timeout = float(gp('scan_timeout').value)

        self.n = int(round(self.size_m / self.res))       # celle per lato (griglia n x n)

        # ---------------- stato ----------------
        self.prev_esdf = None        # ESDF del frame precedente (float32, in metri)
        self.prev_seen = None        # maschera visibilita' del frame precedente
        self.prev_origin = None      # (ox, oy) origine-mondo della griglia precedente
        self.prev_stamp = None       # tempo del frame precedente [s]
        self.last_scan = None

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- I/O ----------------
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos)
        self.pub = self.create_publisher(OccupancyGrid, self.out_topic, 1)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.update)

        self.get_logger().info(
            f'dynamic_esdf_layer avviato | scan={self.scan_topic} -> {self.out_topic} | '
            f'griglia {self.n}x{self.n} @ {self.res} m in frame {self.world_frame} | '
            f'soglia dESDF={self.desdf_thresh} m/s')

    # -------------------------------------------------------------------------
    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg

    # -------------------------------------------------------------------------
    def semantic_gate(self, occ_grid):
        """GANCIO per estensione futura (LiDAR + semantica).
        Ora: nessun filtro (tutti gli ostacoli LiDAR passano).
        In futuro: restituira' una maschera booleana n x n che vale True solo
        dove la classe semantica proiettata e' dinamica (pedone/bici/veicolo),
        cosi' un albero con dESDF spurio (per jitter di localizzazione) viene
        comunque scartato. Il resto della pipeline resta identico."""
        return np.ones_like(occ_grid, dtype=bool)

    # -------------------------------------------------------------------------
    def update(self):
        if self.last_scan is None:
            return
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9

        # posa del robot nel frame mondo (per centrare la finestra rolling)
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.robot_frame, rclpy.time.Time())
        except TransformException:
            return
        rx = tf.transform.translation.x
        ry = tf.transform.translation.y

        # origine-mondo della griglia: finestra centrata sul robot, ALLINEATA
        # alla griglia globale (quantizzo alla risoluzione) cosi' le celle-mondo
        # coincidono tra frame consecutivi -> il confronto ESDF e' coerente.
        ox = np.floor((rx - self.size_m / 2.0) / self.res) * self.res
        oy = np.floor((ry - self.size_m / 2.0) / self.res) * self.res

        # --- costruisci la griglia di occupazione + maschera visibilita' ---
        occ, seen = self.scan_to_grid(self.last_scan, ox, oy, now_s)
        if occ is None:
            self.get_logger().warn(
                'scan_to_grid ha restituito None (scan vecchio o TF mancante) -> non pubblico',
                throttle_duration_sec=2.0)
            return

        # (gancio semantica: per ora passa tutto)
        occ = occ & self.semantic_gate(occ)

        # --- ESDF: distanza euclidea di ogni cella dall'ostacolo piu' vicino ---
        # distanceTransform lavora su celle "libere" (255) verso "occupate" (0):
        # passo l'inverso dell'occupazione. Risultato in celle -> * res = metri.
        free = np.where(occ, 0, 255).astype(np.uint8)
        esdf = cv2.distanceTransform(free, cv2.DIST_L2, 5).astype(np.float32) * self.res

        # smoothing temporale opzionale (riduce jitter)
        if self.esdf_ema > 0.0 and self.prev_esdf is not None \
                and self.prev_esdf.shape == esdf.shape:
            esdf_s = self.esdf_ema * self.prev_esdf + (1.0 - self.esdf_ema) * esdf
        else:
            esdf_s = esdf

        cost = self.compute_cost(esdf_s, seen, ox, oy, now_s)

        # aggiorna stato per il prossimo confronto
        self.prev_esdf = esdf_s
        self.prev_seen = seen
        self.prev_origin = (ox, oy)
        self.prev_stamp = now_s

        self.publish_cost(cost, ox, oy, now)

    # -------------------------------------------------------------------------
    def scan_to_grid(self, scan: LaserScan, ox, oy, now_s):
        """Proietta lo scan in una griglia di occupazione n x n nel frame mondo,
        e costruisce anche la MASCHERA DI VISIBILITA' (celle effettivamente viste
        dal LiDAR in questo frame). Ritorna (occ, seen).
          occ  = True dove un raggio ha COLPITO un ostacolo
          seen = True dove il LiDAR ha OSSERVATO la cella (lungo il raggio, fino
                 all'ostacolo). Le celle nel cono d'ombra o oltre il range restano
                 seen=False (IGNOTE) e vengono escluse dal calcolo del dESDF,
                 cosi' l'ombra che oscilla non genera piu' costo spurio."""
        # timeout: scan troppo vecchio -> niente
        scan_s = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        if now_s - scan_s > self.scan_timeout:
            self.get_logger().warn(
                f'scan troppo vecchio: now={now_s:.2f} scan={scan_s:.2f} '
                f'diff={now_s - scan_s:.2f}s > timeout={self.scan_timeout}s. '
                f'Probabile use_sim_time mancante sul nodo!',
                throttle_duration_sec=2.0)
            return None, None

        # TF dal frame dello scan al mondo
        try:
            t = self.tf_buffer.lookup_transform(
                self.world_frame, scan.header.frame_id, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(
                f'TF {self.world_frame}<-{scan.header.frame_id} non disponibile: {e}',
                throttle_duration_sec=2.0)
            return None, None

        # posa 2D del sensore nel mondo
        import math
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        sx = t.transform.translation.x
        sy = t.transform.translation.y

        occ = np.zeros((self.n, self.n), dtype=bool)
        seen = np.zeros((self.n, self.n), dtype=np.uint8)

        # indice-griglia del sensore (origine dei raggi)
        s_gx = (sx - ox) / self.res
        s_gy = (sy - oy) / self.res

        # raggi validi
        ang = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        r = np.asarray(scan.ranges, dtype=np.float32)
        good = np.isfinite(r) & (r >= scan.range_min) & (r <= scan.range_max)
        ang = ang[good]; r = r[good]
        if r.size == 0:
            return occ, seen.astype(bool)

        # punto colpito, in indici-griglia
        px = r * np.cos(ang); py = r * np.sin(ang)
        wx = sx + px * np.cos(yaw) - py * np.sin(yaw)
        wy = sy + px * np.sin(yaw) + py * np.cos(yaw)
        hx = (wx - ox) / self.res
        hy = (wy - oy) / self.res

        # --- raytracing: marca "seen" lungo ogni raggio dal sensore all'impatto ---
        # campiono ogni raggio a passi di ~1 cella; e' O(raggi * lunghezza), ma
        # con 720 raggi e ~6 m di raggio e' leggero (~decine di migliaia di punti).
        n_steps = int(np.ceil((self.size_m) / self.res))          # passi max
        tt = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)      # parametro lungo il raggio
        # per ogni raggio: punti = sensore + tt*(impatto - sensore)
        rx = s_gx + np.outer(tt, (hx - s_gx))     # shape (n_steps, n_rays)
        ry = s_gy + np.outer(tt, (hy - s_gy))
        rix = np.floor(rx).astype(np.int32).ravel()
        riy = np.floor(ry).astype(np.int32).ravel()
        vin = (rix >= 0) & (rix < self.n) & (riy >= 0) & (riy < self.n)
        seen[riy[vin], rix[vin]] = 1

        # celle di impatto = occupate (e ovviamente viste)
        hix = np.floor(hx).astype(np.int32)
        hiy = np.floor(hy).astype(np.int32)
        hin = (hix >= 0) & (hix < self.n) & (hiy >= 0) & (hiy < self.n)
        occ[hiy[hin], hix[hin]] = True
        seen[hiy[hin], hix[hin]] = 1

        return occ, seen.astype(bool)

    # -------------------------------------------------------------------------
    def compute_cost(self, esdf, seen, ox, oy, now_s):
        """Calcola il costo anticipatorio da dESDF = (esdf - prev_esdf)/dt.
        dESDF < 0  => la distanza dall'ostacolo cala => ostacolo in AVVICINAMENTO
        => costo proporzionale a |dESDF| (oltre la soglia anti-rumore).
        CRITICO: il costo si calcola SOLO nelle celle VISTE in entrambi i frame
        (ora e prima). Le celle nel cono d'ombra (non osservate) sono escluse,
        cosi' l'ombra che oscilla frame-a-frame non genera piu' costo spurio."""
        cost = np.zeros((self.n, self.n), dtype=np.uint8)

        if self.prev_esdf is None or self.prev_origin is None \
                or self.prev_stamp is None or self.prev_seen is None:
            return cost
        if self.prev_esdf.shape != esdf.shape:
            return cost

        dt = max(1e-2, now_s - self.prev_stamp)

        # allinea le due griglie sulle stesse celle-mondo: se l'origine e' shiftata
        # (il robot si e' mosso), traslo la mappa precedente di (shift) celle.
        pox, poy = self.prev_origin
        shift_x = int(round((pox - ox) / self.res))
        shift_y = int(round((poy - oy) / self.res))
        prev_aligned = self.shift_grid(self.prev_esdf, shift_x, shift_y, fill=np.nan)
        prev_seen_al = self.shift_grid(self.prev_seen.astype(np.float32),
                                       shift_x, shift_y, fill=0.0) > 0.5

        # MASCHERA: solo celle viste ORA e PRIMA (osservate in entrambi i frame)
        observed = seen & prev_seen_al & np.isfinite(prev_aligned)

        desdf = np.zeros_like(esdf)
        desdf[observed] = (esdf[observed] - prev_aligned[observed]) / dt

        # avvicinamento = dESDF negativo oltre soglia
        approaching = (-desdf) - self.desdf_thresh    # >0 dove |avvicinamento| supera soglia
        mag = np.clip(approaching, 0.0, None)         # solo avvicinamenti significativi
        mag[~observed] = 0.0                          # niente costo nelle celle non osservate
        c = np.clip(self.cost_gain * mag, 0, self.max_cost).astype(np.uint8)

        # dilata leggermente per dare margine (la zona "davanti" all'ostacolo)
        if self.dilate_cells > 0 and c.any():
            k = 2 * self.dilate_cells + 1
            c = cv2.dilate(c, np.ones((k, k), np.uint8))

        return c

    # -------------------------------------------------------------------------
    @staticmethod
    def shift_grid(grid, sx, sy, fill=0.0):
        """Trasla una griglia di (sx colonne, sy righe), riempiendo i bordi."""
        out = np.full_like(grid, fill)
        n = grid.shape[0]
        # sorgente e destinazione per righe (y) e colonne (x)
        def span(s):
            if s >= 0:
                return slice(s, n), slice(0, n - s)   # dst, src
            else:
                return slice(0, n + s), slice(-s, n)
        dy, syy = span(sy)
        dx, sxx = span(sx)
        out[dy, dx] = grid[syy, sxx]
        return out

    # -------------------------------------------------------------------------
    def publish_cost(self, cost, ox, oy, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp.to_msg()
        msg.header.frame_id = self.world_frame
        msg.info.resolution = self.res
        msg.info.width = self.n
        msg.info.height = self.n
        msg.info.origin.position.x = float(ox)
        msg.info.origin.position.y = float(oy)
        msg.info.origin.orientation.w = 1.0
        # OccupancyGrid vuole int8 0..100 (row-major, riga = y)
        data = cost.astype(np.int8).flatten(order='C')
        msg.data = data.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = DynamicESDFLayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()