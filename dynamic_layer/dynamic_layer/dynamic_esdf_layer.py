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
from nav_msgs.msg import OccupancyGrid, Odometry

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
        # NUOVO: il costo dinamico si applica solo entro questa distanza [m] da un
        # ostacolo OCCUPATO e visto. Elimina i coni/raggi spuri nel vuoto (ombre).
        self.declare_parameter('near_obstacle_m', 0.6)
        # NUOVO: gate sul movimento del robot. Se il robot ruota/trasla oltre queste
        # soglie, il pattern di occlusione cambia troppo -> il dESDF e' inaffidabile
        # -> attenuo/azzero il costo per evitare falsi positivi da "cambio vista".
        self.declare_parameter('gate_ang_vel', 0.4)   # [rad/s] soglia rotazione
        self.declare_parameter('gate_lin_vel', 0.5)   # [m/s] soglia traslazione
        self.declare_parameter('odom_topic', '/odom')
        # NUOVO (Opzione 2 - persistenza): la mappa di occupazione e' PERSISTENTE
        # nel frame mondo. Le celle viste-occupate salgono, le viste-libere calano;
        # le celle NON viste (occluse) MANTENGONO il valore -> niente salti da
        # ri-osservazione. occ_up/occ_down = quanto sale/scende la probabilita'
        # per frame; occ_thresh = soglia per considerare una cella "occupata".
        self.declare_parameter('occ_up', 0.7)      # incremento se vista-occupata
        self.declare_parameter('occ_down', 0.3)    # decremento se vista-libera
        self.declare_parameter('occ_thresh', 0.5)  # soglia occupazione
        # NUOVO: smoothing temporale del COSTO di uscita. Il dESDF (e quindi il
        # costo) e' rumoroso frame-a-frame (balla tra alto e basso -> "sfarfallio").
        # L'EMA sul costo finale lo stabilizza: cost_smooth = a*nuovo + (1-a)*vecchio.
        # Alto = piu' stabile ma meno reattivo. 0 = disattivato.
        self.declare_parameter('cost_ema', 0.5)

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
        self.near_obstacle_m = float(gp('near_obstacle_m').value)
        self.gate_ang_vel = float(gp('gate_ang_vel').value)
        self.gate_lin_vel = float(gp('gate_lin_vel').value)
        self.odom_topic = gp('odom_topic').value
        self.occ_up = float(gp('occ_up').value)
        self.occ_down = float(gp('occ_down').value)
        self.occ_thresh = float(gp('occ_thresh').value)
        self.cost_ema = float(gp('cost_ema').value)

        self.n = int(round(self.size_m / self.res))       # celle per lato (griglia n x n)

        # ---------------- stato ----------------
        self.prev_esdf = None        # ESDF del frame precedente (float32, in metri)
        self.prev_seen = None        # maschera visibilita' del frame precedente
        self.prev_origin = None      # (ox, oy) origine-mondo della griglia precedente
        self.prev_stamp = None       # tempo del frame precedente [s]
        self.last_scan = None
        self.robot_lin = 0.0         # |velocita' lineare| corrente del robot
        self.robot_ang = 0.0         # |velocita' angolare| corrente del robot
        # mappa di occupazione PERSISTENTE (probabilita' 0..1) e sua origine-mondo.
        # Vive tra i frame: le celle occluse mantengono il valore precedente.
        self.persist = None          # griglia float32 n x n, prob. occupazione
        self.persist_origin = None   # (ox, oy) della mappa persistente
        self.cost_prev = None        # costo del frame precedente (per EMA di stabilita')

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- I/O ----------------
        qos = QoSProfile(depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos)
        self.pub = self.create_publisher(OccupancyGrid, self.out_topic, 1)

        self.timer = self.create_timer(1.0 / self.rate_hz, self.update)

        self.get_logger().info(
            f'dynamic_esdf_layer avviato | scan={self.scan_topic} -> {self.out_topic} | '
            f'griglia {self.n}x{self.n} @ {self.res} m in frame {self.world_frame} | '
            f'soglia dESDF={self.desdf_thresh} m/s')

    # -------------------------------------------------------------------------
    def scan_cb(self, msg: LaserScan):
        self.last_scan = msg

    def odom_cb(self, msg: Odometry):
        # modulo delle velocita' correnti (per il gate sul movimento del robot)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.robot_lin = float((vx * vx + vy * vy) ** 0.5)
        self.robot_ang = abs(float(msg.twist.twist.angular.z))

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

        # --- MAPPA PERSISTENTE (Opzione 2): fondi l'osservazione corrente ---
        # Le celle viste-occupate salgono, viste-libere scendono, NON viste
        # mantengono il valore. Cosi' la ESDF si calcola su una mappa stabile e
        # non ci sono salti quando una cella occlusa torna visibile.
        occ_persist = self.update_persistent(occ, seen, ox, oy)

        # --- ESDF sulla mappa PERSISTENTE (non sul singolo frame) ---
        free = np.where(occ_persist, 0, 255).astype(np.uint8)
        esdf = cv2.distanceTransform(free, cv2.DIST_L2, 5).astype(np.float32) * self.res

        # smoothing temporale opzionale (riduce jitter)
        if self.esdf_ema > 0.0 and self.prev_esdf is not None \
                and self.prev_esdf.shape == esdf.shape:
            esdf_s = self.esdf_ema * self.prev_esdf + (1.0 - self.esdf_ema) * esdf
        else:
            esdf_s = esdf

        cost = self.compute_cost(esdf_s, seen, occ_persist, ox, oy, now_s)

        # aggiorna stato per il prossimo confronto
        self.prev_esdf = esdf_s
        self.prev_seen = seen
        self.prev_origin = (ox, oy)
        self.prev_stamp = now_s

        self.publish_cost(cost, ox, oy, now)

    # -------------------------------------------------------------------------
    def update_persistent(self, occ, seen, ox, oy):
        """Fonde l'osservazione corrente (occ, seen) nella mappa di occupazione
        PERSISTENTE, ancorata al mondo. Ritorna la maschera booleana di occupazione
        persistente (prob >= occ_thresh).

        Regola di aggiornamento (log-odds semplificato):
          - cella vista-OCCUPATA  -> prob sale   (verso 1)  [occ_up]
          - cella vista-LIBERA    -> prob scende  (verso 0)  [occ_down]
          - cella NON vista       -> INVARIATA (mantiene il valore)  <-- chiave!
        La persistenza delle celle non viste elimina i salti da ri-osservazione,
        che erano la causa dei coni/raggi spuri quando robot o pedone si muovono."""
        # prima volta: inizializza
        if self.persist is None or self.persist_origin is None:
            self.persist = np.zeros((self.n, self.n), dtype=np.float32)
            self.persist_origin = (ox, oy)

        # se l'origine e' cambiata (robot mosso), trasla la mappa persistente
        # sulle nuove celle-mondo (rolling) prima di fondere.
        pox, poy = self.persist_origin
        shift_x = int(round((pox - ox) / self.res))
        shift_y = int(round((poy - oy) / self.res))
        if shift_x != 0 or shift_y != 0:
            self.persist = self.shift_grid(self.persist, shift_x, shift_y, fill=0.0)
            self.persist_origin = (ox, oy)

        # aggiornamento probabilistico
        p = self.persist
        # celle viste-occupate: prob sale
        seen_occ = seen & occ
        p[seen_occ] = p[seen_occ] + self.occ_up * (1.0 - p[seen_occ])
        # celle viste-libere: prob scende
        seen_free = seen & (~occ)
        p[seen_free] = p[seen_free] - self.occ_down * p[seen_free]
        # celle NON viste: invariate (non tocco nulla) -> persistenza
        self.persist = p

        return p >= self.occ_thresh

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
    def compute_cost(self, esdf, seen, occ, ox, oy, now_s):
        """Calcola il costo anticipatorio da dESDF = (esdf - prev_esdf)/dt.
        dESDF < 0 => distanza cala => ostacolo in AVVICINAMENTO => costo ~ |dESDF|.

        Tre difese contro i falsi positivi (ombre, robot che ruota/trasla):
        1. solo celle VISTE in entrambi i frame (maschera seen);
        2. costo solo in una FASCIA vicino a ostacoli OCCUPATI e visti: nel vuoto
           (coni d'ombra, spazio libero) non c'e' ostacolo -> niente costo, anche
           se la ESDF li' oscilla;
        3. GATE sul moto del robot: se ruota/trasla troppo, il pattern di occlusione
           cambia troppo per fidarsi del dESDF -> attenuo/azzero il costo."""
        cost = np.zeros((self.n, self.n), dtype=np.uint8)

        if self.prev_esdf is None or self.prev_origin is None \
                or self.prev_stamp is None or self.prev_seen is None:
            return cost
        if self.prev_esdf.shape != esdf.shape:
            return cost

        # --- DIFESA 3: gate sul movimento del robot ---
        # se il robot ruota o trasla oltre soglia, in questo frame il cambio di
        # occlusione domina il dESDF -> non pubblico costo (evito falsi positivi).
        if self.robot_ang > self.gate_ang_vel or self.robot_lin > self.gate_lin_vel:
            self.get_logger().info(
                f'gate moto attivo (ang={self.robot_ang:.2f} lin={self.robot_lin:.2f}) '
                f'-> costo dinamico sospeso questo frame',
                throttle_duration_sec=2.0)
            return cost

        dt = max(1e-2, now_s - self.prev_stamp)

        # allinea le due griglie sulle stesse celle-mondo
        pox, poy = self.prev_origin
        shift_x = int(round((pox - ox) / self.res))
        shift_y = int(round((poy - oy) / self.res))
        prev_aligned = self.shift_grid(self.prev_esdf, shift_x, shift_y, fill=np.nan)
        prev_seen_al = self.shift_grid(self.prev_seen.astype(np.float32),
                                       shift_x, shift_y, fill=0.0) > 0.5

        # maschera: viste ORA e PRIMA
        observed = seen & prev_seen_al & np.isfinite(prev_aligned)

        # --- DIFESA 2: solo vicino a ostacoli OCCUPATI e visti ---
        # la ESDF corrente da' la distanza dall'ostacolo piu' vicino: tengo solo
        # le celle entro near_obstacle_m da un ostacolo (esdf piccola). Cosi' il
        # costo sta attorno agli ostacoli reali, non nel vuoto delle ombre.
        near_band = esdf <= self.near_obstacle_m
        valid = observed & near_band

        desdf = np.zeros_like(esdf)
        desdf[valid] = (esdf[valid] - prev_aligned[valid]) / dt

        # avvicinamento = dESDF negativo oltre soglia
        approaching = (-desdf) - self.desdf_thresh
        mag = np.clip(approaching, 0.0, None)
        mag[~valid] = 0.0
        c = np.clip(self.cost_gain * mag, 0, self.max_cost).astype(np.uint8)

        # dilata leggermente per dare margine (la zona "davanti" all'ostacolo)
        if self.dilate_cells > 0 and c.any():
            k = 2 * self.dilate_cells + 1
            c = cv2.dilate(c, np.ones((k, k), np.uint8))

        # --- SMOOTHING TEMPORALE DEL COSTO (stabilizza lo sfarfallio) ---
        # il dESDF e' rumoroso frame-a-frame; l'EMA sul costo fonde i frame forti
        # e deboli in un costo stabile che persiste, invece di ballare 100->0->100.
        cf = c.astype(np.float32)
        if self.cost_ema > 0.0 and self.cost_prev is not None:
            # allinea il costo precedente alle celle-mondo correnti (rolling)
            cprev_al = self.shift_grid(self.cost_prev, shift_x, shift_y, fill=0.0)
            cf = self.cost_ema * cf + (1.0 - self.cost_ema) * cprev_al
        self.cost_prev = cf.copy()
        c = np.clip(cf, 0, self.max_cost).astype(np.uint8)

        # --- DIAGNOSTICA: numeri grezzi per capire cosa succede davvero ---
        n_observed = int(observed.sum())
        n_near = int((observed & near_band).sum())
        desdf_approach = -desdf  # positivo = avvicinamento
        max_approach = float(desdf_approach[valid].max()) if valid.any() else 0.0
        n_over_thresh = int((desdf_approach[valid] > self.desdf_thresh).sum()) if valid.any() else 0
        n_cost = int((c > 0).sum())
        self.get_logger().info(
            f'[diag] osservate={n_observed} vicino_ost={n_near} | '
            f'max_avvicinamento={max_approach:.3f} m/s (soglia={self.desdf_thresh}) | '
            f'celle>soglia={n_over_thresh} celle_costo={n_cost} costo_max={int(c.max())}',
            throttle_duration_sec=1.0)

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