#!/usr/bin/env python3
"""
dynamic_esdf_layer.py -- Rilevamento e anticipazione di ostacoli dinamici (SOLO LiDAR).

  LiDAR -> POSIZIONE, VELOCITA', 360 gradi: clustering (unisce le due gambe in UN
           oggetto), tracking del centroide, stima velocita', proiezione del cono
           di costo anticipatorio.
  Discriminazione statico/dinamico: PER MOVIMENTO (velocita' sopra 'min_speed').
  Un cluster fermo (muro, aiuola) non genera ne' nucleo ne' cono.
"""

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from dynamic_tracker_msgs.msg import Track, TrackArray

import tf2_ros


def transform_to_matrix(t):
    q = t.transform.rotation
    tr = t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


class DynamicTracker(Node):
    def __init__(self):
        super().__init__('dynamic_tracker')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('output_topic', '/dynamic_cost')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('size_m', 8.0)
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('rate_hz', 10.0)
        self.declare_parameter('max_range', 8.0)
        self.declare_parameter('cluster_eps_m', 0.5)
        self.declare_parameter('cluster_min_pts', 2)
        self.declare_parameter('cluster_max_size_m', 1.5)
        self.declare_parameter('gate_dist_m', 0.7)
        self.declare_parameter('vel_ema', 0.4)        # DEPRECATO (sostituito dal Kalman)
        self.declare_parameter('pos_ema', 0.5)        # DEPRECATO (sostituito dal Kalman)
        # ----- Filtro di Kalman (modello a velocita' costante) -----
        self.declare_parameter('kf_sigma_a', 0.6)     # [m/s^2] quanto il pedone puo' accelerare/svoltare
        self.declare_parameter('kf_sigma_z', 0.12)    # [m] quanto BALLA il centroide del cluster LiDAR
        self.declare_parameter('kf_v_init', 1.0)      # [m/s] incertezza iniziale sulla velocita'
        self.declare_parameter('kf_min_snr', 1.0)     # velocita' >= N * sigma_v per fidarsi della DIREZIONE
        self.declare_parameter('min_obs', 3)
        self.declare_parameter('track_timeout_s', 0.6)
        self.declare_parameter('min_speed', 0.30)
        # --- conferma dinamico con isteresi (anti-falsi-positivi statici) ---
        self.declare_parameter('promote_frames', 4)         # frame veloci consecutivi per promuovere
        self.declare_parameter('promote_min_disp_m', 0.25)  # [m] spostamento minimo sulla finestra
        self.declare_parameter('demote_frames', 8)          # frame lenti consecutivi per retrocedere
        self.declare_parameter('disp_window_s', 1.0)        # [s] finestra su cui misurare lo spostamento persistente
        self.declare_parameter('horizon_s', 3.0)
        self.declare_parameter('cone_min_length', 4.0)   # [m] lunghezza minima anche a bassa velocita'
        self.declare_parameter('cone_halfwidth_m', 0.4)
        self.declare_parameter('cone_spread', 1.5)
        self.declare_parameter('max_cost', 100)
        self.declare_parameter('nucleus_radius_m', 0.20)  # raggio NUCLEO (corpo pedone), piccolo: solo il pedone e' letale
        self.declare_parameter('cone_base_cost', 99)      # 99 -> costmap 251: quasi letale MA scavalcabile. 100 = letale/intrappola
        self.declare_parameter('cone_tip_cost', 90)       # costo alla PUNTA: alto = reagisce in anticipo
        self.declare_parameter('cone_lateral_falloff', 0.35)  # 0..1: quanto cala il costo dal centro ai bordi (per USCIRE)
        self.declare_parameter('mark_obstacle', True)
        self.declare_parameter('gate_ang_vel', 1.0)

        gp = self.get_parameter
        self.scan_topic = gp('scan_topic').value
        self.out_topic = gp('output_topic').value
        self.world_frame = gp('world_frame').value
        self.robot_frame = gp('robot_frame').value
        self.odom_topic = gp('odom_topic').value
        self.size_m = float(gp('size_m').value)
        self.res = float(gp('resolution').value)
        self.rate_hz = float(gp('rate_hz').value)
        self.max_range = float(gp('max_range').value)
        self.cluster_eps = float(gp('cluster_eps_m').value)
        self.cluster_min_pts = int(gp('cluster_min_pts').value)
        self.cluster_max_size = float(gp('cluster_max_size_m').value)
        self.gate_dist = float(gp('gate_dist_m').value)
        self.vel_ema = float(gp('vel_ema').value)     # DEPRECATO (non piu' usato)
        self.pos_ema = float(gp('pos_ema').value)     # DEPRECATO (non piu' usato)
        self.kf_sigma_a = float(gp('kf_sigma_a').value)
        self.kf_sigma_z = float(gp('kf_sigma_z').value)
        self.kf_v_init = float(gp('kf_v_init').value)
        self.kf_min_snr = float(gp('kf_min_snr').value)
        self.min_obs = int(gp('min_obs').value)
        self.track_timeout = float(gp('track_timeout_s').value)
        self.min_speed = float(gp('min_speed').value)
        self.promote_frames = int(gp('promote_frames').value)
        self.promote_min_disp = float(gp('promote_min_disp_m').value)
        self.demote_frames = int(gp('demote_frames').value)
        self.disp_window_s = float(gp('disp_window_s').value)
        self.horizon_s = float(gp('horizon_s').value)
        self.cone_min_length = float(gp('cone_min_length').value)
        self.cone_halfwidth = float(gp('cone_halfwidth_m').value)
        self.cone_spread = float(gp('cone_spread').value)
        self.max_cost = int(gp('max_cost').value)
        self.nucleus_radius = float(gp('nucleus_radius_m').value)
        self.cone_base_cost = int(gp('cone_base_cost').value)
        self.cone_tip_cost = int(gp('cone_tip_cost').value)
        self.cone_lateral_falloff = float(gp('cone_lateral_falloff').value)
        self.mark_obstacle = bool(gp('mark_obstacle').value)
        self.gate_ang_vel = float(gp('gate_ang_vel').value)

        self.n = int(round(self.size_m / self.res))

        self.last_scan = None
        self.tracks = []
        self.next_id = 0
        self.robot_ang = 0.0
        self.arrows = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos)

        # TRANSIENT_LOCAL: il layer della costmap chiede durability transient_local;
        # un publisher volatile e' incompatibile e ROS2 blocca il flusso.
        # transient_local qui e' compatibile con qualsiasi subscriber.
        qos_map = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(OccupancyGrid, self.out_topic, qos_map)
        self.markers_pub = self.create_publisher(MarkerArray, '/dynamic_tracks', 1)
        # Tracce Kalman per il critic spazio-temporale MPPI (topic dedicato e leggero).
        self.tracks_pub = self.create_publisher(TrackArray, '/dynamic_tracks_state', 10)
        self.timer = self.create_timer(1.0 / self.rate_hz, self.update)

        self.get_logger().info(
            f'dynamic_esdf_layer avviato (solo LiDAR) | '
            f'griglia {self.n}x{self.n}@{self.res}m frame={self.world_frame}')

    def scan_cb(self, msg): self.last_scan = msg
    def odom_cb(self, msg): self.robot_ang = abs(float(msg.twist.twist.angular.z))

    def lookup(self, frame, stamp=None):
        # stamp=None -> ultima TF disponibile (per origine griglia).
        # stamp=header.stamp -> TF all'ISTANTE dell'osservazione: indispensabile
        # per lo scan, altrimenti in rotazione i punti fermi atterrano ruotati
        # frame dopo frame e sembrano muoversi (velocita' spuria -> falso dinamico).
        try:
            t = rclpy.time.Time() if stamp is None else rclpy.time.Time.from_msg(stamp)
            return self.tf_buffer.lookup_transform(self.world_frame, frame, t)
        except Exception:
            return None

    def update(self):
        if self.last_scan is None:
            return
        now_s = self.get_clock().now().nanoseconds * 1e-9
        T_base = self.lookup(self.robot_frame)
        if T_base is None:
            return
        rx = T_base.transform.translation.x
        ry = T_base.transform.translation.y
        ox = np.floor((rx - self.size_m / 2.0) / self.res) * self.res
        oy = np.floor((ry - self.size_m / 2.0) / self.res) * self.res

        pts = self.scan_points_world(self.last_scan, rx, ry)
        if pts is None:
            return
        clusters = self.cluster_points(pts)
        dets = [c['centroid'] for c in clusters]
        self.track(dets, now_s)

        cost = self.build_cost(ox, oy)
        self.publish_cost(cost, ox, oy)
        self.publish_arrows()
        self.publish_tracks()

    def scan_points_world(self, scan, rx, ry):
        # TF al timestamp dello scan (non l'ultima): allinea i punti al momento
        # in cui il LiDAR li ha misurati. Se non e' ancora nel buffer, T=None e
        # si salta questo frame (il tracker gira a 10Hz, saltarne uno e' innocuo).
        T = self.lookup(scan.header.frame_id, stamp=scan.header.stamp)
        if T is None:
            return None
        M = transform_to_matrix(T)
        sx, sy = M[0, 3], M[1, 3]
        yaw = np.arctan2(M[1, 0], M[0, 0])
        ang = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
        r = np.asarray(scan.ranges, dtype=np.float32)
        good = np.isfinite(r) & (r >= scan.range_min) & (r <= scan.range_max)
        ang = ang[good]; r = r[good]
        if r.size == 0:
            return np.empty((0, 2))
        px = r * np.cos(ang); py = r * np.sin(ang)
        wx = sx + px * np.cos(yaw) - py * np.sin(yaw)
        wy = sy + px * np.sin(yaw) + py * np.cos(yaw)
        d = np.hypot(wx - rx, wy - ry)
        keep = d <= self.max_range
        return np.stack([wx[keep], wy[keep]], axis=1)

    def cluster_points(self, pts):
        clusters = []
        if len(pts) == 0:
            return clusters
        eps = self.cluster_eps
        cell = {}
        keys = np.floor(pts / eps).astype(int)
        for i, (kx, ky) in enumerate(keys):
            cell.setdefault((kx, ky), []).append(i)
        parent = list(range(len(pts)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        eps2 = eps * eps
        for i, (kx, ky) in enumerate(keys):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in cell.get((kx + dx, ky + dy), []):
                        if j <= i:
                            continue
                        if np.sum((pts[i] - pts[j]) ** 2) <= eps2:
                            union(i, j)
        groups = {}
        for i in range(len(pts)):
            groups.setdefault(find(i), []).append(i)
        for idxs in groups.values():
            if len(idxs) < self.cluster_min_pts:
                continue
            cpts = pts[idxs]
            size = np.max(np.ptp(cpts, axis=0)) if len(cpts) > 1 else 0.0
            if size > self.cluster_max_size:
                continue
            centroid = cpts.mean(axis=0)
            clusters.append({'centroid': (float(centroid[0]), float(centroid[1])),
                             'size': float(size)})
        return clusters

    # ================== FILTRO DI KALMAN (modello a VELOCITA' COSTANTE) ==================
    # PERCHE': prima la velocita' era la derivata frame-a-frame del centroide del cluster
    # LiDAR, ripulita con due EMA in cascata. Ma il centroide di un pedone BALLA (le gambe si
    # muovono, il cluster cambia forma, i punti visibili cambiano): derivare quel segnale
    # AMPLIFICA il rumore, e le due EMA lo attenuavano aggiungendo ritardo. Risultato: la
    # DIREZIONE del cono oscillava (e con un cono di 4 m, pochi gradi alla base = decine di cm
    # alla punta).
    # Il Kalman e' il modo principiato di fare la stessa cosa: sa che un pedone si muove a
    # velocita' quasi costante (modello CV), quindi distingue il MOTO VERO dal RUMORE di
    # misura invece di derivare tutto ciecamente. In piu' regala due cose:
    #   1) COASTING: predice anche quando la detection manca (occlusione breve) -> il track
    #      non muore e non riparte da velocita' zero, il cono non si spegne;
    #   2) COVARIANZA: sappiamo QUANTO ci fidiamo della velocita' -> serve ora per non
    #      disegnare coni su stime insignificanti, e servira' per il critic spazio-temporale
    #      (l'incertezza che cresce con l'orizzonte giustifica l'allargamento del cono).
    #
    # Stato: s = [x, y, vx, vy]   Misura: z = [x, y] (solo posizione, dal centroide)
    def _kf_predict(self, tr, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)
        # Q = rumore di processo: il pedone puo' accelerare/svoltare (accelerazione casuale).
        # sigma_a alto = filtro piu' reattivo ma piu' nervoso; basso = piu' liscio ma in ritardo.
        q = self.kf_sigma_a ** 2
        dt2 = dt * dt; dt3 = dt2 * dt; dt4 = dt3 * dt
        Q = q * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                          [0, dt4 / 4, 0, dt3 / 2],
                          [dt3 / 2, 0, dt2, 0],
                          [0, dt3 / 2, 0, dt2]], dtype=float)
        tr['s'] = F @ tr['s']
        tr['P'] = F @ tr['P'] @ F.T + Q

    def _kf_update(self, tr, z):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        R = (self.kf_sigma_z ** 2) * np.eye(2)   # quanto BALLA il centroide del cluster
        y = z - H @ tr['s']                      # innovazione
        S = H @ tr['P'] @ H.T + R
        K = tr['P'] @ H.T @ np.linalg.inv(S)     # guadagno di Kalman
        tr['s'] = tr['s'] + K @ y
        tr['P'] = (np.eye(4) - K @ H) @ tr['P']

    def track(self, dets, now_s):
        # 1) PREDIZIONE: porta ogni traccia all'istante corrente (anche senza detection).
        for tr in self.tracks:
            dt = max(1e-3, min(now_s - tr['t'], 1.0))
            self._kf_predict(tr, dt)
            tr['t'] = now_s

        # 2) ASSOCIAZIONE: il gate usa la posizione PREDETTA, non l'ultima vista.
        #    Per un oggetto in movimento e' molto piu' corretto: cerchiamo la detection
        #    dove il pedone DOVREBBE essere ora, non dove era prima.
        used = set()
        for tr in self.tracks:
            best = None; bd = self.gate_dist
            for j, (dx, dy) in enumerate(dets):
                if j in used:
                    continue
                d = np.hypot(dx - tr['s'][0], dy - tr['s'][1])
                if d < bd:
                    bd = d; best = j
            if best is not None:
                self._kf_update(tr, np.array(dets[best], dtype=float))
                tr['t_seen'] = now_s
                tr['n'] = min(tr['n'] + 1, 999)
                used.add(best)
            # se non associata: nessun update -> la traccia CONTINUA in coasting sulla
            # sola predizione, e P cresce (il filtro sa di essere sempre meno sicuro).

        # 3) NUOVE TRACCE: velocita' ignota -> P grande sulla velocita' (kf_v_init).
        for j, (dx, dy) in enumerate(dets):
            if j in used:
                continue
            s = np.array([dx, dy, 0.0, 0.0], dtype=float)
            P = np.diag([self.kf_sigma_z ** 2, self.kf_sigma_z ** 2,
                         self.kf_v_init ** 2, self.kf_v_init ** 2]).astype(float)
            self.tracks.append(dict(id=self.next_id, s=s, P=P,
                                    n=1, t=now_s, t_seen=now_s,
                                    # --- stato di conferma dinamico (isteresi) ---
                                    is_dynamic=False,     # confermato dinamico?
                                    mov_count=0,          # frame veloci consecutivi
                                    still_count=0,        # frame lenti consecutivi
                                    pos_hist=[(dx, dy, now_s)]))  # storia (x,y,t) per lo spostamento su finestra
            self.next_id += 1

        # 4) SCADENZA: sul tempo dell'ultima MISURA (non dell'ultima predizione),
        #    cosi' il coasting dura al massimo track_timeout.
        self.tracks = [tr for tr in self.tracks
                       if now_s - tr['t_seen'] <= self.track_timeout]

        # 5) Espone lo stato con le stesse chiavi di prima (il resto del nodo non cambia)
        #    + 'sv' = deviazione standard della velocita' stimata (dalla covarianza).
        for tr in self.tracks:
            tr['x'] = float(tr['s'][0]); tr['y'] = float(tr['s'][1])
            tr['vx'] = float(tr['s'][2]); tr['vy'] = float(tr['s'][3])
            tr['sv'] = float(np.sqrt(max(tr['P'][2, 2] + tr['P'][3, 3], 1e-9)))
            tr['sp'] = float(np.sqrt(max(tr['P'][0, 0] + tr['P'][1, 1], 1e-9)))

            # --- CONFERMA DINAMICO con isteresi + spostamento su FINESTRA ---
            # Il discriminante NON e' la direzione istantanea (un pedone a zigzag
            # per schivare la gente la cambia continuamente, ma e' comunque
            # dinamico!). E' lo SPOSTAMENTO PERSISTENTE nel tempo:
            #   - un pedone, anche a zigzag, in ~1s si allontana molto da dov'era;
            #   - uno statico rumoroso oscilla attorno a un punto: spostamento ~0.
            # Cosi' teniamo i pedoni imprevedibili e blocchiamo il rumore.
            # Regola di sicurezza: nel dubbio, dinamico (un falso negativo -pedone
            # non visto- e' molto peggio di un falso positivo -statico evitato-).
            speed = np.hypot(tr['vx'], tr['vy'])
            fast_enough = (speed >= self.min_speed) and \
                          (self.kf_min_snr <= 0.0 or speed >= self.kf_min_snr * tr['sv'])

            # aggiorna la storia delle posizioni e scarta quelle piu' vecchie
            # della finestra (tengo un po' di margine oltre disp_window_s)
            tr['pos_hist'].append((tr['x'], tr['y'], now_s))
            tmin = now_s - max(self.disp_window_s * 1.5, self.disp_window_s + 0.2)
            tr['pos_hist'] = [p for p in tr['pos_hist'] if p[2] >= tmin]

            # spostamento sulla finestra: distanza tra la posizione di ~window_s fa
            # e quella attuale. Cerco il campione piu' vicino a (now - window).
            t_target = now_s - self.disp_window_s
            ref = min(tr['pos_hist'], key=lambda p: abs(p[2] - t_target))
            win_disp = np.hypot(tr['x'] - ref[0], tr['y'] - ref[1])
            # e' valida solo se la finestra e' davvero coperta (track abbastanza vecchio)
            window_covered = (now_s - tr['pos_hist'][0][2]) >= (self.disp_window_s * 0.8)

            if fast_enough:
                tr['mov_count'] += 1
                tr['still_count'] = 0
            else:
                tr['still_count'] += 1
                tr['mov_count'] = 0

            # PROMOZIONE: serve TUTTO insieme:
            #   - visto per abbastanza frame (min_obs): Kalman stabilizzato;
            #   - abbastanza frame veloci (mov_count);
            #   - la finestra temporale e' coperta (track abbastanza vecchio);
            #   - spostamento PERSISTENTE sulla finestra (win_disp): il pedone,
            #     anche a zigzag, si e' allontanato; lo statico rumoroso no.
            if (not tr['is_dynamic']) and tr['n'] >= self.min_obs \
                    and tr['mov_count'] >= self.promote_frames \
                    and window_covered and win_disp >= self.promote_min_disp:
                tr['is_dynamic'] = True

            # RETROCESSIONE: fermo per abbastanza frame consecutivi (isteresi:
            # non retrocede al primo frame lento, cosi' il cono non lampeggia)
            if tr['is_dynamic'] and tr['still_count'] >= self.demote_frames:
                tr['is_dynamic'] = False

    def build_cost(self, ox, oy):
        cost = np.zeros((self.n, self.n), dtype=np.uint8)
        self.arrows = []
        spinning = self.robot_ang > self.gate_ang_vel
        for tr in self.tracks:
            if tr['n'] < self.min_obs:
                continue
            speed = np.hypot(tr['vx'], tr['vy'])
            # DISCRIMINAZIONE PER MOVIMENTO CONFERMATO: non basta la velocita'
            # istantanea (uno statico rumoroso a volte la supera). Serve la
            # conferma con isteresi accumulata in track(): un track e' dinamico
            # solo dopo essersi mosso in modo coerente e con spostamento reale.
            # Muri e aiuole (che ballano ma non si spostano) non passano mai.
            if not tr.get('is_dynamic', False):
                continue
            if self.mark_obstacle:
                self._stamp(cost, tr['x'], tr['y'], ox, oy, self.nucleus_radius, self.max_cost)
            # in rotazione rapida la DIREZIONE della velocita' e' inaffidabile:
            # marca il nucleo (posizione attuale, valida) ma non proiettare il cono.
            if spinning:
                continue
            # GATE DI SIGNIFICATIVITA' (dalla covarianza del Kalman): se la velocita' non
            # supera N volte la propria incertezza, la sua DIREZIONE non e' informazione, e'
            # rumore. Marca il nucleo (dove il pedone e', che sappiamo) ma NON proiettare un
            # cono in una direzione che non conosciamo. Questo e' cio' che prima mancava:
            # con la doppia EMA non c'era modo di sapere QUANTO fidarsi della direzione.
            if self.kf_min_snr > 0.0 and speed < self.kf_min_snr * tr.get('sv', 0.0):
                continue
            self._paint_cone(cost, tr, ox, oy, speed)
            self.arrows.append((tr['x'], tr['y'], tr['vx'], tr['vy']))
        return cost

    def _stamp(self, cost, wx, wy, ox, oy, radius_m, val):
        cxi = int((wx - ox) / self.res); cyi = int((wy - oy) / self.res)
        rr = int(radius_m / self.res)
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if dx*dx + dy*dy <= rr*rr:
                    gx, gy = cxi + dx, cyi + dy
                    if 0 <= gx < self.n and 0 <= gy < self.n and cost[gy, gx] < val:
                        cost[gy, gx] = val

    def _paint_cone(self, cost, tr, ox, oy, speed):
        ux, uy = tr['vx'] / speed, tr['vy'] / speed
        # lunghezza minima: anche un dinamico lento lascia un cono usabile,
        # abbastanza lungo da far reagire il controller PRIMA del nucleo.
        length = min(max(self.cone_min_length, speed * self.horizon_s), self.size_m)
        n_steps = max(1, int(length / self.res))
        px = (tr['x'] - ox) / self.res
        py = (tr['y'] - oy) / self.res
        half0 = self.cone_halfwidth / self.res
        nx, ny = -uy, ux
        for s in range(n_steps):
            frac = s / max(1, n_steps - 1)
            cx = px + ux * s
            cy = py + uy * s
            half = half0 * (1.0 + self.cone_spread * frac)
            # costo longitudinale: base (sul pedone) -> punta. Tenuto <100 apposta:
            # nella costmap resta SOTTO il letale, quindi il controller lo evita
            # fortissimo ma se ci finisce dentro trova ancora traiettorie valide per uscire.
            base_val = self.cone_base_cost - (self.cone_base_cost - self.cone_tip_cost) * frac
            wv = -half
            while wv <= half + 1e-6:
                gx = int(round(cx + nx * wv))
                gy = int(round(cy + ny * wv))
                # gradiente LATERALE: massimo al centro, cala verso i bordi -> chi e'
                # dentro il cono ha un verso "in discesa" per uscire di lato al piu' presto.
                lat = abs(wv) / half if half > 1e-6 else 0.0
                val = int(base_val * (1.0 - self.cone_lateral_falloff * lat))
                if 0 <= gx < self.n and 0 <= gy < self.n and cost[gy, gx] < val:
                    cost[gy, gx] = val
                wv += 1.0

    def publish_tracks(self):
        """Pubblica le tracce Kalman (posizione, velocita', incertezza) per il critic
        spazio-temporale MPPI. E' un topic dedicato e leggero, separato dal cono
        (OccupancyGrid) e dalle frecce (MarkerArray di RViz): il critic ha bisogno dei
        NUMERI, non di una griglia o di un marker."""
        msg = TrackArray()
        msg.header.frame_id = self.world_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for tr in self.tracks:
            if not tr.get('is_dynamic', False):
                continue   # solo dinamici confermati vanno al critic (niente statici)
            t = Track()
            t.id = int(tr['id'])
            t.x = float(tr['x']); t.y = float(tr['y'])
            t.vx = float(tr['vx']); t.vy = float(tr['vy'])
            t.pos_std = float(tr.get('sp', 0.0))
            t.vel_std = float(tr.get('sv', 0.0))
            msg.tracks.append(t)
        self.tracks_pub.publish(msg)

    def publish_arrows(self):
        arr = MarkerArray()
        clear = Marker(); clear.header.frame_id = self.world_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for k, (wx, wy, vx, vy) in enumerate(self.arrows):
            m = Marker()
            m.header.frame_id = self.world_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'dyn'; m.id = k
            m.type = Marker.ARROW; m.action = Marker.ADD
            p0 = Point(x=float(wx), y=float(wy), z=0.1)
            p1 = Point(x=float(wx + vx), y=float(wy + vy), z=0.1)
            m.points = [p0, p1]
            m.scale.x = 0.08; m.scale.y = 0.18; m.scale.z = 0.0
            m.color.r = 0.1; m.color.g = 1.0; m.color.b = 0.1; m.color.a = 1.0
            arr.markers.append(m)
        self.markers_pub.publish(arr)

    def publish_cost(self, cost, ox, oy):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame
        msg.info.resolution = self.res
        msg.info.width = self.n
        msg.info.height = self.n
        msg.info.origin.position.x = float(ox)
        msg.info.origin.position.y = float(oy)
        msg.info.origin.orientation.w = 1.0
        msg.data = cost.astype(np.int8).flatten(order='C').tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = DynamicTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()