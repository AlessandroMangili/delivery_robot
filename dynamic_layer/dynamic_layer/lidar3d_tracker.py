#!/usr/bin/env python3
"""
lidar3d_tracker.py — Core del tracker AB3DMOT (Step 2 della pipeline LiDAR 3D).

Consuma le detection dello Step 1 e produce TRACCE con identità, posizione e
velocità. Riferimento: AB3DMOT (Kalman a velocità costante + associazione),
con le buone pratiche di SimpleTrack (CV + gating).

    /detections (PoseArray, frame sensore base_scan)
        -> TF base_scan -> odom   (frame fisso: nel frame del sensore che ruota
                                    l'ego-moto fingerebbe una velocità)
        -> predizione Kalman CV di tutte le tracce
        -> associazione (Hungarian) con gate di Mahalanobis + gate euclideo
        -> correzione Kalman delle tracce associate
        -> nascita/morte tracce (min_hits per confermare, max_age per il coasting)
        -> GATE DI VELOCITÀ: pubblica solo le tracce in moto (uccide alberi,
           panchine, pali: sono statici, v ~ 0)

Uscita (Step 2, per verifica):
    /tracks/markers   visualization_msgs/MarkerArray   (cilindro + freccia velocità
                                                         + id, colore stabile per id)

Lo Step 4 aggancerà TrackArray(x,y,vx,vy,pos_std,vel_std) su /dynamic_tracks_state
per il critic; qui restiamo sui marker per verificare tracking e gate.

Dipendenze: numpy, scipy (linear_sum_assignment).
"""
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

import tf2_ros
from tf2_ros import TransformException


# ===========================================================================
# NUCLEO DI TRACKING — indipendente da ROS (testabile in isolamento)
# ===========================================================================
class KalmanCV:
    """Filtro di Kalman a velocità costante. Stato [x, y, vx, vy].
    Misura la sola posizione [x, y]; la velocità emerge dalla sequenza."""

    def __init__(self, xy, accel_std, meas_std, init_pos_std, init_vel_std):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([init_pos_std**2, init_pos_std**2,
                          init_vel_std**2, init_vel_std**2]).astype(float)
        self.q = float(accel_std) ** 2      # densità rumore di accelerazione
        self.r = float(meas_std) ** 2       # varianza misura di posizione
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        # rumore di processo a accelerazione bianca discreta (modello CV canonico)
        Q = self.q * np.array([
            [dt**4 / 4, 0,          dt**3 / 2, 0],
            [0,          dt**4 / 4, 0,          dt**3 / 2],
            [dt**3 / 2, 0,          dt**2,      0],
            [0,          dt**3 / 2, 0,          dt**2]], dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.r * np.eye(2)
        return y, S

    def update(self, z):
        y, S = self.innovation(z)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self):
        return self.x[:2].copy()

    @property
    def vel(self):
        return self.x[2:].copy()

    @property
    def speed(self):
        return float(math.hypot(self.x[2], self.x[3]))

    def pos_std(self):
        return float(math.sqrt(0.5 * (self.P[0, 0] + self.P[1, 1])))

    def vel_std(self):
        return float(math.sqrt(0.5 * (self.P[2, 2] + self.P[3, 3])))


class Track:
    def __init__(self, tid, xy, params):
        self.id = tid
        self.kf = KalmanCV(xy, params['accel_std'], params['meas_std'],
                           params['init_pos_std'], params['init_vel_std'])
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.confirmed = False
        self.published_dynamic = False   # isteresi del gate di velocità


class MultiObjectTracker:
    """AB3DMOT: predizione CV -> associazione con gate -> update -> ciclo di vita."""

    def __init__(self, params):
        self.p = params
        self.tracks = []
        self.next_id = 0

    def step(self, dets, dt):
        """dets: ndarray (M,2) nel frame di tracking. Ritorna la lista di tracce."""
        dt = float(min(max(dt, 1e-3), 1.0))

        # 1. predizione di tutte le tracce
        for t in self.tracks:
            t.kf.predict(dt)
            t.age += 1
            t.time_since_update += 1

        # 2. associazione
        matches, un_tracks, un_dets = self._associate(dets)

        # 3. correzione delle tracce associate
        for ti, di in matches:
            self.tracks[ti].kf.update(dets[di])
            self.tracks[ti].hits += 1
            self.tracks[ti].time_since_update = 0
            if self.tracks[ti].hits >= self.p['min_hits']:
                self.tracks[ti].confirmed = True

        # 4. nascita: nuova traccia per ogni detection non associata
        for di in un_dets:
            self.tracks.append(Track(self.next_id, dets[di], self.p))
            self.next_id += 1

        # 5. morte: rimuovi le tracce senza update da troppo tempo
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.p['max_age']]

        # 6. gate di velocità con isteresi (statici -> non pubblicati)
        for t in self.tracks:
            s = t.kf.speed
            if t.published_dynamic:
                if s < self.p['min_speed_off']:
                    t.published_dynamic = False
            else:
                if s >= self.p['min_speed_on']:
                    t.published_dynamic = True

        return self.tracks

    def _associate(self, dets):
        from scipy.optimize import linear_sum_assignment
        T, M = len(self.tracks), len(dets)
        if T == 0 or M == 0:
            return [], list(range(T)), list(range(M))
        BIG = 1e6
        cost = np.full((T, M), BIG)
        gate_maha = self.p['gating_mahalanobis']
        gate_dist = self.p['max_assoc_dist']
        for i, t in enumerate(self.tracks):
            for j in range(M):
                y, S = t.kf.innovation(dets[j])
                if math.hypot(y[0], y[1]) > gate_dist:   # gate euclideo di sicurezza
                    continue
                try:
                    m2 = float(y @ np.linalg.inv(S) @ y)  # distanza di Mahalanobis^2
                except np.linalg.LinAlgError:
                    continue
                if m2 > gate_maha:                        # gate chi-quadro (2 gdl)
                    continue
                cost[i, j] = m2
        rows, cols = linear_sum_assignment(cost)
        matches = []
        un_tracks, un_dets = set(range(T)), set(range(M))
        for r, c in zip(rows, cols):
            if cost[r, c] >= BIG:
                continue
            matches.append((int(r), int(c)))
            un_tracks.discard(int(r))
            un_dets.discard(int(c))
        return matches, list(un_tracks), list(un_dets)


# ===========================================================================
# Utilità geometriche
# ===========================================================================
def quat_to_rot(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1 - (yy + zz), xy - wz,       xz + wy],
        [xy + wz,       1 - (xx + zz), yz - wx],
        [xz - wy,       yz + wx,       1 - (xx + yy)]])


PALETTE = [(0.20, 0.80, 0.95), (0.95, 0.60, 0.20), (0.60, 0.85, 0.25),
           (0.90, 0.35, 0.75), (0.95, 0.85, 0.25), (0.45, 0.55, 0.95),
           (0.30, 0.90, 0.55), (0.95, 0.45, 0.35)]


def color_for(tid, a=0.9):
    r, g, b = PALETTE[tid % len(PALETTE)]
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


# ===========================================================================
# Nodo ROS2
# ===========================================================================
class Lidar3DTracker(Node):
    def __init__(self):
        super().__init__('lidar3d_tracker')

        d = self.declare_parameter
        d('input_topic', '/detections')
        d('tracking_frame', 'odom')
        d('markers_topic', '/tracks/markers')

        # Kalman
        d('accel_std', 0.8)          # densità rumore di accelerazione [m/s^2]
                                     # (tarato: statico fermo resta sotto ~0.12 m/s,
                                     #  ben sotto il gate 0.20; vedi test)
        d('meas_std', 0.10)          # rumore misura di posizione [m]
        d('init_pos_std', 0.30)
        d('init_vel_std', 2.0)

        # associazione
        d('gating_mahalanobis', 9.21)  # chi^2, 2 gdl, ~99%
        d('max_assoc_dist', 1.5)       # gate euclideo di sicurezza [m]

        # ciclo di vita
        d('min_hits', 3)             # frame per confermare una traccia
        d('max_age', 10)             # frame di coasting prima di ucciderla (10 = ~1 s a 10 Hz)

        # gate di velocità (isteresi) — uccide gli statici
        d('min_speed_on', 0.20)
        d('min_speed_off', 0.15)

        # visualizzazione
        d('marker_z', 0.6)
        d('arrow_time_scale', 1.0)   # lunghezza freccia = v * questo [s]
        d('publish_static', False)   # se true, mostra in grigio le tracce ferme (debug)
        d('diag_period_frames', 20)

        g = lambda k: self.get_parameter(k).value
        self.input_topic = g('input_topic')
        self.tracking_frame = g('tracking_frame')
        self.markers_topic = g('markers_topic')
        self.marker_z = float(g('marker_z'))
        self.arrow_time_scale = float(g('arrow_time_scale'))
        self.publish_static = bool(g('publish_static'))
        self.diag_period_frames = int(g('diag_period_frames'))

        self.params = dict(
            accel_std=float(g('accel_std')),
            meas_std=float(g('meas_std')),
            init_pos_std=float(g('init_pos_std')),
            init_vel_std=float(g('init_vel_std')),
            gating_mahalanobis=float(g('gating_mahalanobis')),
            max_assoc_dist=float(g('max_assoc_dist')),
            min_hits=int(g('min_hits')),
            max_age=int(g('max_age')),
            min_speed_on=float(g('min_speed_on')),
            min_speed_off=float(g('min_speed_off')),
        )
        self.mot = MultiObjectTracker(self.params)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE)
        self.pub_mrk = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.sub = self.create_subscription(
            PoseArray, self.input_topic, self.on_dets, qos)

        self._frame = 0
        self.last_t = None
        self.get_logger().info(
            f"lidar3d_tracker avviato: {self.input_topic} -> tracce in "
            f"'{self.tracking_frame}'")

    # -----------------------------------------------------------------------
    def _lookup(self, src, stamp):
        """TF src -> tracking_frame; prova lo stamp esatto, poi l'ultima disponibile."""
        for tp in (rclpy.time.Time.from_msg(stamp), rclpy.time.Time()):
            try:
                return self.tf_buffer.lookup_transform(
                    self.tracking_frame, src, tp,
                    timeout=Duration(seconds=0.05))
            except TransformException:
                continue
        self.get_logger().warn(
            f"TF {self.tracking_frame} <- {src} non disponibile",
            throttle_duration_sec=2.0)
        return None

    @staticmethod
    def _apply_tf(tf, pts):
        q = tf.transform.rotation
        tr = tf.transform.translation
        R = quat_to_rot(q.x, q.y, q.z, q.w)
        return (R @ pts.T).T + np.array([tr.x, tr.y, tr.z])

    def on_dets(self, msg):
        self._frame += 1
        stamp = msg.header.stamp
        t_now = stamp.sec + stamp.nanosec * 1e-9
        dt = 0.1 if self.last_t is None else (t_now - self.last_t)
        self.last_t = t_now

        tf = self._lookup(msg.header.frame_id, stamp)
        if tf is None:
            return

        if len(msg.poses) == 0:
            dets = np.zeros((0, 2))
        else:
            pts = np.array([[p.position.x, p.position.y, p.position.z]
                            for p in msg.poses], dtype=float)
            dets = self._apply_tf(tf, pts)[:, :2]

        tracks = self.mot.step(dets, dt)
        self.publish(stamp, tracks)

        if self.diag_period_frames > 0 and self._frame % self.diag_period_frames == 0:
            n_conf = sum(1 for t in tracks if t.confirmed)
            n_dyn = sum(1 for t in tracks if t.confirmed and t.published_dynamic)
            self.get_logger().info(
                f"[trk] dets={len(dets)} tracce={len(tracks)} "
                f"confermate={n_conf} dinamiche={n_dyn}")

    # -----------------------------------------------------------------------
    def publish(self, stamp, tracks):
        ma = MarkerArray()
        for t in tracks:
            if not t.confirmed:
                continue
            dynamic = t.published_dynamic
            if (not dynamic) and (not self.publish_static):
                continue
            px, py = t.kf.pos
            vx, vy = t.kf.vel
            col = color_for(t.id) if dynamic else ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.4)
            base = t.id * 3

            cyl = self._mk(stamp, 'tracks', base, Marker.CYLINDER,
                           px, py, self.marker_z, col)
            cyl.scale.x = 0.4
            cyl.scale.y = 0.4
            cyl.scale.z = 1.0
            ma.markers.append(cyl)

            if dynamic:
                arr = self._mk(stamp, 'tracks', base + 1, Marker.ARROW,
                               0.0, 0.0, 0.0, col)
                arr.scale.x = 0.06   # diametro asta
                arr.scale.y = 0.12   # diametro testa
                arr.scale.z = 0.18   # lunghezza testa
                arr.points = [
                    Point(x=float(px), y=float(py), z=float(self.marker_z)),
                    Point(x=float(px + vx * self.arrow_time_scale),
                          y=float(py + vy * self.arrow_time_scale),
                          z=float(self.marker_z)),
                ]
                ma.markers.append(arr)

            txt = self._mk(stamp, 'tracks', base + 2, Marker.TEXT_VIEW_FACING,
                           px, py, self.marker_z + 1.0,
                           ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9))
            txt.scale.z = 0.3
            txt.text = f"{t.id}  v={t.kf.speed:.2f}"
            ma.markers.append(txt)

        self.pub_mrk.publish(ma)

    def _mk(self, stamp, ns, mid, mtype, x, y, z, color):
        mk = Marker()
        mk.header.frame_id = self.tracking_frame
        mk.header.stamp = stamp
        mk.ns = ns
        mk.id = int(mid)
        mk.type = mtype
        mk.action = Marker.ADD
        mk.pose.position.x = float(x)
        mk.pose.position.y = float(y)
        mk.pose.position.z = float(z)
        mk.pose.orientation.w = 1.0
        mk.color = color
        mk.frame_locked = True
        mk.lifetime = Duration(seconds=0.4).to_msg()
        return mk


def main(args=None):
    rclpy.init(args=args)
    node = Lidar3DTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
