#!/usr/bin/env python3

import math
import numpy as np

class KalmanCV:
    """Filtro di Kalman a velocita' costante. Stato [x, y, vx, vy].
    Misura la sola posizione [x, y]; la velocita' emerge dalla sequenza."""

    def __init__(self, xy, accel_std, meas_std, init_pos_std, init_vel_std):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([init_pos_std**2, init_pos_std**2,
                          init_vel_std**2, init_vel_std**2]).astype(float)
        self.q = float(accel_std) ** 2
        self.r = float(meas_std) ** 2
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
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
        self.published_dynamic = False   # confermato dinamico (life-cycle)?
        self.mov_count = 0
        self.still_count = 0
        # --- stato semantico (dal ramo YOLO asincrono; STICKY) ---
        self.is_person = False           # confermata persona da una bbox YOLO?
        self.person_hits = 0             # associazioni positive accumulate
        self.person_score = 0.0          # ultima confidenza YOLO associata


class MultiObjectTracker:
    """AB3DMOT: predizione CV -> associazione con gate -> update -> ciclo di vita.
    NUCLEO INVARIATO rispetto alla versione validata."""

    def __init__(self, params):
        self.p = params
        self.tracks = []
        self.next_id = 0

    def step(self, dets, dt):
        dt = float(min(max(dt, 1e-3), 1.0))

        for t in self.tracks:
            t.kf.predict(dt)
            t.age += 1
            t.time_since_update += 1

        matches, un_tracks, un_dets = self._associate(dets)

        for ti, di in matches:
            self.tracks[ti].kf.update(dets[di])
            self.tracks[ti].hits += 1
            self.tracks[ti].time_since_update = 0
            if self.tracks[ti].hits >= self.p['min_hits']:
                self.tracks[ti].confirmed = True

        for di in un_dets:
            self.tracks.append(Track(self.next_id, dets[di], self.p))
            self.next_id += 1

        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.p['max_age']]

        for t in self.tracks:
            speed = t.kf.speed
            sv = t.kf.vel_std()
            moving = ((speed >= self.p['min_speed']) and
                      (self.p['snr_min'] <= 0.0 or speed >= self.p['snr_min'] * sv))
            if moving:
                t.mov_count += 1
                t.still_count = 0
            else:
                t.still_count += 1
                t.mov_count = 0
            if ((not t.published_dynamic) and t.confirmed and
                    t.mov_count >= self.p['promote_frames']):
                t.published_dynamic = True
            if t.published_dynamic and t.still_count >= self.p['demote_frames']:
                t.published_dynamic = False

        return self.tracks

    def _associate(self, dets):
        """Associazione GREEDY (SimpleTrack) + recupero two-stage. INVARIATO."""
        T, M = len(self.tracks), len(dets)
        if T == 0 or M == 0:
            return [], list(range(T)), list(range(M))
        gate_maha = self.p['gating_mahalanobis']
        gate_dist = self.p['max_assoc_dist']
        pairs = []
        for i, t in enumerate(self.tracks):
            for j in range(M):
                y, S = t.kf.innovation(dets[j])
                if math.hypot(y[0], y[1]) > gate_dist:
                    continue
                try:
                    m2 = float(y @ np.linalg.inv(S) @ y)
                except np.linalg.LinAlgError:
                    continue
                if m2 > gate_maha:
                    continue
                pairs.append((m2, i, j))
        pairs.sort(key=lambda p: p[0])
        matched_t, matched_d, matches = set(), set(), []
        for _cost, i, j in pairs:
            if i in matched_t or j in matched_d:
                continue
            matches.append((i, j))
            matched_t.add(i)
            matched_d.add(j)
        un_tracks = [i for i in range(T) if i not in matched_t]
        un_dets = [j for j in range(M) if j not in matched_d]

        rec_dist = self.p['recovery_dist']
        self._n_recovered = 0
        if rec_dist > 0.0 and un_tracks and un_dets:
            rec = []
            for i in un_tracks:
                for j in un_dets:
                    d = math.hypot(*(self.tracks[i].kf.pos - dets[j]))
                    if d <= rec_dist:
                        rec.append((d, i, j))
            rec.sort(key=lambda p: p[0])
            rt, rd = set(), set()
            for _d, i, j in rec:
                if i in rt or j in rd:
                    continue
                matches.append((i, j))
                rt.add(i)
                rd.add(j)
                self._n_recovered += 1
            un_tracks = [i for i in un_tracks if i not in rt]
            un_dets = [j for j in un_dets if j not in rd]

        return matches, un_tracks, un_dets


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


def project_to_pixel(K, p_cam, w, h):
    """Punto 3D (frame ottico camera) -> pixel (u,v). None se dietro/fuori.
    Funzione pura -> testabile."""
    Xc, Yc, Zc = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if Zc <= 0.05:
        return None
    u = int(round(K[0, 0] * Xc / Zc + K[0, 2]))
    v = int(round(K[1, 1] * Yc / Zc + K[1, 2]))
    if 0 <= u < w and 0 <= v < h:
        return (u, v)
    return None


def match_track_to_boxes(px, boxes, margin):
    """Gate persona positivo: se il pixel (u,v) della traccia cade dentro una
    bbox YOLO (espansa di 'margin' per assorbire la latenza), ritorna la
    confidenza della bbox piu' vicina (per centro); altrimenti None.
    boxes: lista di (u0, v0, u1, v1, score). Funzione pura -> testabile."""
    best = None
    u, v = px
    for (u0, v0, u1, v1, score) in boxes:
        if (u0 - margin) <= u <= (u1 + margin) and (v0 - margin) <= v <= (v1 + margin):
            cxc, cyc = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            d = math.hypot(u - cxc, v - cyc)
            if best is None or d < best[0]:
                best = (d, float(score))
    return None if best is None else best[1]


# @@@ROS_BOUNDARY@@@  (i test isolati eseguono solo cio' che sta sopra questa riga)

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray

from dynamic_tracker_msgs.msg import Track as TrackMsg, TrackArray

import tf2_ros
from tf2_ros import TransformException


PALETTE = [(0.20, 0.80, 0.95), (0.95, 0.60, 0.20), (0.60, 0.85, 0.25),
           (0.90, 0.35, 0.75), (0.95, 0.85, 0.25), (0.45, 0.55, 0.95),
           (0.30, 0.90, 0.55), (0.95, 0.45, 0.35)]


def color_for(tid, a=0.9):
    r, g, b = PALETTE[tid % len(PALETTE)]
    return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))


def bbox_to_corners(det):
    """vision_msgs/Detection2D -> (u0, v0, u1, v1, score). Robusto alle due
    varianti di BoundingBox2D.center (vision_msgs/Pose2D .position.x o
    geometry_msgs/Pose2D .x). score dal primo results[] se presente."""
    b = det.bbox
    c = b.center
    if hasattr(c, 'position'):
        cx, cy = float(c.position.x), float(c.position.y)
    else:
        cx, cy = float(c.x), float(c.y)
    hx, hy = 0.5 * float(b.size_x), 0.5 * float(b.size_y)
    score = 1.0
    try:
        if det.results:
            r0 = det.results[0]
            score = float(getattr(r0, 'hypothesis', r0).score)
    except Exception:
        pass
    return (cx - hx, cy - hy, cx + hx, cy + hy, score)


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
        d('tracks_topic', '/dynamic_tracks_state')

        # Kalman
        d('accel_std', 0.8)
        d('meas_std', 0.10)
        d('init_pos_std', 0.30)
        d('init_vel_std', 2.0)

        # associazione
        d('gating_mahalanobis', 9.21)
        d('max_assoc_dist', 1.5)
        d('recovery_dist', 1.0)

        # ciclo di vita
        d('min_hits', 3)
        d('max_age', 10)
        d('coast_publish_frames', 3)
        d('min_speed', 0.30)
        d('snr_min', 1.0)
        d('promote_frames', 4)
        d('demote_frames', 8)

        # --- ramo semantico ASINCRONO: associazione 2D-3D su bbox YOLO ---
        d('use_person_gate', True)
        d('yolo_topic', '/yolo/detections')       # vision_msgs/Detection2DArray
        d('camera_info_topic', '/camera/camera_info')
        d('camera_frame', 'camera_rgb_optical_frame')
        d('probe_height', 0.9)                     # [m] quota traccia proiettata (mezzo busto)
        d('bbox_margin', 12)                       # [px] espansione bbox: assorbe latenza YOLO
        d('person_confirm_hits', 2)                # associazioni positive per confermare persona

        # visualizzazione
        d('marker_z', 0.6)
        d('arrow_time_scale', 1.0)
        d('publish_static', False)
        d('diag_period_frames', 20)

        g = lambda k: self.get_parameter(k).value
        self.input_topic = g('input_topic')
        self.tracking_frame = g('tracking_frame')
        self.markers_topic = g('markers_topic')
        self.tracks_topic = g('tracks_topic')
        self.marker_z = float(g('marker_z'))
        self.arrow_time_scale = float(g('arrow_time_scale'))
        self.publish_static = bool(g('publish_static'))
        self.diag_period_frames = int(g('diag_period_frames'))
        self.coast_publish_frames = int(g('coast_publish_frames'))

        self.use_person_gate = bool(g('use_person_gate'))
        self.camera_frame = g('camera_frame')
        self.probe_height = float(g('probe_height'))
        self.bbox_margin = int(g('bbox_margin'))
        self.person_confirm_hits = int(g('person_confirm_hits'))

        self.params = dict(
            accel_std=float(g('accel_std')),
            meas_std=float(g('meas_std')),
            init_pos_std=float(g('init_pos_std')),
            init_vel_std=float(g('init_vel_std')),
            gating_mahalanobis=float(g('gating_mahalanobis')),
            max_assoc_dist=float(g('max_assoc_dist')),
            recovery_dist=float(g('recovery_dist')),
            min_hits=int(g('min_hits')),
            max_age=int(g('max_age')),
            min_speed=float(g('min_speed')),
            snr_min=float(g('snr_min')),
            promote_frames=int(g('promote_frames')),
            demote_frames=int(g('demote_frames')),
        )
        self.mot = MultiObjectTracker(self.params)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)

        qos = QoSProfile(depth=10,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE)
        self.cb_track = MutuallyExclusiveCallbackGroup()
        self.cb_side = MutuallyExclusiveCallbackGroup()

        self.pub_mrk = self.create_publisher(MarkerArray, self.markers_topic, 10)
        self.pub_tracks = self.create_publisher(TrackArray, self.tracks_topic, 10)
        self.sub = self.create_subscription(
            PoseArray, self.input_topic, self.on_dets, qos,
            callback_group=self.cb_track)

        # ramo camera: bbox YOLO (cache dell'ultima) + intrinseci.
        # sub best-effort: compatibile sia con publisher reliable sia best-effort.
        self.yolo_boxes = []          # lista di (u0,v0,u1,v1,score), ultima ricevuta
        self.K = None
        self.img_w = None
        self.img_h = None
        if self.use_person_gate:
            side_qos = QoSProfile(depth=5,
                                  reliability=ReliabilityPolicy.BEST_EFFORT,
                                  history=HistoryPolicy.KEEP_LAST,
                                  durability=DurabilityPolicy.VOLATILE)
            self.sub_yolo = self.create_subscription(
                Detection2DArray, g('yolo_topic'), self.on_yolo, side_qos,
                callback_group=self.cb_side)
            self.sub_cam = self.create_subscription(
                CameraInfo, g('camera_info_topic'), self.on_caminfo, side_qos,
                callback_group=self.cb_side)

        self._frame = 0
        self.last_t = None
        self._n_person = 0
        self.get_logger().info(
            f"lidar3d_tracker avviato: {self.input_topic} -> tracce in "
            f"'{self.tracking_frame}' | gate persona YOLO:'{g('yolo_topic')}' | "
            f"critic:'{self.tracks_topic}'")

    # -----------------------------------------------------------------------
    def on_yolo(self, msg):
        """Cache dell'ultima Detection2DArray YOLO (person_only). Il ramo camera
        e' asincrono: qui solo si memorizza, l'associazione avviene nel tracking."""
        self.yolo_boxes = [bbox_to_corners(det) for det in msg.detections]

    def on_caminfo(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.img_w = int(msg.width)
        self.img_h = int(msg.height)

    def _lookup_cam(self):
        """TF tracking_frame -> camera_frame (ultima, timeout 0: non blocca on_dets).
        Il gate legge solo la classe: un piccolo sfasamento lo assorbe bbox_margin."""
        try:
            return self.tf_buffer.lookup_transform(
                self.camera_frame, self.tracking_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.0))
        except TransformException:
            return None

    def _update_semantics(self, tracks):
        """Associazione 2D-3D asincrona: proietta le tracce confermate in immagine
        e le confronta con le bbox YOLO in cache. Match -> accumula; a
        person_confirm_hits -> is_person=True (STICKY). Fuori dal cono/dietro:
        nessuna lettura -> nessun cambio (l'etichetta persiste)."""
        if not self.use_person_gate:
            return
        if self.K is None or not self.yolo_boxes:
            return
        tf_cam = self._lookup_cam()
        if tf_cam is None:
            return
        w, h = self.img_w, self.img_h
        for t in tracks:
            if not t.confirmed:
                continue
            p_cam = self._apply_tf(
                tf_cam, np.array([[t.kf.pos[0], t.kf.pos[1], self.probe_height]]))[0]
            px = project_to_pixel(self.K, p_cam, w, h)
            if px is None:
                continue                        # fuori cono -> mantiene lo stato
            score = match_track_to_boxes(px, self.yolo_boxes, self.bbox_margin)
            if score is None:
                continue                        # nessuna bbox: non conferma, non revoca
            t.person_hits += 1
            t.person_score = score
            if t.person_hits >= self.person_confirm_hits:
                t.is_person = True

    # -----------------------------------------------------------------------
    def _lookup(self, src, stamp):
        """TF src -> tracking_frame ALLO STAMP ESATTO del cloud. Niente fallback:
        userebbe una posa diversa da quella di cattura e farebbe 'muovere' gli
        statici in odom (ego-moto non compensato)."""
        try:
            return self.tf_buffer.lookup_transform(
                self.tracking_frame, src,
                rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=0.1))
        except TransformException as e:
            self.get_logger().warn(
                f"TF {self.tracking_frame} <- {src} @stamp non disponibile, "
                f"salto il frame ({e})",
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
        self._update_semantics(tracks)
        self.publish(stamp, tracks)

        if self.diag_period_frames > 0 and self._frame % self.diag_period_frames == 0:
            n_conf = sum(1 for t in tracks if t.confirmed)
            n_dyn = sum(1 for t in tracks if t.confirmed and t.published_dynamic)
            n_per = sum(1 for t in tracks if t.is_person)
            self.get_logger().info(
                f"[trk] dets={len(dets)} tracce={len(tracks)} confermate={n_conf} "
                f"dinamiche={n_dyn} persone={n_per} "
                f"al_critic={self._n_person} "
                f"recuperi={getattr(self.mot, '_n_recovered', 0)}")

    # -----------------------------------------------------------------------
    def _is_valid_dynamic(self, t):
        """UNICA decisione 'pedone dinamico valido' per marker e TrackArray:
        confermata + dinamica (life-cycle) + persona (gate YOLO) + dentro la
        finestra di coasting."""
        if not t.confirmed:
            return False
        dynamic = t.published_dynamic and (t.is_person or not self.use_person_gate)
        if dynamic and t.time_since_update > self.coast_publish_frames:
            dynamic = False
        return dynamic

    def publish(self, stamp, tracks):
        dyn_tracks = []
        ma = MarkerArray()
        for t in tracks:
            if not t.confirmed:
                continue
            dynamic = self._is_valid_dynamic(t)
            if dynamic:
                dyn_tracks.append(t)
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
                arr.scale.x = 0.06
                arr.scale.y = 0.12
                arr.scale.z = 0.18
                arr.points = [
                    Point(x=float(px), y=float(py), z=float(self.marker_z)),
                    Point(x=float(px + vx * self.arrow_time_scale),
                          y=float(py + vy * self.arrow_time_scale),
                          z=float(self.marker_z)),
                ]
                ma.markers.append(arr)

            label = 'person' if t.is_person else '?'
            txt = self._mk(stamp, 'tracks', base + 2, Marker.TEXT_VIEW_FACING,
                           px, py, self.marker_z + 1.0,
                           ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.9))
            txt.scale.z = 0.3
            txt.text = f"{t.id} {label} v={t.kf.speed:.2f}"
            ma.markers.append(txt)

        self.pub_mrk.publish(ma)
        self._publish_tracks(stamp, dyn_tracks)
        self._n_person = len(dyn_tracks)

    def _publish_tracks(self, stamp, dyn_tracks):
        """TrackArray su /dynamic_tracks_state per il critic. Frame = tracking_frame
        (odom): stesso frame delle traiettorie campionate da MPPI."""
        msg = TrackArray()
        msg.header.stamp = stamp
        msg.header.frame_id = self.tracking_frame
        for t in dyn_tracks:
            px, py = t.kf.pos
            vx, vy = t.kf.vel
            tr = TrackMsg()
            tr.id = int(t.id)
            tr.x = float(px)
            tr.y = float(py)
            tr.vx = float(vx)
            tr.vy = float(vy)
            tr.pos_std = float(t.kf.pos_std())
            tr.vel_std = float(t.kf.vel_std())
            msg.tracks.append(tr)
        self.pub_tracks.publish(msg)

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
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()