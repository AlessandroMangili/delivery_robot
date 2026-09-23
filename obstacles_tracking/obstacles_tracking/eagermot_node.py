#!/usr/bin/env python3
"""Nodo ROS2 di tracking EagerMOT.

Lega insieme i pezzi:
    /detections3d     (vision_msgs/Detection3DArray)  <- pointpillars_detector
    /yolo/detections  (vision_msgs/Detection2DArray)  <- yolo_detect_node
    /camera/camera_info
                 |
            fusione in istanze  (eagermot_fusion)
                 |
            tracking a due stadi (eagermot_tracker)
                 |
    /dynamic_tracks_state  (dynamic_tracker_msgs/TrackArray)  -> il critic MPPI
    /tracks/markers        (visualization_msgs/MarkerArray)   -> RViz

E' un sostituto di lidar3d_tracker.py: pubblica gli stessi due topic, quindi il
critic e RViz non vanno toccati. Si lanciano in alternativa, mai insieme.

Il tracking avviene nel frame FISSO (odom), non in quello del robot: un modello
a velocita' costante nel frame di un veicolo che si muove descriverebbe il moto
relativo, non quello del pedone.

Va lanciato col python del venv (serve numpy; non serve torch qui, ma il nodo
convive con il detector che gira nello stesso ambiente).
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray, Detection3DArray
from dynamic_tracker_msgs.msg import Track as TrackMsg, TrackArray

import tf2_ros
from tf2_ros import TransformException

from .eagermot_fusion import fuse_detections, conta_istanze
from .eagermot_tracker import EagerMOT, transform_box3d


# ============================================================================
# Conversioni messaggio -> array (pure, testabili)
# ============================================================================

def quat_to_yaw(q):
    """Yaw da un quaternione (rotazione attorno a z)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def transform_to_matrix(t):
    """geometry_msgs/TransformStamped -> matrice omogenea 4x4."""
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


def det3d_to_boxes(msg):
    """Detection3DArray -> (boxes (N,7), scores (N,)).

    La box esce come [x, y, z, dx, dy, dz, yaw], l'ordine di OpenPCDet, che e'
    quello che il tracker si aspetta.
    """
    boxes, scores = [], []
    for det in msg.detections:
        c = det.bbox.center
        s = det.bbox.size
        boxes.append([c.position.x, c.position.y, c.position.z,
                      s.x, s.y, s.z, quat_to_yaw(c.orientation)])
        sc = 1.0
        if det.results:
            r0 = det.results[0]
            sc = float(getattr(getattr(r0, 'hypothesis', r0), 'score', 1.0))
        scores.append(sc)
    if not boxes:
        return np.zeros((0, 7)), np.zeros(0)
    return np.asarray(boxes, dtype=np.float64), np.asarray(scores, dtype=np.float64)


def det2d_to_boxes(msg):
    """Detection2DArray -> (boxes (M,4) [x1,y1,x2,y2], scores (M,)).

    Robusto alle due varianti di BoundingBox2D.center (vision_msgs/Pose2D con
    .position, oppure geometry_msgs/Pose2D piatta), come gia' fa il tracker
    esistente.
    """
    boxes, scores = [], []
    for det in msg.detections:
        b = det.bbox
        c = b.center
        if hasattr(c, 'position'):
            cx, cy = float(c.position.x), float(c.position.y)
        else:
            cx, cy = float(c.x), float(c.y)
        hx, hy = 0.5 * float(b.size_x), 0.5 * float(b.size_y)
        boxes.append([cx - hx, cy - hy, cx + hx, cy + hy])
        sc = 1.0
        if det.results:
            r0 = det.results[0]
            sc = float(getattr(getattr(r0, 'hypothesis', r0), 'score', 1.0))
        scores.append(sc)
    if not boxes:
        return np.zeros((0, 4)), np.zeros(0)
    return np.asarray(boxes, dtype=np.float64), np.asarray(scores, dtype=np.float64)


# ============================================================================
# Nodo
# ============================================================================

class EagerMOTNode(Node):

    def __init__(self):
        super().__init__('eagermot_tracker_node')

        d = self.declare_parameter
        d('detections3d_topic', '/detections3d')
        d('yolo_topic', '/yolo/detections')
        d('camera_info_topic', '/camera/camera_info')
        d('tracks_topic', '/dynamic_tracks_state')
        d('markers_topic', '/tracks/markers')
        d('tracking_frame', 'odom')
        d('sensor_frame', 'base_scan')
        d('camera_frame', 'camera_rgb_optical_frame')

        # --- soglie EagerMOT (valori del paper, riga pedestrian di NuScenes) ---
        d('theta_fusion', 0.3)     # IoU minima per fondere una 3D con una 2D
        d('theta_3d', 1.8)         # distanza scalata MASSIMA, stadio 1 [m]
        d('theta_2d', 0.5)         # IoU minima, stadio 2
        d('age_max', 3)            # frame senza update dopo i quali si muore
        d('age_2d', 3)             # frame entro cui serve evidenza 2D
        # --- deviazioni dal paper, disattivate di default (vedi confermata()) ---
        # age_3d: -1 = regola del paper (serve evidenza 2D). N >= 0 = accetta
        #   anche evidenza 3D entro N frame. Indispensabile qui, dove il LiDAR
        #   copre 360 gradi e la camera solo ~72: un pedone di lato non potra'
        #   MAI avere evidenza 2D, e con la regola originale resta invisibile
        #   al critic pur essendo visto benissimo dal LiDAR.
        d('age_3d', -1)
        # confirm_coast_frames: 0 = regola del paper (conferma non appiccicosa).
        #   N > 0 = la conferma sopravvive a N frame senza match, contro lo
        #   sfarfallio di un detector che perde qualche frame.
        d('confirm_coast_frames', 0)
        # dup_suppress_radius: 0.0 = regola del paper. R > 0 = un'istanza 3D
        #   orfana a meno di R metri da una traccia viva non apre una traccia
        #   nuova: aggiorna quella (se mancata dallo stadio 1) o si scarta
        #   (se doppia detection). Vedi lo step 2.5 in EagerMOT.step().
        d('dup_suppress_radius', 0.0)
        # min_hits_3d: 0 = nessun vincolo. N > 0 = una traccia confermata solo
        #   dal 3D (via age_3d) deve essere stata associata in almeno N frame.
        d('min_hits_3d', 0)

        # --- covarianze del Kalman (default AB3DMOT, tarati su auto KITTI) ---
        d('kf_p_pos', 10.0)
        d('kf_p_vel', 10000.0)
        d('kf_q_pos', 1.0)
        d('kf_q_vel', 0.01)
        d('kf_r_meas', 1.0)

        # --- pubblicazione ---
        d('min_speed', 0.0)        # 0 = pubblica tutte le confermate; il critic
                                   # ha gia' il suo min_ped_speed
        d('marker_z', 0.6)
        d('arrow_time_scale', 1.0)
        d('diag_period_frames', 20)
        d('tf_timeout_s', 0.10)

        g = lambda k: self.get_parameter(k).value
        self.tracking_frame = g('tracking_frame')
        self.sensor_frame = g('sensor_frame')
        self.camera_frame = g('camera_frame')
        self.min_speed = float(g('min_speed'))
        self.marker_z = float(g('marker_z'))
        self.arrow_time_scale = float(g('arrow_time_scale'))
        self.diag_period_frames = int(g('diag_period_frames'))
        self.tf_timeout = float(g('tf_timeout_s'))
        self.theta_fusion = float(g('theta_fusion'))

        _a3 = int(g('age_3d'))
        self.mot = EagerMOT(
            theta_3d=float(g('theta_3d')), theta_2d=float(g('theta_2d')),
            age_max=int(g('age_max')), age_2d=int(g('age_2d')),
            age_3d=(None if _a3 < 0 else _a3),
            confirm_coast_frames=int(g('confirm_coast_frames')),
            dup_suppress_radius=float(g('dup_suppress_radius')),
            min_hits_3d=int(g('min_hits_3d')),
            kf_kwargs=dict(p_pos=float(g('kf_p_pos')),
                           p_vel=float(g('kf_p_vel')),
                           q_pos=float(g('kf_q_pos')),
                           q_vel=float(g('kf_q_vel')),
                           r_meas=float(g('kf_r_meas'))))

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST,
                         durability=DurabilityPolicy.VOLATILE)
        side_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST,
                              durability=DurabilityPolicy.VOLATILE)
        cb_main = MutuallyExclusiveCallbackGroup()
        cb_side = MutuallyExclusiveCallbackGroup()

        self.pub_tracks = self.create_publisher(TrackArray, g('tracks_topic'), 10)
        self.pub_mrk = self.create_publisher(MarkerArray, g('markers_topic'), 10)

        self.create_subscription(Detection3DArray, g('detections3d_topic'),
                                 self.on_det3d, qos, callback_group=cb_main)
        # Il ramo camera e' ASINCRONO: qui si memorizza soltanto l'ultima
        # Detection2DArray, la fusione avviene quando arriva la nuvola.
        self.yolo_boxes = np.zeros((0, 4))
        self.yolo_scores = np.zeros(0)
        self.K = None
        self.img_w = None
        self.img_h = None
        self.create_subscription(Detection2DArray, g('yolo_topic'),
                                 self.on_yolo, side_qos, callback_group=cb_side)
        self.create_subscription(CameraInfo, g('camera_info_topic'),
                                 self.on_caminfo, side_qos,
                                 callback_group=cb_side)

        self._frame = 0
        self._last_t = None
        self.get_logger().info(
            f"EagerMOT: {g('detections3d_topic')} + {g('yolo_topic')} -> "
            f"{g('tracks_topic')} | theta_fusion={self.theta_fusion} "
            f"theta_3d={self.mot.theta_3d} theta_2d={self.mot.theta_2d} "
            f"age_max={self.mot.age_max} age_2d={self.mot.age_2d} "
            f"age_3d={self.mot.age_3d if self.mot.age_3d is not None else 'off (regola paper)'} "
            f"coast={self.mot.confirm_coast_frames} "
            f"dup_r={self.mot.dup_suppress_radius} "
            f"min_hits_3d={self.mot.min_hits_3d}")

    # -- ramo camera ------------------------------------------------------
    def on_yolo(self, msg):
        self.yolo_boxes, self.yolo_scores = det2d_to_boxes(msg)

    def on_caminfo(self, msg):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.img_w = int(msg.width)
        self.img_h = int(msg.height)

    # -- ciclo principale -------------------------------------------------
    def on_det3d(self, msg):
        self._frame += 1
        stamp = msg.header.stamp
        t_now = stamp.sec + stamp.nanosec * 1e-9
        dt = 0.1 if self._last_t is None else max(t_now - self._last_t, 1e-3)
        self._last_t = t_now

        if self.K is None:
            self.get_logger().warn(
                'CameraInfo non ancora ricevuta: niente fusione ne\' stadio 2.',
                throttle_duration_sec=5.0)
            return

        # --- TF: sensore -> odom (per tracciare) e odom -> camera (per proiettare)
        try:
            tf_odom_sensor = self.tf_buffer.lookup_transform(
                self.tracking_frame, msg.header.frame_id or self.sensor_frame,
                rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=self.tf_timeout))
            tf_cam_odom = self.tf_buffer.lookup_transform(
                self.camera_frame, self.tracking_frame,
                rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=self.tf_timeout))
        except TransformException as e:
            self.get_logger().warn(f'TF non disponibile: {e}',
                                   throttle_duration_sec=2.0)
            return

        T_odom_sensor = transform_to_matrix(tf_odom_sensor)
        T_cam_odom = transform_to_matrix(tf_cam_odom)

        # --- detection 3D dal frame del LiDAR a odom ---
        boxes_s, scores3d = det3d_to_boxes(msg)
        boxes_o = (np.array([transform_box3d(b, T_odom_sensor) for b in boxes_s])
                   if len(boxes_s) else np.zeros((0, 7)))

        # --- FUSIONE (le box 3D sono in odom, quindi si proietta da odom) ---
        istanze = fuse_detections(boxes_o, scores3d,
                                  self.yolo_boxes, self.yolo_scores,
                                  T_cam_odom, self.K, self.img_w, self.img_h,
                                  theta_fusion=self.theta_fusion)

        # --- TRACKING a due stadi ---
        self.mot.step(istanze, dt, T_cam_odom, self.K, self.img_w, self.img_h)

        confermate = [t for t in self.mot.tracks_confermate()
                      if float(np.hypot(*t.velocita[:2])) >= self.min_speed]
        self.publish(stamp, confermate)

        if self.diag_period_frames > 0 and self._frame % self.diag_period_frames == 0:
            b, t3, t2 = conta_istanze(istanze)
            d = self.mot.diagnostica
            self.get_logger().info(
                f"[emot] det3d={len(boxes_o)} yolo={len(self.yolo_boxes)} | "
                f"istanze both={b} solo3d={t3} solo2d={t2} | "
                f"match1={d['match1']} match2={d['match2']} salvate={d['salvate']} soppresse={d['soppresse']} "
                f"nuove={d['nuove']} morte={d['morte']} | "
                f"vive={d['vive']} confermate={len(confermate)} | dt={dt*1000:.0f}ms")

    # -- uscita -----------------------------------------------------------
    def publish(self, stamp, tracce):
        out = TrackArray()
        out.header.stamp = stamp
        out.header.frame_id = self.tracking_frame
        for t in tracce:
            box = t.box3d
            vel = t.velocita
            tr = TrackMsg()
            tr.id = int(t.id)
            tr.x = float(box[0])
            tr.y = float(box[1])
            tr.vx = float(vel[0])
            tr.vy = float(vel[1])
            # deviazioni standard dalla covarianza del filtro: la diagonale di P
            # sulle componenti di posizione (0,1) e di velocita' (7,8).
            tr.pos_std = float(np.sqrt(max(t.kf.P[0, 0], t.kf.P[1, 1], 0.0)))
            tr.vel_std = float(np.sqrt(max(t.kf.P[7, 7], t.kf.P[8, 8], 0.0)))
            out.tracks.append(tr)
        self.pub_tracks.publish(out)
        self.publish_markers(stamp, tracce)

    def publish_markers(self, stamp, tracce):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.tracking_frame
        clear.header.stamp = stamp
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        for t in tracce:
            box = t.box3d
            vel = t.velocita
            # colore: verde se il LiDAR l'ha vista in questo frame, ambra se no.
            # Etichetta: [2D] solo se in questo frame l'ha aggiornata la camera,
            # [coast] se nessuna misura (sola predizione del Kalman).
            coasting = t.eta_da_3d > 0
            tag = ('' if not coasting else
                   ' [2D]' if t.eta_da_2d == 0 else ' [coast]')
            col = (ColorRGBA(r=0.95, g=0.65, b=0.10, a=0.85) if coasting
                   else ColorRGBA(r=0.10, g=0.85, b=0.25, a=0.85))

            m = Marker()
            m.header.frame_id = self.tracking_frame
            m.header.stamp = stamp
            m.ns = 'eagermot'
            m.id = int(t.id) * 3
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(box[0])
            m.pose.position.y = float(box[1])
            m.pose.position.z = self.marker_z
            m.pose.orientation.w = 1.0
            m.scale.x = max(float(box[3]), 0.1)
            m.scale.y = max(float(box[4]), 0.1)
            m.scale.z = max(float(box[5]), 0.1)
            m.color = col
            arr.markers.append(m)

            a = Marker()
            a.header.frame_id = self.tracking_frame
            a.header.stamp = stamp
            a.ns = 'eagermot_vel'
            a.id = int(t.id) * 3 + 1
            a.type = Marker.ARROW
            a.action = Marker.ADD
            a.scale.x = 0.06
            a.scale.y = 0.12
            a.scale.z = 0.12
            a.color = col
            a.points = [
                Point(x=float(box[0]), y=float(box[1]), z=self.marker_z),
                Point(x=float(box[0] + vel[0] * self.arrow_time_scale),
                      y=float(box[1] + vel[1] * self.arrow_time_scale),
                      z=self.marker_z)]
            arr.markers.append(a)

            txt = Marker()
            txt.header.frame_id = self.tracking_frame
            txt.header.stamp = stamp
            txt.ns = 'eagermot_id'
            txt.id = int(t.id) * 3 + 2
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(box[0])
            txt.pose.position.y = float(box[1])
            txt.pose.position.z = self.marker_z + 1.1
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.28
            txt.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
            sped = float(np.hypot(vel[0], vel[1]))
            txt.text = f'{t.id} {sped:.1f}m/s{tag}'
            arr.markers.append(txt)

        self.pub_mrk.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = EagerMOTNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
