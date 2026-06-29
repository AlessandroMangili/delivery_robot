#!/usr/bin/env python3
"""
GOVERNATORE DI VELOCITA' MODULATO DALLA LOCALIZZAZIONE.

Idea (contributo "risk-aware" della tesi, lato localizzazione):
quando la stima di posa e' INCERTA, il robot rallenta. Cosi' da' implicitamente
priorita' al ri-localizzarsi (a bassa velocita' l'odometria deriva meno e le
osservazioni LiDAR/GPS pesano di piu') ed e' piu' prudente quando "non sa bene
dov'e'". Quando la localizzazione e' sicura, torna a velocita' piena.

Con AMCL:
ros2 run navigation localization_speed_governor --ros-args \
  -p use_sim_time:=true \
  -p pose_topic:=/amcl_pose -p pose_type:=pose_cov \
  -p base_max_vel_x:=0.30 -p unc_good:=0.05 -p unc_bad:=0.8 -p f_min:=0.2

Con GPS-EKF, cambi solo le due righe del topic:
-p pose_topic:=/odometry/filtered -p pose_type:=odom

COME:
  - si sottoscrive a un topic di posa-con-covarianza:
      * AMCL  -> /amcl_pose            (PoseWithCovarianceStamped)
      * EKF   -> /odometry/filtered    (Odometry, ha pose.covariance)
    il topic e il tipo sono PARAMETRI: funziona con entrambi.
  - incertezza = traccia della covarianza su (x, y, yaw): C[0,0]+C[1,1]+C[5,5];
    e' quanto e' "spalmata" la stima nel piano.
  - mappa incertezza -> fattore in [f_min, 1]:
      unc <= unc_good -> 1.0 (piena);  unc >= unc_bad -> f_min;  in mezzo lineare.
  - applica il fattore impostando a runtime  max_vel_x  (e opz. max_vel_theta)
    del controller (DWB: FollowPath.max_vel_x) via set_parameters al
    controller_server. In subordine puo' agire sul velocity_smoother.

NOTA: il controller_server di Nav2 espone i parametri del plugin come
  'FollowPath.max_vel_x', 'FollowPath.max_vel_theta'  (per DWB).
  Sono i nomi usati di default; se il tuo plugin si chiama diversamente,
  cambia 'controller_plugin'.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np


class LocalizationSpeedGovernor(Node):
    def __init__(self):
        super().__init__('localization_speed_governor')

        # --- sorgente dell'incertezza ---
        self.declare_parameter('pose_topic', '/amcl_pose')
        # 'pose_cov' = PoseWithCovarianceStamped (AMCL) ; 'odom' = Odometry (EKF)
        self.declare_parameter('pose_type', 'pose_cov')

        # --- mappatura incertezza -> fattore velocita' ---
        self.declare_parameter('unc_good', 0.05)   # <= questo: localizzazione "buona" -> piena
        self.declare_parameter('unc_bad', 0.8)     # >= questo: "pessima" -> f_min
        self.declare_parameter('f_min', 0.2)       # fattore minimo (frazione di vel piena)

        # --- velocita' di riferimento (la "piena") ---
        self.declare_parameter('base_max_vel_x', 0.30)      # m/s a localizzazione buona
        self.declare_parameter('base_max_vel_theta', 1.0)   # rad/s a localizzazione buona
        self.declare_parameter('modulate_theta', False)     # se True scala anche la rotazione

        # --- a chi applicare i parametri ---
        self.declare_parameter('controller_server', '/controller_server')
        self.declare_parameter('controller_plugin', 'FollowPath')

        # --- frequenza di aggiornamento e isteresi ---
        self.declare_parameter('update_period', 0.5)   # s: ogni quanto riapplicare
        self.declare_parameter('min_delta', 0.01)      # m/s: evita spam se cambia poco
        self.declare_parameter('ema_alpha', 0.4)       # smoothing incertezza (0..1)

        gp = self.get_parameter
        self.pose_topic = gp('pose_topic').value
        self.pose_type = gp('pose_type').value
        self.unc_good = gp('unc_good').value
        self.unc_bad = gp('unc_bad').value
        self.f_min = gp('f_min').value
        self.base_vx = gp('base_max_vel_x').value
        self.base_vth = gp('base_max_vel_theta').value
        self.modulate_theta = gp('modulate_theta').value
        self.ctrl_srv = gp('controller_server').value
        self.plugin = gp('controller_plugin').value
        self.update_period = gp('update_period').value
        self.min_delta = gp('min_delta').value
        self.ema_alpha = gp('ema_alpha').value

        self.unc_ema = None
        self.last_applied_vx = None

        # client per impostare i parametri del controller_server
        self.cli = self.create_client(
            SetParameters, f'{self.ctrl_srv}/set_parameters')

        # sottoscrizione alla posa (QoS adatto a entrambe le sorgenti)
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=5,
                         durability=DurabilityPolicy.VOLATILE)
        if self.pose_type == 'odom':
            self.create_subscription(Odometry, self.pose_topic, self.odom_cb, qos)
        else:
            self.create_subscription(PoseWithCovarianceStamped, self.pose_topic,
                                     self.posecov_cb, qos)

        # pubblica l'incertezza corrente (utile per debug / grafici tesi)
        self.unc_pub = self.create_publisher(Float32, '~/uncertainty', 10)
        self.fac_pub = self.create_publisher(Float32, '~/vel_factor', 10)

        # timer di applicazione
        self.timer = self.create_timer(self.update_period, self.apply_timer)

        self.get_logger().info(
            f'Speed governor: {self.pose_topic} ({self.pose_type}) -> '
            f'{self.ctrl_srv} [{self.plugin}.max_vel_x]. '
            f'good={self.unc_good} bad={self.unc_bad} f_min={self.f_min} '
            f'base_vx={self.base_vx}')

    # ---- estrazione incertezza dalle due sorgenti ----
    def _uncertainty_from_cov(self, cov36):
        C = np.array(cov36, dtype=float).reshape(6, 6)
        # traccia su x, y, yaw (indici 0,1,5): dispersione planare della stima
        return float(C[0, 0] + C[1, 1] + C[5, 5])

    def posecov_cb(self, msg: PoseWithCovarianceStamped):
        self._ingest(self._uncertainty_from_cov(msg.pose.covariance))

    def odom_cb(self, msg: Odometry):
        self._ingest(self._uncertainty_from_cov(msg.pose.covariance))

    def _ingest(self, unc):
        if not np.isfinite(unc) or unc < 0:
            return
        if self.unc_ema is None:
            self.unc_ema = unc
        else:
            a = self.ema_alpha
            self.unc_ema = a * unc + (1.0 - a) * self.unc_ema

    # ---- mappatura incertezza -> fattore ----
    def vel_factor(self, unc):
        if unc <= self.unc_good:
            return 1.0
        if unc >= self.unc_bad:
            return self.f_min
        a = (unc - self.unc_good) / (self.unc_bad - self.unc_good)
        return 1.0 + a * (self.f_min - 1.0)

    # ---- applicazione periodica ----
    def apply_timer(self):
        if self.unc_ema is None:
            return  # nessuna posa ancora ricevuta
        f = self.vel_factor(self.unc_ema)
        vx = round(self.base_vx * f, 3)

        # pubblica debug
        m = Float32(); m.data = float(self.unc_ema); self.unc_pub.publish(m)
        m = Float32(); m.data = float(f); self.fac_pub.publish(m)

        if self.last_applied_vx is not None and abs(vx - self.last_applied_vx) < self.min_delta:
            return  # cambia troppo poco, non spammo il servizio

        params = [self._make_double(f'{self.plugin}.max_vel_x', vx)]
        if self.modulate_theta:
            vth = round(self.base_vth * f, 3)
            params.append(self._make_double(f'{self.plugin}.max_vel_theta', vth))

        if not self.cli.service_is_ready():
            # il controller_server potrebbe non essere ancora su: riprovo al giro dopo
            self.get_logger().warn(
                f'set_parameters non pronto su {self.ctrl_srv}, riprovo...',
                throttle_duration_sec=5.0)
            return

        req = SetParameters.Request()
        req.parameters = params
        fut = self.cli.call_async(req)
        fut.add_done_callback(lambda f_: self._on_set_done(f_, vx))

    def _on_set_done(self, fut, vx):
        try:
            res = fut.result()
            ok = res is not None and all(r.successful for r in res.results)
            if ok:
                self.last_applied_vx = vx
                self.get_logger().info(
                    f'max_vel_x -> {vx:.3f} m/s (incertezza={self.unc_ema:.3f})',
                    throttle_duration_sec=2.0)
            else:
                self.get_logger().warn('set_parameters rifiutato dal controller.')
        except Exception as e:
            self.get_logger().warn(f'set_parameters fallito: {e}')

    def _make_double(self, name, value):
        p = Parameter()
        p.name = name
        p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value))
        return p


def main():
    rclpy.init()
    node = LocalizationSpeedGovernor()
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