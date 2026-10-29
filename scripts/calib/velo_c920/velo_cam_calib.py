#!/usr/bin/env python3
"""
velo_cam_calib.py — calibrazione estrinseca Velodyne VLP-16 <-> camera USB (C920),
interattiva: overlay live della nuvola sull'immagine + PnP da corrispondenze.

Parametri stimati: posa di c920_link nel frame velodyne (convenzione URDF,
x avanti / y sinistra / z su, rpy = Rz(y)Ry(p)Rx(r)). Il frame ottico
c920_optical e' ruotato di rpy (-90, 0, -90) rispetto a c920_link.

Flusso:
  1. slider x,y,z,roll,pitch,yaw  -> stima grossolana finche' la nuvola "si
     appoggia" sulla scena (spigoli, pali, persone)
  2. SPAZIO congela immagine+nuvola. Per ogni feature (spigolo di una scatola,
     angolo di una porta...):
       - click sul pixel nella finestra
       - in RViz (Fixed Frame = velodyne) strumento "Publish Point" sullo
         stesso punto della nuvola  -> arriva su /clicked_point
     stesso ordine nei due lati; >= 6 coppie, sparse e a profondita' diverse
  3. 's' risolve PnP, aggiorna gli slider, stampa errore di riproiezione
  4. 'w' salva velo_c920_extrinsic.yaml e stampa il static_transform_publisher

Tasti: SPAZIO freeze/live | s solve | u annulla ultima coppia | c cancella
       coppie | w scrivi | q esci
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from calib_pnp import rpy_to_R, Rt_to_T, invT, solve_extrinsic  # noqa: E402

R_OPT = rpy_to_R(-np.pi / 2, 0.0, -np.pi / 2)   # c920_link -> c920_optical
WIN = "velo -> c920  (SPAZIO freeze, s solve, u undo, c clear, w write, q quit)"

# slider: (nome, min, max, passo)  -> trackbar intera 0..(max-min)/passo
SLIDERS = [("x mm", -1000, 1000, 1), ("y mm", -1000, 1000, 1), ("z mm", -1000, 1000, 1),
           ("roll 0.1deg", -1800, 1800, 1), ("pitch 0.1deg", -1800, 1800, 1),
           ("yaw 0.1deg", -1800, 1800, 1)]


def cloud_xyz(msg):
    off = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                   "offsets": [off["x"], off["y"], off["z"]], "itemsize": msg.point_step})
    a = np.frombuffer(msg.data, dt, count=msg.width * msg.height)
    P = np.column_stack([a["x"], a["y"], a["z"]]).astype(np.float64)
    return P[np.isfinite(P).all(axis=1)]


class Calib(Node):
    def __init__(self):
        super().__init__("velo_cam_calib")
        self.declare_parameter("intrinsics", str(Path.home() / ".ros/camera_info/camera2.yaml"))
        self.declare_parameter("image", "/c920/image_raw")
        self.declare_parameter("cloud", "/velodyne_points")
        self.declare_parameter("out", "velo_c920_extrinsic.yaml")
        # stima iniziale: camera circa sotto/davanti al LiDAR, che guarda avanti
        self.declare_parameter("init", [0.0, 0.0, -0.1, 0.0, 0.0, 0.0])

        d = yaml.safe_load(open(self.get_parameter("intrinsics").value))
        self.K = np.array(d["camera_matrix"]["data"], np.float64).reshape(3, 3)
        self.D = np.array(d["distortion_coefficients"]["data"], np.float64).reshape(1, -1)
        self.wh = (d["image_width"], d["image_height"])

        self.pose = np.array(self.get_parameter("init").value, np.float64)  # x y z r p y
        self.bridge = CvBridge()
        self.img = self.cloud = None
        self.frozen = False
        self.px, self.p3 = [], []
        self.last_err = None

        self.create_subscription(Image, self.get_parameter("image").value, self.on_img, 5)
        self.create_subscription(PointCloud2, self.get_parameter("cloud").value,
                                 self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(PointStamped, "/clicked_point", self.on_click3d, 10)
        self.overlay_pub = self.create_publisher(Image, "/c920/lidar_overlay", 2)

    # ---------------- callback ROS
    def on_img(self, m):
        if not self.frozen:
            self.img = self.bridge.imgmsg_to_cv2(m, "bgr8")
            if (self.img.shape[1], self.img.shape[0]) != self.wh:
                self.get_logger().warn(f"immagine {self.img.shape[1]}x{self.img.shape[0]} "
                                       f"!= intrinseci {self.wh}", once=True)

    def on_cloud(self, m):
        if not self.frozen:
            self.cloud = cloud_xyz(m)

    def on_click3d(self, m):
        if m.header.frame_id != "velodyne":
            self.get_logger().warn(f"/clicked_point in frame '{m.header.frame_id}': "
                                   "metti Fixed Frame = velodyne in RViz")
            return
        self.p3.append([m.point.x, m.point.y, m.point.z])
        self.get_logger().info(f"3D #{len(self.p3)}: {self.p3[-1]}  (pixel: {len(self.px)})")

    # ---------------- geometria
    def T_velo_opt(self):
        T_velo_link = Rt_to_T(rpy_to_R(*self.pose[3:]), self.pose[:3])
        return T_velo_link @ Rt_to_T(R_OPT, [0, 0, 0])

    def project(self, P):
        T = invT(self.T_velo_opt())
        Pc = P @ T[:3, :3].T + T[:3, 3]
        # solo punti davanti e dentro un cono ragionevole: la distorsione
        # polinomiale "ripiega" i punti molto fuori campo dentro l'immagine
        m = (Pc[:, 2] > 0.3) & (np.abs(Pc[:, 0] / Pc[:, 2]) < 1.2) & (np.abs(Pc[:, 1] / Pc[:, 2]) < 1.0)
        Pc = Pc[m]
        if len(Pc) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        uv, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), self.K, self.D)
        return uv.reshape(-1, 2), Pc[:, 2]

    # ---------------- GUI
    def set_sliders(self):
        self._sync = True
        vals = list(self.pose[:3] * 1000) + list(np.degrees(self.pose[3:]) * 10)
        for (name, lo, hi, _), v in zip(SLIDERS, vals):
            v = ((v / 10 + 180) % 360 - 180) * 10 if "deg" in name else v
            cv2.setTrackbarPos(name, WIN, int(round(np.clip(v, lo, hi) - lo)))
        self._sync = False

    def on_slider(self, i, lo):
        def cb(pos):
            if getattr(self, "_sync", False):
                return
            v = pos + lo
            self.pose[i] = v / 1000 if i < 3 else np.radians(v / 10)
        return cb

    def on_mouse(self, ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.px.append([x, y])
            self.get_logger().info(f"pixel #{len(self.px)}: ({x}, {y})  (3D: {len(self.p3)})")

    def draw(self):
        im = self.img.copy()
        if self.cloud is not None:
            uv, z = self.project(self.cloud)
            # scala colori adattata alla scena: i salti di profondita' (bordi degli
            # oggetti) diventano salti di colore, che e' cio' che si allinea
            lo, hi = np.percentile(z, [5, 95]) if len(z) else (0.0, 1.0)
            t = np.clip((z - lo) / max(hi - lo, 0.1), 0, 1)
            col = cv2.applyColorMap((255 - t * 255).astype(np.uint8)
                                    .reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
            for (u, v), c in zip(uv.astype(int), col):
                if 0 <= u < im.shape[1] and 0 <= v < im.shape[0]:
                    cv2.circle(im, (u, v), 2, tuple(int(k) for k in c), -1)
        for i, (u, v) in enumerate(self.px):
            cv2.circle(im, (u, v), 6, (0, 255, 0), 2)
            cv2.putText(im, str(i + 1), (u + 7, v - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if self.p3:
            uv, _ = self.project(np.array(self.p3))
            for u, v in uv.astype(int):
                cv2.drawMarker(im, (u, v), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
        txt = (f"{'FROZEN' if self.frozen else 'LIVE'}  coppie pixel/3D: {len(self.px)}/{len(self.p3)}"
               + (f"  reproj {self.last_err:.2f}px" if self.last_err is not None else ""))
        cv2.putText(im, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return im

    def solve(self):
        n = min(len(self.px), len(self.p3))
        if n < 6:
            self.get_logger().warn(f"servono >= 6 coppie complete, ne hai {n}")
            return
        if len(self.px) != len(self.p3):
            self.get_logger().warn(f"pixel ({len(self.px)}) e 3D ({len(self.p3)}) diversi: uso le prime {n}")
        res = solve_extrinsic(np.array(self.p3[:n]), np.array(self.px[:n], np.float64),
                              self.K, self.D, np.eye(4), R_OPT)
        self.pose[:3] = res["base_to_camera2_xyz"]
        self.pose[3:] = res["base_to_camera2_rpy"]
        self.last_err = res["reproj_mean_px"]
        self.set_sliders()
        self.get_logger().info(f"PnP su {n} coppie: reproj media {res['reproj_mean_px']:.2f} px, "
                               f"max {res['reproj_max_px']:.2f} px")
        self.print_pose()

    def print_pose(self):
        x, y, z, r, p, yw = self.pose
        print("\n=== velodyne -> c920_link ===")
        print(f'  <origin xyz="{x:.4f} {y:.4f} {z:.4f}" rpy="{r:.4f} {p:.4f} {yw:.4f}"/>')
        print(f"  ros2 run tf2_ros static_transform_publisher --x {x:.4f} --y {y:.4f} --z {z:.4f} "
              f"--roll {r:.4f} --pitch {p:.4f} --yaw {yw:.4f} --frame-id velodyne --child-frame-id c920_link")
        print("  (+ c920_link -> c920_optical: rpy -1.5708 0 -1.5708)\n")

    def write(self):
        out = self.get_parameter("out").value
        x, y, z, r, p, yw = (float(v) for v in self.pose)
        yaml.safe_dump({"parent": "velodyne", "child": "c920_link",
                        "xyz": [x, y, z], "rpy": [r, p, yw],
                        "reproj_mean_px": self.last_err,
                        "pairs": {"pixel": self.px, "velodyne_xyz": self.p3}},
                       open(out, "w"), sort_keys=False)
        self.get_logger().info(f"salvato {out}")
        self.print_pose()

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        for i, (name, lo, hi, _) in enumerate(SLIDERS):
            cv2.createTrackbar(name, WIN, 0, hi - lo, self.on_slider(i, lo))
        self.set_sliders()
        cv2.setMouseCallback(WIN, self.on_mouse)
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.img is None:
                cv2.waitKey(10)
                continue
            im = self.draw()
            cv2.imshow(WIN, im)
            self.overlay_pub.publish(self.bridge.cv2_to_imgmsg(im, "bgr8"))
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord(" "):
                self.frozen = not self.frozen
                self.get_logger().info("FROZEN" if self.frozen else "LIVE")
            elif k == ord("s"):
                self.solve()
            elif k == ord("u"):
                # toglie l'elemento "in sospeso" se i due lati sono sbilanciati,
                # altrimenti l'ultima coppia completa
                if len(self.px) > len(self.p3):
                    self.px.pop()
                elif len(self.p3) > len(self.px):
                    self.p3.pop()
                elif self.px:
                    self.px.pop()
                    self.p3.pop()
            elif k == ord("c"):
                self.px, self.p3 = [], []
            elif k == ord("w"):
                self.write()
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = Calib()
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
