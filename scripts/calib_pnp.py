#!/usr/bin/env python3
"""
calib_pnp.py — calibrazione estrinseca camera2 <-> LiDAR via PnP.

Due sottocomandi:
  pick-image  : apri un frame, clicca i pixel delle feature (spigoli, angoli) -> CSV
  solve       : dalle corrispondenze (u,v,X,Y,Z) calcola la posa e la scrive
                pronta per il joint URDF  base_footprint -> camera2_link,
                con errore di riproiezione e overlay di verifica.

Convenzione dei frame (la parte che va fatta giusta):
  solvePnP(punti3D_lidar, pixel, K, D) -> (R,t):  X_camott = R·X_lidar + t
     cioe' la posa  lidar -> camera_ottica.
  L'URDF vuole  base -> camera2_link, quindi:
     T_base_camera2link = T_base_rslidar · inv(R,t) · inv(R_ottica)
  dove R_ottica e' il tuo joint camera2_link->camera2_optical (rpy -90,0,-90).

ACQUISIZIONE dei punti 3D (X,Y,Z nel frame rslidar): in RViz, Fixed Frame=rslidar,
mostra la nuvola, strumento "Publish Point"; leggi i valori con
  ros2 topic echo /clicked_point
e mettili nel CSV accanto al pixel corrispondente (stesso ordine).

CSV: una riga per corrispondenza:  u,v,X,Y,Z   (righe con # o vuote ignorate)
Servono >= 6 corrispondenze non complanari, sparse nell'immagine.
"""

import numpy as np


# ---------- utilità di rotazione (convenzione URDF/tf: R = Rz(y)·Ry(p)·Rx(r)) ----------

def rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_rpy(R):
    # inversa di R = Rz(y)Ry(p)Rx(r)
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, sy))
    p = np.arcsin(sy)
    if abs(np.cos(p)) > 1e-6:
        r = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        r = np.arctan2(-R[1, 2], R[1, 1])
        y = 0.0
    return np.array([r, p, y])


def Rt_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def invT(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


# ---------- solver ----------

def solve_extrinsic(obj_pts, img_pts, K, D,
                    T_base_rslidar, R_optical):
    """obj_pts: Nx3 (frame rslidar). img_pts: Nx2 pixel. K:3x3, D:1xk.
    Ritorna dict con la posa base->camera2_link, base->camera2_optical, errore px."""
    import cv2
    obj = np.asarray(obj_pts, np.float64).reshape(-1, 3)
    img = np.asarray(img_pts, np.float64).reshape(-1, 2)
    # ITERATIVE senza stima iniziale puo' convergere su una soluzione ribaltata
    # (visto sui dati sintetici di camera2, distorsione forte): si prova anche
    # SQPNP (ottimo globale) e si tiene la soluzione con i punti DAVANTI alla
    # camera e l'errore di riproiezione minore
    best = None
    for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rv, tv = cv2.solvePnP(obj, img, K, D, flags=flag)
        except cv2.error:
            continue
        if not ok:
            continue
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, D, rv, tv)
        Rc, _ = cv2.Rodrigues(rv)
        if np.mean((obj @ Rc.T + tv.reshape(1, 3))[:, 2] > 0) < 0.99:
            continue
        pr, _ = cv2.projectPoints(obj, rv, tv, K, D)
        e = float(np.linalg.norm(pr.reshape(-1, 2) - img, axis=1).mean())
        if best is None or e < best[0]:
            best = (e, rv, tv)
    if best is None:
        raise RuntimeError("solvePnP fallito")
    _, rvec, tvec = best

    R_cl, _ = cv2.Rodrigues(rvec)              # camott <- rslidar
    T_camott_rslidar = Rt_to_T(R_cl, tvec)
    T_rslidar_camott = invT(T_camott_rslidar)  # posa camera_ottica in rslidar

    T_camott_link = invT(Rt_to_T(R_optical, [0, 0, 0]))   # camera2_link espresso in camott
    T_rslidar_link = T_rslidar_camott @ T_camott_link
    T_base_link = T_base_rslidar @ T_rslidar_link

    # errore di riproiezione
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)

    return {
        "base_to_camera2_xyz": T_base_link[:3, 3].copy(),
        "base_to_camera2_rpy": R_to_rpy(T_base_link[:3, :3]),
        "rslidar_to_camera2_optical_xyz": T_rslidar_camott[:3, 3].copy(),
        "rslidar_to_camera2_optical_rpy": R_to_rpy(T_rslidar_camott[:3, :3]),
        "reproj_mean_px": float(err.mean()),
        "reproj_max_px": float(err.max()),
        "rvec": rvec, "tvec": tvec,
    }


# ---------- I/O ----------

def load_intrinsics(path):
    import yaml
    d = yaml.safe_load(open(path))
    K = np.array(d["camera_matrix"]["data"], np.float64).reshape(3, 3)
    D = np.array(d["distortion_coefficients"]["data"], np.float64).reshape(1, -1)
    return K, D


def load_corr(path):
    pts = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        vals = [float(x) for x in s.replace(",", " ").split()]
        if len(vals) != 5:
            raise ValueError(f"riga malformata (servono u v X Y Z): {line!r}")
        pts.append(vals)
    a = np.array(pts, np.float64)
    # .copy() forza array C-contigui: le slice di colonne (view non contigue)
    # mandano cv2.solvePnP in errore di shape ("npoints == checkVector(...)").
    return a[:, :2].copy(), a[:, 2:].copy()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("solve")
    ps.add_argument("--intrinsics", required=True)
    ps.add_argument("--corr", required=True)
    ps.add_argument("--rslidar-xyz", nargs=3, type=float, default=[-0.21, 0.0, 0.95])
    ps.add_argument("--rslidar-rpy", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ps.add_argument("--optical-rpy", nargs=3, type=float, default=[-1.5708, 0.0, -1.5708])
    ps.add_argument("--image", default=None, help="frame per l'overlay di verifica")
    ps.add_argument("--out-overlay", default="overlay.png")

    pp = sub.add_parser("pick-image")
    pp.add_argument("--image", required=True)
    pp.add_argument("--out", default="corr.csv")
    pp.add_argument("--gamma", type=float, default=1.0)

    args = ap.parse_args()

    if args.cmd == "pick-image":
        import cv2
        img = cv2.imread(args.image)
        if args.gamma != 1.0:
            x = (img.astype(np.float32) / 255.0) ** (1.0 / args.gamma)
            img = np.clip(x * 255, 0, 255).astype(np.uint8)
        pts = []
        disp = img.copy()

        def on_mouse(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))
                cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)
                cv2.putText(disp, str(len(pts)), (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"pixel {len(pts)}: u={x} v={y}")

        cv2.namedWindow("pick (click feature, q per uscire)")
        cv2.setMouseCallback("pick (click feature, q per uscire)", on_mouse)
        while True:
            cv2.imshow("pick (click feature, q per uscire)", disp)
            if (cv2.waitKey(20) & 0xFF) in (ord("q"), 27):
                break
        cv2.destroyAllWindows()
        with open(args.out, "w") as f:
            f.write("# u,v,X,Y,Z  <- riempi X,Y,Z dal Publish Point di RViz (frame rslidar)\n")
            for (u, v) in pts:
                f.write(f"{u},{v},,,\n")
        print(f"\nScritti {len(pts)} pixel in {args.out}. Ora aggiungi X,Y,Z da RViz.")
        return

    # solve
    import cv2
    K, D = load_intrinsics(args.intrinsics)
    img_pts, obj_pts = load_corr(args.corr)
    if len(img_pts) < 6:
        print(f"ATTENZIONE: solo {len(img_pts)} corrispondenze; per stabilita' servono >=6.")
    T_base_rslidar = Rt_to_T(rpy_to_R(*args.rslidar_rpy), args.rslidar_xyz)
    R_opt = rpy_to_R(*args.optical_rpy)
    res = solve_extrinsic(obj_pts, img_pts, K, D, T_base_rslidar, R_opt)

    x, y, z = res["base_to_camera2_xyz"]
    r, p, yw = res["base_to_camera2_rpy"]
    print("\n=== Joint URDF  base_footprint -> camera2_link ===")
    print(f'  <origin xyz="{x:.4f} {y:.4f} {z:.4f}" rpy="{r:.4f} {p:.4f} {yw:.4f}"/>')
    print("\n(alternativa: rslidar -> camera2_optical diretto)")
    xo, yo, zo = res["rslidar_to_camera2_optical_xyz"]
    ro, po, ywo = res["rslidar_to_camera2_optical_rpy"]
    print(f'  <origin xyz="{xo:.4f} {yo:.4f} {zo:.4f}" rpy="{ro:.4f} {po:.4f} {ywo:.4f}"/>')
    print(f"\nErrore di riproiezione: media {res['reproj_mean_px']:.2f} px, max {res['reproj_max_px']:.2f} px")
    print("  < ~2 px ottimo | 2-4 accettabile | > 5 rivedi le corrispondenze")

    if args.image:
        im = cv2.imread(args.image)
        proj, _ = cv2.projectPoints(obj_pts, res["rvec"], res["tvec"], K, D)
        for (u, v), (pu, pv) in zip(img_pts.astype(int), proj.reshape(-1, 2).astype(int)):
            click, rep = (int(u), int(v)), (int(pu), int(pv))
            cv2.circle(im, click, 5, (0, 255, 0), 1)    # click (verde)
            cv2.circle(im, rep, 3, (0, 0, 255), -1)     # riproiezione (rosso)
            cv2.line(im, click, rep, (255, 0, 0), 1)
        cv2.imwrite(args.out_overlay, im)
        print(f"\nOverlay salvato in {args.out_overlay}: verde=click, rosso=riproiezione. Devono coincidere.")


if __name__ == "__main__":
    main()
