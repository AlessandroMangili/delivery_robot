#!/usr/bin/env python3
"""calib_target.py — calibrazione camera<->LiDAR con un PANNELLO rettangolare.

Registrazione (robot FERMO, pannello opaco W x H noto, es. 60 x 80 cm):
  1. i primi ~5 s SENZA pannello in vista (sfondo)
  2. 6-10 pose del pannello a 2-4 m, ognuna tenuta ferma >= 3 s: sinistra /
     centro / destra, vicino / lontano, ruotato di 20-40 gradi attorno alla
     verticale, pannello DRITTO (non ruotato nel proprio piano oltre ~20 gradi).
     Due serie, perche' le camere guardano ad altezze diverse:
       - camera2 (inclinata di 20 gradi): pannello BASSO, da ~10 cm a <= 70 cm
       - camera1 (orizzontale o poco verso l'alto): pannello su sedia/scatola,
         bordo inferiore >= 50 cm
     Controlla in rqt_image_view che il pannello sia INTERO nell'immagine; in
     'pick' salta (tasto s) le pose che la camera non vede per intero.
  ros2 bag record /rslidar_points /camera1/image_raw /camera1/camera_info \
                  /camera2/image_raw /camera2/camera_info
  (pochi topic: la registrazione completa satura e perde ~40% delle nuvole)

Sottocomandi:
  detect  <db3> --out pose/        trova le pose ferme del pannello (sfondo - scena)
  corners pose/ --board W H        4 spigoli 3D del pannello da piano + rettangolo noto
  pick    pose/ --camera cameraN   clic dei 4 spigoli nell'immagine -> corr_cameraN.csv
Poi:  python3 ../calib_pnp.py solve --intrinsics pose/cameraN_intrinsics.yaml \
          --corr pose/corr_cameraN.csv [--rslidar-xyz ... --rslidar-rpy ...]

Ordine degli spigoli (uguale in LiDAR e immagine): TL, TR, BR, BL come visti
dalla camera (sinistra immagine = +y LiDAR, alto = +z).
"""
import argparse
import glob
import os
import sqlite3

import cv2
import numpy as np
import yaml

BOARD_MIN_PTS = 150


# ------------------------------------------------------------------ geometria pannello

def fit_plane(P, rng, iters=200, tol=0.02):
    best = None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - s[0]) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    Q = P[best]
    c = Q.mean(0)
    n = np.linalg.svd(Q - c, full_matrices=False)[2][2]
    return c, n, best


def _plane_v(c, n, up, az, el):
    """Coordinata 'su' nel piano del punto dove il raggio (az, el) colpisce il piano."""
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    den = d @ n
    return np.nan if abs(den) < 1e-9 else float((((c @ n) / den) * d - c) @ up)


def _center_v(Q, c, n, up, ring_els, H, n_az=7):
    """Centro verticale del pannello (coordinata 'su') dagli intervalli tra anelli.

    Gli anelli NON sono equispaziati (fitti a -6 deg, radi sotto -10): il bordo
    vero sta tra l'ultimo anello che colpisce il pannello e il primo che lo manca.
    Con H noto, ogni lato vincola il centro a un intervallo; si intersecano i
    vincoli di sopra e di sotto su piu' azimut e si prende il punto medio."""
    az_p = np.arctan2(Q[:, 1], Q[:, 0])
    el_p = np.arctan2(Q[:, 2], np.hypot(Q[:, 0], Q[:, 1]))
    ring_of = np.argmin(np.abs(el_p[:, None] - ring_els[None, :]), axis=1)
    lo_b, hi_b = -np.inf, np.inf
    mids = []
    for a in np.percentile(az_p, np.linspace(20, 80, n_az)):
        near = np.abs(np.angle(np.exp(1j * (az_p - a)))) < np.radians(0.3)
        hits = np.unique(ring_of[near])
        if len(hits) < 3:
            continue
        v = np.array([_plane_v(c, n, up, a, e) for e in ring_els])
        order = np.argsort(v)
        pos = {k: i for i, k in enumerate(order)}
        hit_pos = sorted(pos[k] for k in hits)
        i_lo, i_hi = hit_pos[0], hit_pos[-1]
        v_lo_hit, v_hi_hit = v[order[i_lo]], v[order[i_hi]]
        v_lo_miss = v[order[i_lo - 1]] if i_lo > 0 else v_lo_hit - 0.2
        v_hi_miss = v[order[i_hi + 1]] if i_hi + 1 < len(order) else v_hi_hit + 0.2
        lo_a = max(v_lo_miss + H / 2, v_hi_hit - H / 2)
        hi_a = min(v_lo_hit + H / 2, v_hi_miss - H / 2)
        mids.append((lo_a + hi_a) / 2)
        lo_b, hi_b = max(lo_b, lo_a), min(hi_b, hi_a)
    if np.isfinite(lo_b) and np.isfinite(hi_b) and lo_b <= hi_b:
        return (lo_b + hi_b) / 2, hi_b - lo_b
    if mids:            # vincoli incompatibili (rumore/buchi): mediana dei centri per azimut
        return float(np.median(mids)), float(np.ptp(mids))
    return None, None


def board_corners(P, W, H, ring_els=None, rng=None):
    """Punti del pannello (Nx3, frame rslidar) -> 4 spigoli 3D [TL, TR, BR, BL].

    Assi nel piano: 'su' = proiezione di +z sul piano, 'destra' = su x normale
    orientata verso -y (destra dell'immagine per una camera che guarda +x).
    Orizzontale (azimut ogni 0.1 deg, fitto): rettangolo W centrato tra i bordi
    misurati. Verticale: intervalli tra anelli (_center_v) se ring_els e' noto,
    altrimenti centratura tra i bordi misurati (errore fino a mezzo passo anelli).
    """
    rng = rng or np.random.default_rng(0)
    c, n, inl = fit_plane(P, rng)
    Q = P[inl]
    if n[0] > 0:                       # normale verso il sensore
        n = -n
    up = np.array([0.0, 0.0, 1.0]) - n[2] * n
    up /= np.linalg.norm(up)
    right = np.cross(up, n)
    if right[1] > 0:
        right = -right
    uv = np.stack([(Q - c) @ right, (Q - c) @ up], 1)
    lo = np.percentile(uv, 0.5, axis=0)
    hi = np.percentile(uv, 99.5, axis=0)
    mid = (lo + hi) / 2
    meas = hi - lo
    v_width = None
    if ring_els is not None:
        cv_, v_width = _center_v(Q, c, n, up, np.asarray(ring_els), H)
        if cv_ is not None:
            mid[1] = cv_
    half = np.array([W, H]) / 2
    corners2 = mid + np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * half
    X = c + corners2[:, :1] * right + corners2[:, 1:] * up
    return X, {'measured_wh': meas.tolist(), 'plane_rms': float(np.sqrt(np.mean(((Q - c) @ n) ** 2))),
               'n_inliers': int(inl.sum()),
               'v_uncertainty_m': None if v_width is None else float(v_width)}


# ------------------------------------------------------------------ detect

def detect(args):
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    from extract_frames import Bag, closest_image, intrinsics, stamp_ns, to_rgb

    os.makedirs(args.out, exist_ok=True)
    bag = Bag(args.db)
    intrinsics(bag, '/camera1/camera_info', os.path.join(args.out, 'camera1_intrinsics.yaml'))
    intrinsics(bag, '/camera2/camera_info', os.path.join(args.out, 'camera2_intrinsics.yaml'))
    rows = bag.cur.execute('SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp',
                           (bag.ids['/rslidar_points'],)).fetchall()

    bg, bg_stack, t_first = None, [], None
    cur_pose, poses = None, []
    for (ts,) in rows:
        (data,) = bag.cur.execute('SELECT data FROM messages WHERE topic_id=? AND timestamp=?',
                                  (bag.ids['/rslidar_points'], ts)).fetchone()
        cl = deserialize_message(data, PointCloud2)
        arr = np.frombuffer(cl.data, np.float32).reshape(cl.height, cl.width, 4)
        st = stamp_ns(cl)
        t_first = t_first or st
        with np.errstate(invalid='ignore'):
            r = np.linalg.norm(arr[..., :3], axis=2)
        if (st - t_first) / 1e9 < args.bg_seconds:
            bg_stack.append(r)
            continue
        if bg is None:
            bg = np.nanmedian(np.stack(bg_stack), axis=0)
            with np.errstate(invalid='ignore'):
                ring_els = np.sort(np.nanmedian(np.arctan2(arr[..., 2], np.hypot(arr[..., 0], arr[..., 1])), axis=0))
            print(f'sfondo da {len(bg_stack)} nuvole')
        with np.errstate(invalid='ignore'):
            fg = np.isfinite(r) & (r > args.rmin) & (r < args.rmax) & \
                 (~np.isfinite(bg) | (bg - r > args.fg_margin))
        P = arr[..., :3][fg].astype(np.float64)
        ok = False
        if len(P) >= BOARD_MIN_PTS:
            # cluster principale: punti entro 0.6 m dal mediano
            med = np.median(P, 0)
            P = P[np.linalg.norm(P - med, axis=1) < 0.6]
            ok = len(P) >= BOARD_MIN_PTS
        cen = np.median(P, 0) if ok else None
        if ok and cur_pose is not None and np.linalg.norm(cen - cur_pose['cen']) < args.still_m:
            cur_pose['members'].append((st, ts, P))
            continue
        if cur_pose is not None:
            dur = (cur_pose['members'][-1][0] - cur_pose['members'][0][0]) / 1e9
            if dur >= args.min_hold:
                poses.append(cur_pose)
        cur_pose = {'cen': cen, 'members': [(st, ts, P)]} if ok else None
    if cur_pose is not None and (cur_pose['members'][-1][0] - cur_pose['members'][0][0]) / 1e9 >= args.min_hold:
        poses.append(cur_pose)

    for k, pz in enumerate(poses):
        st, ts, P = pz['members'][len(pz['members']) // 2]
        allP = np.concatenate([m[2] for m in pz['members']])      # piu' nuvole = spigoli piu' stabili
        c1 = closest_image(bag, '/camera1/image_raw', st, st)
        c2 = closest_image(bag, '/camera2/image_raw', st, st)
        np.savez_compressed(os.path.join(args.out, f'pose_{k:02d}.npz'), board=allP, ring_els=ring_els,
                            img1=to_rgb(c1[1]), img2=to_rgb(c2[1]), t_rel=(st - t_first) / 1e9)
        print(f'pose_{k:02d}: t={(st - t_first) / 1e9:6.1f}s  nuvole {len(pz["members"])}  '
              f'punti {len(allP)}  centro ({pz["cen"][0]:.2f},{pz["cen"][1]:.2f},{pz["cen"][2]:.2f}) m')
    print(f'\n{len(poses)} pose salvate in {args.out}')


# ------------------------------------------------------------------ corners

def corners(args):
    out = {}
    for f in sorted(glob.glob(os.path.join(args.poses, 'pose_*.npz'))):
        z = np.load(f)
        X, info = board_corners(z['board'], *args.board, ring_els=z['ring_els'])
        name = os.path.basename(f)[:-4]
        out[name] = {'corners_TL_TR_BR_BL': X.tolist(), **info}
        dw, dh = np.array(info['measured_wh']) - np.array(args.board)
        flag = '' if abs(dw) < 0.05 and abs(dh) < 0.08 else '  <-- dimensioni lontane dal pannello: controlla'
        print(f'{name}: misurato {info["measured_wh"][0]:.2f} x {info["measured_wh"][1]:.2f} m, '
              f'piano rms {info["plane_rms"] * 100:.1f} cm, inlier {info["n_inliers"]}{flag}')
    yaml.safe_dump(out, open(os.path.join(args.poses, 'corners_lidar.yaml'), 'w'))
    print(f'-> {args.poses}/corners_lidar.yaml')


# ------------------------------------------------------------------ pick

def pick(args):
    key = {'camera1': 'img1', 'camera2': 'img2'}[args.camera]
    C = yaml.safe_load(open(os.path.join(args.poses, 'corners_lidar.yaml')))
    lut = ((np.arange(256) / 255.0) ** (1 / args.gamma) * 255 + 0.5).astype(np.uint8)
    rows = []
    for name in sorted(C):
        img = cv2.cvtColor(lut[np.load(os.path.join(args.poses, name + '.npz'))[key]], cv2.COLOR_RGB2BGR)
        disp, pts = img.copy(), []
        win = f'{name} {args.camera}: clicca TL, TR, BR, BL (s = salta posa, q = esci)'

        def cb(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((x, y))
                cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(disp, 'TL TR BR BL'.split()[len(pts) - 1], (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, cb)
        k = 0
        while len(pts) < 4 and k not in (ord('s'), ord('q')):
            cv2.imshow(win, disp)
            k = cv2.waitKey(20) & 0xFF
        cv2.destroyWindow(win)
        if k == ord('q'):
            break
        if len(pts) < 4:
            continue
        for (u, v), X in zip(pts, C[name]['corners_TL_TR_BR_BL']):
            rows.append(f'{u},{v},{X[0]:.4f},{X[1]:.4f},{X[2]:.4f}')
    path = os.path.join(args.poses, f'corr_{args.camera}.csv')
    open(path, 'w').write('# u,v,X,Y,Z (rslidar)  generato da calib_target.py pick\n' + '\n'.join(rows) + '\n')
    print(f'{len(rows)} corrispondenze -> {path}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('detect')
    d.add_argument('db')
    d.add_argument('--out', required=True)
    d.add_argument('--bg-seconds', type=float, default=4.0)
    d.add_argument('--fg-margin', type=float, default=0.3, help='[m] piu vicino dello sfondo')
    d.add_argument('--rmin', type=float, default=1.5)
    d.add_argument('--rmax', type=float, default=6.0)
    d.add_argument('--still-m', type=float, default=0.03, help='[m] spostamento max del centro = posa ferma')
    d.add_argument('--min-hold', type=float, default=2.0, help='[s]')
    c = sub.add_parser('corners')
    c.add_argument('poses')
    c.add_argument('--board', nargs=2, type=float, required=True, metavar=('W', 'H'))
    p = sub.add_parser('pick')
    p.add_argument('poses')
    p.add_argument('--camera', choices=['camera1', 'camera2'], required=True)
    p.add_argument('--gamma', type=float, default=1.0)
    a = ap.parse_args()
    {'detect': detect, 'corners': corners, 'pick': pick}[a.cmd](a)


if __name__ == '__main__':
    main()
