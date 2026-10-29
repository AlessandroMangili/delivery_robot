#!/usr/bin/env python3
"""Test sintetico di calib_target.board_corners + solvePnP (nessun bag necessario).

Pannelli W x H campionati come li vede il LiDAR reale (elevazioni degli anelli
prese da una nuvola vera, azimut ogni 0.1 deg, rumore di range 2 cm), spigoli
immagine = proiezione vera + rumore di clic. Verifica:
  * errore degli spigoli 3D stimati
  * errore della posa camera<-LiDAR recuperata con solvePnP
"""
import glob
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, '..')
from calib_pnp import Rt_to_T, invT, rpy_to_R, solve_extrinsic  # noqa: E402
from calib_target import board_corners  # noqa: E402

W, H = 0.60, 0.80
RANGE_NOISE = 0.02
CLICK_NOISE_PX = 1.5


def ring_elevations():
    c = np.load(sorted(glob.glob('frames_static/frame_*.npz'))[0])['cloud'].astype(np.float64)
    with np.errstate(invalid='ignore'):
        el = np.arctan2(c[..., 2], np.hypot(c[..., 0], c[..., 1]))
    return np.sort(np.nanmedian(el, axis=0))


def sample_board(center, yaw, els, rng):
    """Punti LiDAR su un pannello verticale di centro 'center', ruotato di yaw attorno a z."""
    n = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])          # normale verso il sensore
    up = np.array([0, 0, 1.0])
    right = np.cross(up, n)
    if right[1] > 0:
        right = -right
    az = np.radians(np.arange(-60, 60, 0.1))
    A, E = np.meshgrid(az, els)
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    t = (center @ n) / (d @ n)
    X = d * t[:, None]
    loc = X - center
    inside = (t > 0) & (np.abs(loc @ right) <= W / 2) & (np.abs(loc @ up) <= H / 2)
    X, dd = X[inside], d[inside]
    X += dd * rng.normal(0, RANGE_NOISE, len(X))[:, None]
    true = center + np.array([[-W / 2, H / 2], [W / 2, H / 2], [W / 2, -H / 2], [-W / 2, -H / 2]]) @ np.stack([right, up])
    return X, true


def main():
    rng = np.random.default_rng(1)
    els = ring_elevations()
    # 8 pose: 2.2-4 m, a destra/sinistra, bordo basso ~10 cm da terra (LiDAR a 0.99 m)
    poses = [((2.5, 0.8, -0.49), 0.4), ((2.5, -0.8, -0.49), -0.4), ((3.2, 0.0, -0.49), 0.1),
             ((3.8, 1.1, -0.49), 0.6), ((3.8, -1.1, -0.49), -0.5), ((2.2, 0.3, -0.49), -0.2),
             ((3.0, -0.4, -0.40), 0.3), ((3.5, 0.5, -0.45), -0.3)]
    obj, true_all, errs = [], [], []
    for cen, yaw in poses:
        P, true = sample_board(np.array(cen), yaw, els, rng)
        est, info = board_corners(P, W, H, ring_els=els)
        e = np.linalg.norm(est - true, axis=1)
        errs.append(e)
        vu = info['v_uncertainty_m']
        print(f'posa c={cen} yaw={np.degrees(yaw):+5.1f}: {len(P):4d} punti, misurato '
              f'{info["measured_wh"][0]:.3f}x{info["measured_wh"][1]:.3f}, errore spigoli '
              f'{e.mean() * 100:.1f} cm (max {e.max() * 100:.1f}), intervallo verticale '
              f'{"n/d" if vu is None else f"{vu * 100:.1f} cm"}')
        obj.append(est)
        true_all.append(true)
    obj, true_all = np.concatenate(obj), np.concatenate(true_all)

    for cam, link_xyz, link_rpy in (('camera1', (-0.10, 0, 0.83), (0.02, -0.10, 0.03)),
                                     ('camera2', (-0.085, -0.07, 0.80), (0.01, 0.36, -0.02))):
        d = yaml.safe_load(open(f'frames_static/{cam}_intrinsics.yaml'))
        K = np.array(d['camera_matrix']['data']).reshape(3, 3)
        D = np.array(d['distortion_coefficients']['data'])
        T_base_lidar = Rt_to_T(rpy_to_R(0.0374, 0.0127, 0), [-0.21, 0, 0.989])
        T_link_opt = Rt_to_T(rpy_to_R(-1.5708, 0, -1.5708), [0, 0, 0])
        T_true = invT(Rt_to_T(rpy_to_R(*link_rpy), link_xyz) @ T_link_opt) @ T_base_lidar
        rv, _ = cv2.Rodrigues(T_true[:3, :3])
        px, _ = cv2.projectPoints(true_all, rv, T_true[:3, 3], K, D)
        px = px.reshape(-1, 2)
        w, h = d['image_width'], d['image_height']
        vis = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        px = px + rng.normal(0, CLICK_NOISE_PX, px.shape)
        res = solve_extrinsic(obj[vis], px[vis], K, D, T_base_lidar, T_link_opt[:3, :3])
        r, t = res['rvec'], res['tvec']
        R, _ = cv2.Rodrigues(r)
        dR = np.degrees(np.linalg.norm(cv2.Rodrigues(R @ T_true[:3, :3].T)[0]))
        dt = np.linalg.norm(t.ravel() - T_true[:3, 3]) * 100
        rep = np.linalg.norm(cv2.projectPoints(obj[vis], r, t, K, D)[0].reshape(-1, 2) - px[vis], axis=1)
        print(f'{cam}: {vis.sum()}/{len(vis)} spigoli visibili | errore posa: rotazione {dR:.3f} deg, '
              f'traslazione {dt:.1f} cm | riproiezione media {rep.mean():.2f} px')


if __name__ == '__main__':
    main()
