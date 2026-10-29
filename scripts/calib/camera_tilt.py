#!/usr/bin/env python3
"""Inclinazione di una camera rispetto alla VERTICALE e zona cieca a terra.

Metodo indipendente dal LiDAR: gli spigoli verticali della scena (edifici, pali)
nell'immagine non distorta convergono nel punto di fuga verticale V. La sua
direzione K^-1 V e' la verticale nel frame ottico -> pitch e roll della camera.
Il raggio del bordo basso dell'immagine forma un angolo a sotto l'orizzontale:
il terreno (piano) e' visibile da  d = h / tan(a).

  python3 camera_tilt.py --frames frames_static --camera camera1 --height 0.83
"""
import argparse
import glob

import cv2
import numpy as np
import yaml


def vertical_vp(gray, rng, min_len=80, max_tilt_deg=20):
    lsd = cv2.createLineSegmentDetector()
    segs = lsd.detect(gray)[0]
    if segs is None:
        return None, 0
    segs = segs.reshape(-1, 4)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    length = np.hypot(dx, dy)
    tilt = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
    segs = segs[(length > min_len) & (tilt < max_tilt_deg)]
    if len(segs) < 8:
        return None, len(segs)
    p1 = np.c_[segs[:, :2], np.ones(len(segs))]
    p2 = np.c_[segs[:, 2:], np.ones(len(segs))]
    L = np.cross(p1, p2)
    L /= np.linalg.norm(L[:, :2], axis=1, keepdims=True)       # distanza punto-retta in pixel
    best = None
    for _ in range(2000):                                       # RANSAC: alberi/texture sono outlier
        i, j = rng.choice(len(L), 2, replace=False)
        v = np.cross(L[i], L[j])
        if abs(v[2]) < 1e-12:
            v = v / np.linalg.norm(v)                           # punto all'infinito
        # errore angolare: angolo tra il segmento e la retta dal suo centro a V
        mid = (p1 + p2) / 2
        dirs = v[None, :2] - mid[:, :2] * v[2]
        seg_d = p2[:, :2] - p1[:, :2]
        cosang = np.abs(np.sum(dirs * seg_d, 1)) / (np.linalg.norm(dirs, axis=1) * np.linalg.norm(seg_d, axis=1) + 1e-12)
        inl = cosang > np.cos(np.radians(0.5))
        if best is None or inl.sum() > best.sum():
            best = inl
    _, _, vt = np.linalg.svd(L[best])
    return vt[-1], int(best.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='frames_static')
    ap.add_argument('--camera', default='camera1')
    ap.add_argument('--height', type=float, default=0.83, help='altezza centro ottico da terra [m]')
    ap.add_argument('--cam-x', type=float, default=-0.10, help='x della camera rispetto a base_footprint [m]')
    a = ap.parse_args()
    key = {'camera1': 'img1', 'camera2': 'img2'}[a.camera]
    d = yaml.safe_load(open(f'{a.frames}/{a.camera}_intrinsics.yaml'))
    K = np.array(d['camera_matrix']['data']).reshape(3, 3)
    D = np.array(d['distortion_coefficients']['data'])
    rng = np.random.default_rng(0)
    lut = ((np.arange(256) / 255.0) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8)
    res, seen = [], []
    for f in sorted(glob.glob(f'{a.frames}/frame_*.npz')):
        rgb = np.load(f)[key]
        t = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (64, 48)).astype(np.float32).ravel()
        t = (t - t.mean()) / (t.std() + 1e-6)
        if any(float(t @ s) / len(t) > 0.85 for s in seen):
            continue                                            # stesso luogo gia' misurato
        seen.append(t)
        h, w = rgb.shape[:2]
        und = cv2.undistort(cv2.cvtColor(lut[rgb], cv2.COLOR_RGB2GRAY), K, D)
        v, n = vertical_vp(und, rng)
        if v is None or n < 8:
            print(f'{f[-14:-4]}: troppo pochi spigoli verticali ({n})')
            continue
        up = np.linalg.solve(K, v)
        up /= np.linalg.norm(up)
        if up[1] > 0:                      # frame ottico: y verso il basso -> 'su' ha y < 0
            up = -up
        pitch_up = np.degrees(np.arcsin(up[2]))                 # asse ottico sopra l'orizzontale
        roll = np.degrees(np.arctan2(up[0], -up[1]))
        ray = np.linalg.solve(K, [K[0, 2], h - 1, 1.0])         # centro del bordo basso
        ray /= np.linalg.norm(ray)
        below = np.degrees(np.arcsin(-(ray @ up)))              # angolo sotto l'orizzontale
        dist = a.height / np.tan(np.radians(below)) if below > 0 else np.inf
        res.append((pitch_up, roll, dist))
        print(f'{f[-14:-4]}: {n:3d} spigoli verticali | asse ottico {pitch_up:+.2f} deg sopra '
              f"l'orizzontale, roll {roll:+.2f} deg | terreno visibile da {dist:.2f} m davanti alla camera")
    if res:
        p, r, dd = np.median(np.array(res), axis=0)
        print(f'\nMEDIANA su {len(res)} luoghi: pitch {p:+.2f} deg (positivo = guarda in su), roll {r:+.2f} deg')
        print(f'zona cieca a terra: {dd:.2f} m dalla camera = {dd + a.cam_x:.2f} m dal centro robot '
              f'(base_footprint) = {dd + a.cam_x - 0.37:.2f} m davanti all\'asse delle ruote anteriori')


if __name__ == '__main__':
    main()
