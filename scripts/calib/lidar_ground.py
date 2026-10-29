#!/usr/bin/env python3
"""Roll/pitch/altezza del LiDAR rispetto al terreno -> joint base_footprint->rslidar.

Per ogni frame (robot fermo) si stima con RANSAC + SVD il piano del terreno
entro 2-8 m, normale n nel frame rslidar. Con R_base_rslidar = Ry(p)·Rx(r)
(yaw non osservabile dal terreno: resta quello del xacro) vale
    n = (-sin p, sin r cos p, cos r cos p)  ->  p = -asin(nx), r = atan2(ny, nz)
e l'altezza e' la distanza origine-piano. Si prende la MEDIANA sui luoghi
distinti, cosi' un tratto in pendenza non sposta il risultato.
"""
import argparse
import glob

import numpy as np


def fit_ground(xyz, rng, iters=300, tol=0.04):
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    P = xyz[(r > 2.0) & (r < 8.0) & (xyz[:, 2] < -0.5)]
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
    n = np.linalg.svd(Q - c, full_matrices=False)[2][2]   # NON full_matrices: U sarebbe NxN
    n = n if n[2] > 0 else -n
    return n, float(-(n @ c)), best.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default='frames_static')
    ap.add_argument('--xy', nargs=2, type=float, default=[-0.21, 0.0], help='x y del xacro (non osservabili)')
    ap.add_argument('--gap', type=float, default=20.0, help='[s] frame piu vicini = stesso luogo')
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    places = []          # (t, [(r, p, h), ...])
    for f in sorted(glob.glob(f'{args.frames}/frame_*.npz')):
        z = np.load(f)
        xyz = z['cloud'][..., :3].reshape(-1, 3).astype(np.float64)
        xyz = xyz[np.isfinite(xyz).all(1)]
        n, h, frac = fit_ground(xyz, rng)
        r, p = float(np.arctan2(n[1], n[2])), float(-np.arcsin(n[0]))
        t = float(z['t_rel'])
        if not places or t - places[-1][0] > args.gap:
            places.append((t, []))
        places[-1][1].append((r, p, h))
        print(f'{f[-14:-4]} t={t:7.1f}s  roll {np.degrees(r):+.2f}  pitch {np.degrees(p):+.2f} deg  '
              f'h {h:.3f} m  inlier {frac:.0%}')

    per_place = np.array([np.median(np.array(v), axis=0) for _, v in places])
    r, p, h = np.median(per_place, axis=0)
    spread = np.degrees(np.median(np.abs(per_place[:, :2] - [r, p]), axis=0))
    print(f'\n{len(places)} luoghi distinti. Mediana: roll {np.degrees(r):+.2f} deg, pitch {np.degrees(p):+.2f} deg, '
          f'altezza {h:.3f} m  (MAD tra luoghi: roll {spread[0]:.2f}, pitch {spread[1]:.2f} deg)')
    print('\n=== Joint URDF  base_footprint -> rslidar ===')
    print(f'  <origin xyz="{args.xy[0]:.3f} {args.xy[1]:.3f} {h:.3f}" rpy="{r:.4f} {p:.4f} 0"/>')
    print(f'  (argomenti per calib_edges.py: --rslidar-xyz {args.xy[0]} {args.xy[1]} {h:.3f} '
          f'--rslidar-rpy {r:.4f} {p:.4f} 0)')


if __name__ == '__main__':
    main()
