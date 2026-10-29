#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.patches
import yaml

WS = os.path.expanduser('~/Desktop/thesis_ws')
sys.path.insert(0, os.path.join(WS, 'src/delivery_robot/semantic/semantic'))

CLASSI = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
          'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
          'motorcycle', 'bicycle']
COLORI = np.array([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                   [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                   [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
                   [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=np.float32) / 255.0
IGNOTO = 0.93


def grigi(valore, nota, lo, hi):
    g = 1.0 - np.clip((valore - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img = np.repeat(g[..., None], 3, axis=2).astype(np.float32)
    img[~nota] = IGNOTO
    return img


def estensione(z, shape):
    res, gox, goy = float(z['res']), float(z['gox']), float(z['goy'])
    return [gox, gox + shape[1] * res, goy, goy + shape[0] * res]


def semantica(z, ax, plt):
    known, cls, grid = z['known'], z['cls'], z['grid']
    ext = estensione(z, cls.shape)
    img = np.full(cls.shape + (3,), 0.85, dtype=np.float32)
    img[known] = COLORI[np.clip(cls[known], 0, 18)]
    ax[0].imshow(img, origin='lower', extent=ext, interpolation='nearest')
    ax[0].set_title('classe per cella (grigio = ignota)')
    presenti = np.bincount(cls[known].ravel(), minlength=19)
    voci = [matplotlib.patches.Patch(color=COLORI[i], label=f'{CLASSI[i]} {presenti[i]}')
            for i in np.argsort(-presenti) if presenti[i] > 0][:10]
    ax[0].legend(handles=voci, fontsize=7, loc='upper right')
    costo = np.where(known, grid, np.nan)
    im = ax[1].imshow(costo, origin='lower', extent=ext, cmap='RdYlGn_r', vmin=0, vmax=100,
                      interpolation='nearest')
    ax[1].set_title(f'costo 0-100 ({int((known & (grid >= 100)).sum())} celle letali)')
    plt.colorbar(im, ax=ax[1], fraction=0.035)
    tot = int(known.sum())
    print(f'celle note {tot}, area {tot * float(z["res"]) ** 2:.0f} m2')
    for i in np.argsort(-presenti):
        if presenti[i]:
            print(f'  {CLASSI[i]:14s} {presenti[i]:8d}  {100 * presenti[i] / tot:5.1f}%')


def elevazione(z, ax, plt):
    import elevation_costmap_node as E
    Y = yaml.safe_load(open(os.path.join(WS, 'src/delivery_robot/semantic/config/semantic.yaml')))['elevation_costmap_node']['ros__parameters']
    h, n = z['h'].astype(np.float32), z['n']
    ext = estensione(z, h.shape)
    nota = (n > 0) & (z['var_min'] < Y['min_var_for_valid']) if 'var_min' in z.files else n > 0
    lo, hi = np.percentile(h[nota], [2, 98])
    ax[0].imshow(grigi(h, nota, lo, hi), origin='lower', extent=ext, interpolation='nearest')
    ax[0].set_title(f'quota: bianco {lo:+.2f} m, nero {hi:+.2f} m')
    cp = {k: Y[k] for k in ['median_ksize', 'rough_win', 'rough_min_known', 'slope_crit', 'rough_crit',
                            'step_crit', 'slope_dead', 'rough_dead', 'step_dead']}
    cp.update(step_drop_r=Y['step_drop_radius_m'], step_rise_r=Y['step_rise_radius_m'], step_allow=Y['step_slope_allow'])
    costo = E.derive_cost(h, nota, float(z['res']), **cp).astype(np.float32)
    ok = nota & (costo >= 0)
    ax[1].imshow(grigi(costo, ok, 0, 100), origin='lower', extent=ext, interpolation='nearest')
    ax[1].set_title(f'costo come nel nodo: bianco 0, nero 100 (>= 50 nel {100 * np.mean(costo[ok] >= 50):.1f}%)')
    print(f'celle note {int(nota.sum())}, quota da {lo:.2f} a {hi:.2f} m (2-98%), costo >= 50 nel {100 * np.mean(costo[ok] >= 50):.1f}%')


def main():
    p = argparse.ArgumentParser(description='Mostra una mappa semantica_*.npz o elevazione_*.npz')
    p.add_argument('file')
    p.add_argument('--salva', help='salva in PNG invece di aprire la finestra')
    a = p.parse_args()
    if a.salva:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = np.load(a.file, allow_pickle=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    if 'cls' in z.files:
        semantica(z, ax, plt)
    elif 'h' in z.files:
        elevazione(z, ax, plt)
    else:
        sys.exit(f'formato sconosciuto: {z.files}')
    for x in ax:
        x.set_aspect('equal')
        x.set_xlabel('x [m] (frame map)')
    fig.suptitle(os.path.basename(a.file))
    if a.salva:
        fig.savefig(a.salva, dpi=120)
        print('salvato', a.salva)
    else:
        plt.show()


if __name__ == '__main__':
    main()
