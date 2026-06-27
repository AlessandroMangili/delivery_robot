#!/usr/bin/env python3
"""
Visualizza per intero una mappa semantica salvata in .npz dal semantic_costmap_node.
Mostra: la griglia dei costi (con -1 = unknown in grigio) e, opzionale, la confidenza.

USO:
  python3 view_semantic_map.py /percorso/della/mappa.npz
  python3 view_semantic_map.py mappa.npz --save out.png      # salva invece di mostrare
  python3 view_semantic_map.py mappa.npz --conf              # mostra anche la confidenza
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--save', default=None, help='salva su file invece di mostrare')
    ap.add_argument('--conf', action='store_true', help='mostra anche la confidenza')
    args = ap.parse_args()

    d = np.load(args.npz)
    grid = d['grid'].astype(np.float32)          # costi, -1 = unknown
    conf = d['conf'].astype(np.float32) if 'conf' in d else None
    res = float(d['res']) if 'res' in d else 0.05
    gox = float(d['gox']) if 'gox' in d else 0.0
    goy = float(d['goy']) if 'goy' in d else 0.0

    gny, gnx = grid.shape
    # extent in coordinate mondo: [x_min, x_max, y_min, y_max]
    extent = [gox, gox + gnx * res, goy, goy + gny * res]

    # mappa di costo: -1 (unknown) -> mostrata a parte in grigio; 0..100 con colormap
    masked = np.ma.masked_where(grid < 0, grid)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='#3a3a3a')                 # unknown = grigio scuro

    n = 2 if args.conf and conf is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7), squeeze=False)

    ax = axes[0][0]
    im = ax.imshow(masked, origin='lower', extent=extent, cmap=cmap,
                   vmin=0, vmax=100, interpolation='nearest')
    ax.set_title(f'Mappa semantica (costi)  {gnx}x{gny}  res={res} m')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    fig.colorbar(im, ax=ax, label='costo (0=libero ... 100=ostacolo)', shrink=0.8)

    if n == 2:
        ax2 = axes[0][1]
        im2 = ax2.imshow(conf, origin='lower', extent=extent, cmap='magma',
                         interpolation='nearest')
        ax2.set_title('Confidenza')
        ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]')
        fig.colorbar(im2, ax=ax2, label='confidenza', shrink=0.8)

    # statistiche utili a terminale
    known = grid >= 0
    print(f'Griglia: {gnx}x{gny} celle, res={res} m')
    print(f'Estensione mondo: x[{extent[0]:.1f}, {extent[1]:.1f}]  '
          f'y[{extent[2]:.1f}, {extent[3]:.1f}]')
    print(f'Celle note: {known.sum()} / {grid.size} '
          f'({100.0 * known.sum() / grid.size:.1f}%)')
    if known.any():
        vals, cnts = np.unique(grid[known].astype(int), return_counts=True)
        print('Distribuzione costi (valore: n.celle):')
        for v, c in zip(vals, cnts):
            print(f'  {v:4d}: {c}')

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=100, bbox_inches='tight')
        print(f'Salvato in {args.save}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
