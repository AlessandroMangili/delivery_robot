#!/usr/bin/env python3
"""Grafici dalla traccia CSV dello SpatioTemporalCritic.

Uso:
    python3 plot_stc.py /tmp/stc.csv
    python3 plot_stc.py /tmp/stc.csv --out grafici_run1.png

Il CSV lo produce il critic quando gli si passa `diag_csv_path`. Contiene una
riga per OGNI ciclo di controllo, quindi i grafici hanno la risoluzione piena
del controllore (10 Hz) e non quella del log (una ogni diag_period_calls).

Colonne: t, tracce, contatti, coppie, liberi, batch, tau_min, tau_max,
         overlap, ttc, pen_max, cost_max
"""

import argparse
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')          # nessun display: si salva su file
import matplotlib.pyplot as plt


def leggi(path):
    """CSV -> dict di array numpy. Salta le righe malformate senza morire."""
    colonne = None
    righe = []
    with open(path, newline='') as f:
        for r in csv.reader(f):
            if not r:
                continue
            if colonne is None:
                colonne = [c.strip() for c in r]
                continue
            if len(r) != len(colonne):
                continue
            try:
                righe.append([float(v) for v in r])
            except ValueError:
                continue
    if not righe:
        sys.exit(f'nessuna riga valida in {path}')
    a = np.asarray(righe, dtype=np.float64)
    d = {c: a[:, i] for i, c in enumerate(colonne)}
    d['t'] = d['t'] - d['t'][0]          # tempo relativo all'inizio del run
    return d


def media_per_contatto(somma, contatti):
    """Le colonne overlap e ttc sono SOMME su tutte le coppie in contatto.
    Per leggerle serve dividerle: da sole il loro valore dipende da quanti
    campioni collidono, non da quanto e' grave la collisione."""
    out = np.zeros_like(somma)
    ok = contatti > 0
    out[ok] = somma[ok] / contatti[ok]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--out', default=None, help='PNG di uscita')
    ap.add_argument('--orizzonte', type=float, default=3.0,
                    help='prediction_horizon_s, disegnato come riferimento')
    args = ap.parse_args()

    d = leggi(args.csv)
    t = d['t']
    out = args.out or (args.csv.rsplit('.', 1)[0] + '.png')

    fig, ax = plt.subplots(4, 1, figsize=(12, 13), sharex=True)

    # ---- 1. tau: quando arriva il contatto ----
    attivo = d['tau_min'] >= 0.0
    ax[0].axhline(args.orizzonte, color='0.6', ls='--', lw=1,
                  label=f'orizzonte {args.orizzonte:.0f} s')
    ax[0].plot(t[attivo], d['tau_min'][attivo], '.', ms=4, color='#c0392b',
               label='tau_min (contatto piu\' imminente)')
    ax[0].plot(t[attivo], d['tau_max'][attivo], '.', ms=4, color='#2980b9',
               label='tau_max (contatto piu\' lontano)')
    ax[0].fill_between(t, 0, args.orizzonte, where=~attivo, color='#eafaf1',
                       label='nessun contatto')
    ax[0].set_ylabel('tau [s]')
    ax[0].set_title('Tempo al contatto: quando le traiettorie campionate '
                    'incontrano un pedone')
    ax[0].set_ylim(bottom=0)
    ax[0].legend(loc='upper right', fontsize=8)
    ax[0].grid(alpha=0.3)

    # ---- 2. i due termini di costo, NORMALIZZATI per contatto ----
    ovl = media_per_contatto(d['overlap'], d['contatti'])
    ttc = media_per_contatto(d['ttc'], d['contatti'])
    ax[1].plot(t, ovl, lw=1.2, color='#8e44ad', label='overlap / contatto')
    ax[1].plot(t, ttc, lw=1.2, color='#e67e22', label='ttc / contatto')
    ax[1].set_ylabel('costo medio per campione')
    ax[1].set_title('I due termini a confronto: chi domina, e quando')
    ax[1].legend(loc='upper right', fontsize=8)
    ax[1].grid(alpha=0.3)

    # ---- 3. margine di manovra ----
    frazione = np.zeros_like(t)
    ok = d['batch'] > 0
    frazione[ok] = 100.0 * d['liberi'][ok] / d['batch'][ok]
    ax[2].plot(t, frazione, lw=1.2, color='#16a085')
    ax[2].axhline(0, color='#c0392b', lw=1, ls='--')
    ax[2].set_ylabel('campioni liberi [%]')
    ax[2].set_ylim(-3, 103)
    ax[2].set_title('Margine di manovra: percentuale di traiettorie SENZA '
                    'collisione (a 0% MPPI non ha vie d\'uscita)')
    ax[2].grid(alpha=0.3)

    # ---- 4. peso del critic sul costo totale ----
    peso = np.zeros_like(t)
    ok = d['cost_max'] > 1e-9
    peso[ok] = 100.0 * d['pen_max'][ok] / d['cost_max'][ok]
    ax[3].plot(t, peso, lw=1.2, color='#34495e')
    ax[3].set_ylabel('penalita\' / costo tot [%]')
    ax[3].set_xlabel('tempo [s]')
    ax[3].set_title('Quanto pesa questo critic rispetto a tutti gli altri')
    ax[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f'grafico salvato in {out}')

    # ---- riepilogo numerico ----
    n = len(t)
    con_tracce = int((d['tracce'] > 0).sum())
    con_contatti = int((d['contatti'] > 0).sum())
    print()
    print(f'cicli totali              {n}   ({t[-1]:.1f} s)')
    print(f'cicli con almeno 1 traccia {con_tracce:5d}  ({100*con_tracce/n:.1f}%)')
    print(f'cicli con almeno 1 contatto{con_contatti:5d}  ({100*con_contatti/n:.1f}%)')
    if attivo.any():
        print(f'tau_min: minimo {d["tau_min"][attivo].min():.2f} s   '
              f'mediano {np.median(d["tau_min"][attivo]):.2f} s')
    peggiore = frazione.min()
    print(f'margine di manovra: minimo {peggiore:.1f}% di campioni liberi')
    if peggiore <= 0.0:
        print('  ATTENZIONE: in almeno un ciclo TUTTE le traiettorie erano in '
              'collisione. In quei cicli MPPI non aveva alcuna via d\'uscita.')


if __name__ == '__main__':
    main()
