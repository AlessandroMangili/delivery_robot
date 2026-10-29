#!/usr/bin/env python3
"""calib_edges.py — calibrazione estrinseca camera<->LiDAR SENZA target.

Nessun punto da cliccare. Su N frame sincronizzati (extract_frames.py):
  * LiDAR: bordi di PROFONDITA' lungo ogni anello (salto di range tra campioni
    di azimut adiacenti); si tiene il punto in primo piano, peso sqrt(salto).
  * Immagine: schiarita (gamma), gradiente, bordi forti -> mappa di punteggio
    S = exp(-dist_bordo / sigma).
  * Obiettivo: J(T) = sum_i w_i * S(proj_T(p_i)) / sum_i w_i   (media su frame)
    massimizzato sulla posa T = camera_ottica <- rslidar (Levinson & Thrun 2013).

La rotazione e' ben osservabile (0.1 deg ~ 1 px a f=660), la traslazione poco
(1 cm ~ 1 px a 6 m): prima si ottimizza la sola rotazione (griglia grossolana +
Nelder-Mead coarse-to-fine), poi 6 DOF, e la ripetibilita' si verifica
ricalibrando su due meta' disgiunte dei frame. Se la traslazione non e'
ripetibile si tiene quella misurata (xacro).

Uscita: joint URDF base_footprint -> <camera>_link, errore di allineamento
bordi [px] prima/dopo, overlay di verifica, profili del costo per asse.
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from calib_pnp import R_to_rpy, Rt_to_T, invT, rpy_to_R  # noqa: E402

# pose del xacro (hunter_sensors.urdf.xacro) usate come stima iniziale
XACRO_LINK = {
    'camera1': ((-0.10, 0.0, 0.83), (0.0, 0.0, 0.0)),
    'camera2': ((-0.085, -0.07, 0.80), (0.0, 0.3491, 0.0)),
}
IMG_KEY = {'camera1': 'img1', 'camera2': 'img2'}
GRID_DEG, GRID_STEP = 5.0, 1.0
DT_KEY = {'camera1': 'dt1_ms', 'camera2': 'dt2_ms'}


# ---------------------------------------------------------------- geometria

def project(P, R, t, K, D):
    """Punti Nx3 (rslidar) -> (u, v, ok). Modello plumb_bob. ok = davanti alla
    camera e raggio normalizzato < 1.2 (oltre, il polinomio forte della camera2
    puo' ripiegarsi e rimandare dentro l'immagine punti che sono fuori FOV)."""
    Pc = P @ R.T + t
    z = Pc[:, 2]
    ok = z > 0.5
    zs = np.where(ok, z, 1.0)
    x, y = Pc[:, 0] / zs, Pc[:, 1] / zs
    r2 = x * x + y * y
    ok &= r2 < 1.44
    k1, k2, p1, p2 = D[0], D[1], D[2], D[3]
    k3 = D[4] if len(D) > 4 else 0.0
    rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2], ok


def perturb(T0, p):
    """p = [wx, wy, wz (deg), tx, ty, tz (cm)] applicata lato camera: T = E(p) @ T0."""
    Rp, _ = cv2.Rodrigues(np.radians(np.asarray(p[:3], np.float64)).reshape(3, 1))
    E = Rt_to_T(Rp, np.asarray(p[3:6], np.float64) / 100.0)
    return E @ T0


# ---------------------------------------------------------------- feature

SKY_RANGE = 200.0


def align_azimuth(xyz):
    """Riallinea le colonne (laser) su una griglia di azimut comune.

    Nella nuvola organizzata la stessa riga NON e' lo stesso azimut per tutti i
    laser: ognuno ha un offset proprio (su questo sensore da -5.9 a +5.6 deg).
    Con az[i,k] = off_k + step*i, la colonna k va ruotata di round(off_k/step)
    righe; solo dopo si possono confrontare anelli adiacenti."""
    A = xyz.shape[0]
    with np.errstate(invalid='ignore'):
        az = np.arctan2(xyz[..., 1], xyz[..., 0])
    rows = np.arange(A)
    k_ref = int(np.argmax(np.isfinite(az).sum(0)))
    v = np.isfinite(az[:, k_ref])
    i = rows[v]
    step = float(np.median(np.diff(np.unwrap(az[v, k_ref])) / np.diff(i)))
    out = np.empty_like(xyz)
    for k in range(xyz.shape[1]):
        v = np.isfinite(az[:, k])
        if v.sum() < 10:
            out[:, k] = xyz[:, k]
            continue
        off = np.angle(np.mean(np.exp(1j * (az[v, k] - step * rows[v]))))
        out[:, k] = np.roll(xyz[:, k], int(round(off / step)), axis=0)
    return out


def lidar_edges(cloud, rmin, rmax, jump_abs, jump_rel, sky_z=0.3, min_run=3):
    """Bordi di profondita' (punto in primo piano) su nuvola organizzata.

    cloud (A, R, 4): righe = azimut, colonne = laser (NON ordinati per elevazione:
    si riordinano). Vicini lungo l'azimut E tra anelli adiacenti (profili di
    tetti/siepi). Un vicino senza ritorno vale 'lontano' solo se il punto sta
    sopra sky_z (frame rslidar): li' l'assenza di ritorno e' cielo e da' le sagome
    di pali, chiome e tetti; in basso e' piu' spesso vetro/asfalto scuro/rumore.
    """
    xyz = align_azimuth(cloud[..., :3].astype(np.float64))
    with np.errstate(invalid='ignore', all='ignore'):
        el = np.arctan2(xyz[..., 2], np.hypot(xyz[..., 0], xyz[..., 1]))
        xyz = xyz[:, np.argsort(np.nanmedian(el, axis=0))]
        rng = np.linalg.norm(xyz, axis=2)
        valid = np.isfinite(rng)
        r0 = np.where(valid, rng, np.nan)
        # cielo = nessun ritorno in TUTTI gli anelli sopra (non basta il vicino:
        # vetrate e fogliame danno buchi isolati che non sono sagome)
        above_any = np.flip(np.cumsum(np.flip(valid, axis=1), axis=1), axis=1) - valid > 0
        sky_above = valid & (xyz[..., 2] > sky_z) & ~above_any
        thr = np.maximum(jump_abs, jump_rel * np.nan_to_num(r0, nan=0.0))
        d = np.zeros_like(rng)
        for axis, wrap in ((0, True), (1, False)):
            nbs = []
            for sh in (1, -1):
                nb = np.roll(r0, sh, axis=axis)
                if axis == 1 and sh == -1:          # vicino di elevazione SUPERIORE
                    nb = np.where(np.isfinite(nb), nb, np.where(sky_above, SKY_RANGE, np.nan))
                if not wrap:
                    nb[:, 0 if sh == 1 else -1] = np.nan
                nbs.append(nb - r0)
            a, b = nbs
            J = np.fmax(np.nan_to_num(a, nan=-np.inf), np.nan_to_num(b, nan=-np.inf))
            o = np.where(np.nan_to_num(a, nan=-np.inf) >= np.nan_to_num(b, nan=-np.inf), b, a)
            # occlusione vera: salto grande verso un lato e l'ALTRO lato non e' a sua
            # volta molto piu' vicino (su una superficie inclinata continua, es. il
            # terreno tra anelli, il range cresce da un lato e cala dall'altro)
            occl = (J > thr) & (np.nan_to_num(o, nan=0.0) > -0.3 * J)
            d = np.where(occl, np.fmax(d, J), d)
        E = (valid & (d > 0) & (r0 > rmin) & (r0 < rmax)).astype(np.uint8)
    # solo sagome CONTIGUE: >= min_run anelli consecutivi (pali, spigoli) o
    # >= 2*min_run azimut consecutivi (tetti). Vetrate e fogliame danno salti
    # sparsi (telaio/interno, foglia/muro) che nell'immagine non sono bordi.
    vert = cv2.morphologyEx(E, cv2.MORPH_OPEN, np.ones((1, min_run), np.uint8))
    horz = cv2.morphologyEx(E, cv2.MORPH_OPEN, np.ones((2 * min_run, 1), np.uint8))
    sel = (vert | horz).astype(bool)
    return xyz[sel], np.sqrt(np.minimum(d[sel], 3.0))


def gamma_lut(gamma):
    x = np.arange(256, dtype=np.float32) / 255.0
    return (np.power(x, 1.0 / gamma) * 255.0 + 0.5).astype(np.uint8)


def image_edges(rgb, gamma, mask_bottom, edge_pct, min_area=40):
    """Bordo binario (uint8) dell'immagine schiarita + maschera zone valide."""
    g = cv2.cvtColor(gamma_lut(gamma)[rgb], cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.5).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    h, w = g.shape
    valid = np.ones((h, w), bool)
    valid[int(h * (1.0 - mask_bottom)):, :] = False     # corpo del robot
    valid[:3, :] = valid[-3:, :] = valid[:, :3] = valid[:, -3:] = False
    thr = np.percentile(mag[valid], edge_pct)
    edges = ((mag > thr) & valid).astype(np.uint8)
    # via le componenti piccole: di notte sono rumore del sensore, non bordi
    n, lab, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab].astype(np.uint8), valid


class Frame:
    def __init__(self, path, cam, args, K, D, T0):
        f = np.load(path)
        self.name = os.path.basename(path)
        self.t_rel = float(f['t_rel'])
        self.rgb = f[IMG_KEY[cam]]
        if args.img_scale != 1.0:
            # a risoluzione ridotta spariscono le texture fini (malta dei mattoni)
            # e restano le sagome; K arriva gia' scalata da main()
            h0, w0 = self.rgb.shape[:2]
            self.rgb = cv2.resize(self.rgb, (round(w0 / args.img_scale), round(h0 / args.img_scale)),
                                  interpolation=cv2.INTER_AREA)
        P, w = lidar_edges(f['cloud'], args.rmin, args.rmax, args.jump, args.jump_rel)
        # preselezione larga con la stima iniziale (margine = rotazioni della griglia);
        # durante l'ottimizzazione chi esce dall'immagine vale 0, denominatore fisso
        u, v, ok = project(P, T0[:3, :3], T0[:3, 3], K, D)
        h, wd = self.rgb.shape[:2]
        m = 0.35 * wd
        ok &= (u > -m) & (u < wd + m) & (v > -m) & (v < h + m)
        self.P, self.w = P[ok], w[ok]
        self.edges, self.valid = image_edges(self.rgb, args.gamma, args.mask_bottom, args.edge_pct)
        self.dist = cv2.distanceTransform((1 - self.edges).astype(np.uint8), cv2.DIST_L2, 5)
        self.dist[~self.valid] = 1e3
        self.score = {}

    def S(self, sigma):
        if sigma not in self.score:
            self.score[sigma] = np.exp(-self.dist / sigma).astype(np.float32)
        return self.score[sigma]


def objective(frames, T, K, D, sigma):
    R, t = T[:3, :3], T[:3, 3]
    tot = 0.0
    for fr in frames:
        u, v, ok = project(fr.P, R, t, K, D)
        s = np.zeros(len(u))
        if ok.any():
            s[ok] = map_coordinates(fr.S(sigma), [v[ok], u[ok]], order=1, cval=0.0)
        tot += float((fr.w * s).sum() / fr.w.sum())
    return tot / len(frames)


def edge_error_px(frames, T, K, D):
    """Mediana della distanza [px] tra bordo LiDAR proiettato e bordo immagine piu' vicino."""
    ds = []
    for fr in frames:
        u, v, ok = project(fr.P, T[:3, :3], T[:3, 3], K, D)
        h, w = fr.dist.shape
        ok &= (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        d = map_coordinates(fr.dist, [v[ok], u[ok]], order=1)
        ds.append(d[d < 1e2])
    d = np.concatenate(ds)
    return float(np.median(d)), float(np.percentile(d, 75))


# ---------------------------------------------------------------- ottimizzazione

def scenes_of(frames, ncc_same=0.85):
    """Raggruppa i frame per LUOGO: stessa inquadratura se la correlazione
    normalizzata delle miniature supera ncc_same (il tempo non basta: due soste
    nello stesso punto possono distare piu' di due soste in punti diversi)."""
    def thumb(f):
        g = cv2.resize(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2GRAY), (64, 48), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32).ravel()
        return (g - g.mean()) / (g.std() + 1e-6)
    groups = []
    for f in sorted(frames, key=lambda f: f.t_rel):
        t = thumb(f)
        for g in groups:
            if float(t @ g[1]) / len(t) > ncc_same:
                g[0].append(f)
                break
        else:
            groups.append(([f], t))
    return [g[0] for g in groups]


def optimize(frames, T0, K, D, free_translation, s=1.0, log=print):
    """Sigma in pixel. s = fx/660: 1 deg vale ~11.5*s px, quindi le sigma
    grossolane scalano con la focale; quella finale resta 2 px assoluti."""
    def neg(p6, sigma_px):
        return -objective(frames, perturb(T0, p6), K, D, sigma_px)

    # 1) griglia grossolana sulla rotazione (evita massimi locali)
    best, best_p = -1e9, np.zeros(6)
    ax = np.arange(-GRID_DEG, GRID_DEG + 1e-6, GRID_STEP)
    for wx in ax:
        for wy in ax:
            for wz in ax:
                p = np.array([wx, wy, wz, 0, 0, 0])
                j = -neg(p, 10.0 * s)
                if j > best:
                    best, best_p = j, p
    log(f'  griglia: J={best:.4f} a w=({best_p[0]:+.2f},{best_p[1]:+.2f},{best_p[2]:+.2f}) deg')
    if np.any(np.abs(best_p[:3]) >= GRID_DEG - 1e-6):
        log('  ATTENZIONE: ottimo sul BORDO della griglia -> soluzione non affidabile '
            '(stima iniziale troppo lontana o scena ambigua/periodica)')

    # 2) Nelder-Mead coarse-to-fine sulla rotazione
    p = best_p.copy()
    for sigma in (6.0 * s, 3.0 * max(1.0, s / 2), 2.0):
        r = minimize(lambda q: neg(np.r_[q, p[3:]], sigma), p[:3], method='Nelder-Mead',
                     options={'xatol': 0.005, 'fatol': 1e-6, 'initial_simplex':
                              np.vstack([p[:3], p[:3] + np.eye(3) * 0.5])})
        p[:3] = r.x
        log(f'  rot sigma={sigma:.1f}px: J={-r.fun:.4f} w=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) deg')

    # 3) opzionale: 6 DOF
    if free_translation:
        r = minimize(lambda q: neg(q, 2.0), p, method='Nelder-Mead',
                     options={'xatol': 0.01, 'fatol': 1e-6, 'maxiter': 4000, 'initial_simplex':
                              np.vstack([p, p + np.diag([0.3, 0.3, 0.3, 3, 3, 3])])})
        p = r.x
        log(f'  6DOF: J={-r.fun:.4f} w=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) deg '
            f't=({p[3]:+.1f},{p[4]:+.1f},{p[5]:+.1f}) cm')
    return p


def to_urdf(T_camopt_rslidar, T_base_rslidar, T_link_opt):
    T_base_link = T_base_rslidar @ invT(T_camopt_rslidar) @ invT(T_link_opt)
    return T_base_link[:3, 3], R_to_rpy(T_base_link[:3, :3])


# ---------------------------------------------------------------- output

def overlay(fr, T_before, T_after, K, D, path):
    rows = []
    for T, label in ((T_before, 'xacro'), (T_after, 'calibrata')):
        img = cv2.cvtColor(gamma_lut(2.2)[fr.rgb], cv2.COLOR_RGB2BGR)
        img[fr.edges.astype(bool)] = (0.5 * img[fr.edges.astype(bool)] + (0, 127, 0)).astype(np.uint8)
        u, v, ok = project(fr.P, T[:3, :3], T[:3, 3], K, D)
        depth = np.linalg.norm(fr.P, axis=1)
        col = cv2.applyColorMap(np.clip((depth - 1) / 15 * 255, 0, 255).astype(np.uint8).reshape(-1, 1),
                                cv2.COLORMAP_JET).reshape(-1, 3)
        for uu, vv, c in zip(u[ok].astype(int), v[ok].astype(int), col[ok]):
            if 0 <= uu < img.shape[1] and 0 <= vv < img.shape[0]:
                cv2.circle(img, (uu, vv), 1, tuple(int(x) for x in c), -1)
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        rows.append(img)
    cv2.imwrite(path, np.hstack(rows))


def profiles(frames, T, K, D, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['wx [deg]', 'wy [deg]', 'wz [deg]', 'tx [cm]', 'ty [cm]', 'tz [cm]']
    spans = [2.0, 2.0, 2.0, 20.0, 20.0, 20.0]
    fig, axs = plt.subplots(2, 3, figsize=(12, 6))
    for i, a in enumerate(axs.ravel()):
        xs = np.linspace(-spans[i], spans[i], 41)
        js = []
        for x in xs:
            p = np.zeros(6)
            p[i] = x
            js.append(objective(frames, perturb(T, p), K, D, 2.0))
        a.plot(xs, js)
        a.axvline(0, color='r', lw=0.8)
        a.set_title(names[i])
    fig.suptitle('Profilo del costo attorno alla soluzione (picco netto in 0 = asse ben vincolato)')
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', choices=['camera1', 'camera2'], required=True)
    ap.add_argument('--frames', default='frames')
    ap.add_argument('--out', default=None)
    ap.add_argument('--rslidar-xyz', nargs=3, type=float, default=[-0.21, 0.0, 0.95])
    ap.add_argument('--rslidar-rpy', nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ap.add_argument('--optical-rpy', nargs=3, type=float, default=[-1.5708, 0.0, -1.5708])
    ap.add_argument('--gamma', type=float, default=2.2)
    ap.add_argument('--mask-bottom', type=float, default=0.12, help='frazione di immagine in basso da ignorare')
    ap.add_argument('--edge-pct', type=float, default=90.0, help='percentile del gradiente = bordo')
    ap.add_argument('--rmin', type=float, default=1.0)
    ap.add_argument('--rmax', type=float, default=25.0)
    ap.add_argument('--jump', type=float, default=0.3, help='salto di range minimo [m]')
    ap.add_argument('--jump-rel', type=float, default=0.05, help='salto minimo relativo al range')
    ap.add_argument('--max-dt-ms', type=float, default=15.0)
    ap.add_argument('--min-edge-pts', type=int, default=300)
    ap.add_argument('--img-scale', type=float, default=1.0, help='riduzione immagine (2-3 per camera1)')
    ap.add_argument('--grid-deg', type=float, default=5.0, help='semiampiezza griglia rotazione [deg]')
    ap.add_argument('--grid-step', type=float, default=1.0)
    ap.add_argument('--rot-tol', type=float, default=0.3, help='[deg] accordo richiesto tra le meta')
    ap.add_argument('--trans-tol', type=float, default=3.0, help='[cm] accordo richiesto tra le meta')
    args = ap.parse_args()
    out = args.out or f'result_{args.camera}'
    os.makedirs(out, exist_ok=True)
    global GRID_DEG, GRID_STEP
    GRID_DEG, GRID_STEP = args.grid_deg, args.grid_step

    d = yaml.safe_load(open(os.path.join(args.frames, f'{args.camera}_intrinsics.yaml')))
    K = np.array(d['camera_matrix']['data'], np.float64).reshape(3, 3)
    D = np.array(d['distortion_coefficients']['data'], np.float64).ravel()
    K[:2] /= args.img_scale                      # intrinseci dell'immagine ridotta

    T_base_rslidar = Rt_to_T(rpy_to_R(*args.rslidar_rpy), args.rslidar_xyz)
    T_link_opt = Rt_to_T(rpy_to_R(*args.optical_rpy), [0, 0, 0])
    lxyz, lrpy = XACRO_LINK[args.camera]
    T_base_link0 = Rt_to_T(rpy_to_R(*lrpy), lxyz)
    T0 = invT(T_base_link0 @ T_link_opt) @ T_base_rslidar      # camera_ottica <- rslidar

    frames = []
    for p in sorted(glob.glob(os.path.join(args.frames, 'frame_*.npz'))):
        if float(np.load(p)[DT_KEY[args.camera]]) > args.max_dt_ms:
            continue
        fr = Frame(p, args.camera, args, K, D, T0)
        if len(fr.P) >= args.min_edge_pts:
            frames.append(fr)
    print(f'{len(frames)} frame utilizzabili per {args.camera} '
          f'(punti bordo LiDAR per frame: mediana {int(np.median([len(f.P) for f in frames]))})')
    if len(frames) < 6:
        sys.exit('Troppo pochi frame: servono almeno 6.')

    e0 = edge_error_px(frames, T0, K, D)
    print(f'stima iniziale: J={objective(frames, T0, K, D, 2.0):.4f}  errore bordi mediana '
          f'{e0[0] * args.img_scale:.1f} px a piena risoluzione (p75 {e0[1] * args.img_scale:.1f})')

    s = max(1.0, K[0, 0] / 660.0)
    sc = scenes_of(frames)
    print(f'{len(sc)} luoghi distinti (frame per luogo: {[len(g) for g in sc]})')
    if len(sc) < 4:
        sys.exit('Troppo pochi luoghi distinti per un test di ripetibilita: servono almeno 4.')

    print('\n[tutti i frame, sola rotazione]')
    p_rot = optimize(frames, T0, K, D, free_translation=False, s=s)
    print('\n[tutti i frame, 6 DOF]')
    p_6 = optimize(frames, T0, K, D, free_translation=True, s=s)

    # meta' con LUOGHI diversi: test davvero indipendente
    split = [[f for g in sc[i::2] for f in g] for i in (0, 1)]
    quiet = dict(s=s, log=lambda *_: None)
    print('\n[ripetibilita: due meta con luoghi disgiunti]')
    h_rot = [optimize(h, T0, K, D, False, **quiet) for h in split]
    h_6 = [optimize(h, T0, K, D, True, **quiet) for h in split]
    for i in (0, 1):
        print(f'  meta {i} rot : w=({h_rot[i][0]:+.3f},{h_rot[i][1]:+.3f},{h_rot[i][2]:+.3f}) deg')
        print(f'  meta {i} 6DOF: w=({h_6[i][0]:+.3f},{h_6[i][1]:+.3f},{h_6[i][2]:+.3f}) deg '
              f't=({h_6[i][3]:+.1f},{h_6[i][4]:+.1f},{h_6[i][5]:+.1f}) cm')
    drot = float(np.abs(h_rot[0][:3] - h_rot[1][:3]).max())
    drot6 = float(np.abs(h_6[0][:3] - h_6[1][:3]).max())
    dtr = float(np.abs(h_6[0][3:] - h_6[1][3:]).max())
    print(f'  differenza max: rotazione {drot:.3f} deg (sola rot) | 6DOF: rot {drot6:.3f} deg, trasl {dtr:.1f} cm')

    reliable = drot <= args.rot_tol
    use_6 = reliable and drot6 <= args.rot_tol and dtr <= args.trans_tol
    p = p_6 if use_6 else p_rot
    T = perturb(T0, p)
    e1 = edge_error_px(frames, T, K, D)
    xyz, rpy = to_urdf(T, T_base_rslidar, T_link_opt)
    sc_px = args.img_scale
    print(f'\nerrore bordi mediana (px a piena risoluzione): xacro {e0[0] * sc_px:.1f} -> calibrata {e1[0] * sc_px:.1f} '
          f'(p75 {e0[1] * sc_px:.1f} -> {e1[1] * sc_px:.1f})')
    if not reliable:
        print(f'\nNON AFFIDABILE: la rotazione cambia di {drot:.2f} deg tra meta indipendenti '
              f'(soglia {args.rot_tol}). NON usare questo risultato nell URDF.')
    else:
        print(f'\nSCELTA: {"6 DOF (traslazione ripetibile)" if use_6 else "solo rotazione (traslazione del xacro: non ripetibile)"}')
        print(f'=== Joint URDF  base_footprint -> {args.camera}_link ===')
        print(f'  <origin xyz="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}" rpy="{rpy[0]:.4f} {rpy[1]:.4f} {rpy[2]:.4f}"/>')
        print(f'  (xacro attuale: xyz="{lxyz[0]} {lxyz[1]} {lxyz[2]}" rpy="{lrpy[0]} {lrpy[1]} {lrpy[2]}")')

    yaml.safe_dump({
        'camera': args.camera, 'reliable': bool(reliable), 'n_frames': len(frames), 'n_places': len(sc),
        'mode': '6dof' if use_6 else 'rotation_only',
        'rslidar_xyz': [float(x) for x in args.rslidar_xyz], 'rslidar_rpy': [float(x) for x in args.rslidar_rpy],
        'base_to_link_xyz': [float(x) for x in xyz], 'base_to_link_rpy': [float(x) for x in rpy],
        'perturbation_deg_cm': [float(x) for x in p],
        'edge_err_median_px_fullres': {'xacro': e0[0] * sc_px, 'calibrated': e1[0] * sc_px},
        'halves_diff': {'rot_deg': drot, 'rot6_deg': drot6, 'trans_cm': dtr},
    }, open(os.path.join(out, 'result.yaml'), 'w'))
    for fr in frames[:: max(1, len(frames) // 6)]:
        overlay(fr, T0, T, K, D, os.path.join(out, f'overlay_{fr.name[:-4]}.png'))
    profiles(frames, T, K, D, os.path.join(out, 'profiles.png'))
    print(f'\nOverlay (sinistra xacro, destra calibrata) e profiles.png in {out}/')


if __name__ == '__main__':
    main()
