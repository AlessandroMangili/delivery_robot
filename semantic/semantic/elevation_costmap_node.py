#!/usr/bin/env python3

import os
import time
import array
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration

from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import collections
import math
from cv_bridge import CvBridge
import tf2_ros
from std_srvs.srv import Trigger

from semantic.map_jump_gate import MapJumpGate, lookup_via_odom


DEFAULT_DYNAMIC = [11, 12, 13, 14, 15, 16, 17, 18]   # classi dinamiche Cityscapes
PARAMETRI_MODO = {
    'write_outside_camera': (True, False),
    'use_segmentation': (False, True),
    'confirm_s': (2.5, 0.0),
    'process_var_rate': (0.0, 0.005),
    'tilt_compensation': (True, False),
    'raycast_every_n': (10, 5),
    'raycast_stride': (4, 2),
}


# ============================================================================
#  FUNZIONI PURE (testabili in isolamento, senza ROS)
# ============================================================================


def ray_samples(origin, ends, step, start, end_margin, max_range):
    origin = np.asarray(origin, dtype=np.float64)
    d = np.asarray(ends, dtype=np.float64) - origin[None, :]
    length = np.linalg.norm(d, axis=1)
    stop = np.minimum(length - end_margin, max_range)
    keep = np.isfinite(stop) & (stop > start)
    if not np.any(keep):
        return np.empty((0, 3), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
    u = (d[keep] / length[keep, None]).astype(np.float32)
    stop = stop[keep].astype(np.float32)
    n = int(np.ceil((float(stop.max()) - start) / step)) + 1
    t = (start + step * np.arange(n)).astype(np.float32)
    valid = t[None, :] <= stop[:, None]
    tt = np.broadcast_to(t[None, :], valid.shape)[valid]
    counts = valid.sum(axis=1)
    uu = np.repeat(u, counts, axis=0)
    P = origin[None, :].astype(np.float32) + uu * tt[:, None]
    ray = np.repeat(np.flatnonzero(keep), counts)
    return P, tt, ray


def samples_to_cells(P, gox, goy, res, gnx, gny):
    gi = np.floor((P[:, 0] - gox) / res).astype(np.int64)
    gj = np.floor((P[:, 1] - goy) / res).astype(np.int64)
    inside = (gi >= 0) & (gi < gnx) & (gj >= 0) & (gj < gny)
    return gj * gnx + gi, inside

def transform_to_matrix(t):
    """geometry_msgs TransformStamped -> 4x4 omogenea."""
    q = t.transform.rotation
    tr = t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


def transform_points(pts, M):
    """Applica la 4x4 M a punti Nx3 (p' = R p + t). Ritorna Nx3."""
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:3, :3].T + M[:3, 3][None, :]


def project_to_pixel(pts_cam, K, width, height):
    """Punti nel frame OTTICO camera -> (u,v) interi + maschera valida
    (davanti alla camera E dentro l'immagine). Identica al nodo semantico:
    e' cosi' che ogni punto sa su quale pixel (e quindi classe) cade."""
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    in_front = z > 1e-6
    zz = np.where(in_front, z, 1.0)
    u = np.round(fx * (x / zz) + cx).astype(np.int64)
    v = np.round(fy * (y / zz) + cy).astype(np.int64)
    valid = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = np.where(valid, u, 0)
    v = np.where(valid, v, 0)
    return u, v, valid


def sensor_variance(ranges, var_base, var_dist):
    """Modello di rumore di MISURA dell'altezza per punto [m^2].
    Cresce con la distanza: un punto lontano ha altezza meno affidabile
    (raggio radente, footprint del beam piu' grande). Modello quadratico:
        var_meas = var_base + var_dist * range^2
    range = distanza 3D dal sensore [m]. Fedele allo spirito del modello
    sensoriale di Fankhauser (var che sale con la distanza)."""
    ranges = np.asarray(ranges, dtype=np.float64)
    return var_base + var_dist * ranges * ranges


def aggregate_cells(inv, ncells, heights, var_pts):
    w = 1.0 / np.maximum(var_pts, 1e-9)
    sum_w = np.bincount(inv, weights=w, minlength=ncells)
    sum_wh = np.bincount(inv, weights=w * heights, minlength=ncells)
    return sum_wh / sum_w, 1.0 / sum_w


def high_group_mask(inv, ncells, heights, var_pts, k_sigma):
    hmax = np.full(ncells, -np.inf)
    np.maximum.at(hmax, inv, heights)
    return heights >= hmax[inv] - k_sigma * np.sqrt(var_pts)


def kalman_fuse(h, var, h_meas, var_meas):
    """Aggiornamento di misura (fusione di due gaussiane). In-place safe:
    ritorna (h_new, var_new). Applicare solo su celle gia' inizializzate."""
    K = var / (var + var_meas)
    h_new = h + K * (h_meas - h)
    var_new = (1.0 - K) * var
    return h_new, var_new


def median_smooth_masked(h, known, ksize=3):
    """Smoothing MEDIANO di h sulle celle note, PRIMA di derivare slope/rough/step.
    Perche' serve: slope/roughness/step sono DERIVATE spaziali di h, e le derivate
    AMPLIFICANO il rumore. Anche 1-2 cm di rumore residuo in h (normale dopo la
    fusione, e rimescolato ad ogni ri-osservazione per il micro-offset di posa)
    diventano roughness/step visibili -> aloni grigi sul terreno piatto.
    Il filtro MEDIANO e' edge-preserving: uccide il rumore isolato cella-a-cella
    ma PRESERVA i salti reali (un cordolo resta un cordolo). Le celle ignote sono
    riempite temporaneamente con la media locale nota, per non creare falsi bordi."""
    if ksize < 3:
        return h
    h = h.astype(np.float32)
    knownf = known.astype(np.float32)
    h0 = np.where(known, h, 0.0).astype(np.float32)
    cnt = cv2.boxFilter(knownf, -1, (5, 5), normalize=False, borderType=cv2.BORDER_REPLICATE)
    s = cv2.boxFilter(h0, -1, (5, 5), normalize=False, borderType=cv2.BORDER_REPLICATE)
    filled = np.where(known, h, s / np.where(cnt > 0, cnt, 1.0)).astype(np.float32)
    sm = cv2.medianBlur(filled, int(ksize))
    return np.where(known, sm, h).astype(np.float32)


def derive_slope_roughness(h, known, res, rough_win, min_known):
    """Da campo altezza h (float32) + maschera known -> (slope, roughness, valid).
    slope     = |grad h| [m/m] (adimensionale; tan della pendenza).
    roughness = std locale di h in finestra rough_win x rough_win [m].
    valid     = celle con abbastanza vicini noti da fidarsi.
    Le celle ignote vengono riempite con la media locale nota (box filter mascherato)
    per non introdurre gradienti finti ai bordi dell'ignoto."""
    k = int(rough_win)
    knownf = known.astype(np.float32)
    h0 = np.where(known, h, 0.0).astype(np.float32)

    # box filter NON normalizzato = somme locali; conto dei noti nella finestra
    cnt = cv2.boxFilter(knownf, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)
    s = cv2.boxFilter(h0, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)
    ss = cv2.boxFilter(h0 * h0, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)

    valid = cnt >= float(min_known)
    safe_cnt = np.where(cnt > 0, cnt, 1.0)
    mean = s / safe_cnt                       # altezza locale liscia (definita dove cnt>0)
    var_local = np.maximum(ss / safe_cnt - mean * mean, 0.0)
    roughness = np.sqrt(var_local)

    # slope come gradiente spaziale della MEDIA liscia (robusto all'ignoto)
    gy, gx = np.gradient(mean, res)           # d/dy, d/dx
    slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    slope = np.where(valid, slope, 0.0).astype(np.float32)
    roughness = np.where(valid, roughness, 0.0).astype(np.float32)
    return slope, roughness, valid


def elevation_cost(slope, roughness, step, valid,
                   slope_crit, rough_crit, step_crit,
                   slope_dead=0.0, rough_dead=0.0, step_dead=0.0):
    """f(M): combina i pericoli geometrici in un costo 0..100.
      - scoscese : slope alto        -> cost_slope
      - accidentato: roughness alta  -> cost_rough
      - buche/gradini: step alto     -> cost_step
    Ognuno satura al proprio critico; si combina con MAX (il pericolo peggiore
    domina) e si satura a 100. Celle non valide -> -1 (ignoto: il robot le esplora,
    coerente con include_unknown_as_edge:false della confinement)."""
    # DEAD-ZONE: sotto la soglia di rumore il contributo e' ESATTAMENTE 0 (non
    # 'poco'). La normalizzazione lineare da 0 dava un filo di costo anche a 1.5cm
    # di roughness (rumore) -> grigio. Con la dead-zone, sotto *_dead = 0 netto;
    # sopra sale linearmente da *_dead a *_crit. Uccide il residuo che sopravvive
    # al mediano, senza toccare i pericoli veri (che stanno sopra *_dead).
    def _band(x, dead, crit):
        return np.clip((x - dead) / max(crit - dead, 1e-6), 0.0, 1.0)
    cs = _band(slope, slope_dead, slope_crit)
    cr = _band(roughness, rough_dead, rough_crit)
    ck = _band(step, step_dead, step_crit)
    c = np.maximum(np.maximum(cs, cr), ck) * 100.0
    cost = np.where(valid, np.clip(np.round(c), 0, 100), -1).astype(np.int16)
    return cost


def weighted_plane_fit(dx, dy, r, w):
    wx = w * dx
    wy = w * dy
    N = np.array([[w.sum(), wx.sum(), wy.sum()],
                  [wx.sum(), (wx * dx).sum(), (wx * dy).sum()],
                  [wy.sum(), (wy * dx).sum(), (wy * dy).sum()]])
    v = np.array([(w * r).sum(), (wx * r).sum(), (wy * r).sum()])
    return np.linalg.lstsq(N, v, rcond=None)[0]


def roll_pitch_matrix(roll, pitch):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    M = np.eye(4)
    M[:3, :3] = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]) @ np.eye(3)
    M[:3, :3] = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ M[:3, :3]
    return M


def quat_roll_pitch(q):
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x))))
    return roll, pitch


def neighbor_step(h, known):
    step = np.zeros(h.shape, dtype=np.float32)
    d = np.where(known[1:, :] & known[:-1, :], np.abs(h[1:, :] - h[:-1, :]), 0.0).astype(np.float32)
    np.maximum(step[1:, :], d, out=step[1:, :])
    np.maximum(step[:-1, :], d, out=step[:-1, :])
    d = np.where(known[:, 1:] & known[:, :-1], np.abs(h[:, 1:] - h[:, :-1]), 0.0).astype(np.float32)
    np.maximum(step[:, 1:], d, out=step[:, 1:])
    np.maximum(step[:, :-1], d, out=step[:, :-1])
    return step


def derive_cost(h, known, res, median_ksize, rough_win, rough_min_known,
                slope_crit, rough_crit, step_crit, slope_dead, rough_dead, step_dead):
    h_sm = median_smooth_masked(h, known, median_ksize)
    slope, rough, valid = derive_slope_roughness(h_sm, known, res, rough_win, rough_min_known)
    step = neighbor_step(h_sm, known)
    return elevation_cost(slope, rough, step, valid & known,
                          slope_crit, rough_crit, step_crit, slope_dead, rough_dead, step_dead)


class ElevationGrid:
    def __init__(self, res, size_m, grow_margin, var_init, var_max, process_var_rate,
                 maha_thresh, min_var_for_valid, hybrid_k_sigma, cost_params,
                 ray_start, ray_end_margin, ray_max_range, ray_sigma_k, ray_protect_s,
                 confirm_s=0.0, change_min_m=0.08, change_k=2.0, change_max_m=0.25, max_gap_s=3.0,
                 min_hits=3, tilt_comp=False, tilt_min_points=300, tilt_max_offset_m=0.10,
                 tilt_max_deg=1.5, tilt_cell_var=0.0025, tilt_min_obs=5,
                 free_space_filter=False, free_margin_m=0.10):
        self.res = res
        self.confirm_s = float(confirm_s)
        self.change_min_m = float(change_min_m)
        self.change_k = float(change_k)
        self.change_max_m = float(change_max_m)
        self.max_gap_s = float(max_gap_s)
        self.min_hits = int(min_hits)
        self.tilt_comp = bool(tilt_comp)
        self.tilt_min_points = int(tilt_min_points)
        self.tilt_max_offset_m = float(tilt_max_offset_m)
        self.tilt_max = math.tan(math.radians(float(tilt_max_deg)))
        self.tilt_cell_var = float(tilt_cell_var)
        self.tilt_min_obs = int(tilt_min_obs)
        self.tilt_stats = [0, 0]
        self.free_space_filter = bool(free_space_filter)
        self.free_margin_m = float(free_margin_m)
        self.n_dynamic = 0
        self.last_tilt = (0.0, 0.0, 0.0)
        self.grow_margin = grow_margin
        self.var_init = var_init
        self.var_max = var_max
        self.rate = process_var_rate
        self.maha_thresh = maha_thresh
        self.min_var_for_valid = min_var_for_valid
        self.hybrid_k_sigma = hybrid_k_sigma
        self.cost_params = dict(cost_params)
        mk = int(self.cost_params['median_ksize'])
        med = (2 + mk // 2) if mk >= 3 else 0
        self.radius = med + int(self.cost_params['rough_win']) // 2 + 2
        self.ray_start = ray_start
        self.ray_end_margin = ray_end_margin
        self.ray_max_range = ray_max_range
        self.ray_sigma_k = ray_sigma_k
        self.ray_protect_s = ray_protect_s
        n = int(size_m / res)
        self.gnx = n
        self.gny = n
        self.gox = -size_m / 2.0
        self.goy = -size_m / 2.0
        self.h = np.zeros((n, n), np.float32)
        self.var = np.full((n, n), var_init, np.float32)
        self.tv = np.zeros((n, n), np.float64)
        self.var_min = np.full((n, n), np.inf, np.float32)
        self.n = np.zeros((n, n), np.uint16)
        self.t_upd = np.full((n, n), -np.inf, np.float32)
        self.cost = np.full((n, n), -1, np.int8)
        self.hc = np.zeros((n, n), np.float32)
        self.tc0 = np.full((n, n), -np.inf, np.float64)
        self.tcl = np.full((n, n), -np.inf, np.float64)
        self.hits = np.zeros((n, n), np.uint16)
        self.free_floor = np.full((n, n), np.inf, np.float32)
        self.t_ref = None
        self.last_pred = None
        self.tv_pending = False
        self.dirty = None

    def begin(self, t):
        if self.t_ref is None:
            self.t_ref = t
        t_rel = t - self.t_ref
        if self.tv_pending:
            self.tv[:] = t_rel
            self.tv_pending = False
        return t_rel

    def var_now(self, flat, t_rel):
        v = self.var.reshape(-1)[flat].astype(np.float64)
        return np.minimum(v + self.rate * (t_rel - self.tv.reshape(-1)[flat]), self.var_max)

    def var_full(self, t_rel):
        return np.minimum(self.var + self.rate * (t_rel - self.tv), self.var_max).astype(np.float32)

    def known(self):
        return (self.n > 0) & (self.var_min < self.min_var_for_valid)

    def mark_dirty(self, flat):
        if flat.size == 0:
            return
        j = flat // self.gnx
        i = flat - j * self.gnx
        box = [int(i.min()), int(i.max()) + 1, int(j.min()), int(j.max()) + 1]
        if self.dirty is None:
            self.dirty = box
        else:
            d = self.dirty
            self.dirty = [min(d[0], box[0]), max(d[1], box[1]), min(d[2], box[2]), max(d[3], box[3])]

    def ensure_capacity(self, X, Y):
        gi_min = int(np.floor((X.min() - self.gox) / self.res))
        gi_max = int(np.floor((X.max() - self.gox) / self.res))
        gj_min = int(np.floor((Y.min() - self.goy) / self.res))
        gj_max = int(np.floor((Y.max() - self.goy) / self.res))
        m = int(self.grow_margin / self.res)
        pl = (m - gi_min) if gi_min < 0 else 0
        pb = (m - gj_min) if gj_min < 0 else 0
        pr = (gi_max - (self.gnx - 1) + m) if gi_max >= self.gnx else 0
        pt = (gj_max - (self.gny - 1) + m) if gj_max >= self.gny else 0
        if not (pl or pr or pb or pt):
            return False
        ngx = self.gnx + pl + pr
        ngy = self.gny + pb + pt

        def grown(a, fill):
            out = np.full((ngy, ngx), fill, a.dtype)
            out[pb:pb + self.gny, pl:pl + self.gnx] = a
            return out
        self.h = grown(self.h, 0.0)
        self.var = grown(self.var, self.var_init)
        self.tv = grown(self.tv, 0.0)
        self.var_min = grown(self.var_min, np.inf)
        self.n = grown(self.n, 0)
        self.t_upd = grown(self.t_upd, -np.inf)
        self.cost = grown(self.cost, -1)
        self.hc = grown(self.hc, 0.0)
        self.tc0 = grown(self.tc0, -np.inf)
        self.tcl = grown(self.tcl, -np.inf)
        self.hits = grown(self.hits, 0)
        self.free_floor = grown(self.free_floor, np.inf)
        if self.dirty is not None:
            d = self.dirty
            self.dirty = [d[0] + pl, d[1] + pl, d[2] + pb, d[3] + pb]
        self.gnx = ngx
        self.gny = ngy
        self.gox -= pl * self.res
        self.goy -= pb * self.res
        return True

    def clear_rays(self, origin, ends, t_rel):
        P, _, _ = ray_samples(origin, ends, self.res, self.ray_start,
                              self.ray_end_margin, self.ray_max_range)
        if not P.shape[0]:
            return 0
        flat, inside = samples_to_cells(P, self.gox, self.goy, self.res, self.gnx, self.gny)
        flat = flat[inside]
        if self.free_space_filter:
            np.minimum.at(self.free_floor.reshape(-1), flat, P[inside, 2])
        Hs = self.h.reshape(-1)
        Ns = self.n.reshape(-1)
        Vmin = self.var_min.reshape(-1)
        old = t_rel - self.t_upd.reshape(-1)[flat] > self.ray_protect_s
        sig_h = np.sqrt(np.where(np.isfinite(Vmin[flat]), Vmin[flat], self.var_init))
        below = (Ns[flat] > 0) & old & (Hs[flat] - self.ray_sigma_k * sig_h > P[inside, 2])
        if not np.any(below):
            return 0
        cells = np.unique(flat[below])
        Ns[cells] = 0
        Hs[cells] = 0.0
        self.var.reshape(-1)[cells] = self.var_init
        self.tv.reshape(-1)[cells] = self.last_pred if self.last_pred is not None else t_rel
        Vmin[cells] = np.inf
        self.tc0.reshape(-1)[cells] = -np.inf
        self.tcl.reshape(-1)[cells] = -np.inf
        self.hits.reshape(-1)[cells] = 0
        self.mark_dirty(cells)
        return cells.size

    def tilt_correction(self, flat, X, Y, Z, var_pts, base_xy):
        self.tilt_stats[0] += 1
        Hs = self.h.reshape(-1)
        conf = ((self.n.reshape(-1)[flat] >= self.tilt_min_obs)
                & (self.var_min.reshape(-1)[flat] < self.tilt_cell_var))
        r = Z - Hs[flat]
        sel = conf & (np.abs(r) < 0.3)
        if sel.sum() < self.tilt_min_points:
            return Z
        dx = X[sel] - base_xy[0]
        dy = Y[sel] - base_xy[1]
        w = 1.0 / np.maximum(var_pts[sel], 1e-6)
        rs = r[sel]
        coef = None
        keep = np.ones(dx.size, bool)
        for _ in range(2):
            coef = weighted_plane_fit(dx[keep], dy[keep], rs[keep], w[keep])
            keep = np.abs(rs - (coef[0] + coef[1] * dx + coef[2] * dy)) < 0.08
            if keep.sum() < self.tilt_min_points:
                return Z
        a = float(np.clip(coef[0], -self.tilt_max_offset_m, self.tilt_max_offset_m))
        b = float(np.clip(coef[1], -self.tilt_max, self.tilt_max))
        c = float(np.clip(coef[2], -self.tilt_max, self.tilt_max))
        self.tilt_stats[1] += 1
        self.last_tilt = (a, b, c)
        return Z - (a + b * (X - base_xy[0]) + c * (Y - base_xy[1]))

    def integrate(self, X, Y, Z, var_pts, t_rel, base_xy=None):
        self.ensure_capacity(X, Y)
        gi = ((X - self.gox) / self.res).astype(np.int64)
        gj = ((Y - self.goy) / self.res).astype(np.int64)
        inside = (gi >= 0) & (gi < self.gnx) & (gj >= 0) & (gj < self.gny)
        if not np.any(inside):
            return 0
        flat = gj[inside] * self.gnx + gi[inside]
        Z = Z[inside]
        var_pts = var_pts[inside]
        if self.tilt_comp and base_xy is not None:
            Z = self.tilt_correction(flat, X[inside], Y[inside], Z, var_pts, base_xy)
        if self.free_space_filter:
            static = Z <= self.free_floor.reshape(-1)[flat] + self.free_margin_m
            self.n_dynamic += int(static.size - np.count_nonzero(static))
            if not np.all(static):
                flat = flat[static]
                Z = Z[static]
                var_pts = var_pts[static]
                if flat.size == 0:
                    return 0
        self.last_pred = t_rel
        cells, inv = np.unique(flat, return_inverse=True)
        if self.hybrid_k_sigma > 0.0:
            high = high_group_mask(inv, cells.size, Z, var_pts, self.hybrid_k_sigma)
            inv = inv[high]
            Z = Z[high]
            var_pts = var_pts[high]
        h_meas, var_meas = aggregate_cells(inv, cells.size, Z, var_pts)
        if self.confirm_s > 0.0:
            return self.integrate_confirmed(cells, h_meas, var_meas, t_rel)

        Hs = self.h.reshape(-1)
        Vs = self.var.reshape(-1)
        Ns = self.n.reshape(-1)
        Vmin = self.var_min.reshape(-1)
        first = Ns[cells] == 0
        prev = ~first
        h_old = Hs[cells].astype(np.float64)
        v_now = self.var_now(cells, t_rel)
        denom = np.sqrt(v_now + var_meas)
        maha = np.zeros(cells.size)
        good = prev & (denom > 0)
        maha[good] = np.abs(h_meas[good] - h_old[good]) / denom[good]
        fuse = prev & ~(maha > self.maha_thresh)

        h_new = h_old.copy()
        v_new = v_now.copy()
        h_new[first] = h_meas[first]
        v_new[first] = var_meas[first]
        if np.any(fuse):
            h_new[fuse], v_new[fuse] = kalman_fuse(h_old[fuse], v_now[fuse],
                                                   h_meas[fuse], var_meas[fuse])
        touched = first | fuse
        tc = cells[touched]
        Hs[tc] = h_new[touched]
        Vs[tc] = v_new[touched]
        self.tv.reshape(-1)[tc] = t_rel
        Vmin[tc] = np.minimum(Vmin[tc], Vs[tc])
        self.t_upd.reshape(-1)[tc] = t_rel
        Ns[cells[first]] = 1
        fc = cells[fuse]
        Ns[fc] = np.minimum(Ns[fc].astype(np.int64) + 1, 65535)
        self.mark_dirty(tc)
        return int(tc.size)

    def integrate_confirmed(self, cells, h_meas, var_meas, t_rel):
        Hs = self.h.reshape(-1)
        Vs = self.var.reshape(-1)
        Ns = self.n.reshape(-1)
        Vmin = self.var_min.reshape(-1)
        Tv = self.tv.reshape(-1)
        Tu = self.t_upd.reshape(-1)
        HC = self.hc.reshape(-1)
        T0 = self.tc0.reshape(-1)
        TL = self.tcl.reshape(-1)
        HN = self.hits.reshape(-1)
        h_old = Hs[cells].astype(np.float64)
        tol = np.clip(self.change_k * np.sqrt(var_meas), self.change_min_m, self.change_max_m)
        consistent = (Ns[cells] > 0) & (np.abs(h_meas - h_old) <= tol)

        fc = cells[consistent]
        if fc.size:
            hf, vf = kalman_fuse(h_old[consistent], self.var_now(fc, t_rel),
                                 h_meas[consistent], var_meas[consistent])
            Hs[fc] = hf
            Vs[fc] = vf
            Tv[fc] = t_rel
            Vmin[fc] = np.minimum(Vmin[fc], Vs[fc])
            Tu[fc] = t_rel
            Ns[fc] = np.minimum(Ns[fc].astype(np.int64) + 1, 65535)
            T0[fc] = -np.inf
            TL[fc] = -np.inf
            HN[fc] = 0

        ic = ~consistent
        cc = cells[ic]
        hm = h_meas[ic]
        vm = var_meas[ic]
        hc_old = HC[cc].astype(np.float64)
        cont = (t_rel - TL[cc] <= self.max_gap_s) & (np.abs(hm - hc_old) <= tol[ic])
        nc = cc[~cont]
        HC[nc] = hm[~cont]
        T0[nc] = t_rel
        TL[nc] = t_rel
        HN[nc] = 1
        kc = cc[cont]
        hits = HN[kc].astype(np.float64)
        HC[kc] = (hc_old[cont] * hits + hm[cont]) / (hits + 1.0)
        TL[kc] = t_rel
        HN[kc] = np.minimum(HN[kc].astype(np.int64) + 1, 65535)
        commit = cont & (t_rel - T0[cc] >= self.confirm_s) & (HN[cc] >= self.min_hits)
        mc = cc[commit]
        if mc.size:
            Hs[mc] = HC[mc]
            Vs[mc] = vm[commit]
            Tv[mc] = t_rel
            Vmin[mc] = vm[commit]
            Tu[mc] = t_rel
            Ns[mc] = 1
            T0[mc] = -np.inf
            TL[mc] = -np.inf
            HN[mc] = 0
        touched = np.concatenate([fc, mc])
        self.mark_dirty(touched)
        return int(touched.size)

    def update_cost(self):
        if self.dirty is None:
            return False
        r = self.radius
        i0, i1, j0, j1 = self.dirty
        di0, di1 = max(i0 - r, 0), min(i1 + r, self.gnx)
        dj0, dj1 = max(j0 - r, 0), min(j1 + r, self.gny)
        wi0, wi1 = max(i0 - 2 * r, 0), min(i1 + 2 * r, self.gnx)
        wj0, wj1 = max(j0 - 2 * r, 0), min(j1 + 2 * r, self.gny)
        known = ((self.n[wj0:wj1, wi0:wi1] > 0)
                 & (self.var_min[wj0:wj1, wi0:wi1] < self.min_var_for_valid))
        cw = derive_cost(self.h[wj0:wj1, wi0:wi1], known, self.res, **self.cost_params)
        self.cost[dj0:dj1, di0:di1] = np.clip(
            cw[dj0 - wj0:dj1 - wj0, di0 - wi0:di1 - wi0], -1, 100).astype(np.int8)
        self.dirty = None
        return True

    def full_cost(self):
        return np.clip(derive_cost(self.h, self.known(), self.res, **self.cost_params),
                       -1, 100).astype(np.int8)

    def load(self, d):
        self.h = d['h'].astype(np.float32)
        self.var = d['var'].astype(np.float32)
        self.n = d['n'].astype(np.uint16)
        self.var_min = (d['var_min'] if 'var_min' in d.files else d['var']).astype(np.float32)
        self.gny, self.gnx = self.h.shape
        self.gox = float(d['gox'])
        self.goy = float(d['goy'])
        self.tv = np.zeros(self.h.shape, np.float64)
        self.tv_pending = True
        self.t_upd = np.full(self.h.shape, -np.inf, np.float32)
        self.hc = np.zeros(self.h.shape, np.float32)
        self.tc0 = np.full(self.h.shape, -np.inf, np.float64)
        self.tcl = np.full(self.h.shape, -np.inf, np.float64)
        self.hits = np.zeros(self.h.shape, np.uint16)
        self.free_floor = np.full(self.h.shape, np.inf, np.float32)
        self.cost = self.full_cost()
        self.dirty = None

    def arrays_to_save(self):
        t_rel = self.last_pred if self.last_pred is not None else 0.0
        var = self.var if self.tv_pending else self.var_full(t_rel)
        return dict(h=self.h, var=var, var_min=self.var_min, n=self.n,
                    gnx=self.gnx, gny=self.gny, res=self.res, gox=self.gox, goy=self.goy)


# ============================================================================
#  NODO
# ============================================================================

class ElevationCostmapNode(Node):
    def __init__(self):
        super().__init__('elevation_costmap_node')

        # --- Frames ---
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('lidar_frame', 'base_scan')

        # --- Grid ---
        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('initial_size_m', 10.0)
        self.declare_parameter('grow_margin_m', 3.0)

        # --- LiDAR ---
        self.declare_parameter('lidar_topic', '/scan/points')
        self.declare_parameter('lidar_max_age_s', 0.3)
        self.declare_parameter('lidar_min_range', 0.4)
        self.declare_parameter('map_write_max_range', 5.0)
        # banda di quota (rel. al robot) per accettare un punto come TERRENO:
        self.declare_parameter('elev_min_rel', -1.0)   # sotto = buco molto profondo (scartato)
        self.declare_parameter('elev_max_rel', 1.0)    # sopra = overhead (chiome/insegne)

        # --- Camera + segmentazione (co-registrazione col layer semantico) ---
        # I punti LiDAR sono proiettati nel FOV camera e SCARTATI se cadono su un
        # pixel di classe DINAMICA: cosi' i pedoni non lasciano scie nella mappa
        # di elevazione. Copertura = FOV camera (non piu' 360 LiDAR).
        self.declare_parameter('camera_optical_frame', 'camera_rgb_optical_frame')
        self.declare_parameter('seg_topic', '/semantic/segmentation')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('seg_max_age_s', 0.3)
        self.declare_parameter('dynamic_classes', DEFAULT_DYNAMIC)
        self.declare_parameter('dyn_dilate_px', 6)     # alone attorno ai dinamici (bordi seg.)

        # --- Modello di misura Kalman ---
        self.declare_parameter('sensor_var_base', 0.0009)   # (0.03 m)^2 rumore base
        self.declare_parameter('sensor_var_dist', 0.0004)   # coeff. * range^2
        self.declare_parameter('var_init', 1.0)             # varianza cella mai vista
        self.declare_parameter('var_max', 1.0)              # tetto varianza (inflazione)
        self.declare_parameter('maha_thresh', 3.0)          # gate outlier (rumore residuo)
        self.declare_parameter('min_var_for_valid', 0.25)   # var oltre = cella non affidabile

        # --- Derivazione costo (f(M)) ---
        self.declare_parameter('rough_win', 5)              # finestra roughness [celle]
        self.declare_parameter('rough_min_known', 6)        # min vicini noti per validita'
        self.declare_parameter('slope_crit', 0.35)          # [m/m] ~19 deg -> costo max
        self.declare_parameter('rough_crit', 0.06)          # [m] std locale -> costo max
        self.declare_parameter('step_crit', 0.06)           # [m] salto -> costo max (cordolo)
        # smoothing mediano di h prima di derivare (5 = pulisce bene il rumore 1cm;
        # 3 = piu' leggero; 0/1 = off). Con ksize 5 la soglia step affidabile e' ~5cm.
        self.declare_parameter('median_ksize', 5)
        # dead-zone / soglia minima: sotto = rumore (0), sopra sale verso *_crit
        self.declare_parameter('slope_dead', 0.08)          # [m/m] sotto = piatto
        self.declare_parameter('rough_dead', 0.025)         # [m] sotto = liscio (rumore)
        self.declare_parameter('step_dead', 0.02)           # [m] step_min: sotto = rumore

        self.declare_parameter('raycast_enable', True)
        self.declare_parameter('raycast_end_margin', 0.25)
        self.declare_parameter('raycast_sigma_k', 3.0)
        self.declare_parameter('raycast_protect_s', 1.0)
        self.declare_parameter('attitude_topic', '')
        self.declare_parameter('attitude_fallback_topic', '')
        self.declare_parameter('attitude_max_age_s', 0.3)
        self.declare_parameter('change_min_m', 0.08)
        self.declare_parameter('change_k', 2.0)
        self.declare_parameter('change_max_m', 0.25)
        self.declare_parameter('max_gap_s', 3.0)
        self.declare_parameter('min_hits', 3)
        self.declare_parameter('tilt_min_points', 300)
        self.declare_parameter('tilt_max_offset_m', 0.10)
        self.declare_parameter('tilt_max_deg', 1.5)
        self.declare_parameter('free_space_filter', False)
        self.declare_parameter('free_margin_m', 0.10)
        self.declare_parameter('hybrid_k_sigma', 2.5)
        self.declare_parameter('tf_timeout_s', 0.1)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('map_jump_gate_m', 0.25)
        self.declare_parameter('map_jump_gate_deg', 2.0)
        self.declare_parameter('map_jump_hold_s', 1.0)

        # --- Publishing ---
        self.declare_parameter('publish_period_s', 1.0)
        self.declare_parameter('out_topic', '/traversability_cost')

        # --- Gate di localizzazione (stesso schema del semantico) ---
        self.declare_parameter('use_loc_quality_gate', True)
        self.declare_parameter('loc_pose_topic', '/odometry/global')
        self.declare_parameter('loc_std_suspend_m', 1.0)
        self.declare_parameter('loc_std_resume_m', 0.5)
        self.declare_parameter('loc_cov_ema', 0.4)

        # --- Persistenza ---
        self.declare_parameter('map_name', '')

        gp = self.get_parameter
        self.declare_parameter('elevazione_360', True)
        modo = 'modo_360' if gp('elevazione_360').value else 'modo_camera'
        for nome, (v360, vcam) in PARAMETRI_MODO.items():
            self.declare_parameter(f'modo_360.{nome}', v360)
            self.declare_parameter(f'modo_camera.{nome}', vcam)

        def gm(nome):
            return gp(f'{modo}.{nome}')

        self.target = gp('target_frame').value
        self.base_frame = gp('base_frame').value
        self.lidar_frame = gp('lidar_frame').value
        self.res = float(gp('resolution').value)
        self.initial_size_m = float(gp('initial_size_m').value)
        self.grow_margin = float(gp('grow_margin_m').value)

        self.lidar_max_age = float(gp('lidar_max_age_s').value)
        self.lidar_min_range = float(gp('lidar_min_range').value)
        self.map_write_max_range = float(gp('map_write_max_range').value)
        self.elev_min_rel = float(gp('elev_min_rel').value)
        self.elev_max_rel = float(gp('elev_max_rel').value)

        self.cam_frame = gp('camera_optical_frame').value
        self.seg_max_age = float(gp('seg_max_age_s').value)
        self.dyn_dilate_px = int(gp('dyn_dilate_px').value)
        self.dyn_lut = np.zeros(256, dtype=bool)
        for c in gp('dynamic_classes').value:
            self.dyn_lut[int(c)] = True

        self.sensor_var_base = float(gp('sensor_var_base').value)
        self.sensor_var_dist = float(gp('sensor_var_dist').value)
        self.var_init = float(gp('var_init').value)
        self.var_max = float(gp('var_max').value)
        self.process_var_rate = float(gm('process_var_rate').value)
        self.maha_thresh = float(gp('maha_thresh').value)
        self.min_var_for_valid = float(gp('min_var_for_valid').value)

        self.rough_win = int(gp('rough_win').value)
        self.rough_min_known = int(gp('rough_min_known').value)
        self.slope_crit = float(gp('slope_crit').value)
        self.rough_crit = float(gp('rough_crit').value)
        self.step_crit = float(gp('step_crit').value)
        self.median_ksize = int(gp('median_ksize').value)
        self.slope_dead = float(gp('slope_dead').value)
        self.rough_dead = float(gp('rough_dead').value)
        self.step_dead = float(gp('step_dead').value)

        self.raycast_enable = bool(gp('raycast_enable').value)
        self.raycast_every_n = max(1, int(gm('raycast_every_n').value))
        self.raycast_stride = max(1, int(gm('raycast_stride').value))
        self.raycast_end_margin = float(gp('raycast_end_margin').value)
        self.raycast_sigma_k = float(gp('raycast_sigma_k').value)
        self.raycast_protect_s = float(gp('raycast_protect_s').value)
        self.write_outside_camera = bool(gm('write_outside_camera').value)
        self.use_segmentation = bool(gm('use_segmentation').value)
        self.hybrid_k_sigma = float(gp('hybrid_k_sigma').value)
        self.tf_timeout = float(gp('tf_timeout_s').value)
        self.odom_frame = gp('odom_frame').value
        self.jump_gate = (MapJumpGate(gp('map_jump_gate_m').value, gp('map_jump_gate_deg').value,
                                      gp('map_jump_hold_s').value)
                          if self.target != self.odom_frame else None)
        self._ray_calls = 0
        self._ray_cleared = 0
        self._ray_time = 0.0
        self._ray_runs = 0

        self.publish_period = float(gp('publish_period_s').value)

        self.use_loc_gate = bool(gp('use_loc_quality_gate').value)
        self.loc_std_suspend = float(gp('loc_std_suspend_m').value)
        self.loc_std_resume = float(gp('loc_std_resume_m').value)
        self.loc_cov_ema = float(gp('loc_cov_ema').value)
        self.loc_std_smooth = 0.0
        self.loc_ok = True

        self.grid = ElevationGrid(
            self.res, self.initial_size_m, self.grow_margin, self.var_init, self.var_max,
            self.process_var_rate, self.maha_thresh, self.min_var_for_valid, self.hybrid_k_sigma,
            dict(median_ksize=self.median_ksize, rough_win=self.rough_win,
                 rough_min_known=self.rough_min_known, slope_crit=self.slope_crit,
                 rough_crit=self.rough_crit, step_crit=self.step_crit,
                 slope_dead=self.slope_dead, rough_dead=self.rough_dead, step_dead=self.step_dead),
            self.lidar_min_range, self.raycast_end_margin, self.map_write_max_range,
            self.raycast_sigma_k, self.raycast_protect_s,
            confirm_s=gm('confirm_s').value, change_min_m=gp('change_min_m').value,
            change_k=gp('change_k').value, change_max_m=gp('change_max_m').value,
            max_gap_s=gp('max_gap_s').value, min_hits=gp('min_hits').value,
            tilt_comp=gm('tilt_compensation').value, tilt_min_points=gp('tilt_min_points').value,
            tilt_max_offset_m=gp('tilt_max_offset_m').value, tilt_max_deg=gp('tilt_max_deg').value,
            free_space_filter=gp('free_space_filter').value, free_margin_m=gp('free_margin_m').value)
        self._n_clouds = 0
        self._n_skipped = 0
        self._n_integrated = 0
        self._t_integrate = 0.0
        self._t_integrate_max = 0.0
        self._t_publish = 0.0
        self._n_publish = 0
        self._last_cloud_t = None
        self._last_stats = time.monotonic()

        # --- Cache nuvola LiDAR (PULITA una volta sola, come nel semantico) ---
        self.cloud = None
        self.cloud_t = -1e9
        self.cloud_stamp = None

        # --- Cache segmentazione + intrinseci camera (per escludere i dinamici) ---
        self.bridge = CvBridge()
        self.K = None
        self.seg = None
        self.seg_t = -1e9
        self.seg_stamp = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, None, spin_thread=True)

        cloud_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                               history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(PointCloud2, gp('lidar_topic').value,
                                 self.cloud_cb, cloud_qos)

        self.create_subscription(CameraInfo, gp('camera_info_topic').value,
                                 self.info_cb, 1)
        seg_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, gp('seg_topic').value, self.seg_cb, seg_qos)

        self.att = collections.deque(maxlen=200)
        self.att_fb = collections.deque(maxlen=200)
        self.att_max_age = float(gp('attitude_max_age_s').value)
        self.n_att = [0, 0, 0]
        if gp('attitude_topic').value:
            self.create_subscription(PoseWithCovarianceStamped, gp('attitude_topic').value,
                                     lambda m: self.att_cb(self.att, m.header, m.pose.pose.orientation), 10)
        if gp('attitude_fallback_topic').value:
            self.create_subscription(Odometry, gp('attitude_fallback_topic').value,
                                     lambda m: self.att_cb(self.att_fb, m.header, m.pose.pose.orientation), 10)
        if self.use_loc_gate:
            self.create_subscription(Odometry, gp('loc_pose_topic').value,
                                     self.loc_cb, 10)

        latched_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(OccupancyGrid, gp('out_topic').value, latched_qos)

        self._last_pred_t = None

        # --- Persistenza ---
        self.maps_dir = self.resolve_maps_dir()
        map_name = gp('map_name').value
        load_name = save_name = f'elevazione_{map_name}' if map_name else ''
        self.map_save_path = os.path.join(self.maps_dir, save_name) if save_name else ''
        if load_name:
            p = os.path.join(self.maps_dir, load_name)
            if not p.endswith('.npz'):
                p += '.npz'
            if os.path.isfile(p):
                self.load_map(p)
            else:
                self.get_logger().warn(f'Elevation map to load not found: {p}')
        self.autosave = bool(map_name)
        self.create_service(Trigger, '~/save_map', self.save_map_cb)

        # timer di pubblicazione (deriva costo e pubblica, indipendente dal LiDAR)
        self.create_timer(self.publish_period, self.publish_timer)

        self.get_logger().info(
            f'Elevation 2.5D (Kalman-per-cella, LiDAR-primary) pronto. frame={self.target}. '
            f'res={self.res} m. Uscita: {gp("out_topic").value}.')
        
        save_hint = self.map_save_path or os.path.join(self.maps_dir, 'elevation_map')
        if self.autosave:
            self.get_logger().info(f'AUTOSAVE alla chiusura (Ctrl-C) -> {self.map_save_path}')
        self.get_logger().info(
            'Salvataggio su richiesta: '
            f'ros2 service call /elevation_costmap_node/save_map std_srvs/srv/Trigger  ->  {save_hint}')

    # ---------------- persistenza ----------------
    def resolve_maps_dir(self):
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory('semantic')
            ws = share
            for _ in range(8):
                ws = os.path.dirname(ws)
                src = os.path.join(ws, 'src')
                if os.path.isdir(src):
                    for root, dirs, files in os.walk(src):
                        if os.path.basename(root) == 'semantic' and 'package.xml' in files:
                            d = os.path.join(root, 'maps')
                            os.makedirs(d, exist_ok=True)
                            return d
                    break
            d = os.path.join(share, 'maps'); os.makedirs(d, exist_ok=True); return d
        except Exception:
            d = os.path.join(os.path.expanduser('~'), '.semantic_maps')
            os.makedirs(d, exist_ok=True); return d

    def load_map(self, path):
        try:
            d = np.load(path)
            if float(d['res']) != self.res:
                self.get_logger().warn('Elevation map con risoluzione diversa: ignorata.')
                return
            self.grid.load(d)
            self.get_logger().info(
                f'Elevation map caricata da {path} ({self.grid.gnx}x{self.grid.gny}).')
        except Exception as e:
            self.get_logger().warn(f'Load elevation fallito: {e}')

    def save_map(self, path):
        try:
            if not path.endswith('.npz'):
                path += '.npz'
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            np.savez_compressed(path, **self.grid.arrays_to_save())
            self.get_logger().info(f'Elevation map salvata in {path}.')
            return path
        except Exception as e:
            self.get_logger().error(f'Save elevation fallito: {e}')
            return None

    def save_map_cb(self, request, response):
        path = self.map_save_path or os.path.join(self.maps_dir, 'elevation_map')
        saved = self.save_map(path)
        response.success = saved is not None
        response.message = f'Salvata in {saved}' if saved else 'Salvataggio fallito'
        return response

    # ---------------- callbacks ----------------
    def att_cb(self, buf, header, q):
        t = header.stamp.sec + header.stamp.nanosec * 1e-9
        if t > 0:
            buf.append((t, *quat_roll_pitch(q)))

    def attitude_at(self, t):
        for k, buf in enumerate((self.att, self.att_fb)):
            if buf:
                ts = np.fromiter((a[0] for a in buf), float, len(buf))
                i = int(np.argmin(np.abs(ts - t)))
                if abs(ts[i] - t) <= self.att_max_age:
                    self.n_att[k] += 1
                    return buf[i][1], buf[i][2]
        self.n_att[2] += 1
        return None

    def loc_cb(self, msg: Odometry):
        cov = msg.pose.covariance
        var_x = max(0.0, float(cov[0])); var_y = max(0.0, float(cov[7]))
        std_xy = float(np.sqrt(max(var_x, var_y)))
        a = self.loc_cov_ema
        self.loc_std_smooth = a * std_xy + (1.0 - a) * self.loc_std_smooth
        if self.loc_ok and self.loc_std_smooth > self.loc_std_suspend:
            self.loc_ok = False
            self.get_logger().warn(
                f'LOCALIZZAZIONE PERSA (std liscia {self.loc_std_smooth:.2f}m > '
                f'{self.loc_std_suspend}m) -> MAPPATURA ELEVAZIONE SOSPESA.')
        elif (not self.loc_ok) and self.loc_std_smooth < self.loc_std_resume:
            self.loc_ok = True
            self.get_logger().info(
                f'Localizzazione RECUPERATA (std liscia {self.loc_std_smooth:.2f}m) -> RIPRESA.')

    def cloud_cb(self, msg: PointCloud2):
        """Nuvola LiDAR (frame base_scan): pulita e messa in cache.
        Scarta NaN/inf (skip_nans NON toglie gli inf del gpu_lidar) e il rumore
        entro lidar_min_range (come il 'blind' di FAST-LIO)."""
        try:
            rec = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception:
            return
        n = int(rec.shape[0])
        if n == 0:
            return
        pts = np.empty((n, 3), dtype=np.float64)
        pts[:, 0] = rec['x']; pts[:, 1] = rec['y']; pts[:, 2] = rec['z']
        finite = np.isfinite(pts).all(axis=1)
        r2 = pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2
        keep = finite & (r2 >= self.lidar_min_range ** 2)
        pts = pts[keep]
        if pts.shape[0] == 0:
            return
        self.cloud = pts
        self.cloud_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.cloud_stamp = msg.header.stamp
        self._n_clouds += 1
        if self._last_cloud_t is not None:
            gap = self.cloud_t - self._last_cloud_t
            if gap > 0.15:
                self._n_skipped += int(round(gap / 0.1)) - 1
        self._last_cloud_t = self.cloud_t
        t0 = time.perf_counter()
        if self.integrate():
            dt = time.perf_counter() - t0
            self._n_integrated += 1
            self._t_integrate += dt
            self._t_integrate_max = max(self._t_integrate_max, dt)
        self.log_stats()

    def info_cb(self, msg: CameraInfo):
        self.K = np.array(msg.k).reshape(3, 3)

    def seg_cb(self, msg: Image):
        """Immagine di segmentazione (Cityscapes ID, mono8): messa in cache.
        Serve solo a sapere quali pixel sono DINAMICI, per escludere i punti
        LiDAR che ci cadono sopra."""
        try:
            self.seg = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except Exception:
            return
        self.seg_t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.seg_stamp = msg.header.stamp

    def lookup(self, frame, stamp):
        try:
            if self.target != self.odom_frame:
                return lookup_via_odom(self.tf_buffer, self.target, self.odom_frame, frame, stamp, self.tf_timeout)
            return self.tf_buffer.lookup_transform(
                self.target, frame, rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=self.tf_timeout))
        except Exception as e:
            self.get_logger().warn(f'TF non disponibile {frame}->{self.target}: {e}',
                                   throttle_duration_sec=2.0)
            return None

    # ---------------- integrazione (misura Kalman) ----------------
    def integrate(self):
        if self.cloud is None:
            return False
        if self.use_loc_gate and not self.loc_ok:
            return False

        T_scan = self.lookup(self.lidar_frame, self.cloud_stamp)
        T_base = self.lookup(self.base_frame, self.cloud_stamp)
        if T_scan is None or T_base is None:
            return False

        M_scan2map = transform_to_matrix(T_scan)
        if self.att or self.att_fb:
            rp = self.attitude_at(self.cloud_t)
            if rp is not None:
                M_tb = transform_to_matrix(T_base)
                M_scan2map = M_tb @ roll_pitch_matrix(*rp) @ np.linalg.inv(M_tb) @ M_scan2map
        Pm = transform_points(self.cloud, M_scan2map)
        X = Pm[:, 0]; Y = Pm[:, 1]; Z = Pm[:, 2]

        bx = T_base.transform.translation.x
        by = T_base.transform.translation.y
        bz = T_base.transform.translation.z
        if not self.map_jump_ok(self.cloud_stamp, self.cloud_t, bx, by, bz):
            return False

        rng = np.sqrt(self.cloud[:, 0]**2 + self.cloud[:, 1]**2 + self.cloud[:, 2]**2)

        n_pts = self.cloud.shape[0]
        in_img = np.zeros(n_pts, dtype=bool)
        not_dyn = np.ones(n_pts, dtype=bool)
        seg_ok = (self.use_segmentation and self.K is not None and self.seg is not None
                  and (self.cloud_t - self.seg_t) <= self.seg_max_age)
        T_cam = self.lookup(self.cam_frame, self.seg_stamp) if seg_ok else None
        if T_cam is not None:
            M_scan2cam = np.linalg.inv(transform_to_matrix(T_cam)) @ M_scan2map
            Pc = transform_points(self.cloud, M_scan2cam)
            seg = self.seg
            hgt, wid = seg.shape
            uu, vv, in_img = project_to_pixel(Pc, self.K, wid, hgt)
            full_dyn = self.dyn_lut[seg].astype(np.uint8)
            if self.dyn_dilate_px > 0 and full_dyn.any():
                k = 2 * self.dyn_dilate_px + 1
                full_dyn = cv2.dilate(full_dyn, np.ones((k, k), np.uint8), iterations=1)
            not_dyn = ~(full_dyn[vv, uu].astype(bool) & in_img)
        elif not self.write_outside_camera:
            self.get_logger().warn('Segmentazione troppo vecchia in questo frame.',
                                   throttle_duration_sec=10.0)
            return False

        t_rel = self.grid.begin(self.cloud_t)
        usable = (in_img | self.write_outside_camera) & not_dyn
        finite = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        sel = finite & (in_img | self.write_outside_camera)
        self.raycast_clear(M_scan2map[:3, 3], Pm[sel], rng[sel], t_rel)

        rel = Z - bz
        dist = np.hypot(X - bx, Y - by)
        ok = (finite & usable
              & (rel >= self.elev_min_rel) & (rel <= self.elev_max_rel)
              & (dist < self.map_write_max_range))
        if not np.any(ok):
            return False
        var_pts = sensor_variance(rng[ok], self.sensor_var_base, self.sensor_var_dist)
        self.grid.integrate(X[ok], Y[ok], Z[ok], var_pts, t_rel, base_xy=(bx, by))
        return True

    def map_jump_ok(self, stamp, t, bx, by, bz):
        if self.jump_gate is None:
            return True
        T_mo = self.lookup(self.odom_frame, stamp)
        if T_mo is None:
            return False
        ok = self.jump_gate.update(transform_to_matrix(T_mo), (bx, by, bz), t)
        if not ok:
            self.get_logger().warn(
                f'Salto della localizzazione ({self.jump_gate.last_jump_m:.2f} m, '
                f'{self.jump_gate.last_jump_deg:.1f} gradi): scrittura in mappa sospesa '
                f'per {self.jump_gate.hold_s:.1f} s ({self.jump_gate.n_jumps} salti finora).',
                throttle_duration_sec=1.0)
        return ok

    def raycast_clear(self, origin, ends, ranges, t_rel):
        if not self.raycast_enable or ends.shape[0] == 0:
            return
        self._ray_calls += 1
        if self._ray_calls % self.raycast_every_n:
            return
        t0 = time.perf_counter()
        ends = ends[::self.raycast_stride].copy()
        sig_p = np.sqrt(sensor_variance(ranges[::self.raycast_stride],
                                        self.sensor_var_base, self.sensor_var_dist))
        ends[:, 2] += self.raycast_sigma_k * sig_p
        self._ray_cleared += self.grid.clear_rays(origin, ends, t_rel)
        self._ray_time += time.perf_counter() - t0
        self._ray_runs += 1

    def log_stats(self):
        now = time.monotonic()
        if now - self._last_stats < 30.0:
            return
        self._last_stats = now
        ni = max(self._n_integrated, 1)
        npub = max(self._n_publish, 1)
        nr = max(self._ray_runs, 1)
        self.get_logger().info(
            f'nuvole ricevute {self._n_clouds}, integrate {self._n_integrated}, '
            f'saltate {self._n_skipped} | integrazione {1000.0 * self._t_integrate / ni:.1f} ms '
            f'(max {1000.0 * self._t_integrate_max:.1f}) | raycast {1000.0 * self._ray_time / nr:.1f} ms, '
            f'{self._ray_cleared} celle ripulite | costo e pubblicazione '
            f'{1000.0 * self._t_publish / npub:.1f} ms | griglia {self.grid.gnx}x{self.grid.gny}'
            # Agg 
            f' | assetto trovato {self.n_att[0]}, mancante {self.n_att[2]}'
            f' | inclinazione corretta {self.grid.tilt_stats[1]}/{self.grid.tilt_stats[0]}')

    # ---------------- pubblicazione (deriva costo) ----------------
    def publish_timer(self):
        t0 = time.perf_counter()
        self.grid.update_cost()
        if self.grid.t_ref is None:
            return
        g = self.grid
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.target
        msg.info.resolution = self.res
        msg.info.width = g.gnx; msg.info.height = g.gny
        msg.info.origin.position.x = float(g.gox)
        msg.info.origin.position.y = float(g.goy)
        msg.info.origin.orientation.w = 1.0
        msg.data = array.array('b', np.ascontiguousarray(g.cost).tobytes())
        self.pub.publish(msg)
        self._t_publish += time.perf_counter() - t0
        self._n_publish += 1


def main():
    rclpy.init()
    node = ElevationCostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if getattr(node, 'autosave', False) and node.map_save_path:
            node.save_map(node.map_save_path)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()