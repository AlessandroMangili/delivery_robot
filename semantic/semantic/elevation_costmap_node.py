#!/usr/bin/env python3

import array
import collections
import math
import os
import signal
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

from semantic.map_jump_gate import MapJumpGate, lookup_via_odom


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
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ M[:3, :3].T + M[:3, 3][None, :]


def sensor_variance(ranges, var_base, var_dist):
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
    K = var / (var + var_meas)
    h_new = h + K * (h_meas - h)
    var_new = (1.0 - K) * var
    return h_new, var_new


def median_smooth_masked(h, known, ksize=3):
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
    k = int(rough_win)
    knownf = known.astype(np.float32)
    h0 = np.where(known, h, 0.0).astype(np.float32)

    cnt = cv2.boxFilter(knownf, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)
    s = cv2.boxFilter(h0, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)
    ss = cv2.boxFilter(h0 * h0, -1, (k, k), normalize=False, borderType=cv2.BORDER_REPLICATE)

    valid = cnt >= float(min_known)
    safe_cnt = np.where(cnt > 0, cnt, 1.0)
    mean = s / safe_cnt
    var_local = np.maximum(ss / safe_cnt - mean * mean, 0.0)
    roughness = np.sqrt(var_local)

    gy, gx = np.gradient(mean, res)
    slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    slope = np.where(valid, slope, 0.0).astype(np.float32)
    roughness = np.where(valid, roughness, 0.0).astype(np.float32)
    return slope, roughness, valid


def elevation_cost(slope, roughness, step, valid,
                   slope_crit, rough_crit, step_crit,
                   slope_dead=0.0, rough_dead=0.0, step_dead=0.0, step_lethal=0.0, step_for_lethal=None):
    def _band(x, dead, crit):
        return np.clip((x - dead) / max(crit - dead, 1e-6), 0.0, 1.0)
    cs = _band(slope, slope_dead, slope_crit)
    cr = _band(roughness, rough_dead, rough_crit)
    ck = _band(step, step_dead, step_crit)
    c = np.maximum(np.maximum(cs, cr), ck) * 100.0
    if step_lethal > 0.0:
        sl = step if step_for_lethal is None else step_for_lethal
        c = np.where(sl >= step_lethal, 100.0, np.minimum(c, 99.0))
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


def gap_step(h, known, res, drop_r, rise_r, allow):
    hi = np.where(known, h, -1e3).astype(np.float32)
    lo = np.where(known, h, 1e3).astype(np.float32)
    best = np.zeros(h.shape, np.float32)
    for rr in np.arange(0.10, max(drop_r, rise_r) + 1e-6, 0.10):
        k = int(round(rr / res))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        if rr <= rise_r + 1e-6:
            best = np.maximum(best, cv2.dilate(hi, ker) - h - allow * rr)
        if rr <= drop_r + 1e-6:
            best = np.maximum(best, h - cv2.erode(lo, ker) - allow * rr)
    return np.where(known, np.clip(best, 0.0, 1.0), 0.0).astype(np.float32)


def derive_cost(h, known, res, median_ksize, rough_win, rough_min_known,
                slope_crit, rough_crit, step_crit, slope_dead, rough_dead, step_dead,
                step_drop_r=0.0, step_rise_r=0.0, step_allow=0.05, step_lethal=0.0, step_lethal_r=0.0):
    h_sm = median_smooth_masked(h, known, median_ksize)
    slope, rough, valid = derive_slope_roughness(h_sm, known, res, rough_win, rough_min_known)
    step = neighbor_step(h_sm, known)
    step_l = None
    if step_drop_r > 0.0 or step_rise_r > 0.0:
        sn = step
        step = np.maximum(sn, gap_step(h_sm, known, res, step_drop_r, step_rise_r, step_allow))
        if step_lethal > 0.0 and step_lethal_r > 0.0:
            step_l = np.maximum(sn, gap_step(h_sm, known, res, min(step_drop_r, step_lethal_r),
                                             min(step_rise_r, step_lethal_r), step_allow))
    return elevation_cost(slope, rough, step, valid & known,
                          slope_crit, rough_crit, step_crit, slope_dead, rough_dead, step_dead, step_lethal, step_l)


class ElevationGrid:
    def __init__(self, res, size_m, grow_margin, var_init, var_max, process_var_rate,
                 min_var_for_valid, hybrid_k_sigma, cost_params,
                 ray_start, ray_end_margin, ray_max_range, ray_sigma_k, ray_protect_s,
                 confirm_s=2.5, change_min_m=0.08, change_k=2.0, change_max_m=0.25, max_gap_s=3.0,
                 min_hits=3, tilt_min_points=300, tilt_cell_var=0.0025, tilt_min_obs=5,
                 tilt_reject_deg=0.0):
        self.res = res
        self.confirm_s = float(confirm_s)
        self.change_min_m = float(change_min_m)
        self.change_k = float(change_k)
        self.change_max_m = float(change_max_m)
        self.max_gap_s = float(max_gap_s)
        self.min_hits = int(min_hits)
        self.tilt_min_points = int(tilt_min_points)
        self.tilt_cell_var = float(tilt_cell_var)
        self.tilt_min_obs = int(tilt_min_obs)
        self.tilt_reject = math.tan(math.radians(float(tilt_reject_deg))) if tilt_reject_deg > 0.0 else 0.0
        self.reject_stats = [0, 0, 0]
        self.grow_margin = grow_margin
        self.var_init = var_init
        self.var_max = var_max
        self.rate = process_var_rate
        self.min_var_for_valid = min_var_for_valid
        self.hybrid_k_sigma = hybrid_k_sigma
        self.cost_params = dict(cost_params)
        mk = int(self.cost_params['median_ksize'])
        med = (2 + mk // 2) if mk >= 3 else 0
        gap = max(float(self.cost_params.get('step_drop_r', 0.0)), float(self.cost_params.get('step_rise_r', 0.0)))
        self.radius = med + int(self.cost_params['rough_win']) // 2 + 2 + int(math.ceil(gap / res))
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

    def vertical_offset(self, X, Y, Z, t_rel, min_age, max_diff, min_pts):
        gi = ((X - self.gox) / self.res).astype(np.int64)
        gj = ((Y - self.goy) / self.res).astype(np.int64)
        inside = (gi >= 0) & (gi < self.gnx) & (gj >= 0) & (gj < self.gny)
        if np.count_nonzero(inside) < min_pts:
            return None
        flat = gj[inside] * self.gnx + gi[inside]
        ok = ((self.n.reshape(-1)[flat] > 0) & (self.var_min.reshape(-1)[flat] < self.min_var_for_valid)
              & (t_rel - self.t_upd.reshape(-1)[flat] > min_age))
        d = Z[inside][ok] - self.h.reshape(-1)[flat[ok]]
        d = d[np.abs(d) < max_diff]
        if d.size < min_pts:
            return None
        return float(np.median(d))

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

    def scan_tilt(self, flat, X, Y, Z, var_pts, base_xy):
        Hs = self.h.reshape(-1)
        conf = ((self.n.reshape(-1)[flat] >= self.tilt_min_obs)
                & (self.var_min.reshape(-1)[flat] < self.tilt_cell_var))
        r = Z - Hs[flat]
        sel = conf & (np.abs(r) < 0.3)
        if sel.sum() < self.tilt_min_points:
            return None
        dx = X[sel] - base_xy[0]
        dy = Y[sel] - base_xy[1]
        w = 1.0 / np.maximum(var_pts[sel], 1e-6)
        rs = r[sel]
        keep = np.ones(dx.size, bool)
        coef = None
        for _ in range(2):
            coef = weighted_plane_fit(dx[keep], dy[keep], rs[keep], w[keep])
            keep = np.abs(rs - (coef[0] + coef[1] * dx + coef[2] * dy)) < 0.08
            if keep.sum() < self.tilt_min_points:
                return None
        return float(math.hypot(coef[1], coef[2]))

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
        if self.tilt_reject > 0.0 and base_xy is not None:
            self.reject_stats[0] += 1
            sl = self.scan_tilt(flat, X[inside], Y[inside], Z, var_pts, base_xy)
            if sl is not None:
                self.reject_stats[1] += 1
                if sl > self.tilt_reject:
                    self.reject_stats[2] += 1
                    return 0
        self.last_pred = t_rel
        cells, inv = np.unique(flat, return_inverse=True)
        if self.hybrid_k_sigma > 0.0:
            high = high_group_mask(inv, cells.size, Z, var_pts, self.hybrid_k_sigma)
            inv = inv[high]
            Z = Z[high]
            var_pts = var_pts[high]
        h_meas, var_meas = aggregate_cells(inv, cells.size, Z, var_pts)
        return self.integrate_confirmed(cells, h_meas, var_meas, t_rel)

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
        self.cost = self.full_cost()
        self.dirty = None

    def arrays_to_save(self):
        t_rel = self.last_pred if self.last_pred is not None else 0.0
        var = self.var if self.tv_pending else self.var_full(t_rel)
        return dict(h=self.h, var=var, var_min=self.var_min, n=self.n,
                    gnx=self.gnx, gny=self.gny, res=self.res, gox=self.gox, goy=self.goy)


class ElevationCostmapNode(Node):
    def __init__(self):
        super().__init__('elevation_costmap_node')

        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('lidar_frame', 'base_scan')
        self.declare_parameter('lidar_topic', '/scan/points')
        self.declare_parameter('attitude_imu_topic', '')
        self.declare_parameter('out_topic', '/traversability_cost')
        self.declare_parameter('map_name', '')
        self.declare_parameter('update', True)
        self.declare_parameter('publish_period_s', 1.0)
        self.declare_parameter('frozen_publish_period_s', 5.0)
        self.declare_parameter('tf_timeout_s', 0.1)

        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('initial_size_m', 10.0)
        self.declare_parameter('grow_margin_m', 3.0)

        self.declare_parameter('lidar_min_range', 0.4)
        self.declare_parameter('map_write_max_range', 5.0)
        self.declare_parameter('elev_min_rel', -1.0)
        self.declare_parameter('elev_max_rel', 1.0)

        self.declare_parameter('sensor_var_base', 0.0009)
        self.declare_parameter('sensor_var_dist', 0.0004)
        self.declare_parameter('var_init', 1.0)
        self.declare_parameter('var_max', 1.0)
        self.declare_parameter('process_var_rate', 0.0)
        self.declare_parameter('min_var_for_valid', 0.25)
        self.declare_parameter('hybrid_k_sigma', 2.5)
        self.declare_parameter('confirm_s', 2.5)
        self.declare_parameter('change_min_m', 0.08)
        self.declare_parameter('change_k', 2.0)
        self.declare_parameter('change_max_m', 0.25)
        self.declare_parameter('max_gap_s', 3.0)
        self.declare_parameter('min_hits', 3)

        self.declare_parameter('median_ksize', 5)
        self.declare_parameter('rough_win', 5)
        self.declare_parameter('rough_min_known', 6)
        self.declare_parameter('slope_crit', 0.35)
        self.declare_parameter('rough_crit', 0.06)
        self.declare_parameter('step_crit', 0.06)
        self.declare_parameter('slope_dead', 0.08)
        self.declare_parameter('rough_dead', 0.025)
        self.declare_parameter('step_dead', 0.02)
        self.declare_parameter('step_drop_radius_m', 0.0)
        self.declare_parameter('step_rise_radius_m', 0.0)
        self.declare_parameter('step_slope_allow', 0.05)
        self.declare_parameter('step_lethal_m', 0.0)
        self.declare_parameter('step_lethal_radius_m', 0.0)

        self.declare_parameter('raycast_every_n', 10)
        self.declare_parameter('raycast_stride', 4)
        self.declare_parameter('raycast_end_margin', 0.25)
        self.declare_parameter('raycast_sigma_k', 3.0)
        self.declare_parameter('raycast_protect_s', 1.0)

        self.declare_parameter('attitude_max_age_s', 0.3)
        self.declare_parameter('imu_roll_offset_deg', 0.0)
        self.declare_parameter('imu_pitch_offset_deg', 0.0)
        self.declare_parameter('tilt_min_points', 300)
        self.declare_parameter('tilt_reject_deg', 0.0)

        self.declare_parameter('z_align_tau_s', 3.0)
        self.declare_parameter('z_align_min_age_s', 10.0)
        self.declare_parameter('z_align_max_diff_m', 1.5)
        self.declare_parameter('z_align_min_points', 200)
        self.declare_parameter('z_align_max_rel_m', 0.3)
        self.declare_parameter('z_max_step_m', 1.0)
        self.declare_parameter('z_start_range_m', 6.0)
        self.declare_parameter('z_start_max_diff_m', 6.0)
        self.declare_parameter('z_start_min_points', 300)
        self.declare_parameter('z_start_max_wait_s', 10.0)

        self.declare_parameter('still_min_motion_m', 0.0)
        self.declare_parameter('still_min_rotation_deg', 0.0)
        self.declare_parameter('still_period_s', 0.0)

        self.declare_parameter('map_jump_gate_m', 0.25)
        self.declare_parameter('map_jump_gate_deg', 2.0)
        self.declare_parameter('map_jump_hold_s', 1.0)

        gp = self.get_parameter

        self.target = gp('target_frame').value
        self.odom_frame = gp('odom_frame').value
        self.base_frame = gp('base_frame').value
        self.lidar_frame = gp('lidar_frame').value
        self.res = float(gp('resolution').value)
        self.tf_timeout = float(gp('tf_timeout_s').value)
        self.update = bool(gp('update').value)
        self.publish_period = float(gp('publish_period_s').value)

        self.lidar_min_range = float(gp('lidar_min_range').value)
        self.map_write_max_range = float(gp('map_write_max_range').value)
        self.elev_min_rel = float(gp('elev_min_rel').value)
        self.elev_max_rel = float(gp('elev_max_rel').value)
        self.sensor_var_base = float(gp('sensor_var_base').value)
        self.sensor_var_dist = float(gp('sensor_var_dist').value)

        self.raycast_every_n = max(1, int(gp('raycast_every_n').value))
        self.raycast_stride = max(1, int(gp('raycast_stride').value))
        self.raycast_sigma_k = float(gp('raycast_sigma_k').value)

        self.jump_gate = (MapJumpGate(gp('map_jump_gate_m').value, gp('map_jump_gate_deg').value,
                                      gp('map_jump_hold_s').value)
                          if self.target != self.odom_frame else None)

        self.grid = ElevationGrid(
            self.res, float(gp('initial_size_m').value), float(gp('grow_margin_m').value),
            float(gp('var_init').value), float(gp('var_max').value),
            float(gp('process_var_rate').value), float(gp('min_var_for_valid').value),
            float(gp('hybrid_k_sigma').value),
            dict(median_ksize=int(gp('median_ksize').value), rough_win=int(gp('rough_win').value),
                 rough_min_known=int(gp('rough_min_known').value),
                 slope_crit=float(gp('slope_crit').value), rough_crit=float(gp('rough_crit').value),
                 step_crit=float(gp('step_crit').value), slope_dead=float(gp('slope_dead').value),
                 rough_dead=float(gp('rough_dead').value), step_dead=float(gp('step_dead').value),
                 step_drop_r=float(gp('step_drop_radius_m').value),
                 step_rise_r=float(gp('step_rise_radius_m').value),
                 step_allow=float(gp('step_slope_allow').value),
                 step_lethal=float(gp('step_lethal_m').value),
                 step_lethal_r=float(gp('step_lethal_radius_m').value)),
            self.lidar_min_range, float(gp('raycast_end_margin').value), self.map_write_max_range,
            self.raycast_sigma_k, float(gp('raycast_protect_s').value),
            confirm_s=gp('confirm_s').value, change_min_m=gp('change_min_m').value,
            change_k=gp('change_k').value, change_max_m=gp('change_max_m').value,
            max_gap_s=gp('max_gap_s').value, min_hits=gp('min_hits').value,
            tilt_min_points=gp('tilt_min_points').value,
            tilt_reject_deg=float(gp('tilt_reject_deg').value))

        self._ray_calls = 0
        self._ray_cleared = 0
        self._ray_time = 0.0
        self._ray_runs = 0
        self._n_clouds = 0
        self._n_skipped = 0
        self._n_integrated = 0
        self._t_integrate = 0.0
        self._t_integrate_max = 0.0
        self._t_publish = 0.0
        self._n_publish = 0
        self._last_cloud_t = None
        self._last_stats = time.monotonic()

        self.cloud = None
        self.cloud_t = -1e9
        self.cloud_stamp = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, None, spin_thread=True)

        if self.update:
            cloud_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                   history=HistoryPolicy.KEEP_LAST, depth=1)
            self.create_subscription(PointCloud2, gp('lidar_topic').value, self.cloud_cb, cloud_qos)

        self.att = collections.deque(maxlen=200)
        self.att_max_age = float(gp('attitude_max_age_s').value)
        self.n_att = [0, 0]
        self.imu_off = (math.radians(float(gp('imu_roll_offset_deg').value)),
                        math.radians(float(gp('imu_pitch_offset_deg').value)))
        self.z_align_tau = float(gp('z_align_tau_s').value)
        self.z_align_min_age = float(gp('z_align_min_age_s').value)
        self.z_align_max_diff = float(gp('z_align_max_diff_m').value)
        self.z_align_min_points = int(gp('z_align_min_points').value)
        self.z_align_max_rel = float(gp('z_align_max_rel_m').value)
        self.z_corr_sum = 0.0
        self.z_max_step = float(gp('z_max_step_m').value)
        self.zi = None
        self.n_anchor = 0
        self.z_start_range = float(gp('z_start_range_m').value)
        self.z_start_max_diff = float(gp('z_start_max_diff_m').value)
        self.z_start_min_points = int(gp('z_start_min_points').value)
        self.z_start_max_wait = float(gp('z_start_max_wait_s').value)
        self.z_anchored = True
        self.z_anchor_t0 = None
        self.still_min_m = float(gp('still_min_motion_m').value)
        self.still_min_deg = float(gp('still_min_rotation_deg').value)
        self.still_period = float(gp('still_period_s').value)
        self.last_integrated = None
        self.n_still = 0
        if gp('attitude_imu_topic').value:
            self.create_subscription(Imu, gp('attitude_imu_topic').value, self.imu_cb,
                                     qos_profile_sensor_data)

        latched_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(OccupancyGrid, gp('out_topic').value, latched_qos)
        self.pub_range = self.create_publisher(Marker, '/elevation_map_range', 1)
        self.range_pose = None

        self.maps_dir = self.resolve_maps_dir()
        map_name = gp('map_name').value
        file_name = f'elevation_{map_name}' if map_name else ''
        self.map_save_path = os.path.join(self.maps_dir, file_name) if file_name else ''
        if file_name:
            p = os.path.join(self.maps_dir, file_name)
            if not p.endswith('.npz'):
                p += '.npz'
            if os.path.isfile(p):
                self.load_map(p)
                self.z_anchored = not bool(self.grid.n.any())
            else:
                self.get_logger().warn(f'Elevation map to load not found: {p}')
        self.autosave = bool(map_name) and self.update
        self.create_service(Trigger, '~/save_map', self.save_map_cb)

        self.create_timer(self.publish_period if self.update
                          else float(gp('frozen_publish_period_s').value), self.publish_timer)
        if not self.update:
            self.get_logger().info('Frozen elevation map: publishing the loaded map only, no updates.')

        self.get_logger().info(
            f'Elevation map ready: frame {self.target}, resolution {self.res} m, '
            f'output {gp("out_topic").value}.')
        save_hint = self.map_save_path or os.path.join(self.maps_dir, 'elevation_map')
        if self.autosave:
            self.get_logger().info(f'The map is saved on Ctrl-C to {self.map_save_path}')
        self.get_logger().info(
            'Save on request: '
            f'ros2 service call /elevation_costmap_node/save_map std_srvs/srv/Trigger  ->  {save_hint}')

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
            d = os.path.join(share, 'maps')
            os.makedirs(d, exist_ok=True)
            return d
        except Exception:
            d = os.path.join(os.path.expanduser('~'), '.semantic_maps')
            os.makedirs(d, exist_ok=True)
            return d

    def load_map(self, path):
        try:
            d = np.load(path)
            if float(d['res']) != self.res:
                self.get_logger().warn('Elevation map with a different resolution: ignored.')
                return
            self.grid.load(d)
            self.get_logger().info(
                f'Elevation map loaded from {path} ({self.grid.gnx}x{self.grid.gny}).')
        except Exception as e:
            self.get_logger().warn(f'Loading the elevation map failed: {e}')

    def save_map(self, path):
        try:
            if not path.endswith('.npz'):
                path += '.npz'
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            tmp = path[:-4] + '.tmp.npz'
            with open(tmp, 'wb') as f:
                np.savez_compressed(f, **self.grid.arrays_to_save())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            self.get_logger().info(f'Elevation map saved to {path}.')
            return path
        except Exception as e:
            self.get_logger().error(f'Saving the elevation map failed: {e}')
            return None

    def save_map_cb(self, request, response):
        path = self.map_save_path or os.path.join(self.maps_dir, 'elevation_map')
        saved = self.save_map(path)
        response.success = saved is not None
        response.message = f'Saved to {saved}' if saved else 'Save failed'
        return response

    def imu_cb(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if t > 0:
            r, p = quat_roll_pitch(m.orientation)
            self.att.append((t, r - self.imu_off[0], p - self.imu_off[1]))

    def robot_z(self, bx, by, yaw, rp, z_tf, t):
        if self.zi is None:
            self.zi = (z_tf, bx, by, t, t)
            return z_tf
        z, px, py, pt, ta = self.zi
        dx, dy = bx - px, by - py
        if math.hypot(dx, dy) < self.z_max_step:
            c, s = math.cos(yaw), math.sin(yaw)
            d = roll_pitch_matrix(*rp)[:3, :3] @ np.array([c * dx + s * dy, -s * dx + c * dy, 0.0])
            z += float(d[2])
        self.zi = (z, bx, by, t, ta)
        return z

    def anchor_start_z(self, Pm, bx, by, bz):
        if self.z_anchor_t0 is None:
            self.z_anchor_t0 = self.cloud_t
        X, Y, Z = Pm[:, 0], Pm[:, 1], Pm[:, 2]
        g = (np.hypot(X - bx, Y - by) < self.z_start_range) & ((Z - bz) < self.z_align_max_rel) & np.isfinite(Z)
        off = self.grid.vertical_offset(X[g], Y[g], Z[g], 0.0, 0.0, self.z_start_max_diff, self.z_start_min_points)
        if off is not None:
            fine = self.grid.vertical_offset(X[g], Y[g], Z[g] - off, 0.0, 0.0, self.z_align_max_rel, self.z_start_min_points)
            if fine is not None:
                off += fine
            z, px, py, pt, ta = self.zi
            self.zi = (z - off, px, py, pt, ta)
            self.z_anchored = True
            self.get_logger().info(f'Start height aligned to the loaded map: correction {-off:+.3f} m '
                                   f'({int(g.sum())} ground points within {self.z_start_range:.0f} m).')
            return off
        if self.cloud_t - self.z_anchor_t0 > self.z_start_max_wait:
            self.z_anchored = True
            self.get_logger().warn(f'Start height NOT aligned: after {self.z_start_max_wait:.0f} s the robot sees '
                                   'no cell of the loaded map. Writing with the localizer height: '
                                   'start from an area that is already mapped.')
            return 0.0
        self.get_logger().info('Waiting to see the loaded map to align the start height...',
                               throttle_duration_sec=2.0)
        return None

    def align_z(self, X, Y, Z, bz, t, t_rel):
        z, px, py, pt, ta = self.zi
        g = (Z - bz) < self.z_align_max_rel
        off = self.grid.vertical_offset(X[g], Y[g], Z[g], t_rel, self.z_align_min_age,
                                        self.z_align_max_diff, self.z_align_min_points)
        if off is not None:
            fine = self.grid.vertical_offset(X[g], Y[g], Z[g] - off, t_rel, self.z_align_min_age,
                                             0.10, self.z_align_min_points)
            if fine is not None:
                off += fine
        if off is None or self.z_align_tau <= 0.0 or t <= ta:
            self.zi = (z, px, py, pt, t)
            return 0.0
        corr = (1.0 - math.exp(-(t - ta) / self.z_align_tau)) * off
        self.zi = (z - corr, px, py, pt, t)
        self.n_anchor += 1
        self.z_corr_sum += abs(corr)
        return corr

    def attitude_at(self, t):
        if self.att:
            ts = np.fromiter((a[0] for a in self.att), float, len(self.att))
            i = int(np.argmin(np.abs(ts - t)))
            if abs(ts[i] - t) <= self.att_max_age:
                self.n_att[0] += 1
                return self.att[i][1], self.att[i][2]
        self.n_att[1] += 1
        return None

    def cloud_cb(self, msg: PointCloud2):
        try:
            rec = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        except Exception:
            return
        n = int(rec.shape[0])
        if n == 0:
            return
        pts = np.empty((n, 3), dtype=np.float64)
        pts[:, 0] = rec['x']
        pts[:, 1] = rec['y']
        pts[:, 2] = rec['z']
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

    def publish_range_marker(self, stamp):
        if self.range_pose is None:
            return
        bx, by, yaw = self.range_pose
        R = self.map_write_max_range
        ang = np.linspace(-math.pi, math.pi, 73)
        m = Marker()
        m.header.frame_id = self.target
        m.header.stamp = stamp
        m.ns = 'elevation_range'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color.r = 0.1
        m.color.g = 0.8
        m.color.b = 1.0
        m.color.a = 0.9
        m.points = [Point(x=float(bx + R * math.cos(a)), y=float(by + R * math.sin(a)), z=0.05) for a in ang]
        self.pub_range.publish(m)

    def lookup(self, frame, stamp):
        try:
            if self.target != self.odom_frame:
                return lookup_via_odom(self.tf_buffer, self.target, self.odom_frame, frame, stamp, self.tf_timeout)
            return self.tf_buffer.lookup_transform(
                self.target, frame, rclpy.time.Time.from_msg(stamp),
                timeout=Duration(seconds=self.tf_timeout))
        except Exception as e:
            self.get_logger().warn(f'Transform {frame} -> {self.target} not available: {e}',
                                   throttle_duration_sec=2.0)
            return None

    def integrate(self):
        if self.cloud is None:
            return False

        T_scan = self.lookup(self.lidar_frame, self.cloud_stamp)
        T_base = self.lookup(self.base_frame, self.cloud_stamp)
        if T_scan is None or T_base is None:
            return False

        M_scan2map = transform_to_matrix(T_scan)
        rp = None
        if self.att:
            rp = self.attitude_at(self.cloud_t)
            if rp is not None:
                M_tb = transform_to_matrix(T_base)
                M_scan2map = M_tb @ roll_pitch_matrix(*rp) @ np.linalg.inv(M_tb) @ M_scan2map

        bx = T_base.transform.translation.x
        by = T_base.transform.translation.y
        bz_tf = T_base.transform.translation.z
        qr = T_base.transform.rotation
        self.range_pose = (bx, by, math.atan2(2 * (qr.w * qr.z + qr.x * qr.y), 1 - 2 * (qr.y * qr.y + qr.z * qr.z)))
        if self.still_period > 0.0 and self.last_integrated is not None:
            lx, ly, lyaw, lt = self.last_integrated
            dyaw = abs(math.atan2(math.sin(self.range_pose[2] - lyaw), math.cos(self.range_pose[2] - lyaw)))
            if (math.hypot(bx - lx, by - ly) < self.still_min_m and math.degrees(dyaw) < self.still_min_deg
                    and self.cloud_t - lt < self.still_period):
                self.n_still += 1
                return False
        self.last_integrated = (bx, by, self.range_pose[2], self.cloud_t)
        if not self.map_jump_ok(self.cloud_stamp, self.cloud_t, bx, by, bz_tf):
            return False
        bz = bz_tf
        if rp is not None:
            q = T_base.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            bz = self.robot_z(bx, by, yaw, rp, bz_tf, self.cloud_t)
            M_scan2map[2, 3] += bz - bz_tf
        Pm = transform_points(self.cloud, M_scan2map)
        if rp is not None and not self.z_anchored:
            off = self.anchor_start_z(Pm, bx, by, bz)
            if off is None:
                return False
            Pm[:, 2] -= off
            M_scan2map[2, 3] -= off
            bz -= off
        if rp is not None:
            t_ref = self.grid.t_ref if self.grid.t_ref is not None else self.cloud_t
            corr = self.align_z(Pm[:, 0], Pm[:, 1], Pm[:, 2], bz, self.cloud_t, self.cloud_t - t_ref)
            if corr != 0.0:
                Pm[:, 2] -= corr
                M_scan2map[2, 3] -= corr
                bz -= corr
        X = Pm[:, 0]
        Y = Pm[:, 1]
        Z = Pm[:, 2]

        rng = np.sqrt(self.cloud[:, 0]**2 + self.cloud[:, 1]**2 + self.cloud[:, 2]**2)

        t_rel = self.grid.begin(self.cloud_t)
        finite = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
        self.raycast_clear(M_scan2map[:3, 3], Pm[finite], rng[finite], t_rel)

        rel = Z - bz
        dist = np.hypot(X - bx, Y - by)
        ok = (finite
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
                f'Localization jump ({self.jump_gate.last_jump_m:.2f} m, '
                f'{self.jump_gate.last_jump_deg:.1f} deg): map writing paused '
                f'for {self.jump_gate.hold_s:.1f} s ({self.jump_gate.n_jumps} jumps so far).',
                throttle_duration_sec=1.0)
        return ok

    def raycast_clear(self, origin, ends, ranges, t_rel):
        if ends.shape[0] == 0:
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
            f'clouds received {self._n_clouds}, integrated {self._n_integrated}, '
            f'skipped {self._n_skipped} | integration {1000.0 * self._t_integrate / ni:.1f} ms '
            f'(max {1000.0 * self._t_integrate_max:.1f}) | raycast {1000.0 * self._ray_time / nr:.1f} ms, '
            f'{self._ray_cleared} cells cleared | cost and publishing '
            f'{1000.0 * self._t_publish / npub:.1f} ms | grid {self.grid.gnx}x{self.grid.gny}'
            f' | attitude found {self.n_att[0]}, missing {self.n_att[1]}'
            + (f' | IMU height {self.zi[0]:+.3f} m, alignments {self.n_anchor}, total correction {self.z_corr_sum:.2f} m' if self.zi is not None else '')
            + f' | scans skipped while still {self.n_still}'
            + f' | tilt checked {self.grid.reject_stats[1]}/{self.grid.reject_stats[0]}, rejected {self.grid.reject_stats[2]}')

    def publish_timer(self):
        t0 = time.perf_counter()
        self.grid.update_cost()
        if self.grid.t_ref is None and self.update:
            return
        g = self.grid
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.target
        msg.info.resolution = self.res
        msg.info.width = g.gnx
        msg.info.height = g.gny
        msg.info.origin.position.x = float(g.gox)
        msg.info.origin.position.y = float(g.goy)
        msg.info.origin.orientation.w = 1.0
        msg.data = array.array('b', np.ascontiguousarray(g.cost).tobytes())
        self.pub.publish(msg)
        self.publish_range_marker(msg.header.stamp)
        self._t_publish += time.perf_counter() - t0
        self._n_publish += 1


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main():
    rclpy.init()
    node = ElevationCostmapNode()
    signal.signal(signal.SIGHUP, _interrupt)
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
