import numpy as np

N_CLASSES = 19


def cost_vector(class_cost, n_classes=N_CLASSES):
    c = np.full(n_classes, np.nan, dtype=np.float32)
    for k, v in class_cost.items():
        if 0 <= int(k) < n_classes:
            c[int(k)] = float(v)
    return c


def fuse(alpha, flat, cls, w, alpha_max, valid):
    cls = np.asarray(cls)
    ok = (cls >= 0) & (cls < alpha.shape[1]) & (w > 0)
    ok[ok] = valid[cls[ok]]
    flat = flat[ok]
    cls = cls[ok].astype(np.int64)
    w = w[ok].astype(np.float32)
    if flat.size == 0:
        return np.empty(0, dtype=np.int64)
    np.add.at(alpha, (flat, cls), w)
    cells = np.unique(flat)
    s = alpha[cells].sum(axis=1)
    over = s > alpha_max
    if np.any(over):
        alpha[cells[over]] *= (alpha_max / s[over])[:, None]
    return cells


def decay(alpha, unseen, factor):
    if factor < 1.0 and np.any(unseen):
        alpha[unseen] *= np.float32(factor)


def expected_cost(alpha, costs, known, lethal_classes, p_lethal):
    usable = np.isfinite(costs)
    a = alpha[:, usable]
    s = a.sum(axis=1)
    safe = np.where(s > 0, s, 1.0)
    exp_cost = (a @ costs[usable]) / safe
    lethal_mask = np.isin(np.flatnonzero(usable), lethal_classes)
    p_obst = a[:, lethal_mask].sum(axis=1) / safe
    out = np.where(p_obst >= p_lethal, 100.0, np.minimum(exp_cost, 99.0))
    return np.where(known & (s > 0), np.round(out), -1.0).astype(np.float32)


def update_known(known, alpha, alpha_min):
    known |= alpha.sum(axis=1) >= alpha_min
    return known


def majority_class(alpha):
    s = alpha.sum(axis=1)
    return np.where(s > 0, np.argmax(alpha, axis=1), 255).astype(np.uint8)
