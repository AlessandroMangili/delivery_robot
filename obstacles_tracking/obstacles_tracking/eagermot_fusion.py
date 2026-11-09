from dataclasses import dataclass
from typing import List, Optional

import numpy as np


def box3d_corners(box):
    x, y, z, dx, dy, dz, yaw = [float(v) for v in box[:7]]
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    local_corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return local_corners @ R.T + np.array([x, y, z])


def project_box3d(box, T_cam_lidar, K, width, height):
    corners = box3d_corners(box)
    homogeneous = np.hstack([corners, np.ones((8, 1))])
    in_cam = (homogeneous @ T_cam_lidar.T)[:, :3]

    if np.any(in_cam[:, 2] <= 1e-6):
        return None

    u = K[0, 0] * (in_cam[:, 0] / in_cam[:, 2]) + K[0, 2]
    v = K[1, 1] * (in_cam[:, 1] / in_cam[:, 2]) + K[1, 2]

    x1 = max(float(u.min()), 0.0)
    y1 = max(float(v.min()), 0.0)
    x2 = min(float(u.max()), float(width))
    y2 = min(float(v.max()), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def iou_2d(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def greedy_match(affinity, threshold, maximize):
    aff = np.asarray(affinity, dtype=np.float64)
    if aff.size == 0:
        R = aff.shape[0] if aff.ndim == 2 else 0
        C = aff.shape[1] if aff.ndim == 2 else 0
        return [], list(range(R)), list(range(C))

    R, C = aff.shape
    valid = aff > threshold if maximize else aff < threshold
    rr, cc = np.nonzero(valid)
    if len(rr) == 0:
        return [], list(range(R)), list(range(C))

    val = aff[rr, cc]
    order = np.argsort(-val if maximize else val, kind='stable')

    row_taken = np.zeros(R, dtype=bool)
    col_taken = np.zeros(C, dtype=bool)
    pairs = []
    for k in order:
        r, c = int(rr[k]), int(cc[k])
        if row_taken[r] or col_taken[c]:
            continue
        row_taken[r] = True
        col_taken[c] = True
        pairs.append((r, c))

    return (pairs,
            [int(i) for i in np.flatnonzero(~row_taken)],
            [int(j) for j in np.flatnonzero(~col_taken)])


@dataclass
class Instance:
    idx_3d: Optional[int] = None
    idx_2d: Optional[int] = None
    box3d: Optional[np.ndarray] = None
    score3d: Optional[float] = None
    box2d: Optional[np.ndarray] = None
    score2d: Optional[float] = None
    box3d_proj: Optional[np.ndarray] = None

    @property
    def has_3d(self):
        return self.idx_3d is not None

    @property
    def has_2d(self):
        return self.idx_2d is not None

    @property
    def kind(self):
        if self.has_3d and self.has_2d:
            return 'both'
        return '3d' if self.has_3d else '2d'


def fuse_detections(boxes3d, scores3d, boxes2d, scores2d,
                    T_cam_lidar, K, width, height, theta_fusion=0.3):
    b3 = np.zeros((0, 7)) if boxes3d is None else np.asarray(boxes3d, dtype=np.float64).reshape(-1, 7)
    b2 = np.zeros((0, 4)) if boxes2d is None else np.asarray(boxes2d, dtype=np.float64).reshape(-1, 4)
    s3 = np.zeros(len(b3)) if scores3d is None else np.asarray(scores3d, dtype=np.float64).reshape(-1)
    s2 = np.zeros(len(b2)) if scores2d is None else np.asarray(scores2d, dtype=np.float64).reshape(-1)
    if len(s3) != len(b3) or len(s2) != len(b2):
        raise ValueError('scores and boxes have different lengths')

    N, M = len(b3), len(b2)

    projections = [project_box3d(b3[i], T_cam_lidar, K, width, height)
                  for i in range(N)]

    aff = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        if projections[i] is None:
            continue
        for j in range(M):
            aff[i, j] = iou_2d(projections[i], b2[j])

    pairs, only_3d, only_2d = greedy_match(aff, theta_fusion, maximize=True)

    instances: List[Instance] = []
    for i, j in pairs:
        instances.append(Instance(
            idx_3d=i, idx_2d=j,
            box3d=b3[i].copy(), score3d=float(s3[i]),
            box2d=b2[j].copy(), score2d=float(s2[j]),
            box3d_proj=None if projections[i] is None else projections[i].copy()))
    for i in only_3d:
        instances.append(Instance(
            idx_3d=i, box3d=b3[i].copy(), score3d=float(s3[i]),
            box3d_proj=None if projections[i] is None else projections[i].copy()))
    for j in only_2d:
        instances.append(Instance(
            idx_2d=j, box2d=b2[j].copy(), score2d=float(s2[j])))

    return instances


def count_instances(instances):
    b = sum(1 for x in instances if x.kind == 'both')
    t = sum(1 for x in instances if x.kind == '3d')
    d = sum(1 for x in instances if x.kind == '2d')
    return b, t, d
