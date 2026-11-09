from typing import List, Optional

import numpy as np

from .eagermot_fusion import Instance, greedy_match, iou_2d, project_box3d


def normalize_angle(a):
    return float((float(a) + np.pi) % (2.0 * np.pi) - np.pi)


def transform_box3d(box, T):
    box = np.asarray(box, dtype=np.float64).reshape(7)
    T = np.asarray(T, dtype=np.float64)
    out = box.copy()
    out[:3] = T[:3, :3] @ box[:3] + T[:3, 3]
    out[6] = normalize_angle(box[6] + float(np.arctan2(T[1, 0], T[0, 0])))
    return out


def scaled_distance(box_a, box_b):
    a = np.asarray(box_a, dtype=np.float64).reshape(7)
    b = np.asarray(box_b, dtype=np.float64).reshape(7)
    d = float(np.linalg.norm(a[:6] - b[:6]))
    alpha = 2.0 - abs(float(np.cos(a[6] - b[6])))
    return d * alpha


class KalmanBox3D:
    def __init__(self, box3d, p_pos=10.0, p_vel=10000.0,
                 q_pos=1.0, q_vel=0.01, r_meas=1.0):
        self.x = np.zeros(10, dtype=np.float64)
        self.x[:7] = np.asarray(box3d, dtype=np.float64).reshape(7)
        self.x[6] = normalize_angle(self.x[6])

        self.P = np.eye(10) * float(p_pos)
        self.P[7:, 7:] = np.eye(3) * float(p_vel)

        self.Q = np.eye(10) * float(q_pos)
        self.Q[7:, 7:] = np.eye(3) * float(q_vel)

        self.R = np.eye(7) * float(r_meas)

        self.H = np.zeros((7, 10))
        self.H[:7, :7] = np.eye(7)

    def predict(self, dt):
        F = np.eye(10)
        F[0, 7] = F[1, 8] = F[2, 9] = float(dt)
        self.x = F @ self.x
        self.x[6] = normalize_angle(self.x[6])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, box3d):
        z = np.asarray(box3d, dtype=np.float64).reshape(7).copy()
        z[6] = normalize_angle(z[6])
        self.x[6] = normalize_angle(self.x[6])

        diff = abs(z[6] - self.x[6])
        if (np.pi / 2.0) < diff < (np.pi * 3.0 / 2.0):
            self.x[6] = normalize_angle(self.x[6] + np.pi)
        if abs(z[6] - self.x[6]) >= (np.pi * 3.0 / 2.0):
            self.x[6] += 2.0 * np.pi if z[6] > 0 else -2.0 * np.pi

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[6] = normalize_angle(self.x[6])
        self.P = (np.eye(10) - K @ self.H) @ self.P

    @property
    def box(self):
        return self.x[:7].copy()

    @property
    def velocity(self):
        return self.x[7:10].copy()


class EagerTrack:
    NEVER = 10 ** 6

    def __init__(self, tid, instance: Instance, kf_kwargs=None):
        self.id = int(tid)
        self._kf_kwargs = dict(kf_kwargs or {})
        self.kf: Optional[KalmanBox3D] = None
        self.score3d: Optional[float] = None
        self.box2d: Optional[np.ndarray] = None
        self.score2d: Optional[float] = None

        if instance.has_3d:
            self.kf = KalmanBox3D(instance.box3d, **self._kf_kwargs)
            self.score3d = instance.score3d
        if instance.has_2d:
            self.box2d = np.asarray(instance.box2d, dtype=np.float64).copy()
            self.score2d = instance.score2d

        self.age_since_update = 0
        self.age_since_2d = 0 if instance.has_2d else self.NEVER
        self.age_since_3d = 0 if instance.has_3d else self.NEVER
        self.updated_now = True
        self.hits = 1
        self.birth_frame = 0

    @property
    def has_3d(self):
        return self.kf is not None

    @property
    def box3d(self):
        return None if self.kf is None else self.kf.box

    @property
    def velocity(self):
        return None if self.kf is None else self.kf.velocity

    def predict(self, dt):
        if self.kf is not None:
            self.kf.predict(dt)
        self.updated_now = False
        self.age_since_update += 1
        self.age_since_2d = min(self.age_since_2d + 1, self.NEVER)
        self.age_since_3d = min(self.age_since_3d + 1, self.NEVER)

    def update_3d(self, box3d, score3d):
        if self.kf is None:
            self.kf = KalmanBox3D(box3d, **self._kf_kwargs)
        else:
            self.kf.update(box3d)
        self.score3d = score3d
        self.age_since_3d = 0
        self._mark_update()

    def update_2d(self, box2d, score2d):
        self.box2d = np.asarray(box2d, dtype=np.float64).copy()
        self.score2d = score2d
        self.age_since_2d = 0
        self._mark_update()

    def _mark_update(self):
        if not self.updated_now:
            self.hits += 1
        self.updated_now = True
        self.age_since_update = 0

    def box2d_for_matching(self, T_cam_world, K, width, height):
        if self.kf is not None:
            p = project_box3d(self.kf.box, T_cam_world, K, width, height)
            if p is not None:
                return p
        return self.box2d

    def is_confirmed(self, age_2d, age_3d=None, coast_frames=0, min_hits_3d=0):
        if self.updated_now:
            associated = True
        else:
            associated = (int(coast_frames) > 0 and
                         self.age_since_update <= int(coast_frames))
        if not associated:
            return False
        if self.age_since_2d <= int(age_2d):
            return True
        if (age_3d is not None and self.age_since_3d <= int(age_3d)
                and self.hits >= int(min_hits_3d)):
            return True
        return False


class EagerMOT:
    def __init__(self, theta_3d=1.8, theta_2d=0.5, age_max=3, age_2d=3,
                 age_3d=None, confirm_coast_frames=0, dup_suppress_radius=0.0,
                 min_hits_3d=0, kf_kwargs=None):
        self.theta_3d = float(theta_3d)
        self.theta_2d = float(theta_2d)
        self.age_max = int(age_max)
        self.age_2d = int(age_2d)
        self.age_3d = None if age_3d is None else int(age_3d)
        self.confirm_coast_frames = int(confirm_coast_frames)
        self.dup_suppress_radius = float(dup_suppress_radius)
        self.min_hits_3d = int(min_hits_3d)
        self.kf_kwargs = dict(kf_kwargs or {})
        self.tracks: List[EagerTrack] = []
        self._next_id = 0
        self._frame = 0
        self.diagnostics = {}

    def step(self, instances: List[Instance], dt, T_cam_world, K, width, height):
        self._frame += 1
        instances = list(instances)

        for t in self.tracks:
            t.predict(dt)

        used = [False] * len(instances)

        idx_i3 = [i for i, x in enumerate(instances) if x.has_3d]
        idx_t3 = [j for j, t in enumerate(self.tracks) if t.has_3d]

        D = np.zeros((len(idx_i3), len(idx_t3)), dtype=np.float64)
        for a, i in enumerate(idx_i3):
            for b, j in enumerate(idx_t3):
                D[a, b] = scaled_distance(instances[i].box3d,
                                          self.tracks[j].box3d)
        pairs1, _, free_tracks1 = greedy_match(D, self.theta_3d, maximize=False)

        for a, b in pairs1:
            inst = instances[idx_i3[a]]
            trk = self.tracks[idx_t3[b]]
            trk.update_3d(inst.box3d, inst.score3d)
            if inst.has_2d:
                trk.update_2d(inst.box2d, inst.score2d)
            used[idx_i3[a]] = True

        orphans1 = [idx_t3[b] for b in free_tracks1]

        idx_i2 = [i for i, x in enumerate(instances) if x.has_2d and not x.has_3d]
        idx_t2 = orphans1 + [j for j, t in enumerate(self.tracks)
                            if not t.has_3d]

        box_t2 = [self.tracks[j].box2d_for_matching(T_cam_world, K,
                                                    width, height)
                  for j in idx_t2]

        I2 = np.zeros((len(idx_i2), len(idx_t2)), dtype=np.float64)
        for a, i in enumerate(idx_i2):
            for b in range(len(idx_t2)):
                if box_t2[b] is not None:
                    I2[a, b] = iou_2d(instances[i].box2d, box_t2[b])
        pairs2, _, _ = greedy_match(I2, self.theta_2d, maximize=True)

        for a, b in pairs2:
            inst = instances[idx_i2[a]]
            trk = self.tracks[idx_t2[b]]
            trk.update_2d(inst.box2d, inst.score2d)
            used[idx_i2[a]] = True

        rescued = suppressed = 0
        if self.dup_suppress_radius > 0.0:
            for i in idx_i3:
                if used[i]:
                    continue
                px, py = instances[i].box3d[0], instances[i].box3d[1]
                nearest, d_min = None, self.dup_suppress_radius
                for t in self.tracks:
                    if not t.has_3d:
                        continue
                    d = float(np.hypot(t.box3d[0] - px, t.box3d[1] - py))
                    if d < d_min:
                        nearest, d_min = t, d
                if nearest is None:
                    continue
                used[i] = True
                if nearest.updated_now:
                    suppressed += 1
                else:
                    nearest.update_3d(instances[i].box3d, instances[i].score3d)
                    if instances[i].has_2d:
                        nearest.update_2d(instances[i].box2d, instances[i].score2d)
                    rescued += 1

        born = 0
        for i, u in enumerate(used):
            if u:
                continue
            t = EagerTrack(self._next_id, instances[i], self.kf_kwargs)
            t.birth_frame = self._frame
            self._next_id += 1
            self.tracks.append(t)
            born += 1

        before = len(self.tracks)
        self.tracks = [t for t in self.tracks
                       if t.age_since_update <= self.age_max]

        self.diagnostics = dict(
            frame=self._frame, instances=len(instances),
            match1=len(pairs1), match2=len(pairs2),
            rescued=rescued, suppressed=suppressed,
            born=born, died=before - len(self.tracks),
            alive=len(self.tracks), confirmed=len(self.confirmed_tracks()))
        return self.tracks

    def confirmed_tracks(self):
        return [t for t in self.tracks
                if t.has_3d and t.is_confirmed(self.age_2d, self.age_3d,
                                            self.confirm_coast_frames,
                                            self.min_hits_3d)]
