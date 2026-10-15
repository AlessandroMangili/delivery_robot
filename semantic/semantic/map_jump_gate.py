import math

import numpy as np


class MapJumpGate:
    def __init__(self, max_jump_m, max_jump_deg, hold_s):
        self.max_jump_m = float(max_jump_m)
        self.max_jump_rad = math.radians(float(max_jump_deg))
        self.hold_s = float(hold_s)
        self.prev = None
        self.hold_until = -math.inf
        self.n_jumps = 0
        self.last_jump_m = 0.0
        self.last_jump_deg = 0.0

    def update(self, M_map_odom, base_map_xyz, t):
        M = np.asarray(M_map_odom, dtype=np.float64)
        if self.prev is not None:
            b = np.array([base_map_xyz[0], base_map_xyz[1], base_map_xyz[2], 1.0])
            b_prev = self.prev @ (np.linalg.inv(M) @ b)
            jump_m = math.hypot(b[0] - b_prev[0], b[1] - b_prev[1])
            yaw_c = math.atan2(M[1, 0], M[0, 0])
            yaw_p = math.atan2(self.prev[1, 0], self.prev[0, 0])
            jump_rad = abs((yaw_c - yaw_p + math.pi) % (2.0 * math.pi) - math.pi)
            if jump_m > self.max_jump_m or jump_rad > self.max_jump_rad:
                self.hold_until = t + self.hold_s
                self.n_jumps += 1
                self.last_jump_m = jump_m
                self.last_jump_deg = math.degrees(jump_rad)
        self.prev = M
        return t >= self.hold_until


def tf_matrix(t):
    q = t.transform.rotation
    tr = t.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    M = np.eye(4)
    M[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M[:3, 3] = [tr.x, tr.y, tr.z]
    return M


def matrix_to_transform(M, stamp, parent, child):
    from geometry_msgs.msg import TransformStamped
    R = M[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent
    t.child_frame_id = child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in M[:3, 3])
    t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = float(x), float(y), float(z), float(w)
    return t


def lookup_via_odom(buf, target, odom, frame, stamp, timeout_s):
    import rclpy.time
    from rclpy.duration import Duration
    when = rclpy.time.Time.from_msg(stamp) if stamp is not None else rclpy.time.Time()
    T_of = buf.lookup_transform(odom, frame, when, timeout=Duration(seconds=timeout_s))
    T_mo = buf.lookup_transform(target, odom, rclpy.time.Time())
    return matrix_to_transform(tf_matrix(T_mo) @ tf_matrix(T_of), T_of.header.stamp, target, frame)
