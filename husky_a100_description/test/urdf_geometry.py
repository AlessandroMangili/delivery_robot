"""Strumenti geometrici per verificare l'URDF dell'Husky A100 senza ROS in esecuzione.

Usa solo xacro (Python), numpy e PyYAML. Calcola la cinematica diretta a giunti fermi,
le pose delle primitive di collisione e una serie di controlli geometrici rispetto
alle dimensioni ufficiali del file YAML.

Formule usate:
  R(rpy) = Rz(yaw) * Ry(pitch) * Rx(roll)            (convenzione URDF)
  T_mondo_link = T_mondo_padre * T_giunto              (giunti a angolo zero)
  punto dentro un box:      |p_i| <= size_i / 2 - tol  per i = x, y, z (nel frame del box)
  punto dentro un cilindro: |p_z| <= L / 2 - tol  e  p_x^2 + p_y^2 <= (r - tol)^2
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import xacro
import yaml

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_XACRO = os.path.join(PKG_DIR, 'urdf', 'husky_a100.urdf.xacro')
DEFAULT_YAML = os.path.join(PKG_DIR, 'config', 'a100_dimensions.yaml')

WHEEL_JOINTS = [
    'front_left_wheel_joint', 'middle_left_wheel_joint', 'rear_left_wheel_joint',
    'front_right_wheel_joint', 'middle_right_wheel_joint', 'rear_right_wheel_joint',
]
# Nomi pubblicati dal diff_drive del container Clearpath (joint_states)
CONTAINER_JOINTS = [
    'front_left_wheel_joint', 'rear_left_wheel_joint',
    'front_right_wheel_joint', 'rear_right_wheel_joint',
]


def load_dims(path=DEFAULT_YAML):
    with open(path) as f:
        return yaml.safe_load(f)


def expand(dims_file=None, use_gazebo=False, urdf_extras=None):
    """Espande l'xacro principale e restituisce l'URDF come stringa XML."""
    mappings = {'use_gazebo': 'true' if use_gazebo else 'false'}
    if dims_file:
        mappings['dimensions_file'] = dims_file
    if urdf_extras:
        mappings['urdf_extras'] = urdf_extras
    doc = xacro.process_file(MAIN_XACRO, mappings=mappings)
    return doc.toprettyxml(indent='  ')


def rpy_to_matrix(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def origin_to_T(elem):
    t = np.eye(4)
    if elem is None:
        return t
    xyz = [float(v) for v in elem.get('xyz', '0 0 0').split()]
    rpy = [float(v) for v in elem.get('rpy', '0 0 0').split()]
    t[:3, :3] = rpy_to_matrix(rpy)
    t[:3, 3] = xyz
    return t


class Model:
    """URDF analizzato: link, giunti, pose dei link a giunti fermi, primitive."""

    def __init__(self, urdf_xml):
        self.root_elem = ET.fromstring(urdf_xml)
        self.links = {l.get('name'): l for l in self.root_elem.findall('link')}
        self.joints = self.root_elem.findall('joint')
        children = {j.find('child').get('link') for j in self.joints}
        roots = [n for n in self.links if n not in children]
        if len(roots) != 1:
            raise ValueError(f'URDF con {len(roots)} radici: {roots}')
        self.root = roots[0]
        self.T = {self.root: np.eye(4)}
        pending = list(self.joints)
        while pending:
            progressed = False
            for j in list(pending):
                parent = j.find('parent').get('link')
                child = j.find('child').get('link')
                if parent in self.T:
                    self.T[child] = self.T[parent] @ origin_to_T(j.find('origin'))
                    pending.remove(j)
                    progressed = True
            if not progressed:
                raise ValueError('Albero dei giunti non connesso')

    def joint(self, name):
        for j in self.joints:
            if j.get('name') == name:
                return j
        return None

    def primitives(self, kind='collision'):
        """Lista di primitive con posa nel frame radice: dict(link, type, dims, T)."""
        prims = []
        for name, link in self.links.items():
            for el in link.findall(kind):
                geom = el.find('geometry')
                t = self.T[name] @ origin_to_T(el.find('origin'))
                box = geom.find('box')
                cyl = geom.find('cylinder')
                if box is not None:
                    dims = [float(v) for v in box.get('size').split()]
                    prims.append({'link': name, 'type': 'box', 'dims': dims, 'T': t})
                elif cyl is not None:
                    dims = [float(cyl.get('radius')), float(cyl.get('length'))]
                    prims.append({'link': name, 'type': 'cylinder', 'dims': dims, 'T': t})
        return prims


def sample_points(prim, n=9):
    """Punti su superficie e interno della primitiva, nel frame radice."""
    if prim['type'] == 'box':
        sx, sy, sz = prim['dims']
        g = np.linspace(-0.5, 0.5, n)
        xx, yy, zz = np.meshgrid(g * sx, g * sy, g * sz, indexing='ij')
        local = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    else:
        r, length = prim['dims']
        rad = np.linspace(0.0, r, n)
        ang = np.linspace(0.0, 2 * math.pi, 4 * n, endpoint=False)
        hz = np.linspace(-length / 2, length / 2, n)
        rr, aa, zz = np.meshgrid(rad, ang, hz, indexing='ij')
        local = np.stack([(rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel(), zz.ravel()], axis=1)
    t = prim['T']
    return local @ t[:3, :3].T + t[:3, 3]


def inside(prim, pts, tol=5e-4):
    """Maschera dei punti strettamente dentro la primitiva (con tolleranza tol)."""
    t = prim['T']
    local = (pts - t[:3, 3]) @ t[:3, :3]
    if prim['type'] == 'box':
        half = np.array(prim['dims']) / 2.0 - tol
        return np.all(np.abs(local) <= half, axis=1)
    r, length = prim['dims']
    return (np.abs(local[:, 2]) <= length / 2 - tol) & \
        (local[:, 0] ** 2 + local[:, 1] ** 2 <= (r - tol) ** 2)


def intersect(a, b):
    return bool(inside(b, sample_points(a)).any() or inside(a, sample_points(b)).any())


def check_geometry(model, dims, tol=1e-3):
    """Restituisce la lista delle violazioni geometriche (vuota se tutto torna)."""
    v = []
    L = dims['overall']['length']
    W = dims['overall']['width']
    H = dims['overall']['height']
    gc = dims['overall']['ground_clearance']
    r = dims['wheels']['radius']
    w = dims['wheels']['width']

    # 1) sei ruote, giunti continui, asse Y, nomi compatibili con il container
    for name in WHEEL_JOINTS:
        j = model.joint(name)
        if j is None:
            v.append(f'giunto ruota mancante: {name}')
            continue
        if j.get('type') != 'continuous':
            v.append(f'{name}: tipo {j.get("type")} invece di continuous')
        axis = [float(a) for a in j.find('axis').get('xyz').split()]
        if not np.allclose(axis, [0, 1, 0]):
            v.append(f'{name}: asse {axis} invece di 0 1 0')
    for name in CONTAINER_JOINTS:
        if model.joint(name) is None:
            v.append(f'nome giunto del container assente: {name}')
    for side in ('left', 'right'):
        j = model.joint(f'middle_{side}_wheel_joint')
        if j is not None:
            mim = j.find('mimic')
            if mim is None or mim.get('joint') != f'front_{side}_wheel_joint':
                v.append(f'middle_{side}_wheel_joint: mimic assente o errato')

    prims = model.primitives('collision')

    # 2) dimensioni delle primitive positive
    for p in prims:
        if any(d <= 0 for d in p['dims']):
            v.append(f'primitiva con dimensione non positiva in {p["link"]}: {p["dims"]}')

    # 3) ruote appoggiate a terra e coerenti con il YAML
    wheels = [p for p in prims if p['link'].endswith('_wheel_link')]
    if len(wheels) != 6:
        v.append(f'trovate {len(wheels)} collisioni ruota invece di 6')
    for p in wheels:
        zmin = sample_points(p)[:, 2].min()
        if abs(zmin) > tol:
            v.append(f'{p["link"]}: punto più basso a z={zmin:.4f} invece di 0')
        if not np.allclose(p['dims'], [r, w]):
            v.append(f'{p["link"]}: dimensioni {p["dims"]} diverse da r={r}, w={w}')

    # 4) ingombro totale = quote ufficiali (le facce esterne devono toccare L/2, W/2, H)
    pts = np.vstack([sample_points(p) for p in prims if all(d > 0 for d in p['dims'])])
    ext = {'x': (pts[:, 0].min(), pts[:, 0].max(), L / 2), 'y': (pts[:, 1].min(), pts[:, 1].max(), W / 2)}
    for ax, (lo, hi, half) in ext.items():
        if hi > half + tol or lo < -half - tol:
            v.append(f'ingombro {ax} [{lo:.4f}, {hi:.4f}] oltre la quota ufficiale +-{half:.4f}')
        elif hi < half - 3 * tol or lo > -half + 3 * tol:
            v.append(f'ingombro {ax} [{lo:.4f}, {hi:.4f}] non raggiunge la quota ufficiale +-{half:.4f}')
    if pts[:, 2].max() > H + tol or pts[:, 2].max() < H - tol:
        v.append(f'altezza massima {pts[:, 2].max():.4f} diversa dalla quota ufficiale {H}')

    # 5) luce da terra del corpo
    body = [p for p in prims if p['link'] == 'base_link']
    if body:
        zb = np.vstack([sample_points(p) for p in body])[:, 2].min()
        if abs(zb - gc) > tol:
            v.append(f'fondo del corpo a z={zb:.4f} invece della luce da terra ufficiale {gc}')

    # 6) nessuna compenetrazione delle ruote (tra loro e con il resto)
    others = [p for p in prims if not p['link'].endswith('_wheel_link')]
    for i, a in enumerate(wheels):
        for b in wheels[i + 1:]:
            if intersect(a, b):
                v.append(f'compenetrazione ruota-ruota: {a["link"]} / {b["link"]}')
        for b in others:
            if intersect(a, b):
                v.append(f'compenetrazione ruota-corpo: {a["link"]} / {b["link"]}')

    # 7) scala della mesh ruota coerente con raggio e larghezza
    mesh_d = dims['mesh']['wheel_native_diameter']
    mesh_w = dims['mesh']['wheel_native_width']
    for link_name, link in model.links.items():
        if not link_name.endswith('_wheel_link'):
            continue
        mesh = link.find('visual/geometry/mesh')
        sc = [float(s) for s in mesh.get('scale').split()]
        if not np.allclose([sc[0] * mesh_d, sc[1] * mesh_w, sc[2] * mesh_d], [2 * r, w, 2 * r]):
            v.append(f'{link_name}: scala mesh {sc} non coerente con r={r}, w={w}')

    # 8) inerzie fisicamente valide (positive e disuguaglianza triangolare)
    for link_name, link in model.links.items():
        it = link.find('inertial/inertia')
        if it is None:
            continue
        ixx, iyy, izz = (float(it.get(k)) for k in ('ixx', 'iyy', 'izz'))
        if min(ixx, iyy, izz) <= 0 or ixx + iyy < izz or ixx + izz < iyy or iyy + izz < ixx:
            v.append(f'{link_name}: inerzia non valida ({ixx}, {iyy}, {izz})')
    return v
