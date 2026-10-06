"""Test geometrici dell'URDF Husky A100.

Caso nominale: il modello espanso con il YAML del pacchetto rispetta gli ingombri del YAML
(lunghezza, larghezza, altezza, luce da terra), ha 6 ruote a terra senza compenetrazioni,
nomi dei giunti compatibili con lo stack Clearpath e mesh scalata correttamente.

Casi di errore sintetici (uno per test, YAML modificato in una cartella temporanea):
  - ruote che si sovrappongono (passo tra assi < diametro)
  - ruote che entrano nel corpo (carreggiata troppo stretta)
  - ruote che escono dall'ingombro ufficiale (raggio troppo grande)
  - chiave mancante nel YAML (xacro deve fallire)
  - paraurti geometricamente impossibile (smusso con lato negativo)
Più due controlli di integrazione: plugin Gazebo e meccanismo urdf_extras.
"""

import copy
import os
import xml.etree.ElementTree as ET

import pytest
import yaml

from urdf_geometry import (DEFAULT_YAML, PKG_DIR, Model, check_geometry, expand,
                           load_dims)


def write_yaml(tmp_path, dims, name='dims.yaml'):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(dims))
    return str(p)


def variant(tmp_path, section, key, value):
    dims = copy.deepcopy(load_dims())
    dims[section][key] = value
    return dims, expand(dims_file=write_yaml(tmp_path, dims))


# ---------------------------------------------------------------- caso nominale

def test_nominale_nessuna_violazione():
    dims = load_dims()
    violations = check_geometry(Model(expand()), dims)
    assert violations == [], '\n'.join(violations)


def test_radice_e_base_link_all_altezza_degli_assi():
    model = Model(expand())
    r = load_dims()['wheels']['radius']
    assert model.root == 'base_footprint'
    assert model.T['base_link'][2, 3] == pytest.approx(r)


def test_mesh_ruota_presente_su_disco():
    model = Model(expand())
    for name, link in model.links.items():
        mesh = link.find('visual/geometry/mesh')
        if mesh is None:
            continue
        uri = mesh.get('filename')
        assert uri.startswith('package://husky_a100_description/')
        path = os.path.join(PKG_DIR, uri.split('package://husky_a100_description/')[1])
        assert os.path.isfile(path), f'mesh mancante: {path}'


# ---------------------------------------------------------- casi di errore

# I valori dei casi di errore sono derivati dal YAML, così restano validi se cambiano le misure.

def test_rileva_ruote_sovrapposte(tmp_path):
    d = load_dims()
    spacing = 2 * d['wheels']['radius'] - 0.01                       # passo < diametro
    dims, urdf = variant(tmp_path, 'wheels', 'axle_spacing', spacing)
    v = check_geometry(Model(urdf), dims)
    assert any('ruota-ruota' in m for m in v), v


def test_rileva_ruote_dentro_il_corpo(tmp_path):
    d = load_dims()
    track = d['chassis']['width']                                     # centro gomma sul fianco del corpo
    dims, urdf = variant(tmp_path, 'wheels', 'track', track)
    v = check_geometry(Model(urdf), dims)
    assert any('ruota-corpo' in m and 'base_link' in m for m in v), v


def test_rileva_ruote_fuori_ingombro(tmp_path):
    d = load_dims()
    radius = d['overall']['length'] / 2 - d['wheels']['axle_spacing'] + 0.03   # s + r > L/2
    dims, urdf = variant(tmp_path, 'wheels', 'radius', radius)
    v = check_geometry(Model(urdf), dims)
    assert any(m.startswith('ingombro x') for m in v), v


def test_chiave_mancante_fa_fallire_xacro(tmp_path):
    dims = copy.deepcopy(load_dims())
    del dims['wheels']['track']
    with pytest.raises(Exception):
        expand(dims_file=write_yaml(tmp_path, dims))


def test_rileva_paraurti_impossibile(tmp_path):
    # tratto dritto più largo dei tratti laterali (ys = W/2 - db/2) -> smusso con lato negativo
    d = load_dims()
    a = d['overall']['width'] / 2 - d['bumpers']['tube_diameter'] / 2 + 0.005
    dims, urdf = variant(tmp_path, 'bumpers', 'straight_half_width', a)
    v = check_geometry(Model(urdf), dims)
    assert any('non positiva' in m and 'bumper' in m for m in v), v


# ------------------------------------------------------- integrazione

def test_plugin_gazebo_sei_ruote():
    dims = load_dims()
    root = ET.fromstring(expand(use_gazebo=True))
    plugins = [p for p in root.iter('plugin') if 'DiffDrive' in p.get('name', '')]
    assert len(plugins) == 1
    p = plugins[0]
    assert len(p.findall('left_joint')) == 3 and len(p.findall('right_joint')) == 3
    assert float(p.find('wheel_separation').text) == pytest.approx(dims['wheels']['track'])
    assert float(p.find('wheel_radius').text) == pytest.approx(dims['wheels']['radius'])
    # senza use_gazebo il modello per il robot reale non contiene plugin
    assert not list(ET.fromstring(expand()).iter('plugin'))


def test_urdf_extras_viene_incluso(tmp_path):
    extras = tmp_path / 'extras.urdf.xacro'
    extras.write_text(
        '<?xml version="1.0"?>\n'
        '<robot xmlns:xacro="http://ros.org/wiki/xacro">\n'
        '  <link name="velodyne"/>\n'
        '  <joint name="velodyne_joint" type="fixed">\n'
        '    <parent link="top_chassis_link"/><child link="velodyne"/>\n'
        '    <origin xyz="0.1 0 0.05" rpy="0 0 0"/>\n'
        '  </joint>\n'
        '</robot>\n')
    model = Model(expand(urdf_extras=str(extras)))
    H = load_dims()['overall']['height']
    assert model.T['velodyne'][2, 3] == pytest.approx(H + 0.05)
