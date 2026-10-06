# husky_a100_description

URDF parametrico del **Clearpath Husky A100** (2010): 6 ruote, tre per lato, accoppiate meccanicamente.
Pacchetto ROS 2 Humble (`ament_cmake`).

Non esiste un URDF ufficiale dell'A100: i `husky_description` pubblicati (stack storico
`clearpath_husky` e `husky/husky`) descrivono l'A200 a 4 ruote. Questo pacchetto usa:

- **dimensioni**: le misure prese sul robot reale (larghezza 620 mm, altezza 340 mm, gomme da 210 mm),
  le quote del manuale (§2.2, Fig. 1) e le misure ricavate dal suo disegno in scala;
- **dal branch `noetic-devel` di `husky/husky_description`**: mesh della ruota, struttura della macro
  `husky_wheel`, materiali, parametri di contatto e meccanismo degli "extras" (licenza BSD, vedi
  `meshes/LICENSE.husky_description`).

Il corpo è fatto di primitive, perché le mesh del corpo A200 hanno un'altra forma.

## Provenienza delle dimensioni

Tutto sta in `config/a100_dimensions.yaml`, con l'origine di ogni valore:

| Etichetta | Significato |
|---|---|
| MISURATO | misurato sul robot reale, prevale sul manuale: larghezza 0.620 m, altezza 0.340 m, diametro gomma 0.210 m |
| UFFICIALE | quote del manuale: lunghezza 860 mm, luce da terra 92 mm (larghezza 605 e altezza 350 sostituite dalle misure reali) |
| DISEGNO | misurate sul disegno della Fig. 1, scala 1.8175 mm/pixel calibrata su 860 e 605 mm (incertezza circa ±3 mm) |
| MESH | bounding box della mesh importata |
| STIMA | da misurare o pesare sul robot (masse) |

`docs/overlay_disegno_ufficiale.png` mostra la prima versione del modello (con le quote del manuale)
sovrapposta alle tre viste del disegno: serve a verificare le misure ricavate dal disegno.

La carreggiata geometrica (0.484 m) è diversa dalla "larghezza effettiva" di 0.50 m che il manuale usa
per la cinematica: quella include lo slittamento della sterzata a pattinamento.

## Frame

```
base_footprint (a terra, sotto il centro del robot)        <- radice
  └ base_link (all'altezza degli assi, z = r)
      ├ {front,middle,rear}_{left,right}_wheel_link        giunti continui, asse Y
      ├ {left,right}_rail_link                             piastre laterali
      ├ {front,rear}_bumper_link                           paraurti tubolari
      └ top_chassis_link                                   centro del piano superiore (montaggio sensori)
```

- X avanti, Y sinistra, Z su (REP 103). Il manuale (§3.5) mette il pannello di stato sul retro: +X è il lato opposto.
- I nomi `front_left_wheel_joint`, `rear_left_wheel_joint`, `front_right_wheel_joint`, `rear_right_wheel_joint`
  sono gli stessi che pubblica lo stack Clearpath nel container, quindi con i suoi `joint_states` le ruote
  girano anche in visualizzazione. Le ruote centrali sono `mimic` delle anteriori.

## Formule principali

- Ruote rispetto a `base_link`: `(x_i, ±T/2, 0)` con `x_i ∈ {+s, 0, −s}`; `base_footprint → base_link` = `(0, 0, r)`.
- Scala della mesh ruota: `s_r = 2r / D_mesh` (assi X, Z), `s_w = w / W_mesh` (asse Y).
- Smussi del corpo: `dx = (Lc − Lt)/2`, `dz = H − z_ch`, `hyp = √(dx² + dz²)`, spessore `d = dx·dz/hyp`,
  rotazione `θ = atan2(−dx, dz)`. Il box ruotato riempie esattamente il triangolo tra parte bassa e parte alta.
- Paraurti: `xb = L/2 − db/2`, `ys = W/2 − db/2`, smusso a 45° di lato `ys − a`. Le facce esterne coincidono
  con l'ingombro ufficiale.
- Inerzie: box `m(b² + c²)/12`; cilindro con asse Y `Ixx = Izz = m(3r² + w²)/12`, `Iyy = m r²/2`.

## Uso

Dipendenze:

```bash
sudo apt install ros-humble-xacro ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
                 ros-humble-joint-state-publisher-gui ros-humble-rviz2 ros-humble-ros-gz \
                 ros-humble-teleop-twist-keyboard
```

Build (copia la cartella in `~/ros2_ws/src/`):

```bash
cd ~/ros2_ws && colcon build --packages-select husky_a100_description && source install/setup.bash
```

Visualizzazione in RViz2, con i cursori per far girare le ruote:

```bash
ros2 launch husky_a100_description display.launch.py
```

Simulazione in Gazebo Fortress:

```bash
ros2 launch husky_a100_description gazebo.launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard     # in un altro terminale
```

Solo l'URDF espanso:

```bash
xacro urdf/husky_a100.urdf.xacro > /tmp/husky_a100.urdf && check_urdf /tmp/husky_a100.urdf
```

Argomenti comuni a entrambi i launch: `dimensions_file:=` (YAML alternativo, per esempio con le tue misure
reali) e `urdf_extras:=` (file xacro con i sensori).

## Aggiungere sensori

Crea un file xacro che aggancia i sensori a `top_chassis_link` (o a `base_footprint`) e passalo con
`urdf_extras:=/percorso/sensori.urdf.xacro`. Il modello base non va toccato. È lo stesso meccanismo di
`HUSKY_URDF_EXTRAS` del pacchetto noetic.

## Test

```bash
colcon test --packages-select husky_a100_description && colcon test-result --verbose
# oppure, senza colcon:
cd test && python3 -m pytest -v
```

I test espandono l'xacro e controllano:

- ingombro uguale a quello del file YAML;
- ruote a terra e luce da terra del corpo;
- nessuna compenetrazione tra le ruote né tra ruote e corpo;
- nomi dei giunti e `mimic`;
- scala della mesh e validità delle inerzie;
- plugin Gazebo e `urdf_extras`.

Ci sono anche casi di errore sintetici (ruote sovrapposte, dentro il corpo, fuori ingombro, chiave YAML
mancante, paraurti impossibile), per verificare che i controlli scattino davvero.
