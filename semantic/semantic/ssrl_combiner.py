#!/usr/bin/env python3

import numpy as np
import array


def grid_key(info):
    return (info.width, info.height,
            round(info.origin.position.x, 3), round(info.origin.position.y, 3),
            round(info.resolution, 4))


def check_grid_compatible(l_res, l_ox, l_oy, l_quat,
                          r_res, r_ox, r_oy, r_quat,
                          tol_cells=0.01):
    """Verifica le IPOTESI su cui si regge align_to_ref. Ritorna (ok, motivo).

    align_to_ref allinea un layer alla griglia di riferimento con un semplice
    shift INTERO di celle:

        dx = round((l_ox - r_ox) / res)

    Questo e' corretto SOLO se valgono quattro cose, che finora nessuno
    verificava e la cui violazione non produce alcun errore -- solo mappe
    sbagliate in silenzio:

      1. stessa risoluzione. align_to_ref usa la `res` del RIFERIMENTO anche
         per il layer: se differiscono, le celle hanno dimensioni diverse e sia
         lo shift sia la sovrapposizione sono privi di senso.
      2. orientamento identita' su ENTRAMBE le griglie. Una griglia ruotata non
         si allinea con una traslazione di celle.
      3. origini sullo STESSO RETICOLO, cioe' distanti un multiplo intero di
         res. Un residuo sub-cella viene assorbito dal round() -> errore
         sistematico fino a mezza cella (2.5 cm a res 0.05).
      4. risoluzione positiva.

    NB: con i nodi attuali la (3) e' garantita per costruzione (origine iniziale
    = -initial_size_m/2, crescita per numero INTERO di celle). Questa guardia
    non corregge nulla: rende VISIBILE la rottura se un domani cambia un
    parametro (es. una `resolution` diversa fra i due nodi, o un
    `initial_size_m` la cui meta' non e' multipla di res).

    Funzione PURA -> testabile senza ROS.
    l_quat / r_quat: tuple (x, y, z, w).
    """
    if r_res <= 0.0 or l_res <= 0.0:
        return False, f"risoluzione non positiva (layer={l_res}, rif={r_res})"

    if abs(l_res - r_res) > 1e-6:
        return False, (f"risoluzione diversa: layer={l_res:.4f} m, "
                       f"riferimento={r_res:.4f} m")

    for nome, q in (("layer", l_quat), ("riferimento", r_quat)):
        qx, qy, qz, qw = q
        if (abs(qx) > 1e-6 or abs(qy) > 1e-6 or abs(qz) > 1e-6
                or abs(abs(qw) - 1.0) > 1e-6):
            return False, (f"orientamento non identita' sulla griglia {nome}: "
                           f"({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})")

    for asse, lo, ro in (("x", l_ox, r_ox), ("y", l_oy, r_oy)):
        celle = (lo - ro) / r_res
        residuo = abs(celle - round(celle))
        if residuo > tol_cells:
            return False, (f"origini fuori reticolo su {asse}: residuo "
                           f"{residuo:.3f} celle ({residuo * r_res * 100:.1f} cm)")

    return True, ""


def align_to_ref(layer, l_ox, l_oy, ref_ox, ref_oy, res, ref_h, ref_w, fill=-1):
    """Rimappa 'layer' (origine l_ox,l_oy) sulla griglia di riferimento (semantica).
    Semantica ed elevation sono nodi separati con griglie di dimensione/origine
    diverse ma STESSA risoluzione e orientamento -> basta uno shift intero di celle,
    niente interpolazione. Celle di ref non coperte dal layer = fill (-1 = ignoto)."""
    out = np.full((ref_h, ref_w), fill, dtype=np.int16)
    dx = int(round((l_ox - ref_ox) / res))
    dy = int(round((l_oy - ref_oy) / res))
    lh, lw = layer.shape
    r0 = max(dy, 0); r1 = min(dy + lh, ref_h)
    c0 = max(dx, 0); c1 = min(dx + lw, ref_w)
    if r0 >= r1 or c0 >= c1:
        return out
    out[r0:r1, c0:c1] = layer[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
    return out


def fuse_elevation(base, elev, w_elev, soft_ceiling, elev_lethal_thr,
                   elev_can_be_lethal):
    """Fonde il costo di traversabilita' (elevation) sopra la base
    (semantica + confinement). Funzione PURA -> testabile senza ROS.

    base:  ndarray (H,W) float, 0..100, gia' combinata e clippata
    elev:  ndarray (H,W) float, 0..100, gia' riallineata su base; None = disattiva

    REGOLA 1 (mai demozione): il costo finale non e' MAI inferiore alla base.
      Il soft_ceiling limita QUANTO l'elevation puo' AGGIUNGERE, non e' un tetto
      sulla base. Senza questo, una cella letale (100) della semantica veniva
      schiacciata a soft_ceiling (97) -> in Nav2 perde lo stato LETHAL_OBSTACLE
      e non semina piu' inflation.

    REGOLA 2 (letale solo se voluto): elev_can_be_lethal=False significa che
      l'elevation contribuisce SOLO costo soft. Rugosita' e pendenza sono
      terreno scomodo, non muri: promuoverle a 100 le rende letali E le fa
      gonfiare dall'inflation di Nav2 (alone spurio).
    """
    if elev is None:
        return np.clip(base, 0.0, 100.0)

    spazio = np.maximum(soft_ceiling - base, 0.0)
    incremento = (elev / 100.0) * spazio * w_elev
    soft = np.minimum(base + incremento, soft_ceiling)
    # REGOLA 1: l'elevation puo' solo alzare, mai abbassare.
    soft = np.maximum(base, soft)

    if elev_can_be_lethal:
        # REGOLA 2: solo qui l'elevation puo' dichiarare ostacolo fisico.
        soft = np.where(elev >= elev_lethal_thr, 100.0, soft)

    return np.clip(soft, 0.0, 100.0)


# @@@ROS_BOUNDARY@@@  (i test isolati eseguono solo cio' che sta sopra questa riga)

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from nav_msgs.msg import OccupancyGrid


def quat_of(info):
    """(x, y, z, w) dell'orientamento di una MapMetaData."""
    o = info.origin.orientation
    return (float(o.x), float(o.y), float(o.z), float(o.w))


class SSRLCombiner(Node):
    def __init__(self):
        super().__init__('ssrl_combiner')
        self.declare_parameter('semantic_topic', '/semantic_costmap')
        self.declare_parameter('layer_topics', ['/confinement_cost'])
        self.declare_parameter('elevation_topic', '/elevation_cost')
        self.declare_parameter('out_topic', '/ssrl_costmap')
        # --- parametri fusione elevation (traversabilita') ---
        self.declare_parameter('use_elevation', True)        # OFF = semantica+confinement soltanto
        self.declare_parameter('w_elev', 0.6)                # aggressivita' incremento soft
        self.declare_parameter('soft_ceiling', 97)           # tetto di cio' che l'elevation AGGIUNGE
        self.declare_parameter('elev_can_be_lethal', False)  # l'elevation puo' dichiarare 100?
        self.declare_parameter('elev_lethal_thr', 92)        # soglia, usata solo se sopra e' True
        # --- guardia di compatibilita' delle griglie ---
        self.declare_parameter('grid_tol_cells', 0.01)       # residuo di reticolo tollerato [celle]

        self.semantic_topic = self.get_parameter('semantic_topic').value
        self.layer_topics = list(self.get_parameter('layer_topics').value)
        self.elevation_topic = self.get_parameter('elevation_topic').value
        out_topic = self.get_parameter('out_topic').value
        self.use_elevation = bool(self.get_parameter('use_elevation').value)
        self.w_elev = float(self.get_parameter('w_elev').value)
        self.soft_ceiling = float(self.get_parameter('soft_ceiling').value)
        self.elev_can_be_lethal = bool(self.get_parameter('elev_can_be_lethal').value)
        self.elev_lethal_thr = float(self.get_parameter('elev_lethal_thr').value)
        self.grid_tol_cells = float(self.get_parameter('grid_tol_cells').value)

        # la semantica e' pubblicata LATCHED (transient_local): sottoscrivo
        # transient_local per ricevere subito l'ultima mappa anche partendo dopo.
        qos_latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        # i layer (confinement, dinamici) sono volatile normali
        qos_vol = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.layers = {}        # topic -> (info, ndarray)  [layer PURI: confinement]
        self.elev = None        # (info, ndarray) elevation, gestita a parte
        self.last_sem = None    # (key, ndarray, header, info) ultima semantica

        self.create_subscription(OccupancyGrid, self.semantic_topic,
                                 self.semantic_cb, qos_latched)
        for tp in self.layer_topics:
            self.create_subscription(
                OccupancyGrid, tp, lambda m, t=tp: self.layer_cb(t, m), qos_vol)
        # elevation: subscription creata SOLO se abilitata. Cosi' use_elevation=false
        # non lascia residui di stato e non spende CPU sul topic.
        if self.use_elevation:
            self.create_subscription(OccupancyGrid, self.elevation_topic,
                                     self.elevation_cb, qos_vol)

        # output LATCHED: il costmap di Nav2 (o chiunque parta dopo) riceve
        # subito l'ultima /ssrl_costmap.
        self.pub = self.create_publisher(OccupancyGrid, out_topic, qos_latched)
        self.get_logger().info(
            f'SSRL combiner: {self.semantic_topic} + {self.layer_topics} -> {out_topic} | '
            f'elevation={"ON" if self.use_elevation else "OFF"} '
            f'(letale={"si" if self.elev_can_be_lethal else "no"})')

    # -------------------------------------------------------------------------
    def layer_cb(self, topic, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        arr = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.layers[topic] = (msg.info, arr)
        if self.last_sem is not None:
            self.combine_and_publish()

    def elevation_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        arr = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.elev = (msg.info, arr)
        if self.last_sem is not None:
            self.combine_and_publish()

    def semantic_cb(self, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        sem = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.last_sem = (grid_key(msg.info), sem, msg.header, msg.info)
        self.combine_and_publish()

    # -------------------------------------------------------------------------
    def _compatible(self, nome, linfo, ref_info):
        """Guardia: un layer incompatibile viene SCARTATO con un errore chiaro,
        invece di essere fuso producendo una mappa sbagliata in silenzio."""
        ok, motivo = check_grid_compatible(
            float(linfo.resolution), float(linfo.origin.position.x),
            float(linfo.origin.position.y), quat_of(linfo),
            float(ref_info.resolution), float(ref_info.origin.position.x),
            float(ref_info.origin.position.y), quat_of(ref_info),
            tol_cells=self.grid_tol_cells)
        if not ok:
            self.get_logger().error(
                f'Layer "{nome}" INCOMPATIBILE con la griglia semantica: {motivo}. '
                f'Layer SCARTATO da questa fusione.', throttle_duration_sec=5.0)
        return ok

    def combine_and_publish(self):
        key, sem, header, info = self.last_sem
        known = sem >= 0
        ref_ox = info.origin.position.x
        ref_oy = info.origin.position.y
        res = info.resolution
        ref_h, ref_w = sem.shape

        # --- BASE: semantica + layer PURI (confinement), RIALLINEATI alla semantica ---
        base = sem.astype(np.float32).copy()
        for topic, (linfo, arr) in self.layers.items():
            if not self._compatible(topic, linfo, info):
                continue
            a = align_to_ref(arr, linfo.origin.position.x, linfo.origin.position.y,
                             ref_ox, ref_oy, res, ref_h, ref_w, fill=0)  # fill 0: non aggiunge
            add = np.clip(a, 0, 100).astype(np.float32)
            base[known] = base[known] + add[known]
        base = np.clip(base, 0, 100)

        # --- ELEVATION: riallineata, poi fusa (funzione pura) ---
        elev = None
        if self.use_elevation and self.elev is not None:
            einfo, earr = self.elev
            if self._compatible(self.elevation_topic, einfo, info):
                elev = align_to_ref(earr, einfo.origin.position.x,
                                    einfo.origin.position.y,
                                    ref_ox, ref_oy, res, ref_h, ref_w, fill=0)
                elev = np.clip(elev, 0, 100).astype(np.float32)

        total = fuse_elevation(base, elev, self.w_elev, self.soft_ceiling,
                               self.elev_lethal_thr, self.elev_can_be_lethal)

        total = np.clip(np.round(total), 0, 100)
        total[~known] = -1

        out = OccupancyGrid()
        out.header = header
        out.info = info
        out.data = array.array('b', np.ascontiguousarray(
            total.astype(np.int8)).tobytes())
        self.pub.publish(out)


def main():
    rclpy.init()
    node = SSRLCombiner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()