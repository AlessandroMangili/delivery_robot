#!/usr/bin/env python3
"""Combina semantica + confinamento + elevation in /ssrl_costmap.

CONFINEMENT COLLASSATO DENTRO IL COMBINER
-----------------------------------------
Il campo di confinamento e' una FUNZIONE PURA della mappa semantica: maschera
delle celle marciapiede, distance transform, esponenziale decrescente. Finche'
viveva in un nodo separato, la semantica raggiungeva il combiner per DUE strade
di durata diversa -- una diretta (~1 ms) e una attraverso il nodo confinement
(~30 ms) -- che riconvergevano portando istantanee DIVERSE. E' un rombo di
dipendenza, e l'incoerenza che produce si chiama glitch.

L'effetto concreto: quando una cella di bordo cambia classe (tipicamente
sidewalk -> terrain mentre si scopre una chiazza d'erba), per qualche decina di
ms il combiner aveva la semantica NUOVA (terrain, 70) e il confinamento
calcolato sulla VECCHIA (35, il massimo, perche' la cella e' proprio sul bordo).
La somma 70 + 35 saturava a 100 = LETHAL_OBSTACLE, con inflation al seguito:
la cella nera transitoria.

Calcolando il campo QUI, dalla stessa `sem` che viene fusa, lo stato sbagliato
diventa inesprimibile: non esiste una variabile che possa contenere "un
confinamento di un'altra istantanea". L'invariante

    conf > 0  =>  la cella e' marciapiede  =>  sem <= sidewalk_cost_max

smette di essere una convenzione fra due nodi e diventa una proprieta' di tre
righe di numpy. Il massimo possibile della somma e'
    sidewalk_cost_max + conf_max * exp(-risoluzione/falloff)
cioe' 15 + 35.3 = 50.3 con i parametri attuali: strutturalmente lontano da 100.
"""

import numpy as np
import cv2
import array


def grid_key(info):
    return (info.width, info.height,
            round(info.origin.position.x, 3), round(info.origin.position.y, 3),
            round(info.resolution, 4))


def stamp_key(header):
    """Identita' dell'istantanea a cui un messaggio appartiene.
    Serve solo per eventuali layer ESTERNI derivati dalla semantica: il
    confinamento non ne ha piu' bisogno, essendo calcolato in locale."""
    return (int(header.stamp.sec), int(header.stamp.nanosec))


def confinement_field(sem, res, sidewalk_cost_max, conf_max, falloff,
                      include_unknown_as_edge):
    """Campo di confinamento dal marciapiede. Funzione PURA -> testabile senza ROS.

    sem: ndarray (H,W) di costo semantico, -1 = ignoto, 0..100 = noto
    res: [m/cella]

    Marciapiede = celle NOTE con costo basso. La distance transform da', per
    ogni cella di marciapiede, la distanza dal bordo (= dalla cella non
    marciapiede piu' vicina). Il costo decade esponenzialmente verso l'interno:
    alto al bordo, trascurabile al centro.

    L'ultima riga e' l'invariante che rende impossibile la saturazione letale:
    il campo esiste SOLO sul marciapiede, quindi non puo' mai sommarsi a una
    classe di costo alto (terrain 70, road 90).
    """
    sem = np.asarray(sem)
    sidewalk = (sem >= 0) & (sem <= sidewalk_cost_max)
    if not sidewalk.any():
        return np.zeros(sem.shape, dtype=np.float32)

    src = sidewalk.astype(np.uint8) * 255
    if not include_unknown_as_edge:
        # l'ignoto non genera bordo: lo "tappiamo" perche' non faccia da zero
        src[sem < 0] = 255

    # distanceTransform misura la distanza dallo zero piu' vicino: senza zeri il
    # risultato non e' definito. Nessun bordo -> nessun confinamento.
    if not (src == 0).any():
        return np.zeros(sem.shape, dtype=np.float32)

    dist_m = cv2.distanceTransform(src, cv2.DIST_L2, 5) * float(res)
    conf = float(conf_max) * np.exp(-dist_m / float(falloff))
    conf[~sidewalk] = 0.0        # INVARIANTE: confinamento solo sul marciapiede
    return conf.astype(np.float32)


def check_grid_compatible(l_res, l_ox, l_oy, l_quat,
                          r_res, r_ox, r_oy, r_quat,
                          tol_cells=0.01):
    """Verifica le IPOTESI su cui si regge align_to_ref. Ritorna (ok, motivo).

    align_to_ref allinea un layer con uno shift INTERO di celle:
        dx = round((l_ox - r_ox) / res)
    Corretto solo se: stessa risoluzione, orientamento identita' su entrambe le
    griglie, origini sullo stesso reticolo, risoluzione positiva. La violazione
    non produce errori -- solo mappe sbagliate in silenzio.

    NB: il confinamento non passa piu' di qui (nasce gia' sulla griglia di
    riferimento). Resta per l'elevation e per eventuali layer esterni.

    Funzione PURA. l_quat / r_quat: tuple (x, y, z, w).
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
    Stessa risoluzione e orientamento -> basta uno shift intero di celle, niente
    interpolazione. Celle di ref non coperte dal layer = fill."""
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
    (semantica + confinamento). Funzione PURA.

    REGOLA 1 (mai demozione): il costo finale non e' MAI inferiore alla base.
      Il soft_ceiling limita QUANTO l'elevation puo' AGGIUNGERE, non e' un tetto
      sulla base. Senza questo una cella letale (100) veniva schiacciata a 97 ->
      in Nav2 perde lo stato LETHAL_OBSTACLE e non semina piu' inflation.

    REGOLA 2 (letale solo se voluto): con elev_can_be_lethal=False l'elevation
      contribuisce SOLO costo soft. Rugosita' e pendenza sono terreno scomodo,
      non muri: promuoverle a 100 le rende letali E le fa gonfiare
      dall'inflation di Nav2 (alone spurio).
    """
    if elev is None:
        return np.clip(base, 0.0, 100.0)

    spazio = np.maximum(soft_ceiling - base, 0.0)
    incremento = (elev / 100.0) * spazio * w_elev
    soft = np.minimum(base + incremento, soft_ceiling)
    soft = np.maximum(base, soft)          # REGOLA 1

    if elev_can_be_lethal:
        soft = np.where(elev >= elev_lethal_thr, 100.0, soft)   # REGOLA 2

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
        self.declare_parameter('elevation_topic', '/elevation_cost')
        self.declare_parameter('out_topic', '/ssrl_costmap')
        # layer ESTERNI aggiuntivi: vuoto di default, il confinamento non
        # arriva piu' da fuori. Il meccanismo resta per usi futuri.
        self.declare_parameter('layer_topics', [''])
        self.declare_parameter('require_layer_stamp_match', True)

        # --- confinamento, calcolato QUI (stessi parametri del vecchio nodo) ---
        self.declare_parameter('use_confinement', True)
        self.declare_parameter('sidewalk_cost_max', 15)   # celle <= questo = marciapiede
        self.declare_parameter('conf_max', 40.0)          # costo al bordo
        self.declare_parameter('falloff', 0.4)            # [m] lunghezza di decadimento
        self.declare_parameter('include_unknown_as_edge', False)
        self.declare_parameter('publish_confinement', True)   # ripubblica per RViz
        self.declare_parameter('confinement_topic', '/confinement_cost')

        # --- fusione elevation ---
        self.declare_parameter('use_elevation', True)
        self.declare_parameter('w_elev', 0.6)
        self.declare_parameter('soft_ceiling', 97)
        self.declare_parameter('elev_can_be_lethal', False)
        self.declare_parameter('elev_lethal_thr', 92)
        # --- guardia di compatibilita' delle griglie ---
        self.declare_parameter('grid_tol_cells', 0.01)

        gp = self.get_parameter
        self.semantic_topic = gp('semantic_topic').value
        self.elevation_topic = gp('elevation_topic').value
        out_topic = gp('out_topic').value
        self.layer_topics = [t for t in gp('layer_topics').value if t]
        self.require_stamp_match = bool(gp('require_layer_stamp_match').value)

        self.use_confinement = bool(gp('use_confinement').value)
        self.sidewalk_cost_max = int(gp('sidewalk_cost_max').value)
        self.conf_max = float(gp('conf_max').value)
        self.falloff = float(gp('falloff').value)
        self.include_unknown_as_edge = bool(gp('include_unknown_as_edge').value)
        self.publish_confinement = bool(gp('publish_confinement').value)

        self.use_elevation = bool(gp('use_elevation').value)
        self.w_elev = float(gp('w_elev').value)
        self.soft_ceiling = float(gp('soft_ceiling').value)
        self.elev_can_be_lethal = bool(gp('elev_can_be_lethal').value)
        self.elev_lethal_thr = float(gp('elev_lethal_thr').value)
        self.grid_tol_cells = float(gp('grid_tol_cells').value)

        qos_latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_vol = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self.layers = {}        # topic -> (info, ndarray, stamp)  [layer esterni]
        self.elev = None        # (info, ndarray)
        self.last_sem = None    # (key, ndarray, header, info)

        self.create_subscription(OccupancyGrid, self.semantic_topic,
                                 self.semantic_cb, qos_latched)
        for tp in self.layer_topics:
            self.create_subscription(
                OccupancyGrid, tp, lambda m, t=tp: self.layer_cb(t, m), qos_vol)
        if self.use_elevation:
            self.create_subscription(OccupancyGrid, self.elevation_topic,
                                     self.elevation_cb, qos_vol)

        self.pub = self.create_publisher(OccupancyGrid, out_topic, qos_latched)
        self.pub_conf = None
        if self.publish_confinement:
            self.pub_conf = self.create_publisher(
                OccupancyGrid, gp('confinement_topic').value, qos_latched)

        self.get_logger().info(
            f'SSRL combiner: {self.semantic_topic} -> {out_topic} | '
            f'confinamento CALCOLATO QUI: '
            f'{"ON" if self.use_confinement else "OFF"} '
            f'(sidewalk<={self.sidewalk_cost_max}, max={self.conf_max:.0f}, '
            f'falloff={self.falloff}) | '
            f'elevation={"ON" if self.use_elevation else "OFF"} '
            f'(letale={"si" if self.elev_can_be_lethal else "no"}) | '
            f'layer esterni={self.layer_topics if self.layer_topics else "nessuno"}')

    # -------------------------------------------------------------------------
    def layer_cb(self, topic, msg: OccupancyGrid):
        h, w = msg.info.height, msg.info.width
        arr = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.layers[topic] = (msg.info, arr, stamp_key(msg.header))
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
        """Un layer incompatibile viene SCARTATO con un errore chiaro, invece di
        essere fuso producendo una mappa sbagliata in silenzio."""
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

    def _grid_msg(self, arr, header, info):
        msg = OccupancyGrid()
        msg.header = header
        msg.info = info
        msg.data = array.array('b', np.ascontiguousarray(
            np.clip(np.round(arr), -1, 100).astype(np.int8)).tobytes())
        return msg

    def combine_and_publish(self):
        key, sem, header, info = self.last_sem
        known = sem >= 0
        ref_ox = info.origin.position.x
        ref_oy = info.origin.position.y
        res = info.resolution
        ref_h, ref_w = sem.shape
        sem_stamp = stamp_key(header)

        base = sem.astype(np.float32).copy()

        # --- CONFINAMENTO: derivato da QUESTA sem, non da un topic ---
        # Nessun allineamento e nessuna guardia: nasce gia' sulla griglia di
        # riferimento. Nessuno stamp da confrontare: e' la stessa istantanea
        # per costruzione.
        if self.use_confinement:
            conf = confinement_field(
                sem, res, self.sidewalk_cost_max, self.conf_max,
                self.falloff, self.include_unknown_as_edge)
            # conf e' gia' 0 fuori dal marciapiede, e il marciapiede e' un
            # sottoinsieme di `known`: sommare direttamente e' equivalente a
            # mascherare, senza il costo della maschera.
            base = base + conf
            if self.pub_conf is not None:
                self.pub_conf.publish(self._grid_msg(conf, header, info))

        # --- layer ESTERNI (nessuno di default) ---
        for topic, (linfo, arr, lstamp) in self.layers.items():
            if self.require_stamp_match and lstamp != sem_stamp:
                self.get_logger().debug(
                    f'Layer "{topic}" da un\'altra istantanea: saltato.')
                continue
            if not self._compatible(topic, linfo, info):
                continue
            a = align_to_ref(arr, linfo.origin.position.x, linfo.origin.position.y,
                             ref_ox, ref_oy, res, ref_h, ref_w, fill=0)
            base[known] = base[known] + np.clip(a, 0, 100).astype(np.float32)[known]

        base = np.clip(base, 0, 100)

        # --- ELEVATION: costruita dal LiDAR con un ciclo proprio, quindi NON
        # derivata dalla semantica e non soggetta al problema delle istantanee.
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
        self.pub.publish(self._grid_msg(total, header, info))


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