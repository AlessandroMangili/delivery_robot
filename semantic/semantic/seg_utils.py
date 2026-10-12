"""Helper puri per il wrapper di segmentazione semantica.

Nessun ROS, nessun torch, nessuna GPU: solo numpy e cv2, cosi' si testano in
isolamento. Coprono le tre trasformazioni tra la mappa di classe grezza di
YOLO26-sem e l'immagine mono8 di trainId che `semantic_costmap_node` consuma:

  1. remap   - relabel opzionale: id di classe del modello -> trainId Cityscapes
  2. resize  - SOLO nearest-neighbour (un label map non va mai interpolato)
  3. pack    - uint8 C-contiguo, pronto per cv_bridge mono8

Il consumer legge seg[vv, uu] come trainId e lo cerca in una LUT di costo dove
ogni id che non conosce mappa a costo negativo e viene scartato: percio' l'id
di ignore (default 255) e' un valore sicuro di "qui non ho informazione".
"""
import numpy as np
import cv2


def build_remap_lut(pairs, ignore_id=255):
    """LUT uint8 a 256 voci: id di classe del modello -> trainId.

    Identita' di default (lut[k] = k). Ogni (src, dst) in `pairs` sovrascrive
    una voce; gli id non citati passano invariati. Serve ad adattare un
    checkpoint il cui ordine di classe NON e' quello canonico dei trainId
    Cityscapes; per i pesi ufficiali yolo26*-sem `pairs` e' vuoto e la LUT e'
    l'identita'.

    Formula: lut[k] = dst se (k, dst) in pairs, altrimenti k, per k in [0,255].
    """
    lut = np.arange(256, dtype=np.uint8)
    for src, dst in pairs:
        s = int(src)
        d = int(dst)
        if not (0 <= s <= 255):
            raise ValueError('remap source fuori range [0,255]: %d' % s)
        if not (0 <= d <= 255):
            raise ValueError('remap dest fuori range [0,255]: %d' % d)
        lut[s] = d
    _ = ignore_id  # l'ignore sopravvive alla LUT salvo remap esplicito
    return lut


def apply_remap(label_map, lut):
    """Applica una LUT a 256 voci a un label map. Ritorna uint8, stessa shape.

    I valori sono id di classe; il cast a indice uint8 e' sicuro perche'
    Cityscapes (<=18), ADE20K (<=149) e l'ignore (255) stanno tutti in [0,255].
    """
    lm = np.asarray(label_map)
    if lm.dtype != np.uint8:
        if lm.min() < 0 or lm.max() > 255:
            raise ValueError('id di label fuori [0,255]: non impacchettabili in uint8')
        lm = lm.astype(np.uint8)
    return lut[lm]


def resize_label_map(label_map, target_w, target_h):
    """Resize nearest-neighbour di un label map a (target_h, target_w).

    NEAREST e' obbligatorio: qualunque interpolazione (bilinear/area) medierebbe
    id di classe vicini inventando id mai predetti (es. sidewalk=1 accanto a
    terrain=9 -> un 5 spurio sul bordo). Se la mappa e' gia' della dimensione
    richiesta e' una copia no-op.
    """
    lm = np.asarray(label_map)
    h, w = lm.shape[:2]
    if (w, h) == (int(target_w), int(target_h)):
        return lm.copy()
    return cv2.resize(lm, (int(target_w), int(target_h)),
                      interpolation=cv2.INTER_NEAREST)


def to_mono8(label_map):
    """Ritorna una vista uint8 C-contigua pronta per la pubblicazione mono8."""
    lm = np.asarray(label_map)
    if lm.ndim != 2:
        raise ValueError('atteso label map 2D, ottenuto shape %s' % (lm.shape,))
    if lm.dtype != np.uint8:
        if lm.min() < 0 or lm.max() > 255:
            raise ValueError('id di label fuori [0,255]: non impacchettabili in uint8')
        lm = lm.astype(np.uint8)
    return np.ascontiguousarray(lm)
