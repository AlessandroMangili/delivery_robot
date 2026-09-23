#!/usr/bin/env python3
"""Fusione delle osservazioni secondo EagerMOT (Kim, Osep, Leal-Taixe, ICRA 2021).

Implementa il modulo "A. Fusion" del paper: date le detection 3D del LiDAR e le
detection 2D della camera, produce un insieme di ISTANZE, ciascuna delle quali
porta l'informazione di entrambe le modalita' oppure di una sola.

    "Il modulo di fusione associa avidamente le detection 3D a quelle 2D in
     base alla loro sovrapposizione 2D nel piano immagine [...] Ci riferiamo
     alle detection rimanenti (non accoppiate) come osservazioni parziali,
     contenenti informazione su una sola delle due modalita'."

Il punto del metodo -- ed e' il motivo del nome EAGER -- e' che NESSUNA
detection viene scartata. Ogni detection entra in un'istanza, in uno di tre
modi: fusa (3D+2D), solo 3D, solo 2D. La soglia decide unicamente se due
detection descrivono lo STESSO oggetto, non se un oggetto merita di esistere.

Modulo PURO: nessuna dipendenza da ROS, tutto testabile in isolamento.

Convenzioni
-----------
box 3D : ndarray (7,) = [x, y, z, dx, dy, dz, yaw] nel frame del LiDAR,
         con (x,y,z) CENTRO della box (convenzione OpenPCDet) e yaw attorno
         all'asse verticale.
box 2D : ndarray (4,) = [x1, y1, x2, y2] in pixel, angoli alto-sinistra e
         basso-destra.
T_cam_lidar : ndarray (4,4), trasformazione omogenea che porta un punto dal
         frame del LiDAR al frame OTTICO della camera (z in avanti, y in giu').
K      : ndarray (3,3), matrice intrinseca della camera.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ============================================================================
# Geometria: dalla box 3D alla sua impronta nell'immagine
# ============================================================================

def box3d_corners(box):
    """Gli 8 vertici di una box 3D orientata. Ritorna (8,3).

    Ordine locale dei vertici (prima i quattro in basso, poi i quattro in alto),
    con dx lungo l'asse x locale, dy lungo y, dz lungo z (verticale).
    """
    x, y, z, dx, dy, dz, yaw = [float(v) for v in box[:7]]
    hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
    locali = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy,  hz], [hx, -hy,  hz], [hx, hy,  hz], [-hx, hy,  hz],
    ], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return locali @ R.T + np.array([x, y, z])


def project_box3d(box, T_cam_lidar, K, width, height):
    """Proiezione della box 3D nel piano immagine -> box 2D che la racchiude.

    Ritorna (4,) [x1,y1,x2,y2] oppure None se la box non e' visibile.

    Il paper, nella parte multi-camera, dice: "nel caso in cui una detection 3D
    non sia visibile in una particolare camera, consideriamo la sua
    sovrapposizione con le detection 2D in quel piano immagine come vuota".
    Qui "non visibile" significa due cose, entrambe trattate come None:
      - almeno un vertice sta DIETRO il piano immagine (z <= 0): la proiezione
        prospettica di quel vertice non ha senso e produrrebbe una box assurda;
      - la proiezione, ritagliata ai bordi dell'immagine, ha area nulla.
    """
    corners = box3d_corners(box)
    omog = np.hstack([corners, np.ones((8, 1))])
    in_cam = (omog @ T_cam_lidar.T)[:, :3]

    if np.any(in_cam[:, 2] <= 1e-6):
        return None                      # almeno un vertice dietro la camera

    u = K[0, 0] * (in_cam[:, 0] / in_cam[:, 2]) + K[0, 2]
    v = K[1, 1] * (in_cam[:, 1] / in_cam[:, 2]) + K[1, 2]

    x1 = max(float(u.min()), 0.0)
    y1 = max(float(v.min()), 0.0)
    x2 = min(float(u.max()), float(width))
    y2 = min(float(v.max()), float(height))
    if x2 <= x1 or y2 <= y1:
        return None                      # interamente fuori dall'immagine
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def iou_2d(a, b):
    """IoU fra due box 2D allineate agli assi. Ritorna 0.0 se non si toccano."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = ix2 - ix1
    ih = iy2 - iy1
    if iw <= 0.0 or ih <= 0.0:
        return 0.0
    inter = iw * ih
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    unione = area_a + area_b - inter
    return float(inter / unione) if unione > 0.0 else 0.0


# ============================================================================
# Accoppiamento avido (greedy)
# ============================================================================

def greedy_match(affinity, threshold, maximize):
    """Accoppiamento avido uno-a-uno su una matrice di affinita' (R x C).

    Il paper descrive la procedura cosi', per la fusione:
        "ordiniamo tutte le possibili coppie per sovrapposizione decrescente.
         Le coppie si considerano una alla volta e si combinano [...] quando
         (i) la loro sovrapposizione e' sopra una soglia e (ii) ne' la
         detection 2D ne' la 3D sono gia' state accoppiate."
    e poi, per il secondo stadio: "questo stadio di associazione e' identico al
    primo, tranne che qui usiamo l'IoU delle box 2D come metrica". Da cui una
    sola implementazione riusata tre volte (fusione, stadio 1, stadio 2).

    maximize=True  -> affinita' tipo IoU: coppia valida se valore >  threshold
    maximize=False -> affinita' tipo distanza: valida se valore <  threshold
                      (lo stadio 1 usa la distanza scalata, dove piu' piccolo
                      e' meglio)

    NB: la soglia e' STRETTA in entrambi i casi. Con IoU = 0 esatto (box che non
    si toccano affatto) e threshold = 0, la coppia viene rifiutata: e' il
    comportamento voluto, altrimenti due box disgiunte verrebbero fuse.

    Ritorna (coppie, righe_libere, colonne_libere) con coppie = [(r, c), ...]
    ordinate dalla migliore alla peggiore.
    """
    aff = np.asarray(affinity, dtype=np.float64)
    if aff.size == 0:
        R = aff.shape[0] if aff.ndim == 2 else 0
        C = aff.shape[1] if aff.ndim == 2 else 0
        return [], list(range(R)), list(range(C))

    R, C = aff.shape
    valide = aff > threshold if maximize else aff < threshold
    rr, cc = np.nonzero(valide)
    if len(rr) == 0:
        return [], list(range(R)), list(range(C))

    val = aff[rr, cc]
    # migliore per primo: decrescente se massimizziamo, crescente se distanza
    ordine = np.argsort(-val if maximize else val, kind='stable')

    riga_presa = np.zeros(R, dtype=bool)
    col_presa = np.zeros(C, dtype=bool)
    coppie = []
    for k in ordine:
        r, c = int(rr[k]), int(cc[k])
        if riga_presa[r] or col_presa[c]:
            continue
        riga_presa[r] = True
        col_presa[c] = True
        coppie.append((r, c))

    return (coppie,
            [int(i) for i in np.flatnonzero(~riga_presa)],
            [int(j) for j in np.flatnonzero(~col_presa)])


# ============================================================================
# Istanze
# ============================================================================

@dataclass
class Istanza:
    """Un'osservazione fusa, o parziale.

    idx_3d / idx_2d: indice nella lista di detection originale, None se assente.
    Le tre forme possibili corrispondono ai tre insiemi del paper:
        both -> idx_3d e idx_2d entrambi presenti
        3d   -> solo idx_3d   (osservazione parziale)
        2d   -> solo idx_2d   (osservazione parziale)
    """
    idx_3d: Optional[int] = None
    idx_2d: Optional[int] = None
    box3d: Optional[np.ndarray] = None       # (7,) [x,y,z,dx,dy,dz,yaw]
    score3d: Optional[float] = None
    box2d: Optional[np.ndarray] = None       # (4,) [x1,y1,x2,y2]
    score2d: Optional[float] = None
    box3d_proj: Optional[np.ndarray] = None  # proiezione della box3d, per debug

    @property
    def ha_3d(self):
        return self.idx_3d is not None

    @property
    def ha_2d(self):
        return self.idx_2d is not None

    @property
    def tipo(self):
        if self.ha_3d and self.ha_2d:
            return 'both'
        return '3d' if self.ha_3d else '2d'


def fuse_detections(boxes3d, scores3d, boxes2d, scores2d,
                    T_cam_lidar, K, width, height, theta_fusion=0.3):
    """Modulo di fusione di EagerMOT. Ritorna la lista di istanze.

    boxes3d : (N,7) o None/vuoto
    scores3d: (N,)  o None
    boxes2d : (M,4) o None/vuoto
    scores2d: (M,)  o None
    theta_fusion: soglia di IoU sopra la quale una coppia 3D-2D viene fusa.
        Riferimento del paper: 0.3 su NuScenes (0.01 su KITTI).

    GARANZIA (verificata dai test): ogni detection compare in ESATTAMENTE una
    istanza. Il numero di istanze e' N + M - (numero di coppie fuse).
    """
    b3 = np.zeros((0, 7)) if boxes3d is None else np.asarray(boxes3d, dtype=np.float64).reshape(-1, 7)
    b2 = np.zeros((0, 4)) if boxes2d is None else np.asarray(boxes2d, dtype=np.float64).reshape(-1, 4)
    s3 = np.zeros(len(b3)) if scores3d is None else np.asarray(scores3d, dtype=np.float64).reshape(-1)
    s2 = np.zeros(len(b2)) if scores2d is None else np.asarray(scores2d, dtype=np.float64).reshape(-1)
    if len(s3) != len(b3) or len(s2) != len(b2):
        raise ValueError('scores e boxes hanno lunghezze diverse')

    N, M = len(b3), len(b2)

    # --- proiezione di ogni box 3D nel piano immagine ---
    proiezioni = [project_box3d(b3[i], T_cam_lidar, K, width, height)
                  for i in range(N)]

    # --- matrice di sovrapposizione: IoU(proiezione della 3D, box 2D) ---
    # Le box 3D non visibili nell'immagine hanno sovrapposizione VUOTA con
    # tutto, quindi non possono essere fuse: restano istanze solo-3D.
    aff = np.zeros((N, M), dtype=np.float64)
    for i in range(N):
        if proiezioni[i] is None:
            continue
        for j in range(M):
            aff[i, j] = iou_2d(proiezioni[i], b2[j])

    coppie, soli_3d, soli_2d = greedy_match(aff, theta_fusion, maximize=True)

    istanze: List[Istanza] = []
    for i, j in coppie:
        istanze.append(Istanza(
            idx_3d=i, idx_2d=j,
            box3d=b3[i].copy(), score3d=float(s3[i]),
            box2d=b2[j].copy(), score2d=float(s2[j]),
            box3d_proj=None if proiezioni[i] is None else proiezioni[i].copy()))
    for i in soli_3d:
        istanze.append(Istanza(
            idx_3d=i, box3d=b3[i].copy(), score3d=float(s3[i]),
            box3d_proj=None if proiezioni[i] is None else proiezioni[i].copy()))
    for j in soli_2d:
        istanze.append(Istanza(
            idx_2d=j, box2d=b2[j].copy(), score2d=float(s2[j])))

    return istanze


def conta_istanze(istanze):
    """(both, solo_3d, solo_2d) -- comodo per la diagnostica."""
    b = sum(1 for x in istanze if x.tipo == 'both')
    t = sum(1 for x in istanze if x.tipo == '3d')
    d = sum(1 for x in istanze if x.tipo == '2d')
    return b, t, d
