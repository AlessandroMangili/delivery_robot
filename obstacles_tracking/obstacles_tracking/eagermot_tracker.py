#!/usr/bin/env python3
"""Tracking-by-detection secondo EagerMOT (Kim, Osep, Leal-Taixe, ICRA 2021).

Consuma le ISTANZE prodotte da eagermot_fusion.fuse_detections e mantiene le
tracce nel tempo. Implementa i moduli "B. Matching" e "C. Track lifecycle".

Struttura di un frame, nell'ordine esatto del paper:

  0. PREDICT   ogni traccia con stato 3D avanza il proprio Kalman.
               "Per ogni nuovo frame t+1, le tracce esistenti predicono la
                loro posizione nel frame corrente."
               Le box PREDETTE sono cio' che lo stadio 1 confronta.

  1. STADIO 1  istanze CON box 3D  vs  tracce CON stato 3D, accoppiate
               avidamente sulla DISTANZA SCALATA (eq. 1-2 del paper).
               "Dopo questo primo stadio, tutte le istanze rilevate in 3D
                devono essere associate o etichettate come non accoppiate, e
                non parteciperanno a ulteriori accoppiamenti."

  2. STADIO 2  istanze SOLO-2D  vs  (tracce orfane dello stadio 1 + tracce
               senza stato 3D), accoppiate sull'IoU 2D nell'immagine.

  3. UPDATE    lo stato 2D si sovrascrive; lo stato 3D si filtra con Kalman
               SOLO se e' arrivata una misura 3D. Altrimenti resta il solo
               predict del passo 0: la traccia coasta.
               "Quando l'informazione di detection 3D non e' disponibile,
                eseguiamo solo il passo di predizione del filtro di Kalman."

  4. LIFECYCLE nascita, conferma, morte.

Modulo PURO: nessuna dipendenza da ROS.
"""

from typing import List, Optional

import numpy as np

from .eagermot_fusion import Istanza, greedy_match, iou_2d, project_box3d


# ============================================================================
# Utilita' geometriche
# ============================================================================

def normalizza_angolo(a):
    """Riporta un angolo in [-pi, pi)."""
    return float((float(a) + np.pi) % (2.0 * np.pi) - np.pi)


def transform_box3d(box, T):
    """Porta una box [x,y,z,dx,dy,dz,yaw] in un altro frame con una 4x4.

    Le dimensioni non cambiano; centro e yaw si'. Va usata per portare le
    detection dal frame del LiDAR a quello FISSO (odom) PRIMA di tracciare: un
    modello a velocita' costante nel frame di un robot che si muove non
    descrive il moto del pedone, ma quello relativo -- e sarebbe sbagliato.
    """
    box = np.asarray(box, dtype=np.float64).reshape(7)
    T = np.asarray(T, dtype=np.float64)
    out = box.copy()
    out[:3] = T[:3, :3] @ box[:3] + T[:3, 3]
    # lo yaw ruota di quanto ruota l'asse x del frame attorno alla verticale
    out[6] = normalizza_angolo(box[6] + float(np.arctan2(T[1, 0], T[0, 0])))
    return out


def scaled_distance(box_a, box_b):
    """Distanza scalata di EagerMOT fra due box 3D orientate (eq. 1 e 2).

        d(Bi, Bj) = || Brho_i - Brho_j || * alpha(Bi, Bj)
        alpha     = 2 - cos< Bgamma_i , Bgamma_j >        in [1, 2]

    dove Brho = [x, y, z, h, w, l] contiene posizione E DIMENSIONI della box, e
    Bgamma e' l'orientamento attorno all'asse verticale.

    Due cose facili da sbagliare, entrambe volute:
      - la norma e' su SEI componenti, non tre: due box nello stesso punto ma
        di taglia diversa hanno distanza non nulla;
      - alpha e' un MOLTIPLICATORE, non un addendo: un orientamento ortogonale
        RADDOPPIA la distanza invece di spostarla di una costante.

    DISCREPANZA NEL PAPER, e come la risolviamo. Preso alla lettera,
    2 - cos(x) con il coseno in [-1,1] varia in [1,3], mentre il paper dichiara
    esplicitamente alpha in [1,2]. Perche' il campo dichiarato valga, il coseno
    dev'essere in [0,1], cioe' l'orientamento va considerato MODULO PI: si usa
    quindi |cos|. Non e' una forzatura ma la lettura geometricamente corretta,
    perche' una box con yaw theta e una con yaw theta+pi sono la stessa box --
    la stessa ambiguita' che AB3DMOT disambigua nell'update del Kalman. Con
    |cos|: orientamenti uguali o opposti danno alpha=1 (nessuna penalita',
    giustamente), orientamenti ortogonali danno alpha=2.

    Il paper la preferisce all'IoU 3D e alla distanza di Mahalanobis, "in
    particolare a basso frame rate", perche' non degenera quando la box
    predetta e quella osservata non si sovrappongono affatto -- caso in cui
    l'IoU 3D vale zero per tutte le coppie e non ordina piu' nulla.
    """
    a = np.asarray(box_a, dtype=np.float64).reshape(7)
    b = np.asarray(box_b, dtype=np.float64).reshape(7)
    d = float(np.linalg.norm(a[:6] - b[:6]))
    alpha = 2.0 - abs(float(np.cos(a[6] - b[6])))
    return d * alpha


# ============================================================================
# Filtro di Kalman a velocita' costante (come AB3DMOT)
# ============================================================================

class KalmanBox3D:
    """Stato 3D di una traccia: box orientata + velocita' POSIZIONALE.

        x = [x, y, z, dx, dy, dz, yaw, vx, vy, vz]      (10)
        z = [x, y, z, dx, dy, dz, yaw]                  (7)

    ATTENZIONE all'ordine: lo yaw e' in posizione 6, non 3. AB3DMOT usa
    [x,y,z,theta,l,w,h] ma OpenPCDet -- e quindi PointPillars -- produce
    [x,y,z,dx,dy,dz,heading]. Tenere l'ordine di OpenPCDet evita una
    conversione a ogni chiamata; scambiarli per sbaglio fa filtrare una
    dimensione come se fosse un angolo, senza che nulla segnali l'errore.

    "Rappresentiamo lo stato 3D di una traccia con una box 3D orientata e un
     vettore di velocita' posizionale (escludendo la velocita' angolare, come
     in [35])" -- cioe' AB3DMOT. Anche le dimensioni fanno parte dello stato e
     vengono filtrate.

    I default delle covarianze sono quelli di AB3DMOT, tarati su auto KITTI a
    10 Hz. Per pedoni sono generosi: esposti come parametri apposta.
    """

    def __init__(self, box3d, p_pos=10.0, p_vel=10000.0,
                 q_pos=1.0, q_vel=0.01, r_meas=1.0):
        self.x = np.zeros(10, dtype=np.float64)
        self.x[:7] = np.asarray(box3d, dtype=np.float64).reshape(7)
        self.x[6] = normalizza_angolo(self.x[6])

        self.P = np.eye(10) * float(p_pos)
        self.P[7:, 7:] = np.eye(3) * float(p_vel)   # velocita' iniziale ignota

        self.Q = np.eye(10) * float(q_pos)
        self.Q[7:, 7:] = np.eye(3) * float(q_vel)

        self.R = np.eye(7) * float(r_meas)

        self.H = np.zeros((7, 10))
        self.H[:7, :7] = np.eye(7)

    def predict(self, dt):
        """Predizione a velocita' costante.

        AB3DMOT mette 1 al posto di dt (passo unitario). Qui il dt e' esplicito
        perche' in ROS il periodo fra due nuvole non e' costante: assumerlo
        unitario introdurrebbe un errore proporzionale al jitter.
        """
        F = np.eye(10)
        F[0, 7] = F[1, 8] = F[2, 9] = float(dt)
        self.x = F @ self.x
        self.x[6] = normalizza_angolo(self.x[6])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, box3d):
        """Correzione con una misura 3D.

        Prima si applica la disambiguazione angolare di AB3DMOT: una box e la
        stessa box ruotata di pi descrivono lo stesso oggetto, quindi se
        predizione e misura differiscono di piu' di 90 gradi si ruota la
        PREDIZIONE, invece di far compiere al filtro un salto di mezzo giro.
        """
        z = np.asarray(box3d, dtype=np.float64).reshape(7).copy()
        z[6] = normalizza_angolo(z[6])
        self.x[6] = normalizza_angolo(self.x[6])

        diff = abs(z[6] - self.x[6])
        if (np.pi / 2.0) < diff < (np.pi * 3.0 / 2.0):
            self.x[6] = normalizza_angolo(self.x[6] + np.pi)
        # angoli quasi opposti a cavallo di +-pi: riportali sullo stesso giro
        if abs(z[6] - self.x[6]) >= (np.pi * 3.0 / 2.0):
            self.x[6] += 2.0 * np.pi if z[6] > 0 else -2.0 * np.pi

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[6] = normalizza_angolo(self.x[6])
        self.P = (np.eye(10) - K @ self.H) @ self.P

    @property
    def box(self):
        return self.x[:7].copy()

    @property
    def velocita(self):
        return self.x[7:10].copy()


# ============================================================================
# Traccia
# ============================================================================

class EagerTrack:
    """Traccia con stato 3D e 2D mantenuti IN PARALLELO e INDIPENDENTI.

    "Manteniamo lo stato 2D e 3D delle tracce in parallelo. Tuttavia li
     trattiamo in modo indipendente." Non c'e' accoppiamento fra i due: lo
     stato 2D e' l'ultima box osservata, sovrascritta, senza alcun filtro (il
     paper lascia un eventuale modello di predizione 2D come lavoro futuro).
    """

    MAI = 10 ** 6      # sentinella: "non e' mai successo"

    def __init__(self, tid, istanza: Istanza, kf_kwargs=None):
        self.id = int(tid)
        self._kf_kwargs = dict(kf_kwargs or {})
        self.kf: Optional[KalmanBox3D] = None
        self.score3d: Optional[float] = None
        self.box2d: Optional[np.ndarray] = None
        self.score2d: Optional[float] = None

        if istanza.ha_3d:
            self.kf = KalmanBox3D(istanza.box3d, **self._kf_kwargs)
            self.score3d = istanza.score3d
        if istanza.ha_2d:
            self.box2d = np.asarray(istanza.box2d, dtype=np.float64).copy()
            self.score2d = istanza.score2d

        # --- contatori del ciclo di vita, in FRAME ---
        self.eta_da_update = 0
        self.eta_da_2d = 0 if istanza.ha_2d else self.MAI
        self.eta_da_3d = 0 if istanza.ha_3d else self.MAI
        self.aggiornata_ora = True
        self.hits = 1
        self.frame_nascita = 0

    # -- stato ------------------------------------------------------------
    @property
    def ha_3d(self):
        return self.kf is not None

    @property
    def box3d(self):
        return None if self.kf is None else self.kf.box

    @property
    def velocita(self):
        return None if self.kf is None else self.kf.velocita

    # -- passi del ciclo --------------------------------------------------
    def predict(self, dt):
        if self.kf is not None:
            self.kf.predict(dt)
        self.aggiornata_ora = False
        self.eta_da_update += 1
        self.eta_da_2d = min(self.eta_da_2d + 1, self.MAI)
        self.eta_da_3d = min(self.eta_da_3d + 1, self.MAI)

    def update_3d(self, box3d, score3d):
        """Correzione di Kalman con una misura 3D."""
        if self.kf is None:
            self.kf = KalmanBox3D(box3d, **self._kf_kwargs)
        else:
            self.kf.update(box3d)
        self.score3d = score3d
        self.eta_da_3d = 0
        self._segna_update()

    def update_2d(self, box2d, score2d):
        """Lo stato 2D si SOVRASCRIVE, non si filtra.
        Nessun effetto sullo stato 3D: quello resta al solo predict."""
        self.box2d = np.asarray(box2d, dtype=np.float64).copy()
        self.score2d = score2d
        self.eta_da_2d = 0
        self._segna_update()

    def _segna_update(self):
        if not self.aggiornata_ora:
            self.hits += 1          # un frame vale un hit, anche se 3D e 2D
        self.aggiornata_ora = True  # aggiornano entrambi
        self.eta_da_update = 0

    # -- proiezione per lo stadio 2 ---------------------------------------
    def box2d_per_matching(self, T_cam_world, K, width, height):
        """Box 2D con cui la traccia si presenta allo stadio 2.

        "valutiamo la sovrapposizione fra la box 2D dell'istanza nel frame
         corrente e la proiezione 2D della box 3D predetta della traccia,
         oppure l'ultima box 2D osservata nel caso in cui una predizione 3D
         non sia disponibile."
        Puo' ritornare None: la traccia semplicemente non e' accoppiabile
        in questo frame.
        """
        if self.kf is not None:
            p = project_box3d(self.kf.box, T_cam_world, K, width, height)
            if p is not None:
                return p
        return self.box2d

    # -- conferma ---------------------------------------------------------
    def confermata(self, age_2d, age_3d=None, coast_frames=0, min_hits_3d=0):
        """Regola di conferma. Con i default e' quella del paper, alla lettera:

        "una traccia e' considerata confermata se e' stata associata a
         un'istanza nel frame corrente ED e' stata aggiornata con informazione
         2D negli ultimi Age_2d frame."

        La condizione sul 2D esiste perche' "i detector 3D di solito non sono
        affidabili quanto quelli basati su immagine in termini di precisione":
        e' la difesa contro i falsi positivi del LiDAR, integrata nel ciclo di
        vita invece che aggiunta come filtro esterno.

        DUE DEVIAZIONI OPZIONALI, entrambe disattivate di default.

        age_3d (None = regola del paper). Il paper assume che camera e LiDAR
          abbiano FOV SOVRAPPOSTI -- su KITTI e NuScenes e' cosi'. Con un LiDAR
          a 360 gradi e una camera a ~72, un pedone di lato e' visto benissimo
          in 3D ma non potra' MAI avere evidenza 2D: la regola del paper lo
          scarta, buttando via l'80% della copertura del sensore migliore.
          Con age_3d=N basta evidenza 2D entro age_2d OPPURE 3D entro N frame.
          Rischio: passano i falsi positivi del detector 3D. Mitigazione: sono
          su ostacoli STATICI, e a valle il critic filtra per velocita' minima.

        coast_frames (0 = regola del paper). Nel paper la conferma e'
          rivalutata a ogni frame e NON e' appiccicosa: basta un frame in cui
          il detector manca la traccia e questa sparisce dall'uscita, per
          riapparire subito dopo. Su un benchmark valutato frame per frame va
          bene; per alimentare un controllore in continuo produce sfarfallio.
          Con coast_frames=N la conferma sopravvive a N frame senza match.

        min_hits_3d (0 = nessun vincolo). Corregge il rischio di age_3d: una
          traccia confermata SOLO dal 3D deve essere stata associata in almeno
          N frame. Su un bag reale 9 statici fantasma su 13 e tutti i doppioni
          vicino a un pedone in moto avevano UN solo aggiornamento in tutta la
          vita (una detection spuria di PointPillars), ma venivano pubblicati
          per ~1 s. La via 2D (prova dalla camera) non ha questo vincolo.
        """
        if self.aggiornata_ora:
            associata = True
        else:
            associata = (int(coast_frames) > 0 and
                         self.eta_da_update <= int(coast_frames))
        if not associata:
            return False
        if self.eta_da_2d <= int(age_2d):
            return True
        if (age_3d is not None and self.eta_da_3d <= int(age_3d)
                and self.hits >= int(min_hits_3d)):
            return True
        return False


# ============================================================================
# Tracker
# ============================================================================

class EagerMOT:
    """Tracker a due stadi.

    Parametri, con i valori del paper per la classe pedestrian su NuScenes:
        theta_3d : soglia MASSIMA di distanza scalata, stadio 1   [1.8]
        theta_2d : soglia MINIMA di IoU 2D, stadio 2              [0.5]
        age_max  : frame senza alcun update dopo i quali si muore [3]
        age_2d   : frame entro cui serve evidenza 2D per confermare [3]

    Deviazioni opzionali (default = comportamento del paper):
        age_3d               : None = disattivo. Con N, l'evidenza 3D recente
                               basta a confermare anche senza 2D.
        confirm_coast_frames : 0 = disattivo. Con N, la conferma sopravvive a
                               N frame senza alcun match.

    age_max e age_2d sono in FRAME, come nel paper: 3 frame sono 0.3 s a 10 Hz
    ma 1.5 s a 2 Hz. Se il detector cambia frequenza vanno riscalati.
    """

    def __init__(self, theta_3d=1.8, theta_2d=0.5, age_max=3, age_2d=3,
                 age_3d=None, confirm_coast_frames=0, dup_suppress_radius=0.0,
                 min_hits_3d=0, kf_kwargs=None):
        self.theta_3d = float(theta_3d)
        self.theta_2d = float(theta_2d)
        self.age_max = int(age_max)
        self.age_2d = int(age_2d)
        # deviazioni opzionali dal paper: vedi EagerTrack.confermata
        self.age_3d = None if age_3d is None else int(age_3d)
        self.confirm_coast_frames = int(confirm_coast_frames)
        # 0.0 = regola del paper (nessun salvataggio). Vedi il commento nello
        # step 2.5 qui sotto.
        self.dup_suppress_radius = float(dup_suppress_radius)
        self.min_hits_3d = int(min_hits_3d)
        self.kf_kwargs = dict(kf_kwargs or {})
        self.tracks: List[EagerTrack] = []
        self._prossimo_id = 0
        self._frame = 0
        self.diagnostica = {}

    # ------------------------------------------------------------------
    def step(self, istanze: List[Istanza], dt, T_cam_world, K, width, height):
        """Un frame completo.

        Le box 3D delle istanze devono essere GIA' nel frame fisso (odom);
        T_cam_world porta da quel frame all'ottico della camera.
        Ritorna la lista di tracce vive dopo l'aggiornamento.
        """
        self._frame += 1
        istanze = list(istanze)

        # ---- 0. PREDICT --------------------------------------------------
        for t in self.tracks:
            t.predict(dt)

        usata = [False] * len(istanze)

        # ---- 1. STADIO 1: istanze 3D vs tracce 3D, distanza scalata -------
        idx_i3 = [i for i, x in enumerate(istanze) if x.ha_3d]
        idx_t3 = [j for j, t in enumerate(self.tracks) if t.ha_3d]

        D = np.zeros((len(idx_i3), len(idx_t3)), dtype=np.float64)
        for a, i in enumerate(idx_i3):
            for b, j in enumerate(idx_t3):
                D[a, b] = scaled_distance(istanze[i].box3d,
                                          self.tracks[j].box3d)
        coppie1, _, t_liberi1 = greedy_match(D, self.theta_3d, maximize=False)

        for a, b in coppie1:
            ist = istanze[idx_i3[a]]
            trk = self.tracks[idx_t3[b]]
            trk.update_3d(ist.box3d, ist.score3d)
            if ist.ha_2d:               # istanza fusa: aggiorna anche il 2D
                trk.update_2d(ist.box2d, ist.score2d)
            usata[idx_i3[a]] = True

        orfane1 = [idx_t3[b] for b in t_liberi1]

        # ---- 2. STADIO 2: istanze SOLO-2D vs tracce rimaste, IoU 2D -------
        # "le istanze che erano state rilevate in 3D non partecipano a questo
        #  stadio anche se erano state rilevate anche in 2D": qui entrano solo
        #  le istanze prive di box 3D. Le istanze 3D che hanno fallito lo
        #  stadio 1 hanno gia' avuto la loro occasione e apriranno una traccia.
        idx_i2 = [i for i, x in enumerate(istanze) if x.ha_2d and not x.ha_3d]
        idx_t2 = orfane1 + [j for j, t in enumerate(self.tracks)
                            if not t.ha_3d]

        box_t2 = [self.tracks[j].box2d_per_matching(T_cam_world, K,
                                                    width, height)
                  for j in idx_t2]

        I2 = np.zeros((len(idx_i2), len(idx_t2)), dtype=np.float64)
        for a, i in enumerate(idx_i2):
            for b in range(len(idx_t2)):
                if box_t2[b] is not None:
                    I2[a, b] = iou_2d(istanze[i].box2d, box_t2[b])
        coppie2, _, _ = greedy_match(I2, self.theta_2d, maximize=True)

        for a, b in coppie2:
            ist = istanze[idx_i2[a]]
            trk = self.tracks[idx_t2[b]]
            trk.update_2d(ist.box2d, ist.score2d)
            # NESSUN update_3d: lo stato 3D resta quello del solo predict.
            usata[idx_i2[a]] = True

        # ---- 2.5 DEDUPLICA (deviazione OPZIONALE, spenta di default) ------
        # Nel paper ogni istanza 3D orfana dello stadio 1 apre una traccia
        # nuova. Osservato dal vivo: una traccia sparisce e una nuova nasce
        # 0.5 s dopo a 0.3 m -- lo stesso pedone con un altro ID. Due cause
        # possibili, entrambe gestite qui con una distanza euclidea sul piano
        # (non la scaled_distance: dimensioni e yaw di un pedone da
        # PointPillars sono rumorosi, e con alpha fino a 2 possono far
        # fallire lo stadio 1 anche a pochi cm):
        #   - la traccia vicina NON e' stata aggiornata in questo frame: lo
        #     stadio 1 l'ha mancata -> la si aggiorna con questa istanza;
        #   - la traccia vicina E' GIA' stata aggiornata: e' una doppia
        #     detection dello stesso oggetto -> l'istanza si scarta.
        # Lo stadio 1 resta quello del paper: qui si toccano solo le istanze
        # che avrebbero aperto una traccia nuova.
        salvate = soppresse = 0
        if self.dup_suppress_radius > 0.0:
            for i in idx_i3:
                if usata[i]:
                    continue
                px, py = istanze[i].box3d[0], istanze[i].box3d[1]
                vicina, d_min = None, self.dup_suppress_radius
                for t in self.tracks:
                    if not t.ha_3d:
                        continue
                    d = float(np.hypot(t.box3d[0] - px, t.box3d[1] - py))
                    if d < d_min:
                        vicina, d_min = t, d
                if vicina is None:
                    continue
                usata[i] = True
                if vicina.aggiornata_ora:
                    soppresse += 1
                else:
                    vicina.update_3d(istanze[i].box3d, istanze[i].score3d)
                    if istanze[i].ha_2d:
                        vicina.update_2d(istanze[i].box2d, istanze[i].score2d)
                    salvate += 1

        # ---- 3. NASCITE ---------------------------------------------------
        # "tutte le istanze mai accoppiate iniziano nuove tracce": comprese le
        # istanze 3D orfane dello stadio 1, altrimenti nessun oggetto nuovo
        # entrerebbe mai in tracking.
        nuove = 0
        for i, u in enumerate(usata):
            if u:
                continue
            t = EagerTrack(self._prossimo_id, istanze[i], self.kf_kwargs)
            t.frame_nascita = self._frame
            self._prossimo_id += 1
            self.tracks.append(t)
            nuove += 1

        # ---- 4. MORTI -----------------------------------------------------
        prima = len(self.tracks)
        self.tracks = [t for t in self.tracks
                       if t.eta_da_update <= self.age_max]

        self.diagnostica = dict(
            frame=self._frame, istanze=len(istanze),
            match1=len(coppie1), match2=len(coppie2),
            salvate=salvate, soppresse=soppresse,
            nuove=nuove, morte=prima - len(self.tracks),
            vive=len(self.tracks), confermate=len(self.tracks_confermate()))
        return self.tracks

    # ------------------------------------------------------------------
    def tracks_confermate(self):
        """Le tracce da pubblicare: confermate E con uno stato 3D.

        Il paper riporta le stime 3D per le tracce confermate; una traccia
        senza stato 3D non ha una posizione nel mondo da pubblicare.
        """
        return [t for t in self.tracks
                if t.ha_3d and t.confermata(self.age_2d, self.age_3d,
                                            self.confirm_coast_frames,
                                            self.min_hits_3d)]
