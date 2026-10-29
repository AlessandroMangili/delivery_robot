import sys
import os
import re
import math
import time
import heapq
import yaml
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.duration import Duration
from rclpy.time import Time
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener

QUI = os.path.dirname(os.path.abspath(__file__))
PUNTI = os.path.join(QUI, 'punti.yaml')
REGISTRO = os.path.join(QUI, 'vai_registro.csv')
LONTANO = 70.0
SALTO = 13.0
TAPPA = 45.0
PASSAGGIO = 2.0

ESITI = {
    GoalStatus.STATUS_SUCCEEDED: 'arrivato',
    GoalStatus.STATUS_ABORTED: 'fallito',
    GoalStatus.STATUS_CANCELED: 'annullato o sostituito da un altro goal',
}

ZONE = {
    'S': 'corridoio sud, da B verso G',
    'N': "anello nord, da G a D, lato nord, C, lato est fino a sopra la partenza",
    'V': 'strada sotto il corridoio sud, da est verso ovest',
    'O': 'piazzale ovest',
    'L': 'viale lungo verso sud, da nord a sud',
    'F': 'viale in fondo a sud: corsia nord verso ovest, poi corsia sud verso est',
    'E': 'vialetto a nord-est, da C verso nord-est',
}

USO = '''uso:
  python3 vai.py NOME                    un goal (es. A, partenza, L4)
  python3 vai.py NOME NOME ...           sequenza: ogni goal parte quando il precedente e' arrivato
  python3 vai.py L1-L6                   intervallo di una zona (anche L6-L1 al contrario)
  python3 vai.py X Y [YAW]               goal in coordinate mappa
  opzioni: --attendi  aspetta l'esito anche con un goal solo
           --inverso  gira di 180 gradi il verso di arrivo
           --continua in una sequenza va avanti anche se un goal fallisce
           --diretto  niente tappe automatiche per i goal oltre 70 m
           --lista [ZONA]  elenca i punti        --vicino  punti piu' vicini al robot
  Ctrl+C mentre aspetta = annulla il goal in corso (il robot si ferma).
  Le tappe automatiche non si raggiungono di precisione: a 2 m parte gia' il goal successivo.'''


def zona(nome):
    m = re.fullmatch(r'([A-Z])(\d+)', nome)
    return m.group(1) if m and m.group(1) in ZONE else None


def ordina(nomi):
    return sorted(nomi, key=lambda n: (zona(n) is not None, list(ZONE).index(zona(n)) if zona(n) else 0, int(n[1:]) if zona(n) else 0))


def espandi(tok, punti):
    m = re.fullmatch(r'([A-Z])(\d+)-([A-Z]?)(\d+)', tok)
    if not m or (m.group(3) and m.group(3) != m.group(1)):
        return None
    z, a, b = m.group(1), int(m.group(2)), int(m.group(4))
    passo = 1 if b >= a else -1
    nomi = ['%s%d' % (z, k) for k in range(a, b + passo, passo)]
    mancanti = [n for n in nomi if n not in punti]
    if mancanti:
        raise ValueError('punti inesistenti: ' + ' '.join(mancanti))
    return nomi


def normalizza(a):
    return math.atan2(math.sin(a), math.cos(a))


def posa_robot(n, tf, attesa=2.0):
    t0 = time.time()
    while time.time() - t0 < attesa:
        rclpy.spin_once(n, timeout_sec=0.1)
        try:
            t = tf.lookup_transform('map', 'base_footprint', Time(), Duration(seconds=0.1))
            q = t.transform.rotation
            return t.transform.translation.x, t.transform.translation.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        except Exception:
            pass
    return None


def tappe(punti, da, a):
    P = {k: (p['x'], p['y']) for k, p in punti.items()}
    P['@'], P['#'] = da, a
    dist = {k: math.inf for k in P}
    prec = {}
    dist['@'] = 0.0
    h = [(0.0, '@')]
    while h:
        d, k = heapq.heappop(h)
        if d > dist[k]:
            continue
        if k == '#':
            break
        for j in P:
            e = math.hypot(P[j][0] - P[k][0], P[j][1] - P[k][1])
            if j != k and e <= (25.0 if '@' in (j, k) else SALTO) and d + e < dist[j]:
                dist[j] = d + e
                prec[j] = k
                heapq.heappush(h, (d + e, j))
    if not math.isfinite(dist['#']):
        return None
    cammino = ['#']
    while cammino[-1] != '@':
        cammino.append(prec[cammino[-1]])
    cammino = cammino[::-1][1:]
    scelte, base, prima = [], 0.0, None
    for k in cammino:
        if dist[k] - base > TAPPA and prima not in (None, '#'):
            scelte.append(prima)
            base = dist[prima]
        prima = k
    return [k for k in scelte if math.hypot(P[k][0] - a[0], P[k][1] - a[1]) > 15.0]


def registra(riga):
    nuovo = not os.path.exists(REGISTRO)
    try:
        with open(REGISTRO, 'a') as f:
            if nuovo:
                f.write('ora,goal,x,y,yaw_gradi,esito,secondi,recuperi,distanza_rimasta_m\n')
            f.write(','.join(str(v) for v in riga) + '\n')
    except OSError:
        pass


def attendi(n, futuro, limite):
    t0 = time.time()
    while not futuro.done() and time.time() - t0 < limite:
        rclpy.spin_once(n, timeout_sec=0.1)
    return futuro.done()


def chiudi(n, codice):
    n.destroy_node()
    rclpy.shutdown()
    return codice


def lista(punti, filtro):
    nomi = [k for k in ordina(punti) if not filtro or (zona(k) or '') == filtro or (filtro == '-' and not zona(k))]
    ultima = '-'
    for k in nomi:
        z = zona(k)
        if z != ultima:
            print('\n' + ('goal con nome' if z is None else '%s: %s' % (z, ZONE[z])))
            ultima = z
        p = punti[k]
        print('  %-9s x %7.2f  y %7.2f  yaw %5.0f gradi' % (k, p['x'], p['y'], math.degrees(p['yaw'])))


def main():
    opzioni = {'--inverso', '--attendi', '--continua', '--lista', '--vicino', '--diretto'}
    argv = sys.argv[1:]
    args = [a for a in argv if a not in opzioni]
    punti = yaml.safe_load(open(PUNTI))
    if '--lista' in argv:
        lista(punti, args[0].upper() if args else None)
        return 0
    inverso = '--inverso' in argv
    continua = '--continua' in argv
    goal = []
    if len(args) in (2, 3) and all(re.fullmatch(r'-?\d+(\.\d*)?', a) for a in args):
        x, y = float(args[0]), float(args[1])
        goal.append(('(%.2f, %.2f)' % (x, y), x, y, float(args[2]) if len(args) == 3 else 0.0, False, False))
    elif args:
        for a in args:
            try:
                nomi = espandi(a, punti)
            except ValueError as e:
                print(e)
                return 1
            for k in nomi or [a]:
                if k not in punti:
                    print('punto sconosciuto: %s (python3 vai.py --lista per vederli)' % k)
                    return 1
                p = punti[k]
                goal.append((k, p['x'], p['y'], p['yaw'], zona(k) is not None, False))
    elif '--vicino' not in argv:
        print(USO)
        return 1
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    n = Node('vai')
    tf = Buffer()
    TransformListener(tf, n)
    if '--vicino' in argv:
        r = posa_robot(n, tf, 3.0)
        if r is None:
            print('nessuna posa del robot (map -> base_footprint): la localizzazione e accesa?')
            return chiudi(n, 1)
        print('robot in x %.2f y %.2f yaw %.0f gradi; punti piu vicini:' % (r[0], r[1], math.degrees(r[2])))
        for k in sorted(punti, key=lambda k: math.hypot(punti[k]['x'] - r[0], punti[k]['y'] - r[1]))[:6]:
            print('  %-9s %5.1f m' % (k, math.hypot(punti[k]['x'] - r[0], punti[k]['y'] - r[1])))
        if not goal:
            return chiudi(n, 0)
    client = ActionClient(n, NavigateToPose, 'navigate_to_pose')
    if not client.wait_for_server(timeout_sec=10.0):
        print('Nav2 non risponde su navigate_to_pose: e acceso? ROS_DOMAIN_ID giusto?')
        return chiudi(n, 1)
    r = posa_robot(n, tf)
    prima = (r[0], r[1]) if r else None
    if '--diretto' not in argv:
        espansa, pos = [], prima
        for gl in goal:
            d = math.hypot(gl[1] - pos[0], gl[2] - pos[1]) if pos else 0.0
            if pos and d > LONTANO:
                tt = tappe(punti, pos, (gl[1], gl[2]))
                if tt:
                    print('goal %s a %.0f m in linea d aria: il planner oltre ~95 m di percorso non ce la fa, passo da %s' % (gl[0], d, ' '.join(tt)))
                    espansa += [(k, punti[k]['x'], punti[k]['y'], punti[k]['yaw'], True, True) for k in tt]
            elif not pos and len(espansa) == 0:
                print('posa del robot non disponibile: niente tappe automatiche, mando i goal cosi come sono')
            espansa.append(gl)
            pos = (gl[1], gl[2])
        goal = espansa
    aspetta = '--attendi' in argv or len(goal) > 1
    falliti = 0
    for i, (nome, x, y, th, automatico, tappa) in enumerate(goal):
        if automatico and prima is not None and math.hypot(x - prima[0], y - prima[1]) > 0.5:
            verso = math.atan2(y - prima[1], x - prima[0])
            if math.cos(th - verso) < 0:
                th = normalizza(th + math.pi)
        if inverso:
            th = normalizza(th + math.pi)
        prima = (x, y)
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = 'map'
        g.pose.pose.position.x = float(x)
        g.pose.pose.position.y = float(y)
        g.pose.pose.orientation.z = math.sin(th / 2)
        g.pose.pose.orientation.w = math.cos(th / 2)
        conta = '[%d/%d] ' % (i + 1, len(goal)) if len(goal) > 1 else ''
        descrizione = '%s%s %s: x %.2f y %.2f yaw %.0f gradi' % (conta, 'tappa' if tappa else 'goal', nome, x, y, math.degrees(th))
        fb = {}
        invio = client.send_goal_async(g, feedback_callback=lambda m: fb.update(d=m.feedback.distance_remaining, r=m.feedback.number_of_recoveries,
                                                                            px=m.feedback.current_pose.pose.position.x, py=m.feedback.current_pose.pose.position.y))
        if not attendi(n, invio, 10.0):
            print('%s: nessuna risposta da Nav2 entro 10 s, goal non confermato' % descrizione)
            return chiudi(n, 1)
        handle = invio.result()
        if not handle.accepted:
            print('%s: RIFIUTATO da Nav2' % descrizione)
            return chiudi(n, 1)
        t0 = time.time()
        print('%s: accettato alle %s' % (descrizione, time.strftime('%H:%M:%S')), flush=True)
        if not aspetta:
            registra([time.strftime('%H:%M:%S'), nome, x, y, round(math.degrees(th)), 'inviato', '', '', ''])
            return chiudi(n, 0)
        esito = handle.get_result_async()
        ultimo = time.time()
        superata = False
        try:
            while not esito.done():
                rclpy.spin_once(n, timeout_sec=0.1)
                if tappa and i + 1 < len(goal) and 'px' in fb and math.hypot(fb['px'] - x, fb['py'] - y) < PASSAGGIO:
                    superata = True
                    break
                if time.time() - ultimo >= 10.0 and 'd' in fb:
                    ultimo = time.time()
                    print('   %3.0f s: mancano %.1f m, recuperi %d' % (time.time() - t0, fb['d'], fb['r']), flush=True)
        except KeyboardInterrupt:
            print('\nCtrl+C: annullo il goal %s' % nome)
            annulla = handle.cancel_goal_async()
            attendi(n, annulla, 3.0)
            registra([time.strftime('%H:%M:%S'), nome, x, y, round(math.degrees(th)), 'annullato_da_me', round(time.time() - t0), fb.get('r', ''), round(fb['d'], 1) if 'd' in fb else ''])
            return chiudi(n, 130)
        if superata:
            dist = math.hypot(fb['px'] - x, fb['py'] - y)
            print('%s: superata a %.1f m alle %s dopo %.0f s, recuperi %s; mando il goal successivo' % (descrizione, dist, time.strftime('%H:%M:%S'), time.time() - t0, fb.get('r', '?')), flush=True)
            registra([time.strftime('%H:%M:%S'), nome, x, y, round(math.degrees(th)), 'tappa_superata', round(time.time() - t0), fb.get('r', ''), round(dist, 1)])
            continue
        stato = esito.result().status
        durata = time.time() - t0
        print('%s: %s alle %s dopo %.0f s, recuperi %s' % (descrizione, ESITI.get(stato, 'stato %d' % stato), time.strftime('%H:%M:%S'), durata, fb.get('r', '?')), flush=True)
        registra([time.strftime('%H:%M:%S'), nome, x, y, round(math.degrees(th)), ESITI.get(stato, stato) if stato != GoalStatus.STATUS_CANCELED else 'annullato',
                  round(durata), fb.get('r', ''), round(fb['d'], 1) if 'd' in fb else ''])
        if stato != GoalStatus.STATUS_SUCCEEDED:
            falliti += 1
            if not continua and i + 1 < len(goal):
                print('sequenza interrotta (usa --continua per andare avanti dopo un fallimento)')
                return chiudi(n, 1)
    if len(goal) > 1:
        print('sequenza finita: %d goal, %d non arrivati' % (len(goal), falliti))
    return chiudi(n, 0 if falliti == 0 else 1)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\ninterrotto')
        sys.exit(130)
