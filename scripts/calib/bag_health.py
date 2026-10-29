#!/usr/bin/env python3
"""bag_health.py — controllo rapido di una rosbag2 sqlite3: perdite e ritardi per topic.

Per ogni topic con header: frequenza registrata, frequenza pubblicata (dal passo
degli stamp), % persa, ritardo bag_time - header.stamp. Legge solo i primi 12
byte di ogni messaggio (header CDR), quindi e' veloce anche su bag enormi.

  python3 bag_health.py ~/bags/prova [--seconds 60]
  python3 bag_health.py ~/bags/prova/prova_0.db3 [--seconds 60]

Una registrazione sana: persi ~0% e ritardo mediano di pochi ms (LiDAR: ~100 ms
se ts_first_point=true, e' il tempo di una spazzata).
"""
import argparse
import glob
import os
import sqlite3
import struct

import numpy as np

HEADER_TYPES = ('sensor_msgs/msg/', 'nav_msgs/msg/Odometry', 'geometry_msgs/msg/')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('db')
    ap.add_argument('--seconds', type=float, default=0, help='analizza solo i primi N secondi (0 = tutto)')
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(a.db, '*.db3')), key=lambda f: int(f.rsplit('_', 1)[-1].split('.')[0])) if os.path.isdir(a.db) else [a.db]
    curs = [sqlite3.connect(f'file:{f}?mode=ro', uri=True).cursor() for f in paths]
    span = [c.execute('SELECT min(timestamp), max(timestamp) FROM messages').fetchone() for c in curs]
    span = [x for x in span if x[0] is not None]
    t0, t1 = (min(x[0] for x in span), max(x[1] for x in span)) if span else (None, None)
    if t0 is None or t1 == t0:
        raise SystemExit('bag vuoto: nessun topic registrato (i driver sono accesi?)')
    if a.seconds > 0:
        t1 = min(t1, t0 + int(a.seconds * 1e9))
    dur = (t1 - t0) / 1e9
    print(f'durata analizzata {dur:.0f} s ({len(paths)} file)\n')
    print(f'{"topic":28s} {"reg Hz":>7s} {"pub Hz":>7s} {"persi":>6s} {"ritardo med":>11s} {"p95":>7s} {"MB/s":>6s} {"buco max":>9s}')
    tot = 0
    buchi = []
    topics = {}
    for c in curs:
        for tid, name, typ in c.execute('SELECT id, name, type FROM topics').fetchall():
            topics.setdefault((name, typ), []).append((c, tid))
    for (name, typ), refs in topics.items():
        rows = sorted(r for c, tid in refs for r in c.execute(
            'SELECT timestamp, substr(data,1,12), length(data) FROM messages '
            'WHERE topic_id=? AND timestamp<=?', (tid, t1)).fetchall())
        if not rows:
            continue
        mb = sum(r[2] for r in rows) / dur / 1e6
        tot += mb
        if not any(typ.startswith(h) or typ == h for h in HEADER_TYPES) or len(rows) < 3:
            print(f'{name:28s} {len(rows) / dur:7.1f} {"-":>7s} {"-":>6s} {"-":>11s} {"-":>7s} {mb:6.1f} {"-":>9s}')
            continue
        bt = np.array([r[0] for r in rows], np.int64)
        st = np.array([s * 1_000_000_000 + ns for s, ns in (struct.unpack('<iI', r[1][4:12]) for r in rows)], np.int64)
        d = np.diff(np.sort(st))
        gap = d.max() / 1e9 if len(d) else 0.0
        if gap > 1.0:
            buchi.append((name, gap, int(np.sum(d > 1_000_000_000))))
        burst = np.mean(d < 1_000_000)          # stamp presi all'arrivo, a raffiche (< 1 ms)
        d = d[d >= 1_000_000]
        per = np.median(d) / 1e9 if len(d) else np.nan
        if burst > 0.3:
            name = name + ' (*)'
        pub_hz = 1 / per if per > 0 else np.nan
        lost = 100 * (1 - (len(rows) / dur) / pub_hz) if pub_hz == pub_hz else np.nan
        lag = (bt - st) / 1e6
        print(f'{name:28s} {len(rows) / dur:7.1f} {pub_hz:7.1f} {max(lost, 0):5.0f}% {np.median(lag):9.0f}ms '
              f'{np.percentile(lag, 95):5.0f}ms {mb:6.1f} {gap:8.2f}s')
    print(f'\nbanda totale scritta: {tot:.0f} MB/s')
    for name, gap, n in buchi:
        print(f'ATTENZIONE: {name} ha {n} buchi oltre 1 s (il piu lungo {gap:.1f} s)')
    print('(*) stamp a raffiche (presi all\'arrivo, es. IMU con time_option 2): "pub Hz" e "persi" indicativi')


if __name__ == '__main__':
    main()
