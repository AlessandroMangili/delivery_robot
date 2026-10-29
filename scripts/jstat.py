#!/usr/bin/env python3
import glob
import os
import sys
import time

V, G, R, D, B, N = '\033[32m', '\033[33m', '\033[31m', '\033[90m', '\033[1m', '\033[0m'
PERIODO = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
HZ = os.sysconf('SC_CLK_TCK')


def leggi(p, d=None):
    try:
        with open(p) as f:
            return f.read().strip()
    except Exception:
        return d


def cpu_righe():
    out = {}
    for l in leggi('/proc/stat', '').split('\n'):
        if l.startswith('cpu'):
            p = l.split()
            v = [int(x) for x in p[1:8]]
            out[p[0]] = (sum(v) - v[3] - v[4], sum(v))
    return out


def processi():
    out = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        s = leggi(f'/proc/{d}/stat')
        if not s:
            continue
        i = s.rfind(')')
        f = s[i + 2:].split()
        out[int(d)] = (s[s.find('(') + 1:i], int(f[11]) + int(f[12]))
    return out


def colore(x, g, r):
    return V if x < g else (G if x < r else R)


zone = {leggi(z + '/type'): z + '/temp' for z in glob.glob('/sys/class/thermal/thermal_zone*')}
ina = {}
for d in glob.glob('/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*'):
    for i in (1, 2, 3):
        lab = leggi(f'{d}/in{i}_label')
        if lab:
            ina[lab] = (f'{d}/in{i}_input', f'{d}/curr{i}_input')
gpu_p = next((p for p in ('/sys/devices/platform/bus@0/17000000.gpu/load', '/sys/devices/platform/gpu.0/load') if os.path.exists(p)), None)
nomi_lunghi = {}
c0, p0, t0 = cpu_righe(), processi(), time.monotonic()
time.sleep(PERIODO)
io = os.getpid()
while True:
    c1, p1, t1 = cpu_righe(), processi(), time.monotonic()
    dt = t1 - t0
    tot = {k: 100.0 * (c1[k][0] - c0[k][0]) / max(c1[k][1] - c0[k][1], 1) for k in c1 if k in c0}
    core = [tot[k] for k in sorted(c1) if k != 'cpu' and k in tot]
    mem = dict(l.split(':') for l in leggi('/proc/meminfo', '').split('\n') if ':' in l)
    ram = 100 * (1 - int(mem['MemAvailable'].split()[0]) / int(mem['MemTotal'].split()[0]))
    t = lambda k: float(leggi(zone[k], 'nan')) / 1000 if k in zone else float('nan')
    w = float('nan')
    if 'VDD_IN' in ina:
        v, i = (float(leggi(p, 'nan')) for p in ina['VDD_IN'])
        w = v * i / 1e6
    gpu = float(leggi(gpu_p, 'nan')) / 10 if gpu_p else float('nan')
    freq = float(leggi('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'nan')) / 1000
    load = leggi('/proc/loadavg', '? ? ?').split()[:3]
    top = []
    for pid, (nome, tk) in p1.items():
        if pid in p0 and p0[pid][0] == nome:
            u = 100.0 * (tk - p0[pid][1]) / HZ / dt
            if u > 0.5:
                top.append((u, pid, nome))
    top.sort(reverse=True)
    righe = [f'{B}jstat{N}  {time.strftime("%H:%M:%S")}   carico {" ".join(load)} (6 core)']
    righe.append(f'CPU {colore(tot["cpu"], 70, 90)}{tot["cpu"]:5.1f}%{N}  per core ' + ' '.join(f'{colore(x, 70, 90)}{x:3.0f}{N}' for x in core) + f'   {freq:.0f} MHz')
    righe.append(f'GPU {colore(gpu, 70, 90)}{gpu:5.1f}%{N}   RAM {colore(ram, 75, 90)}{ram:4.1f}%{N}   temp CPU {t("cpu-thermal"):.0f}  GPU {t("gpu-thermal"):.0f}  Tj {colore(t("tj-thermal"), 70, 85)}{t("tj-thermal"):.0f}°C{N}   potenza {w:.1f} W')
    righe.append('')
    righe.append(f'{B}{"%CPU":>6s}  {"PID":>7s}  processo{N}')
    for u, pid, nome in top[:12]:
        if pid not in nomi_lunghi:
            cmd = (leggi(f'/proc/{pid}/cmdline', '') or '').replace('\0', ' ').split()
            nomi_lunghi[pid] = next((os.path.basename(x) for x in cmd if '/' in x and not x.startswith('--') and 'python' not in os.path.basename(x) and 'ros2' != os.path.basename(x)), nome)[:48]
        righe.append(f'{colore(u, 30, 70)}{u:6.1f}{N}  {pid:7d}  {nomi_lunghi[pid]}' + (f'  {D}(questo jstat){N}' if pid == io else ''))
    sys.stdout.write('\033[H\033[2J' + '\n'.join(righe) + '\n')
    sys.stdout.flush()
    c0, p0, t0 = c1, p1, t1
    time.sleep(PERIODO)
