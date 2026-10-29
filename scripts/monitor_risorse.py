import os, sys, time, signal, threading, subprocess, datetime, glob, csv

PI = 'ubuntu@192.168.131.1'
PI_LOOP = r'''
trap 'kill 0' EXIT PIPE HUP TERM
docker logs -f --tail 0 husky-platform 2>&1 | grep --line-buffered "Overrun detected" | while read l; do echo O; done &
while true; do
  s=$(head -1 /proc/stat | cut -d' ' -f3-)
  m=$(awk '/MemTotal/{t=$2}/MemAvailable/{a=$2}END{printf "%.1f", 100*(t-a)/t}' /proc/meminfo)
  t=$(cat /sys/class/thermal/thermal_zone0/temp)
  f=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)
  adc=$(vcgencmd pmic_read_adc)
  v=$(echo "$adc" | awk -F= '/EXT5V_V/{gsub("V","",$2); print $2}')
  p=$(echo "$adc" | awk '{n=$1; sub(/_[AV]$/,"",n); split($2,b,"="); x=b[2]+0; if ($1 ~ /_A$/) I[n]=x; else V[n]=x} END {s=0; for (k in I) if (k in V) s+=I[k]*V[k]; printf "%.2f", s}')
  th=$(vcgencmd get_throttled | cut -d= -f2)
  echo "S $s;$m;$t;$f;$v;$p;$th"
  sleep 1
done
'''
COLS = ['ora', 'j_cpu', 'j_ram', 'j_gpu', 'j_t_cpu', 'j_t_gpu', 'j_t_tj', 'j_vin_v', 'j_in_w', 'j_cpugpu_w', 'j_soc_w',
        'j_oc_eventi', 'j_freq_mhz', 'pi_cpu', 'pi_ram', 'pi_temp', 'pi_freq_mhz', 'pi_ext5v_v', 'pi_soc_w', 'pi_throttled',
        'pi_overrun', 'husky_ciclo_hz']


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def cpu_times(line):
    v = [int(x) for x in line.split()[:8]]
    idle = v[3] + v[4]
    return sum(v) - idle, sum(v)


class Jetson:
    def __init__(self):
        self.prev = None
        self.zones = {read(z + '/type'): z + '/temp' for z in glob.glob('/sys/class/thermal/thermal_zone*')}
        self.ina = {}
        for d in glob.glob('/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*'):
            for i in (1, 2, 3):
                label = read(f'{d}/in{i}_label')
                if label:
                    self.ina[label] = (f'{d}/in{i}_input', f'{d}/curr{i}_input')
        oc = [d for d in glob.glob('/sys/class/hwmon/hwmon*') if read(d + '/name') == 'soctherm_oc']
        self.oc = [f'{oc[0]}/oc{i}_event_cnt' for i in (1, 2, 3)] if oc else []
        gpu = [p for p in ('/sys/devices/platform/bus@0/17000000.gpu/load', '/sys/devices/platform/gpu.0/load') if os.path.exists(p)]
        self.gpu = gpu[0] if gpu else None

    def power(self, label):
        if label not in self.ina:
            return None, None
        v, i = (float(read(p, 'nan')) for p in self.ina[label])
        return v / 1000, v * i / 1e6

    def sample(self):
        busy, total = cpu_times(read('/proc/stat').split('\n')[0].split(None, 1)[1])
        cpu = None if self.prev is None else 100 * (busy - self.prev[0]) / max(total - self.prev[1], 1)
        self.prev = (busy, total)
        mem = dict(l.split(':') for l in read('/proc/meminfo').split('\n') if ':' in l)
        ram = 100 * (1 - int(mem['MemAvailable'].split()[0]) / int(mem['MemTotal'].split()[0]))
        temp = lambda k: float(read(self.zones[k], 'nan')) / 1000 if k in self.zones else None
        vin, pin = self.power('VDD_IN')
        _, pcg = self.power('VDD_CPU_GPU_CV')
        _, psoc = self.power('VDD_SOC')
        oc = sum(int(read(p, '0')) for p in self.oc)
        gpu = float(read(self.gpu, 'nan')) / 10 if self.gpu else None
        freq = float(read('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', 'nan')) / 1000
        return [cpu, ram, gpu, temp('cpu-thermal'), temp('gpu-thermal'), temp('tj-thermal'), vin, pin, pcg, psoc, oc, freq]


class Pi:
    def __init__(self):
        self.last, self.t_last, self.prev, self.overrun = None, 0.0, None, 0
        self.lock = threading.Lock()
        self.proc = subprocess.Popen(['ssh', '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=5', PI, 'bash', '-s'],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.proc.stdin.write(PI_LOOP)
        self.proc.stdin.close()
        threading.Thread(target=self.reader, daemon=True).start()

    def reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            with self.lock:
                if line == 'O':
                    self.overrun += 1
                elif line.startswith('S '):
                    try:
                        s, m, t, f, v, p, th = line[2:].split(';')
                        busy, total = cpu_times(s)
                        cpu = None if self.prev is None else 100 * (busy - self.prev[0]) / max(total - self.prev[1], 1)
                        self.prev = (busy, total)
                        self.last = [cpu, float(m), float(t) / 1000, float(f) / 1000, float(v), float(p), th]
                        self.t_last = time.time()
                    except ValueError:
                        pass

    def sample(self):
        with self.lock:
            n, self.overrun = self.overrun, 0
            fresh = time.time() - self.t_last < 3
            return (self.last if fresh else [None] * 7) + [n]

    def stop(self):
        self.proc.terminate()


class HuskyLoop:
    def __init__(self):
        self.n, self.ok = 0, False
        self.lock = threading.Lock()
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from nav_msgs.msg import Odometry
            rclpy.init()
            self.node = rclpy.create_node('monitor_risorse')
            self.node.create_subscription(Odometry, '/a200_0000/platform/odom', self.cb, qos_profile_sensor_data)
            threading.Thread(target=self.spin, args=(rclpy,), daemon=True).start()
            self.ok = True
        except Exception:
            pass

    def spin(self, rclpy):
        try:
            rclpy.spin(self.node)
        except Exception:
            pass

    def cb(self, msg):
        with self.lock:
            self.n += 1

    def sample(self):
        if not self.ok:
            return [None]
        with self.lock:
            n, self.n = self.n, 0
        return [float(n)]


def record(path):
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    signal.signal(signal.SIGTERM, lambda *a: stop.set())
    jet, pi, loop = Jetson(), Pi(), HuskyLoop()
    fmt = lambda x: '' if x is None else (x if isinstance(x, str) else f'{x:.3f}')
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        k = 0
        next_t = time.time()
        while not stop.is_set():
            row = [datetime.datetime.now().strftime('%H:%M:%S')] + jet.sample() + pi.sample() + loop.sample()
            w.writerow([row[0]] + [fmt(x) for x in row[1:]])
            fh.flush()
            if k % 10 == 0 and row[1] is not None:
                pi_txt = 'Pi non raggiungibile' if row[13] is None else \
                    f'Pi: CPU {row[13] or 0:.0f}% {row[15]:.0f}C, 5V {row[17]:.2f} V, flag {row[19]} | ciclo motori Husky {row[21] or 0:.0f} Hz (deve essere 20)'
                print(f'{row[0]}  Jetson: CPU {row[1]:.0f}% GPU {row[3] or 0:.0f}% {row[6] or 0:.0f}C, {row[8]:.1f} W  |  {pi_txt}', flush=True)
            k += 1
            next_t += 1.0
            stop.wait(max(0.0, next_t - time.time()))
    pi.stop()


def plot(path):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(path)))
    if len(rows) < 3:
        print('troppo pochi campioni')
        return
    col = lambda k: np.array([float(r[k]) if r[k] not in ('', None) else np.nan for r in rows])
    t = np.arange(len(rows)) / 60.0
    th = [int(r['pi_throttled'], 16) if r['pi_throttled'] else 0 for r in rows]
    uv_now = np.array([(x & 0x1) > 0 for x in th]); thr_now = np.array([(x & 0x6) > 0 for x in th])
    fig, ax = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
    ax[0].plot(t, col('j_in_w'), color='tab:red', label='Jetson: potenza totale (W)')
    ax[0].plot(t, col('j_cpugpu_w'), color='tab:orange', lw=0.8, label='Jetson: CPU+GPU (W)')
    ax[0].set_ylabel('W'); ax[0].legend(loc='upper left'); ax[0].grid(alpha=0.3)
    a0 = ax[0].twinx(); a0.plot(t, col('j_vin_v'), color='tab:blue', lw=0.8); a0.set_ylabel('tensione modulo Jetson (V)', color='tab:blue')
    ax[1].plot(t, col('j_cpu'), label='CPU %'); ax[1].plot(t, col('j_gpu'), label='GPU %'); ax[1].plot(t, col('j_ram'), label='RAM %')
    ax[1].plot(t, col('j_t_tj'), label='temperatura max (C)', color='k', lw=0.8)
    ax[1].set_ylabel('Jetson'); ax[1].legend(loc='upper left', ncol=4); ax[1].grid(alpha=0.3)
    ax[2].plot(t, col('pi_ext5v_v'), color='tab:blue', label='Pi: tensione in ingresso (V)')
    ax[2].axhline(4.75, color='tab:red', ls='--', lw=0.8, label='soglia bassa tensione (circa 4,75 V)')
    if uv_now.any():
        ax[2].plot(t[uv_now], col('pi_ext5v_v')[uv_now], 'rv', label='bassa tensione segnalata')
    ax[2].set_ylabel('V'); ax[2].legend(loc='lower left'); ax[2].grid(alpha=0.3)
    a2 = ax[2].twinx(); a2.plot(t, col('pi_soc_w'), color='tab:green', lw=0.8); a2.set_ylabel('Pi: potenza SoC stimata (W)', color='tab:green')
    ax[3].plot(t, col('pi_cpu'), label='Pi CPU %'); ax[3].plot(t, col('pi_temp'), label='Pi temperatura (C)')
    ov = col('pi_overrun')
    ax[3].plot(t, col('husky_ciclo_hz'), label='ciclo controllo motori Husky (Hz, nominale 20)', color='tab:red')
    if thr_now.any():
        ax[3].plot(t[thr_now], np.zeros(thr_now.sum()), 'k|', ms=12, label='CPU del Pi rallentata')
    ax[3].set_ylabel('Pi'); ax[3].set_xlabel(f"minuti dall'inizio ({rows[0]['ora']})"); ax[3].legend(loc='upper left', ncol=2); ax[3].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.splitext(path)[0] + '.png'
    fig.savefig(out, dpi=90)
    q = lambda k, f: f(col(k)[~np.isnan(col(k))]) if np.any(~np.isnan(col(k))) else float('nan')
    print(f'durata {len(rows) / 60:.1f} min, grafico in {out}')
    print(f"Jetson: potenza media {q('j_in_w', np.mean):.1f} W, max {q('j_in_w', np.max):.1f} W | tensione modulo min {q('j_vin_v', np.min):.2f} V | "
          f"CPU media {q('j_cpu', np.mean):.0f}%, p95 {q('j_cpu', lambda a: np.percentile(a, 95)):.0f}% | GPU media {q('j_gpu', np.mean):.0f}% | "
          f"temperatura max {q('j_t_tj', np.max):.0f} C | eventi di sovracorrente {int(q('j_oc_eventi', np.max) - q('j_oc_eventi', np.min))}")
    print(f"Pi: tensione min {q('pi_ext5v_v', np.min):.2f} V, media {q('pi_ext5v_v', np.mean):.2f} V | secondi con bassa tensione {uv_now.sum()} | "
          f"secondi con CPU rallentata {thr_now.sum()} | CPU media {q('pi_cpu', np.mean):.0f}% | temperatura max {q('pi_temp', np.max):.0f} C | "
          f"Pi non raggiungibile per {int(np.isnan(col('pi_cpu')).sum())} s")
    hz = col('husky_ciclo_hz')
    if np.any(~np.isnan(hz)):
        print(f"ciclo di controllo dei motori Husky: media {np.nanmean(hz):.1f} Hz, minimo {np.nanmin(hz):.0f} Hz (nominale 20 Hz), "
              f"secondi sotto 18 Hz: {int(np.sum(hz < 18))} su {int(np.sum(~np.isnan(hz)))}, avvisi di overrun nel log del Pi {int(np.nansum(ov))}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'registra'
    if mode == 'registra':
        out = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser(
            f"~/bags/risorse_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(f'registro le risorse in {out} (Ctrl-C per fermare)', flush=True)
        record(out)
        plot(out)
    elif mode == 'grafico':
        plot(sys.argv[2])
    else:
        print('uso: monitor_risorse.py [registra [file.csv] | grafico file.csv]')
