#!/usr/bin/env bash
WS=$HOME/Desktop/thesis_ws
ok() { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
no() { printf '  \033[31mNO\033[0m   %s\n' "$1"; ERR=1; }
av() { printf '  \033[33mATT\033[0m  %s\n' "$1"; }
ERR=0
echo "Controllo prima del run ($(date '+%d/%m %H:%M:%S'))"

anno=$(date +%Y)
if [ "$anno" -lt 2026 ]; then no "orologio del Jetson sbagliato ($(date)): cron e timestamp vanno fuori, collegare internet o impostare l'ora"; else ok "orologio del Jetson $(date '+%d/%m/%Y %H:%M:%S')"; fi
pi=$(timeout 4 ssh -o ConnectTimeout=3 -o BatchMode=yes ubuntu@192.168.131.1 date +%s 2>/dev/null)
if [ -n "$pi" ]; then
  d=$(( pi - $(date +%s) )); d=${d#-}
  if [ "$d" -le 1 ]; then ok "Pi raggiungibile, orologi allineati (scarto ${d} s)"; else no "Pi e Jetson sfasati di ${d} s: i comandi al Husky scadono male"; fi
else
  no "Pi non raggiungibile via ssh (192.168.131.1)"
fi
nvpmodel -q 2>/dev/null | grep -q MAXN && ok "Jetson in MAXN" || av "Jetson non in MAXN: sudo nvpmodel -m 2 && sudo jetson_clocks"
lib=$(df -BG --output=avail "$HOME/bags" | tail -1 | tr -dc 0-9)
[ "$lib" -ge 20 ] && ok "disco: ${lib} GB liberi" || av "disco: solo ${lib} GB liberi"
carico=$(awk '{print $1}' /proc/loadavg)
if awk "BEGIN{exit !($carico > 1.5)}"; then av "carico gia' alto prima di partire ($carico): $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,4p | tr -s ' ' | tr '\n' ';')"; else ok "carico a riposo $carico"; fi
[ -e /dev/ttyACM0 ] && ok "Pixhawk su /dev/ttyACM0" || no "Pixhawk non collegato (/dev/ttyACM0 assente)"
ping -c1 -W1 192.168.1.201 >/dev/null 2>&1 && ok "Velodyne risponde (192.168.1.201)" || no "Velodyne non risponde (192.168.1.201): cavo o batteria 12 V"
L=$WS/install/localization/share/localization/maps
[ -s "$L/mappa.pcd" ] && [ -s "$L/griglia.yaml" ] && ok "mappa di localizzazione installata ($(du -h "$L/mappa.pcd" | cut -f1))" || no "manca mappa.pcd o griglia.yaml in $L"
nome=$(grep -o 'map_name:=[A-Za-z0-9_]*' "$WS/robot_tmux.sh" | head -1 | cut -d= -f2)
for f in semantica elevazione; do
  [ -s "$WS/src/delivery_robot/semantic/maps/${f}_${nome}.npz" ] && ok "mappa ${f}_${nome}.npz presente" || no "manca ${f}_${nome}.npz (robot_tmux usa map_name:=$nome)"
done
[ -s "$WS/prove/punti.yaml" ] && ok "goal: $(grep -c '^[A-Za-z0-9_]*:' "$WS/prove/punti.yaml") punti in prove/punti.yaml" || no "manca prove/punti.yaml"
grep -q "record_run.sh navigazione" "$WS/robot_tmux.sh" && ok "registrazione precompilata: navigazione (leggera solo per giri di mappatura)" || av "la finestra bag non usa il profilo navigazione"
if tmux has-session -t robot 2>/dev/null; then av "sessione tmux robot gia' aperta"; fi
if pgrep -x mavros_node >/dev/null; then
  set +u; source /opt/ros/humble/setup.bash; source "$WS/install/setup.bash"; set -u 2>/dev/null
  PYTHONDONTWRITEBYTECODE=1 python3 "$WS/controllo_pixhawk.py" 6 || ERR=1
else
  av "MAVROS non attivo: dopo la finestra sensori, a robot fermo, lancia python3 controllo_pixhawk.py"
fi
echo
echo "Ricorda: dopo l'accensione il Pixhawk puo' impiegare fino a ~10 minuti ad assestare l'assetto (24/10 pomeriggio: 2,8 gradi di errore all'avvio)."
echo "Conta solo per i giri di mappatura dell'elevazione: per la navigazione con mappe fisse si puo' partire subito."
[ "$ERR" -eq 0 ] && echo "Tutto pronto." || echo "Ci sono problemi da sistemare (righe NO)."
