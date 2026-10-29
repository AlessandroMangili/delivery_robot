#!/usr/bin/env bash
set -uo pipefail
M=$HOME/Desktop/thesis_ws/src/delivery_robot/semantic/maps
BASE=${1:-}
if [ -z "$BASE" ]; then echo "uso: ./prove/salva_mappe_vivo.sh NOME_MAPPA [NOME_COPIA]   (es. campus_2510_centro)"; exit 1; fi
NUOVO=${2:-${BASE}_vivo_$(date +%Y%m%d_%H%M)}
set +u
source /opt/ros/humble/setup.bash
source "$HOME/Desktop/thesis_ws/install/setup.bash"
set -u
export PYTHONDONTWRITEBYTECODE=1
NODI='lib/semantic/(elevation|semantic)_costmap_node'

for f in elevazione semantica; do
  [ -f "$M/${f}_${BASE}.npz" ] || { echo "Non trovo $M/${f}_${BASE}.npz"; exit 1; }
  [ -e "$M/${f}_${NUOVO}.npz" ] && { echo "Esiste già ${f}_${NUOVO}.npz: scegli un altro nome"; exit 1; }
done

controlla() {
  python3 - "$1" "$2" <<'PY'
import sys, numpy as np
tipo, path = sys.argv[1], sys.argv[2]
try:
    z = np.load(path)
    if tipo == 'elevazione':
        n = z['n']; area = float((n > 0).sum()) * float(z['res']) ** 2
    else:
        k = z['known']; z['alpha']; z['cls']; area = float(k.astype(bool).sum()) * float(z['res']) ** 2
    print(f'  {tipo}: ok, {area:.0f} m² noti')
except Exception as e:
    print(f'  {tipo}: FILE ROTTO ({e})'); sys.exit(1)
PY
}

if pgrep -f "$NODI" >/dev/null; then
  echo "Mappe ancora accese: salvo adesso con i servizi dei nodi."
  for n in elevation_costmap_node semantic_costmap_node; do
    r=$(timeout 180 ros2 service call /$n/save_map std_srvs/srv/Trigger 2>&1 | tail -2)
    echo "$r" | grep -q "success=True" || { echo "Salvataggio di $n NON riuscito:"; echo "$r"; exit 1; }
    echo "  $n salvata"
  done
else
  for f in elevazione semantica; do
    t="$M/${f}_${BASE}.tmp.npz"
    if [ -f "$t" ] && [ "$t" -nt "$M/${f}_${BASE}.npz" ]; then
      echo "ATTENZIONE: c'è ${f}_${BASE}.tmp.npz più recente della mappa: l'ultimo salvataggio si è interrotto."
      echo "Nel file c'è la mappa di prima, non quella dal vivo. Non copio niente e non ripristino niente."
      exit 1
    fi
  done
fi

echo "Controllo i file salvati:"
for f in elevazione semantica; do controlla $f "$M/${f}_${BASE}.npz" || exit 1; done

for f in elevazione semantica; do
  cp "$M/${f}_${BASE}.npz" "$M/${f}_${NUOVO}.parziale"
  mv "$M/${f}_${NUOVO}.parziale" "$M/${f}_${NUOVO}.npz"
done
echo "Copia della mappa dal vivo: ${NUOVO}"
for f in elevazione semantica; do controlla $f "$M/${f}_${NUOVO}.npz" || exit 1; done

if [ ! -f "$M/elevazione_${BASE}_backup.npz" ] || [ ! -f "$M/semantica_${BASE}_backup.npz" ]; then
  echo "Non c'è ${BASE}_backup: la mappa ${BASE} resta quella dal vivo."
else
  if pgrep -f "$NODI" >/dev/null; then
    echo
    echo "Ora ferma la finestra mappe con Ctrl-C. Aspetto che si chiuda, poi rimetto ${BASE} com'era prima (${BASE}_backup)."
    for i in $(seq 1 300); do pgrep -f "$NODI" >/dev/null || break; sleep 1; done
    if pgrep -f "$NODI" >/dev/null; then echo "Le mappe sono ancora accese dopo 5 minuti: non ripristino. Rilancia lo script quando sono ferme."; exit 1; fi
    sleep 2
  fi
  for f in elevazione semantica; do
    cp "$M/${f}_${BASE}_backup.npz" "$M/${f}_${BASE}.parziale"
    mv "$M/${f}_${BASE}.parziale" "$M/${f}_${BASE}.npz"
    rm -f "$M/${f}_${BASE}.tmp.npz"
  done
  echo "${BASE} rimessa com'era prima del giro dal vivo:"
  for f in elevazione semantica; do controlla $f "$M/${f}_${BASE}.npz" || exit 1; done
fi

echo
echo "Per riusare la mappa dal vivo a mappa ferma, nella finestra mappe:"
echo "  ros2 launch semantic start_mapping.launch.py map_name:=${NUOVO} aggiorna_mappe:=false raggio_gradino_letale:=0.3"
