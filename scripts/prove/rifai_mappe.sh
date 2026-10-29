#!/usr/bin/env bash
set -uo pipefail
BAG=${1:-}
BASE=${2:-}
if [ -z "$BAG" ] || [ -z "$BASE" ]; then
  echo "uso: ./prove/rifai_mappe.sh CARTELLA_BAG MAPPA_DI_PARTENZA [NOME_NUOVO]"
  echo "  MAPPA_DI_PARTENZA = per esempio campus_2510_centro, oppure vuota"
  echo "  il bag deve essere registrato con ./record_run.sh mappe (LiDAR e camera)"
  exit 1
fi
BAG=$(realpath "$BAG")
NUOVO=${3:-${BASE}_rifatta_$(basename "$BAG" | sed 's/^[a-z]*_//')}
WS=$HOME/Desktop/thesis_ws
M=$WS/src/delivery_robot/semantic/maps
LOG=$HOME/bags/rifai_mappe/$NUOVO
set +u
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
set -u
export ROS_DOMAIN_ID=97 ROS_LOCALHOST_ONLY=1 PYTHONDONTWRITEBYTECODE=1
cd "$WS"

robot_acceso() { pgrep -x fastlio_mapping >/dev/null || pgrep -x velodyne_driver >/dev/null; }
if robot_acceso; then echo "Il robot è acceso (FAST-LIO o driver del Velodyne): non rigioco niente."; exit 1; fi
for t in /velodyne_points /c920/image_raw/compressed /tf; do
  grep -q "name: $t\$" "$BAG/metadata.yaml" || { echo "Nel bag manca $t: serve un bag registrato con ./record_run.sh mappe"; exit 1; }
done
for f in elevazione semantica; do
  [ -e "$M/${f}_${NUOVO}.npz" ] && { echo "Esiste già ${f}_${NUOVO}.npz: scegli un altro nome"; exit 1; }
done
if [ "$BASE" != vuota ]; then
  for f in elevazione semantica; do
    [ -f "$M/${f}_${BASE}.npz" ] || { echo "Non trovo ${f}_${BASE}.npz"; exit 1; }
    cp "$M/${f}_${BASE}.npz" "$M/${f}_${NUOVO}.npz"
  done
fi

mkdir -p "$LOG"
echo "Rigioco $BAG partendo da $BASE: le mappe nuove si chiameranno $NUOVO. Log in $LOG"
setsid ros2 launch husky_a100_description display.launch.py replay:=true rviz:=false > "$LOG/sensori.log" 2>&1 &
P0=$!
setsid ros2 launch semantic start_mapping.launch.py replay:=true map_name:="$NUOVO" aggiorna_mappe:=true raggio_gradino_letale:=0.3 core_mappe:=0-5 > "$LOG/mappe.log" 2>&1 &
P1=$!
for i in $(seq 1 60); do
  ros2 service list 2>/dev/null | grep -q /semantic_costmap_node/save_map && ros2 service list 2>/dev/null | grep -q /elevation_costmap_node/save_map && break
  sleep 2
done
sleep 10
setsid timeout -s INT ${DURATA:-0} ros2 bag play "$BAG" --clock --rate 1.0 --topics /tf /tf_static /velodyne_points /c920/image_raw/compressed /c920/camera_info /pcl_pose /odometry/local /mavros/imu/data > "$LOG/bag.log" 2>&1 &
P2=$!
INTERROTTA=0
while kill -0 $P2 2>/dev/null; do
  if robot_acceso; then echo "Il robot si è acceso: interrompo."; kill -INT -$P2 2>/dev/null; INTERROTTA=1; break; fi
  sleep 5
done
wait $P2 2>/dev/null
sleep 3
for n in elevation_costmap_node semantic_costmap_node; do
  timeout 180 ros2 service call /$n/save_map std_srvs/srv/Trigger 2>&1 | grep -q "success=True" && echo "  $n salvata" || echo "  $n: salvataggio NON riuscito"
done
kill -INT -$P1 2>/dev/null
for i in $(seq 1 30); do kill -0 $P1 2>/dev/null || break; sleep 1; done
kill -INT -$P0 2>/dev/null
sleep 3
kill -KILL -$P1 -$P0 2>/dev/null
for f in elevazione semantica; do rm -f "$M/${f}_${NUOVO}.tmp.npz"; done
[ $INTERROTTA = 1 ] && echo "Rigioco interrotto a metà: le mappe $NUOVO contengono solo la prima parte."
echo
echo "Fatto. Per guardarle a mappa ferma (robot spento o al posto della mappa solita):"
echo "  ros2 launch semantic start_mapping.launch.py map_name:=${NUOVO} aggiorna_mappe:=false raggio_gradino_letale:=0.3"
