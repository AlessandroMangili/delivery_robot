#!/usr/bin/env bash
# Registra un run del robot e alla fine controlla perdite/ritardi (calib/bag_health.py).
#
#   ./record_run.sh            run completo (Ctrl-C per fermare)
#   ./record_run.sh prova 30   prova di 30 s: da fare PRIMA del run vero
#   ./record_run.sh calib      solo camere + LiDAR, per la calibrazione con il pannello
#
# Bag in $BAG_DIR (default ~/bags). Durante la registrazione NON tenere aperti RViz
# o la segmentazione: sulla Orin Nano rubano la CPU al registratore (il run del
# 20/09 ha perso ~55% dei messaggi su tutti i topic, anche /odom).
set -euo pipefail

MODE=${1:-run}
DUR=${2:-0}
OUT_DIR=${BAG_DIR:-$HOME/bags}
WS=$HOME/Desktop/thesis_ws

# messaggi grandi (nuvole, camera1) in memoria condivisa invece che a frammenti UDP
export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-$WS/dds/fastdds_big_msgs.xml}
set +u
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"
set -u

SENSORS=(/velodyne_points /c920/image_raw/compressed /c920/camera_info)
NAV=(/mavros/imu/data /mavros/imu/data_raw /mavros/imu/mag
     /mavros/global_position/raw/fix /mavros/global_position/raw/gps_vel
     /mavros/gpsstatus/gps1/raw /mavros/global_position/compass_hdg /mavros/global_position/global
     /a200_0000/platform/odom /a200_0000/platform/odom/filtered /a200_0000/cmd_vel /a200_0000/joy_teleop/cmd_vel /a200_0000/platform/cmd_vel
     /a200_0000/platform/joint_states /a200_0000/tf /a200_0000/tf_static /tf /tf_static)
LOC=(/Odometry /odometry/local /odometry/global /odometry/gps /gps/filtered /pcl_pose /alignment_status /reinitialization_requested)
NAV2=(/cmd_vel_nav /cmd_vel /cmd_vel_safe /plan /goal_pose /navigate_to_pose/_action/status /polygon_stop /polygon_slowdown)
LEGGERA=(/velodyne_points /c920/image_raw/compressed /c920/camera_info /mavros/imu/data /mavros/global_position/raw/fix
     /a200_0000/platform/odom /a200_0000/cmd_vel /a200_0000/joy_teleop/cmd_vel /a200_0000/platform/joint_states /tf /tf_static
     /Odometry /odometry/local /pcl_pose /alignment_status /reinitialization_requested /cmd_vel_nav /cmd_vel_safe /plan /goal_pose
     /local_costmap/costmap /behavior_tree_log /detections /dynamic_tracks_state /camera_detections /trajectories)
NAVIGAZIONE=(/tf /tf_static /mavros/imu/data /mavros/global_position/raw/fix /a200_0000/platform/odom /a200_0000/cmd_vel
     /a200_0000/joy_teleop/cmd_vel /a200_0000/platform/joint_states /Odometry /odometry/local /pcl_pose /alignment_status
     /reinitialization_requested /startup_initialization/status /cmd_vel_nav /cmd_vel_safe /plan /goal_pose
     /navigate_to_pose/_action/status /local_costmap/costmap /behavior_tree_log /detections /dynamic_tracks_state
     /camera_detections /trajectories)
MAPPE=("${NAVIGAZIONE[@]}" /velodyne_points /c920/image_raw/compressed /c920/camera_info)
case "$MODE" in
  calib)     TOPICS=("${SENSORS[@]}") ;;
  gps)       TOPICS=("${NAV[@]}" "${LOC[@]}") ;;
  run|prova) TOPICS=("${SENSORS[@]}" "${NAV[@]}" "${LOC[@]}" "${NAV2[@]}") ;;
  leggera)   TOPICS=("${LEGGERA[@]}") ;;
  navigazione) TOPICS=("${NAVIGAZIONE[@]}") ;;
  mappe)     TOPICS=("${MAPPE[@]}") ;;
  *) echo "modo sconosciuto: $MODE (run | leggera | navigazione | mappe | prova [secondi] | calib | gps)"; exit 1 ;;
esac
if [ "$MODE" = prova ] && [ "$DUR" -eq 0 ]; then DUR=30; fi

nvpmodel -q 2>/dev/null | grep -q MAXN || \
  echo "ATTENZIONE: Jetson non in MAXN_SUPER -> sudo nvpmodel -m 2 && sudo jetson_clocks"
if pgrep -f "rviz2|seg_infer_node" >/dev/null; then
  echo "ATTENZIONE: rviz2 o seg_infer_node attivi: tolgono CPU al registratore."
fi

if [ "$MODE" = mappe ]; then
  LIBERI=$(df -BG --output=avail "$OUT_DIR" | tail -1 | tr -dc 0-9)
  echo "Profilo mappe: LiDAR e camera, circa 9 MB/s = 0,5 GB al minuto. Liberi ${LIBERI} GB = circa $(( (LIBERI - 5) * 18 / 10 )) minuti lasciando 5 GB."
  if [ "$LIBERI" -lt 10 ]; then echo "Meno di 10 GB liberi: non registro."; exit 1; fi
fi

NAME=${MODE}_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT_DIR"
echo "Registro in $OUT_DIR/$NAME: ${TOPICS[*]}"
PYTHONDONTWRITEBYTECODE=1 python3 "$WS/monitor_risorse.py" registra "$OUT_DIR/${NAME}_risorse.csv" > "$OUT_DIR/${NAME}_risorse.log" 2>&1 &
MON=$!
nice -n 19 ionice -c3 ros2 bag record -o "$OUT_DIR/$NAME" --max-cache-size 200000000 --max-bag-duration 60 "${TOPICS[@]}" &
PID=$!
trap 'kill -INT $PID 2>/dev/null || true' INT TERM
if [ "$DUR" -gt 0 ]; then sleep "$DUR"; kill -INT $PID 2>/dev/null || true; fi
wait $PID 2>/dev/null || true
wait $PID 2>/dev/null || true      # dopo Ctrl-C il primo wait ritorna subito
trap - INT TERM
kill -INT $MON 2>/dev/null || true
wait $MON 2>/dev/null || true

DB="$OUT_DIR/$NAME"
echo; echo "=== controllo $DB"
python3 "$WS/calib/bag_health.py" "$DB"
echo "Sano: 'persi' ~0% e ritardi di pochi ms (LiDAR ~100 ms: timestamp a inizio spazzata)."
echo; echo "=== risorse di Jetson e Pi ($OUT_DIR/${NAME}_risorse.png)"
tail -3 "$OUT_DIR/${NAME}_risorse.log"
