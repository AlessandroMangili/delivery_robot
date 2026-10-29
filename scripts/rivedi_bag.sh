#!/usr/bin/env bash
BAG=${1:-}
DA=${2:-0}
VEL=${3:-0.5}
S=rivedi
if [ -z "$BAG" ] || [ ! -f "$BAG/metadata.yaml" ]; then
  echo "Uso: $0 <cartella della bag> [secondo di partenza] [velocità]"
  echo "Esempio: $0 ~/bags/leggera_20261017_193540 200 0.5"
  exit 1
fi
BAG=$(readlink -f "$BAG")
if tmux has-session -t robot 2>/dev/null || pgrep -f "start_localization|velodyne_driver_node" >/dev/null; then
  echo "Lo stack del robot è acceso: chiudilo prima (tmux kill-session -t robot), la bag gira sullo stesso dominio."
  exit 1
fi
if tmux has-session -t $S 2>/dev/null; then
  echo "La sessione '$S' è già aperta: tmux attach -t $S, oppure chiudila con tmux kill-session -t $S"
  exit 1
fi
INIT="source /opt/ros/humble/setup.bash && source ~/Desktop/thesis_ws/install/setup.bash && unset ROS_DOMAIN_ID; export PYTHONDONTWRITEBYTECODE=1 && cd ~/Desktop/thesis_ws"
TOPIC="/tf /tf_static /pcl_pose /odometry/local /Odometry /velodyne_points /plan /c920/image_raw/compressed /c920/camera_info /mavros/imu/data"
finestra() {
  if [ "$1" = sensori ]; then tmux new-session -d -s $S -n sensori; else tmux new-window -t $S -n "$1"; fi
  tmux send-keys -t $S:"$1" "$INIT && $2" C-m
}
finestra sensori 'ros2 launch husky_a100_description display.launch.py replay:=true rviz:=false'
finestra mappe 'ros2 launch semantic start_mapping.launch.py replay:=true map_name:=giro2_riempita aggiorna_mappe:=false raggio_gradino_letale:=0.3'
finestra nav 'ros2 launch navigation start_navigation.launch.py use_sim_time:=true muovi_robot:=false'
finestra dinamico 'ros2 launch dynamic_layer start_dynamic_navigation_layer.launch.py use_sim_time:=true immagine:=/c920/image_raw/compressed'
finestra bag "sleep 25 && ros2 bag play $BAG --clock --rate $VEL --start-offset $DA --topics $TOPIC /local_costmap/costmap --remap /local_costmap/costmap:=/robot/local_costmap"
echo "Revisione di $(basename $BAG) dal secondo $DA a velocità $VEL, dominio 0."
echo "La bag parte fra 25 s. Nella finestra 'bag': spazio = pausa, freccia destra = un passo."
echo "Per chiudere:   tmux kill-session -t $S"
tmux set -t $S mouse on
tmux select-window -t $S:bag
exec tmux attach -t $S
