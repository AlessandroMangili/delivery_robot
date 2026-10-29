#!/usr/bin/env bash
S=robot
if tmux has-session -t $S 2>/dev/null; then exec tmux attach -t $S; fi
INIT='source /opt/ros/humble/setup.bash && source ~/Desktop/thesis_ws/install/setup.bash && export PYTHONDONTWRITEBYTECODE=1 && cd ~/Desktop/thesis_ws'
finestra() {
  if [ "$1" = sensori ]; then tmux new-session -d -s $S -n sensori; else tmux new-window -t $S -n "$1"; fi
  tmux send-keys -t $S:"$1" "$INIT" C-m
  tmux send-keys -t $S:"$1" "$2"
}
finestra sensori 'ros2 launch husky_a100_description display.launch.py rviz:=false'
finestra localizz 'ros2 launch localization start_localization.launch.py use_map_localization:=true avvio:=globale use_global:=false'
finestra mappe 'ros2 launch semantic start_mapping.launch.py map_name:=campus_2510 aggiorna_mappe:=false raggio_gradino_letale:=0.3'
finestra nav 'ros2 launch navigation start_navigation.launch.py libera_impronta:=true'
finestra dinamico 'ros2 launch dynamic_layer start_dynamic_navigation_layer.launch.py'
finestra cpu './prove/registra_cpu.sh'
finestra bag './record_run.sh navigazione'
finestra vai 'python3 prove/vai.py '
finestra jstat 'python3 ~/Desktop/thesis_ws/jstat.py'
finestra imu 'python3 controllo_pixhawk.py'
finestra salva './prove/salva_mappe_vivo.sh campus_2510_centro'
tmux set -t $S mouse on
tmux select-window -t $S:sensori
exec tmux attach -t $S
