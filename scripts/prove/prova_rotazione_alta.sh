#!/usr/bin/env bash
set -m
set +u; source /opt/ros/humble/setup.bash; source $HOME/Desktop/thesis_ws/install/setup.bash
export PYTHONDONTWRITEBYTECODE=1
NOME=$HOME/bags/rotazione_alta_$(date +%Y%m%d_%H%M%S)
echo "Prova di rotazione. Robot in uno spazio libero di 3 m, Nav2 SPENTO, joystick in mano (LB = fermi tu). Invio per partire, Ctrl-C per fermare."
read
python3 $HOME/Desktop/thesis_ws/install/navigation/lib/navigation/cmd_vel_ponte.py > $NOME.ponte.log 2>&1 &
PONTE=$!
ros2 bag record -o $NOME /mavros/imu/data /cmd_vel_safe /a200_0000/cmd_vel /a200_0000/platform/odom /a200_0000/platform/joint_states /Odometry /velodyne_points < /dev/null > $NOME.log 2>&1 &
REC=$!
sleep 5
if [ ! -d $NOME ]; then
  echo "ERRORE: la registrazione non parte, il robot NON si muove. Log: $NOME.log"
  grep -v "sequence size" $NOME.log | tail -5
  kill -INT $PONTE $REC 2>/dev/null
  exit 1
fi
echo "Registro in $NOME"
python3 $HOME/Desktop/thesis_ws/prove/prova_rotazione_alta.py
sleep 2
kill -INT $REC; kill -INT $PONTE
wait $REC 2>/dev/null
python3 $HOME/Desktop/thesis_ws/prove/analizza_rotazione.py $NOME
python3 $HOME/Desktop/thesis_ws/prove/analizza_montaggio.py $NOME
