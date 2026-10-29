#!/usr/bin/env bash
MR=${1:-0.3}
if pgrep -f "velodyne_driver_node" >/dev/null; then echo "Il driver del Velodyne gira gia': chiudi lo stack del robot prima (tmux kill-session -t robot)."; exit 1; fi
source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=96 ROS_LOCALHOST_ONLY=1 PYTHONDONTWRITEBYTECODE=1
D=$(ros2 pkg prefix velodyne_driver)/share/velodyne_driver/config/VLP16-velodyne_driver_node-params.yaml
T=$(ros2 pkg prefix velodyne_pointcloud)/share/velodyne_pointcloud/config/VLP16-velodyne_transform_node-params.yaml
C=$(ros2 pkg prefix velodyne_pointcloud)/share/velodyne_pointcloud/params/VLP16db.yaml
setsid ros2 run velodyne_driver velodyne_driver_node --ros-args --params-file $D -p timestamp_first_packet:=true > /tmp/prova_mr_driver.log 2>&1 &
P1=$!
setsid ros2 run velodyne_pointcloud velodyne_transform_node --ros-args --params-file $T -p min_range:=$MR -p calibration:=$C > /tmp/prova_mr_transform.log 2>&1 &
P2=$!
sleep 4
python3 $(dirname $(readlink -f $0))/punti_vicini.py 10 $HOME/bags/punti_vicini_mr$MR.png
kill -INT -$P1 -$P2 2>/dev/null; sleep 2; kill -KILL -$P1 -$P2 2>/dev/null
