"""Avvia la pipeline EagerMOT: detector 3D (PointPillars), detector 2D (YOLO),
tracker a due stadi.

Alternativo a dynamic_layer/lidar3d_tracker: MAI insieme, pubblicano entrambi
/dynamic_tracks_state (verificare con `ros2 topic info /dynamic_tracks_state`
che ci sia un solo publisher).

pointpillars_detector ha bisogno di torch/pcdet, tipicamente in un venv
separato da quello di sistema (vedi il docstring del nodo). Questo launch lo
avvia con l'interprete indicato da `pcdet_python` (di default il python di
sistema, quasi certamente SBAGLIATO): passare
    pcdet_python:=/percorso/al/venv/bin/python3
e assicurarsi che quel venv veda anche i pacchetti ROS (--system-site-packages
o install/setup.bash sourcato nella stessa shell). Con
`launch_pointpillars:=false` non lo avvia: comodo per lanciarlo a mano in un
altro terminale col venv attivato, guardando i suoi log [pp] direttamente.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('obstacles_tracking')
    default_config = os.path.join(pkg_share, 'config', 'obstacles_tracking.yaml')

    config_arg = DeclareLaunchArgument(
        'config_file', default_value=default_config,
        description='Parametri YAML per yolo_detect_node, pointpillars_detector, eagermot_node')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Flag use_sim_time')

    launch_pointpillars_arg = DeclareLaunchArgument(
        'launch_pointpillars', default_value='true',
        description='Se false, non avvia pointpillars_detector (lanciarlo a mano nel venv)')

    pcdet_python_arg = DeclareLaunchArgument(
        'pcdet_python', default_value='python3',
        description='Interprete python del venv con torch/pcdet per pointpillars_detector')

    config_file = LaunchConfiguration('config_file')

    # 4 thread: YOLO su CPU usa davvero il parallelismo, ma senza limite
    # torch/OpenMP si dimensionano su tutti i 48 core (120 thread osservati).
    yolo_node = Node(
        package='obstacles_tracking',
        executable='yolo_detect_node',
        name='yolo_detect_node',
        output='screen',
        parameters=[config_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        additional_env={'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'},
    )

    # 1 thread BLAS: il Kalman lavora su matrici 10x10, dove il multithreading
    # non serve. Senza limite OpenBLAS teneva 48 thread in attesa attiva, e il
    # tracker usava ~3.5 core.
    eagermot_node = Node(
        package='obstacles_tracking',
        executable='eagermot_node',
        name='eagermot_tracker_node',
        output='screen',
        parameters=[config_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        additional_env={'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
    )

    # Avviato come processo con l'interprete del venv (non `ros2 run`, che
    # userebbe il python di sistema): -m richiede che quel venv veda anche il
    # pacchetto installato di obstacles_tracking (PYTHONPATH da install/setup.bash).
    pointpillars_process = ExecuteProcess(
        cmd=[
            LaunchConfiguration('pcdet_python'), '-m',
            'obstacles_tracking.pointpillars_detector',
            '--ros-args', '--params-file', config_file,
            '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')],
        ],
        output='screen',
        condition=IfCondition(LaunchConfiguration('launch_pointpillars')),
    )

    return LaunchDescription([
        config_arg,
        use_sim_time_arg,
        launch_pointpillars_arg,
        pcdet_python_arg,
        yolo_node,
        eagermot_node,
        pointpillars_process,
    ])
