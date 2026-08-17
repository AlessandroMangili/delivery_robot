"""
navsat.launch.py — SOLO navsat_transform, in un launch a parte.

Serve per il test del fallback GNSS: lo lanci in un terminale suo e lo spegni con
Ctrl-C quando vuoi simulare la perdita del GPS, senza toccare gli EKF ne' RViz.
Usalo insieme a:  ros2 launch localization localization.launch.py use_gps:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('localization')
    ekf_yaml = os.path.join(pkg, 'config', 'ekf.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(
            package='robot_localization', executable='navsat_transform_node',
            name='navsat_transform_node', output='screen',
            parameters=[ekf_yaml, {'use_sim_time': use_sim_time}],
            remappings=[
                ('imu', '/imu'),
                ('gps/fix', '/navsat'),
                ('odometry/filtered', '/odometry/global'),
                ('odometry/gps', '/odometry/gps'),
            ],
        ),
    ])