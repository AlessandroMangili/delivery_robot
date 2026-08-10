import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dynamic = get_package_share_directory('dynamic_layer')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Usa il clock di simulazione (Gazebo)'
    )
    enable_arg = DeclareLaunchArgument(
        'dynamic_layer', default_value='true',
        description='Avvia il nodo dynamic_tracker (costo anticipatorio dinamici)'
    )
    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dynamic, 'config', 'dynamic_tracker.yaml'),
        description='File dei parametri del layer ESDF differenziale'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')

    # nodo del layer anticipatorio: legge i parametri dal esdf.yaml e
    # riceve use_sim_time dal launch (il param nel yaml e' ridondante ma innocuo).
    pointcloud = Node(
        package='dynamic_layer',
        executable='pointcloud_detector',
        name='pointcloud_detector',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time},
        ],
        condition=IfCondition(LaunchConfiguration('dynamic_layer')),
    )
    
    dynamic_tracker = Node(
        package='dynamic_layer',
        executable='lidar3d_tracker',
        name='lidar3d_tracker',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time},
        ],
        condition=IfCondition(LaunchConfiguration('dynamic_layer')),
    )

    return LaunchDescription([
        use_sim_time_arg,
        enable_arg,
        params_arg,
        pointcloud,
        dynamic_tracker,
    ])