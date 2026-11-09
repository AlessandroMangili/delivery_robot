import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('obstacles_tracking')
    default_config = os.path.join(pkg_share, 'config', 'obstacles_tracking.yaml')

    config_arg = DeclareLaunchArgument(
        'config_file', default_value=default_config,
        description='Parameter file of yolo_detect_node, pointpillars_detector and eagermot_tracker_node')
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use the simulation clock')
    launch_pointpillars_arg = DeclareLaunchArgument(
        'launch_pointpillars', default_value='true',
        description='Start pointpillars_detector (false = start it by hand in its virtual environment)')
    pcdet_python_arg = DeclareLaunchArgument(
        'pcdet_python', default_value='python3',
        description='Python interpreter of the virtual environment with torch and OpenPCDet')

    config_file = LaunchConfiguration('config_file')

    yolo_node = Node(
        package='obstacles_tracking',
        executable='yolo_detect_node',
        name='yolo_detect_node',
        output='screen',
        parameters=[config_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        additional_env={'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4',
                        'OMP_WAIT_POLICY': 'PASSIVE'},
    )

    eagermot_node = Node(
        package='obstacles_tracking',
        executable='eagermot_node',
        name='eagermot_tracker_node',
        output='screen',
        parameters=[config_file, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        additional_env={'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
    )

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
