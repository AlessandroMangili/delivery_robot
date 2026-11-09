import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Use the simulation clock')

    nav2_navigation_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')
    navigation_params_path = os.path.join(
        get_package_share_directory('navigation'), 'config', 'navigation.yaml')

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_navigation_launch_path),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'params_file': navigation_params_path,
        }.items()
    )

    return LaunchDescription([sim_time_arg, navigation_launch])
