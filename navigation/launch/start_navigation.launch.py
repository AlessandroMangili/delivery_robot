import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from nav2_common.launch import RewrittenYaml


def robot_footprint(margin=0.02):
    dimensions_path = os.path.join(
        get_package_share_directory('husky_a100_description'), 'config', 'a100_dimensions.yaml')
    with open(dimensions_path) as f:
        overall = yaml.safe_load(f)['overall']
    half_length = round(overall['length'] / 2 + margin, 3)
    half_width = round(overall['width'] / 2 + margin, 3)
    return (f'[[{half_length}, {half_width}], [{half_length}, {-half_width}], '
            f'[{-half_length}, {-half_width}], [{-half_length}, {half_width}]]')


def generate_launch_description():
    pkg_navigation = get_package_share_directory('navigation')
    use_sim_time = LaunchConfiguration('use_sim_time')

    params = RewrittenYaml(
        source_file=os.path.join(pkg_navigation, 'config', 'navigation.yaml'),
        param_rewrites={'use_sim_time': use_sim_time,
                        'footprint': robot_footprint(),
                        'clear_footprint': LaunchConfiguration('clear_footprint'),
                        'default_nav_to_pose_bt_xml': os.path.join(pkg_navigation, 'config', 'bt_no_spin.xml')},
        convert_types=True)

    return LaunchDescription([
        DeclareLaunchArgument(
            'clear_footprint', default_value='false',
            description='true ignores the lethal cells of the stored map under the robot footprint, '
                        'so that an old obstacle saved in the map cannot trap the robot'),
        DeclareLaunchArgument(
            'move_robot', default_value='true',
            description='false does not forward the commands to the robot, for tests on bags'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='true when replaying a bag with --clock'),
        SetParameter(name='bond_timeout', value=20.0),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('nav2_bringup'), 'launch', 'navigation_launch.py')),
            launch_arguments={'use_sim_time': use_sim_time, 'params_file': params}.items()),
        Node(
            package='navigation', executable='cmd_vel_bridge.py', name='cmd_vel_bridge',
            output='screen', parameters=[params],
            condition=IfCondition(LaunchConfiguration('move_robot'))),
    ])
