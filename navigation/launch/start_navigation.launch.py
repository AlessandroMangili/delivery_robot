import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from ament_index_python.packages import get_package_share_directory
from nav2_common.launch import RewrittenYaml

def robot_footprint(margin=0.02):
    dimensions_path = os.path.join(
        get_package_share_directory('husky_a100_description'),
        'config',
        'a100_dimensions.yaml'
    )
    with open(dimensions_path) as f:
        overall = yaml.safe_load(f)['overall']
    half_length = round(overall['length'] / 2 + margin, 3)
    half_width = round(overall['width'] / 2 + margin, 3)
    return (f'[[{half_length}, {half_width}], [{half_length}, {-half_width}], '
            f'[{-half_length}, {-half_width}], [{-half_length}, {half_width}]]')

def generate_launch_description():

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='true sulle bag con --clock, false sul robot reale'
    )

    muovi_robot_arg = DeclareLaunchArgument(
        'muovi_robot', default_value='true',
        description='false = non inoltra i comandi al Pi (prove sulle bag)'
    )

    libera_impronta_arg = DeclareLaunchArgument(
        'libera_impronta', default_value='false',
        description='true = le celle letali della mappa ferma sotto l impronta attuale del robot non bloccano MPPI'
    )

    rotazione_minima_arg = DeclareLaunchArgument(
        'rotazione_minima', default_value='0.0',
        description='rad/s; 0 = spenta. Sotto 0,10 m/s il ponte alza a questo valore le rotazioni chieste, con rampa'
    )

    nav2_navigation_launch_path = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'launch',
        'navigation_launch.py'
    )

    navigation_params = RewrittenYaml(
        source_file=os.path.join(get_package_share_directory('navigation'), 'config', 'navigation.yaml'),
        param_rewrites={'use_sim_time': LaunchConfiguration('use_sim_time'),
                        'footprint': robot_footprint(),
                        'libera_impronta': LaunchConfiguration('libera_impronta'),
                        'rotazione_minima': LaunchConfiguration('rotazione_minima'),
                        'default_nav_to_pose_bt_xml': os.path.join(
                            get_package_share_directory('navigation'), 'config', 'bt_senza_spin.xml')},
        convert_types=True
    )

    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_navigation_launch_path),
        launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'params_file': navigation_params,
        }.items()
    )

    collision_monitor_node = Node(
        package='nav2_collision_monitor',
        executable='collision_monitor',
        name='collision_monitor',
        output='screen',
        parameters=[navigation_params]
    )

    impronta_sicurezza = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub', '--rate', '2', '/impronta_sicurezza', 'geometry_msgs/msg/PolygonStamped',
             '{header: {frame_id: base_footprint}, polygon: {points: [{x: 0.60, y: 0.36}, {x: 0.60, y: -0.36}, '
             '{x: -0.55, y: -0.36}, {x: -0.55, y: 0.36}]}}'],
        output='log'
    )

    collision_monitor_lifecycle = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_collision_monitor',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                     'autostart': True,
                     'node_names': ['collision_monitor']}]
    )

    cmd_vel_ponte_node = Node(
        package='navigation',
        executable='cmd_vel_ponte.py',
        name='cmd_vel_ponte',
        output='screen',
        parameters=[navigation_params],
        condition=IfCondition(LaunchConfiguration('muovi_robot'))
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(sim_time_arg)
    launchDescriptionObject.add_action(muovi_robot_arg)
    launchDescriptionObject.add_action(libera_impronta_arg)
    launchDescriptionObject.add_action(rotazione_minima_arg)
    launchDescriptionObject.add_action(SetParameter(name='bond_timeout', value=20.0))
    launchDescriptionObject.add_action(navigation_launch)
    launchDescriptionObject.add_action(impronta_sicurezza)
    launchDescriptionObject.add_action(collision_monitor_node)
    launchDescriptionObject.add_action(collision_monitor_lifecycle)
    launchDescriptionObject.add_action(cmd_vel_ponte_node)

    return launchDescriptionObject
