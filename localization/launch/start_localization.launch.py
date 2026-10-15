import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import (AndSubstitution, EqualsSubstitution, LaunchConfiguration,
                                  NotSubstitution, PythonExpression)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    pkg_localization = get_package_share_directory('localization')

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='true sulle bag con --clock, false sul robot reale'
    )

    use_gps_arg = DeclareLaunchArgument(
        'use_gps', default_value='true',
        description='false = non avvia navsat'
    )

    use_fastlio_arg = DeclareLaunchArgument(
        'use_fastlio', default_value='true',
        description='false = non avvia FAST-LIO (sulle bag si usa /Odometry registrato)'
    )

    use_global_arg = DeclareLaunchArgument(
        'use_global', default_value='true',
        description='false = solo ekf_local, senza ekf_global e navsat'
    )

    use_map_localization_arg = DeclareLaunchArgument(
        'use_map_localization', default_value='false',
        description='true = lidar_localization_ros2 sulla mappa .pcd pubblica map -> odom'
    )

    map_path_arg = DeclareLaunchArgument(
        'map_path', default_value=os.path.join(pkg_localization, 'maps', 'mappa.pcd'),
        description='nuvola di punti della mappa'
    )

    initial_x_arg = DeclareLaunchArgument('initial_x', default_value='0.0', description='posa iniziale nella mappa [m]')
    initial_y_arg = DeclareLaunchArgument('initial_y', default_value='0.0', description='posa iniziale nella mappa [m]')
    initial_z_arg = DeclareLaunchArgument('initial_z', default_value='-0.57', description='quota di base_footprint nella mappa [m]')
    initial_yaw_arg = DeclareLaunchArgument('initial_yaw_deg', default_value='0.0', description='orientamento iniziale nella mappa [gradi]')

    avvio_arg = DeclareLaunchArgument(
        'avvio', default_value='fissa',
        description='fissa = posa iniziale dagli argomenti initial_*; globale = ultima posa salvata se verificata, altrimenti ricerca su tutta la mappa'
    )

    occupancy_yaml_arg = DeclareLaunchArgument(
        'occupancy_yaml', default_value=os.path.join(pkg_localization, 'maps', 'griglia.yaml'),
        description='griglia 2D della mappa per la ricerca globale'
    )

    pose_state_path_arg = DeclareLaunchArgument(
        'pose_state_path', default_value=os.path.join(os.path.expanduser('~'), '.ros', 'posa_localizzatore.json'),
        description='file dove salvare l ultima posa nella mappa'
    )

    ekf_params_path = os.path.join(pkg_localization, 'config', 'localization.yaml')
    fastlio_params_path = os.path.join(pkg_localization, 'config', 'fast_lio_vlp16.yaml')
    lidar_localization_params_path = os.path.join(pkg_localization, 'config', 'lidar_localization.yaml')

    sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    use_map_localization = LaunchConfiguration('use_map_localization')
    avvio_globale = AndSubstitution(use_map_localization, EqualsSubstitution(LaunchConfiguration('avvio'), 'globale'))
    initial_z = ParameterValue(LaunchConfiguration('initial_z'), value_type=float)
    initial_qz = PythonExpression(['sin(radians(', LaunchConfiguration('initial_yaw_deg'), ') / 2)'])
    initial_qw = PythonExpression(['cos(radians(', LaunchConfiguration('initial_yaw_deg'), ') / 2)'])

    fastlio_node = Node(
        package='fast_lio',
        executable='fastlio_mapping',
        name='laser_mapping',
        output='screen',
        parameters=[fastlio_params_path, sim_time],
        condition=IfCondition(LaunchConfiguration('use_fastlio'))
    )

    fastlio_frame_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_camera_init',
        arguments=['--x', '-0.12', '--y', '0', '--z', '0.35', '--frame-id', 'odom', '--child-frame-id', 'camera_init'],
        parameters=[sim_time]
    )

    ekf_local_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_local',
        output='screen',
        parameters=[ekf_params_path, sim_time],
        remappings=[('/odometry/filtered', '/odometry/local')]
    )

    ekf_global_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_global',
        output='screen',
        parameters=[ekf_params_path, sim_time,
                    {'publish_tf': NotSubstitution(use_map_localization)}],
        remappings=[('/odometry/filtered', '/odometry/global')],
        condition=IfCondition(LaunchConfiguration('use_global'))
    )

    navsat_transform_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        parameters=[ekf_params_path, sim_time],
        remappings=[
            ('imu', '/mavros/imu/data'),
            ('gps/fix', '/mavros/global_position/raw/fix'),
            ('odometry/filtered', '/odometry/global'),
            ('odometry/gps', '/odometry/gps'),
        ],
        condition=IfCondition(AndSubstitution(LaunchConfiguration('use_gps'), LaunchConfiguration('use_global')))
    )

    lidar_localization_node = Node(
        package='lidar_localization_ros2',
        executable='lidar_localization_node',
        name='lidar_localization',
        output='screen',
        parameters=[lidar_localization_params_path, sim_time,
                    {'map_path': LaunchConfiguration('map_path'),
                     'set_initial_pose': NotSubstitution(EqualsSubstitution(LaunchConfiguration('avvio'), 'globale')),
                     'initial_pose_x': ParameterValue(LaunchConfiguration('initial_x'), value_type=float),
                     'initial_pose_y': ParameterValue(LaunchConfiguration('initial_y'), value_type=float),
                     'initial_pose_z': initial_z,
                     'initial_pose_qz': ParameterValue(initial_qz, value_type=float),
                     'initial_pose_qw': ParameterValue(initial_qw, value_type=float)}],
        remappings=[('/cloud', '/velodyne_points')],
        condition=IfCondition(use_map_localization)
    )

    lidar_localization_start = Node(
        package='lidar_localization_ros2',
        executable='start_lifecycle_node.py',
        arguments=['lidar_localization'],
        output='screen',
        condition=IfCondition(use_map_localization)
    )

    global_localization_node = Node(
        package='lidar_localization_ros2',
        executable='global_localization_node.py',
        name='global_localization_node',
        output='screen',
        parameters=[sim_time,
                    {'occupancy_yaml': LaunchConfiguration('occupancy_yaml'),
                     'map_path': LaunchConfiguration('map_path'),
                     'cloud_topic': '/velodyne_points',
                     'global_frame_id': 'map',
                     'base_frame_id': 'base_footprint',
                     'odom_frame_id': 'odom',
                     'use_cpp_backend': True,
                     'enable_registration_scoring': True,
                     'registration_score_gate': 6.0,
                     'registration_refine_candidates': False,
                     'registration_seed_z_m': initial_z,
                     'seed_z_m': initial_z,
                     'max_scan_points': 256,
                     'angular_resolution_deg': 5.0,
                     'max_candidates': 8,
                     'nms_radius_m': 3.0}],
        condition=IfCondition(avvio_globale)
    )

    startup_initialization_node = Node(
        package='lidar_localization_ros2',
        executable='startup_initialization_node.py',
        name='startup_initialization',
        output='screen',
        parameters=[sim_time,
                    {'map_path': LaunchConfiguration('map_path'),
                     'pose_state_path': LaunchConfiguration('pose_state_path'),
                     'restore_saved_pose': True,
                     'initial_pose_preconfigured': False,
                     'enable_global_initialization': True,
                     'require_global_registration_scoring': True,
                     'cloud_topic': '/velodyne_points',
                     'pose_topic': '/pcl_pose',
                     'global_frame_id': 'map',
                     'odom_frame_id': 'odom',
                     'base_frame_id': 'base_footprint',
                     'default_z_m': initial_z}],
        condition=IfCondition(avvio_globale)
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(sim_time_arg)
    launchDescriptionObject.add_action(use_gps_arg)
    launchDescriptionObject.add_action(use_fastlio_arg)
    launchDescriptionObject.add_action(use_global_arg)
    launchDescriptionObject.add_action(use_map_localization_arg)
    launchDescriptionObject.add_action(map_path_arg)
    launchDescriptionObject.add_action(initial_x_arg)
    launchDescriptionObject.add_action(initial_y_arg)
    launchDescriptionObject.add_action(initial_z_arg)
    launchDescriptionObject.add_action(initial_yaw_arg)
    launchDescriptionObject.add_action(avvio_arg)
    launchDescriptionObject.add_action(occupancy_yaml_arg)
    launchDescriptionObject.add_action(pose_state_path_arg)
    launchDescriptionObject.add_action(fastlio_node)
    launchDescriptionObject.add_action(fastlio_frame_tf)
    launchDescriptionObject.add_action(ekf_local_node)
    launchDescriptionObject.add_action(ekf_global_node)
    launchDescriptionObject.add_action(navsat_transform_node)
    launchDescriptionObject.add_action(lidar_localization_node)
    launchDescriptionObject.add_action(lidar_localization_start)
    launchDescriptionObject.add_action(global_localization_node)
    launchDescriptionObject.add_action(startup_initialization_node)

    return launchDescriptionObject
