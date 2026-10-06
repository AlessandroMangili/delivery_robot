"""Simula l'Husky A100 in Gazebo Fortress (default di ROS 2 Humble).

    ros2 launch husky_a100_description gazebo.launch.py
    ros2 launch husky_a100_description gazebo.launch.py rviz:=false world:=empty.sdf

Per guidarlo, in un altro terminale:
    ros2 run teleop_twist_keyboard teleop_twist_keyboard

Topic ROS 2 disponibili (via ros_gz_bridge):
    /cmd_vel       geometry_msgs/Twist        ROS -> Gazebo
    /odom          nav_msgs/Odometry          Gazebo -> ROS
    /tf            odom -> base_footprint     Gazebo -> ROS
    /joint_states  stato delle 6 ruote        Gazebo -> ROS
    /clock         tempo simulato             Gazebo -> ROS
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _resource_path(var_name, extra):
    """Aggiunge una cartella al percorso risorse di Gazebo senza perdere quello esistente."""
    current = os.environ.get(var_name, '')
    return extra if not current else extra + os.pathsep + current


def generate_launch_description():
    pkg = get_package_share_directory('husky_a100_description')
    # Gazebo risolve package://husky_a100_description/... cercando la cartella del pacchetto
    # dentro i percorsi risorse: serve la cartella 'share' che la contiene.
    share_dir = os.path.dirname(pkg)

    model = LaunchConfiguration('model')
    dimensions_file = LaunchConfiguration('dimensions_file')
    urdf_extras = LaunchConfiguration('urdf_extras')
    world = LaunchConfiguration('world')
    spawn_z = LaunchConfiguration('spawn_z')
    rviz = LaunchConfiguration('rviz')

    robot_description = ParameterValue(
        Command(['xacro ', model,
                 ' dimensions_file:=', dimensions_file,
                 ' urdf_extras:=', urdf_extras,
                 ' use_gazebo:=true']),
        value_type=str)

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r ', world]}.items())

    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-name', 'husky_a100', '-topic', 'robot_description', '-z', spawn_z])

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V',
            '/joint_states@sensor_msgs/msg/JointState[ignition.msgs.Model',
        ])

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}])

    rviz_node = Node(
        package='rviz2', executable='rviz2', output='screen', condition=IfCondition(rviz),
        arguments=['-d', os.path.join(pkg, 'rviz', 'a100.rviz'), '-f', 'odom'],
        parameters=[{'use_sim_time': True}])

    return LaunchDescription([
        DeclareLaunchArgument(
            'model', default_value=os.path.join(pkg, 'urdf', 'husky_a100.urdf.xacro'),
            description='File xacro del robot'),
        DeclareLaunchArgument(
            'dimensions_file', default_value=os.path.join(pkg, 'config', 'a100_dimensions.yaml'),
            description='File YAML con le dimensioni'),
        DeclareLaunchArgument(
            'urdf_extras', default_value='',
            description='File xacro aggiuntivo con i sensori (vuoto = nessuno)'),
        DeclareLaunchArgument(
            'world', default_value='empty.sdf',
            description='Mondo Gazebo (file .sdf)'),
        DeclareLaunchArgument(
            'spawn_z', default_value='0.05',
            description='Quota di spawn [m], un po\' sopra il suolo'),
        DeclareLaunchArgument(
            'rviz', default_value='true',
            description='Avvia anche RViz2 con fixed frame odom'),

        # Fortress legge IGN_GAZEBO_RESOURCE_PATH; le versioni più nuove GZ_SIM_RESOURCE_PATH
        SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH',
                               _resource_path('IGN_GAZEBO_RESOURCE_PATH', share_dir)),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH',
                               _resource_path('GZ_SIM_RESOURCE_PATH', share_dir)),

        gz_sim,
        robot_state_publisher,
        spawn,
        bridge,
        rviz_node,
    ])
