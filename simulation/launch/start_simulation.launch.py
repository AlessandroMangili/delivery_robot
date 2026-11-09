import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, LaunchConfiguration, PathJoinSubstitution,
                                  PythonExpression, TextSubstitution)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_simulation = get_package_share_directory('simulation')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    gazebo_models_path, _ = os.path.split(pkg_simulation)
    os.environ['GZ_SIM_RESOURCE_PATH'] = (
        os.environ.get('GZ_SIM_RESOURCE_PATH', '') + os.pathsep + gazebo_models_path)

    rviz_launch_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Open RViz')
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value='navigation_gps.rviz',
        description='RViz config file')
    world_arg = DeclareLaunchArgument(
        'world', default_value='sidewalk.world',
        description='World file in simulation/worlds')
    model_arg = DeclareLaunchArgument(
        'model', default_value='turtlebot3_waffle_pi_gps.urdf',
        description='Robot description in simulation/urdf')
    x_arg = DeclareLaunchArgument(
        'x', default_value='0',
        description='x coordinate of the spawned robot')
    y_arg = DeclareLaunchArgument(
        'y', default_value='0',
        description='y coordinate of the spawned robot')
    yaw_arg = DeclareLaunchArgument(
        'yaw', default_value='0',
        description='yaw angle of the spawned robot')
    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Use the simulation clock')
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Start Gazebo without its GUI')

    urdf_file_path = PathJoinSubstitution([
        pkg_simulation, 'urdf', LaunchConfiguration('model')])

    gz_args_value = PythonExpression([
        "'-s -r -v -v1' if '",
        LaunchConfiguration('headless'),
        "'.lower() == 'true' else '-r -v -v1'"
    ])

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': [
                PathJoinSubstitution([pkg_simulation, 'worlds', LaunchConfiguration('world')]),
                TextSubstitution(text=' '),
                gz_args_value],
            'on_exit_shutdown': 'true'}.items()
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', PathJoinSubstitution(
            [pkg_simulation, 'rviz', LaunchConfiguration('rviz_config')])],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'robot_description': ParameterValue(
                Command(['xacro', ' ', urdf_file_path]), value_type=str),
             'use_sim_time': LaunchConfiguration('use_sim_time')},
        ],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')],
    )

    spawn_urdf_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'turtlebot3_waffle',
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', '0.2',
            '-Y', LaunchConfiguration('yaw'),
        ],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    gz_bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            'imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/navsat@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/scan/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/semantic/labels_map@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
    )

    return LaunchDescription([
        rviz_launch_arg,
        rviz_config_arg,
        world_arg,
        model_arg,
        x_arg,
        y_arg,
        yaw_arg,
        sim_time_arg,
        headless_arg,
        gazebo_launch,
        rviz_node,
        spawn_urdf_node,
        gz_bridge_node,
        robot_state_publisher_node,
    ])
