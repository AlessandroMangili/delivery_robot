import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('husky_a100_description')

    model = LaunchConfiguration('model')
    dimensions_file = LaunchConfiguration('dimensions_file')
    urdf_extras = LaunchConfiguration('urdf_extras')
    rviz_config = LaunchConfiguration('rviz_config')
    rviz = LaunchConfiguration('rviz')
    sensors = LaunchConfiguration('sensors')
    replay = LaunchConfiguration('replay')
    sim_time = ParameterValue(replay, value_type=bool)

    def live(cond):
        return IfCondition(PythonExpression(["'", cond, "' == 'true' and '", replay, "' != 'true'"]))

    robot_description = ParameterValue(
        Command(['xacro ', model,
                 ' dimensions_file:=', dimensions_file,
                 ' urdf_extras:=', urdf_extras]),
        value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument(
            'model', default_value=os.path.join(pkg, 'urdf', 'husky_a100.urdf.xacro'),
            description='Robot xacro file'),
        DeclareLaunchArgument(
            'dimensions_file', default_value=os.path.join(pkg, 'config', 'a100_dimensions.yaml'),
            description='YAML file with the robot dimensions'),
        DeclareLaunchArgument(
            'urdf_extras', default_value=os.path.join(pkg, 'urdf', 'sensors.urdf.xacro'),
            description='Extra xacro file with the sensors (empty for none)'),
        DeclareLaunchArgument(
            'rviz_config', default_value=os.path.join(pkg, 'rviz', 'a100.rviz'),
            description='RViz configuration'),
        DeclareLaunchArgument(
            'replay', default_value='false',
            description='true when replaying a bag with ros2 bag play --clock: simulated time, no drivers'),
        DeclareLaunchArgument(
            'sensors', default_value='true',
            description='true also starts the LiDAR, camera and Pixhawk drivers (sensors.launch.py)'),

        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[{'robot_description': robot_description, 'use_sim_time': sim_time}],
             remappings=[('joint_states', '/a200_0000/platform/joint_states')], output='screen'),
        DeclareLaunchArgument('rviz', default_value='true', description='true opens RViz'),
        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_config], output='screen',
             parameters=[{'use_sim_time': sim_time}], condition=IfCondition(rviz)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sensors.launch.py')),
            condition=live(sensors)),
    ])
