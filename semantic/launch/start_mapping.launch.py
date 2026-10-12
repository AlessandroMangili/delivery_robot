import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(get_package_share_directory('semantic'), 'config', 'semantic.yaml')
    replay = LaunchConfiguration('replay')
    brighten = LaunchConfiguration('brighten')
    sim_time = {'use_sim_time': ParameterValue(replay, value_type=bool)}
    image = PythonExpression(["'/c920/image_bright' if '", brighten, "' == 'true' else '/c920/image_raw'"])

    def node(executable, extra=None):
        return Node(package='semantic', executable=executable, name=executable, output='screen',
                    parameters=[params, sim_time] + ([extra] if extra else []))

    return LaunchDescription([
        DeclareLaunchArgument('replay', default_value='false',
                              description='true = rosbag con --clock: tempo simulato e immagini decompresse'),
        DeclareLaunchArgument('brighten', default_value='false',
                              description='true = schiarisce le immagini prima della segmentazione (notte)'),
        Node(package='image_transport', executable='republish', name='c920_decompress',
             arguments=['compressed', 'raw'],
             remappings=[('in/compressed', '/c920/image_raw/compressed'), ('out', '/c920/image_raw')],
             parameters=[sim_time], condition=IfCondition(replay)),
        Node(package='semantic', executable='image_brighten_node', name='image_brighten_node', output='screen',
             parameters=[params, sim_time], condition=IfCondition(brighten)),
        node('seg_infer_node', {'image_topic': image}),
        node('semantic_costmap_node'),
        node('elevation_costmap_node'),
        node('semantic_overlay_node', {'rgb_topic': image}),
        node('ssrl_combiner'),
    ])
