import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('semantic'), 'config', 'params.yaml')
    return LaunchDescription([
        Node(package='semantic', executable='segmentation_node',
             name='segmentation_node', output='screen', parameters=[params]),
        Node(package='semantic', executable='semantic_costmap_node',
             name='semantic_costmap_node', output='screen', parameters=[params]),
    ])