import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dynamic = get_package_share_directory('dynamic_layer')

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='true solo per rivedere una rosbag con --clock'
    )
    enable_arg = DeclareLaunchArgument(
        'dynamic_layer', default_value='true',
        description='Avvia il nodo dynamic_tracker (costo anticipatorio dinamici)'
    )
    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dynamic, 'config', 'dynamic_tracker.yaml'),
        description='File dei parametri del layer ESDF differenziale'
    )

    modello_arg = DeclareLaunchArgument(
        'modello', default_value='yolo26n.pt',
        description='modello YOLO della detection (in share/dynamic_layer/models o percorso assoluto)'
    )
    immagine_arg = DeclareLaunchArgument(
        'immagine', default_value='/c920/image_raw',
        description='topic della camera; con /compressed alla fine legge le immagini compresse'
    )

    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    un_thread = {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}

    camera = Node(
        package='dynamic_layer',
        executable='camera_detector',
        name='camera_detector',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time,
             'model_path': LaunchConfiguration('modello'),
             'image_topic': LaunchConfiguration('immagine')},
        ],
        additional_env=un_thread,
        condition=IfCondition(LaunchConfiguration('dynamic_layer')),
    )

    pointcloud = Node(
        package='dynamic_layer',
        executable='pointcloud_detector',
        name='pointcloud_detector',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time},
        ],
        additional_env=un_thread,
        condition=IfCondition(LaunchConfiguration('dynamic_layer')),
    )
    
    dynamic_tracker = Node(
        package='dynamic_layer',
        executable='lidar3d_tracker',
        name='lidar3d_tracker',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': use_sim_time},
        ],
        additional_env=un_thread,
        condition=IfCondition(LaunchConfiguration('dynamic_layer')),
    )

    return LaunchDescription([
        use_sim_time_arg,
        enable_arg,
        params_arg,
        modello_arg,
        immagine_arg,
        pointcloud,
        camera,
        dynamic_tracker,
    ])