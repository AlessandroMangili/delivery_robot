import os
import subprocess

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
                            LogInfo, OpaqueFunction, TimerAction)
from launch.launch_description_sources import (AnyLaunchDescriptionSource,
                                               PythonLaunchDescriptionSource)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

C920_DEVICE = '/dev/v4l/by-id/usb-046d_HD_Pro_Webcam_C920_979BF06F-video-index0'
C920_CALIBRATION = os.path.expanduser('~/.ros/camera_info/camera2.yaml')


def is_running(cmdline_pattern):
    return subprocess.run(['pgrep', '-f', cmdline_pattern], capture_output=True).returncode == 0


def is_true(context, name):
    return LaunchConfiguration(name).perform(context).lower() in ('true', '1', 'yes')


def lidar_actions(context):
    if not is_true(context, 'lidar'):
        return []
    if is_running('^[^ ]*/lib/velodyne_driver/velodyne_driver_node( |$)'):
        return [LogInfo(msg='[sensors] driver Velodyne gia\' in esecuzione: non lo rilancio')]
    launch_file = os.path.join(get_package_share_directory('velodyne'),
                               'launch', 'velodyne-all-nodes-VLP16-launch.py')
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(launch_file))]


def camera_node(namespace, by_id, params):
    if is_running(f'^[^ ]*/usb_cam_node_exe .*__ns:=/{namespace}( |$)'):
        return LogInfo(msg=f'[sensors] camera /{namespace} gia\' in esecuzione: non la rilancio')
    if not os.path.exists(by_id):
        return LogInfo(msg=f'[sensors] camera /{namespace} non collegata ({by_id})')
    params = dict(params, video_device=os.path.realpath(by_id), io_method='mmap')
    return Node(package='usb_cam', executable='usb_cam_node_exe', name='usb_cam',
                namespace=namespace, parameters=[params], output='screen')


def camera_actions(context):
    actions = []
    if is_true(context, 'c920'):
        actions.append(camera_node('c920', C920_DEVICE, {
            'image_width': 640, 'image_height': 480, 'framerate': 30.0,
            'pixel_format': 'mjpeg2rgb',          
            'frame_id': 'c920_optical',
            'camera_name': 'camera2',             
            'camera_info_url': 'file://' + C920_CALIBRATION,
        }))
    return actions


def imu_actions(context):
    if not is_true(context, 'imu'):
        return []
    if is_running('^[^ ]*/lib/mavros/mavros_node( |$)'):
        return [LogInfo(msg='[sensors] MAVROS gia\' in esecuzione: non la rilancio')]
    try:
        mavros_share = get_package_share_directory('mavros')
    except PackageNotFoundError:
        return [LogInfo(msg='[sensors] MAVROS non trovata: source ~/mavros_ws/install/setup.bash')]
    fcu_url = LaunchConfiguration('fcu_url').perform(context)
    port = fcu_url.split(':')[0]
    if port.startswith('/dev/') and not os.path.exists(port):
        return [LogInfo(msg=f'[sensors] Pixhawk non collegata ({port}): MAVROS non avviata')]

    rate = LaunchConfiguration('stream_rate').perform(context)
    request = ['ros2', 'service', 'call', '/mavros/set_stream_rate', 'mavros_msgs/srv/StreamRate',
               f'{{stream_id: 0, message_rate: {rate}, on_off: true}}']
    mavros = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(mavros_share, 'launch', 'node.launch')),
        launch_arguments={
            'fcu_url': fcu_url, 'gcs_url': '', 'tgt_system': '1', 'tgt_component': '1',
            'pluginlists_yaml': os.path.join(get_package_share_directory('husky_a100_description'),
                                             'config', 'mavros_pluginlists.yaml'),
            'config_yaml': os.path.join(mavros_share, 'launch', 'apm_config.yaml'),
        }.items())
    return [mavros] + [
        TimerAction(period=t, actions=[
            LogInfo(msg=f'[sensors] richiesta stream a {rate} Hz alla Pixhawk'),
            ExecuteProcess(cmd=request, output='log')])
        for t in (8.0, 20.0)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('lidar', default_value='true', description='driver VLP-16'),
        DeclareLaunchArgument('c920', default_value='true', description='camera C920 (usb_cam)'),
        DeclareLaunchArgument('imu', default_value='true', description='MAVROS: IMU, GPS, bussola della Pixhawk'),
        DeclareLaunchArgument('fcu_url', default_value='/dev/ttyACM0:57600',
                              description='porta della Pixhawk (USB: il baud e\' ignorato)'),
        DeclareLaunchArgument('stream_rate', default_value='50', description='Hz richiesti alla Pixhawk'),
        OpaqueFunction(function=lidar_actions),
        OpaqueFunction(function=camera_actions),
        OpaqueFunction(function=imu_actions),
    ])
