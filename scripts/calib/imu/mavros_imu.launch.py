"""MAVROS per l'IMU del Pixhawk 1 (ArduPilot) via USB, con richiesta automatica degli stream.

    source ~/mavros_ws/install/setup.bash
    ros2 launch ~/Desktop/thesis_ws/calib/imu/mavros_imu.launch.py
    ros2 launch ~/Desktop/thesis_ws/calib/imu/mavros_imu.launch.py stream_rate:=20

ArduPilot non invia i dati finche' qualcuno non li chiede: dopo `stream_delay` secondi
si chiama /mavros/set_stream_rate (tutti gli stream, `stream_rate` Hz).
Topic: /mavros/imu/data (orientamento EKF + gyro + acc), /mavros/imu/data_raw, /mavros/imu/mag.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    fcu_url = LaunchConfiguration('fcu_url')
    stream_rate = LaunchConfiguration('stream_rate')
    stream_delay = LaunchConfiguration('stream_delay')

    return LaunchDescription([
        DeclareLaunchArgument('fcu_url', default_value='/dev/ttyACM0:57600',
                              description='Porta del Pixhawk (USB: il baud e\' ignorato)'),
        DeclareLaunchArgument('stream_rate', default_value='50', description='Hz richiesti'),
        DeclareLaunchArgument('stream_delay', default_value='8.0',
                              description='secondi di attesa della connessione prima della richiesta'),

        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution([FindPackageShare('mavros'), 'launch', 'apm.launch'])),
            launch_arguments={'fcu_url': fcu_url}.items()),

        TimerAction(period=stream_delay, actions=[ExecuteProcess(
            cmd=['ros2', 'service', 'call', '/mavros/set_stream_rate', 'mavros_msgs/srv/StreamRate',
                 ['{stream_id: 0, message_rate: ', stream_rate, ', on_off: true}']],
            output='screen')]),
    ])
