from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'target_object',
            default_value='blue block',
            description='Target object description for VLM detection'),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'),

        Node(
            package='vision_grasp',
            executable='vlm_picker',
            name='vlm_picker',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }]),

        Node(
            package='vision_grasp',
            executable='grasp_executor',
            name='grasp_executor',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
            }]),
    ])
