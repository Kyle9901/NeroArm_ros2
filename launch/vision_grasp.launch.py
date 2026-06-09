from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Arguments
    target_frame_arg = DeclareLaunchArgument(
        'target_frame',
        default_value='base_link',
        description='Target frame for grasping (arm base frame)'
    )

    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera_color_optical_frame',
        description='Camera optical frame'
    )

    use_handeye_arg = DeclareLaunchArgument(
        'use_handeye',
        default_value='true',
        description='Use easy_handeye2 calibration result'
    )

    handeye_name_arg = DeclareLaunchArgument(
        'handeye_name',
        default_value='nero_handeye',
        description='Name of the handeye calibration'
    )

    # TF: Use handeye calibration or static transform
    # Option 1: Use easy_handeye2 publisher (if calibration exists)
    handeye_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('easy_handeye2'),
                'launch',
                'publish.launch.py'
            )
        ),
        launch_arguments={
            'name': LaunchConfiguration('handeye_name'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('use_handeye'))
    )

    # Option 2: Static transform (fallback)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_base_tf',
        arguments=[
            '0.3',    # x: 30cm in front of base
            '0.0',    # y: centered
            '0.5',    # z: 50cm above base
            '0.0',    # roll
            '0.0',    # pitch
            '0.0',    # yaw
            'base_link',
            'camera_color_optical_frame'
        ],
        condition=UnlessCondition(LaunchConfiguration('use_handeye'))
    )

    # Vision Grasp Node
    grasp_node = Node(
        package='vision_grasp',
        executable='grasp_node',
        name='vision_grasp_node',
        parameters=[{
            'target_frame': LaunchConfiguration('target_frame'),
            'camera_frame': LaunchConfiguration('camera_frame'),
            'grasp_offset_z': 0.05,
            'grasp_depth': 0.02,
            'gripper_open_width': 0.08,
            'gripper_close_width': 0.0,
            'gripper_force': 1.5,
            'approach_height': 0.10,
            'lift_height': 0.15,
            'place_x_offset': 0.15,
        }],
        output='screen'
    )

    return LaunchDescription([
        target_frame_arg,
        camera_frame_arg,
        use_handeye_arg,
        handeye_name_arg,
        handeye_launch,
        static_tf_node,
        grasp_node,
    ])
