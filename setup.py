from setuptools import setup

package_name = 'vision_grasp'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/vlm_grasp.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alkaid',
    maintainer_email='alkaid@example.com',
    description='Vision-based grasping for AGX Arm',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_picker = vision_grasp.vlm_picker_node:main',
            'grasp_executor = vision_grasp.grasp_executor:main',
            'test_move_to = vision_grasp.test_move_to:main',
        ],
    },
)
