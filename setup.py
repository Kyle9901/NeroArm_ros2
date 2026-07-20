import os
from glob import glob

from setuptools import find_packages, setup


package_name = "vision_grasp"


setup(
    name=package_name,
    version="0.2.1",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/robot.yaml"]),
        (
            os.path.join("share", package_name, "launch"),
            glob(os.path.join("launch", "*.launch.py")),
        ),
    ],
    install_requires=[
        "setuptools",
        "opencv-python",
        "numpy",
        "requests",
        "PyYAML",
        "langgraph>=1.2.0",
        "langchain-core>=1.4.0",
        "mcp>=1.27.0",
        "ultralytics>=8.3.0",
    ],
    zip_safe=False,
    maintainer="Alkaid",
    maintainer_email="alkaid@example.com",
    description="LLM-orchestrated vision grasping for AGX Arm",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "robot_arm_mcp = mcp_server.task_server:main",
        ],
    },
)
