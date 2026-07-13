from setuptools import find_packages, setup


package_name = "vision_grasp"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "opencv-python", "numpy", "requests"],
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
