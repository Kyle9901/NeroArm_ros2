"""External ROS launch-process management and readiness checks."""

import os
import signal
import subprocess
import threading
import time


class BringupManager:
    def __init__(self, node_provider):
        self._node_provider = node_provider
        self._processes = {}
        self._lock = threading.Lock()

    def _spawn(self, name: str, command: list[str]) -> tuple[bool, str]:
        with self._lock:
            current = self._processes.get(name)
            if current is not None and current.poll() is None:
                return False, f"{name} is already running (pid={current.pid})"
            log_dir = "/tmp/robot_arm_bringup"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{name}.log")
            with open(log_path, "w") as output:
                process = subprocess.Popen(
                    command,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    start_new_session=True,
                )
            self._processes[name] = process
            return True, f"started (pid={process.pid}, log={log_path})"

    @staticmethod
    def _format_tcp_offset(values) -> str:
        """Format the configured six-dimensional TCP transform for ros2 launch."""
        return "[" + ", ".join(f"{float(value):.10g}" for value in values) + "]"

    @staticmethod
    def _signal_process_group(process, sig) -> str | None:
        if process.poll() is not None:
            return None
        try:
            os.killpg(os.getpgid(process.pid), sig)
            return None
        except ProcessLookupError:
            return None
        except Exception as exc:
            return str(exc)

    @staticmethod
    def _wait_remaining(processes: dict, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        remaining = dict(processes)
        while remaining and time.monotonic() < deadline:
            remaining = {
                name: process for name, process in remaining.items()
                if process.poll() is None
            }
            if remaining:
                time.sleep(0.05)
        return {
            name: process for name, process in remaining.items()
            if process.poll() is None
        }

    def stop_all(self) -> dict:
        """Stop only launch process groups created by this manager."""
        with self._lock:
            active = {
                name: process for name, process in self._processes.items()
                if process.poll() is None
            }
        if not active:
            return {"success": True, "stopped": [], "forced": [], "errors": {}}

        errors = {}
        forced = []
        for name, process in active.items():
            error = self._signal_process_group(process, signal.SIGINT)
            if error:
                errors[name] = error
        remaining = self._wait_remaining(active, 5.0)

        for name, process in remaining.items():
            error = self._signal_process_group(process, signal.SIGTERM)
            if error:
                errors[name] = error
        remaining = self._wait_remaining(remaining, 3.0)

        for name, process in remaining.items():
            forced.append(name)
            error = self._signal_process_group(process, signal.SIGKILL)
            if error:
                errors[name] = error
        remaining = self._wait_remaining(remaining, 1.0)
        for name in remaining:
            errors[name] = errors.get(name, "process did not exit after SIGKILL")

        with self._lock:
            for name, process in list(self._processes.items()):
                if process.poll() is not None:
                    self._processes.pop(name, None)
        stopped = [name for name in active if name not in remaining]
        return {
            "success": not errors,
            "stopped": stopped,
            "forced": forced,
            "errors": errors,
        }

    @staticmethod
    def check_can(can_port: str = "can0") -> dict:
        try:
            output = subprocess.check_output(
                ["ip", "link", "show", can_port],
                stderr=subprocess.STDOUT,
                timeout=2.0,
            ).decode()
            if "state UP" in output:
                return {"up": True, "message": f"{can_port} is UP"}
            return {
                "up": False,
                "message": f"{can_port} exists but is DOWN. Run: sudo ip link set {can_port} up",
            }
        except subprocess.CalledProcessError:
            return {"up": False, "message": f"{can_port} not found. Check CAN hardware and driver"}
        except Exception as exc:
            return {"up": False, "message": f"CAN check failed: {exc}"}

    def _wait_endpoint(self, endpoint_type: str, name: str, timeout: float) -> bool:
        started = time.monotonic()
        node = self._node_provider()
        while time.monotonic() - started < timeout:
            try:
                if endpoint_type == "action":
                    from moveit_msgs.action import MoveGroup
                    from rclpy.action import ActionClient

                    client = ActionClient(node, MoveGroup, name)
                    if client.wait_for_server(timeout_sec=0.3):
                        return True
                elif endpoint_type == "topic":
                    if name in [topic[0] for topic in node.get_topic_names_and_types()]:
                        return True
                elif endpoint_type == "service":
                    if name in [service[0] for service in node.get_service_names_and_types()]:
                        return True
                elif endpoint_type == "node":
                    names = [
                        f"{namespace.rstrip('/')}/{node_name}".replace("//", "/")
                        for node_name, namespace in node.get_node_names_and_namespaces()
                    ]
                    if name in names:
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def start(self, can_port: str = "can0", calib_name: str = "my_eih_calib_park",
              octomap_enabled: bool = False) -> dict:
        can_status = self.check_can(can_port)
        if not can_status["up"]:
            return {
                "success": False,
                "can": can_status,
                "hint": "CAN接口未就绪。请手动执行: sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up",
                "arm": "skipped",
                "camera": "skipped",
                "pointcloud_filter": "skipped",
                "calib": "skipped",
            }

        results = {"success": True, "can": can_status}
        if self._wait_endpoint("action", "/move_action", 0.5):
            results["arm_launch"] = "not started: /move_action already exists"
            results["arm"] = "already_ready"
        else:
            tcp_offset = self._format_tcp_offset(
                self._node_provider().get_parameter("tcp_offset").value
            )
            ok, message = self._spawn("arm", [
                "ros2", "launch", "agx_arm_ctrl", "start_single_agx_arm_moveit.launch.py",
                f"can_port:={can_port}", "arm_type:=nero", "effector_type:=agx_gripper",
                f"tcp_offset:={tcp_offset}",
            ])
            results["arm_launch"] = message
            results["arm"] = "ready" if ok and self._wait_endpoint("action", "/move_action", 10.0) else "started_but_not_ready"
            if results["arm"] == "started_but_not_ready":
                results["hint"] = "MoveIt启动超时(10s)。检查CAN、机械臂电源和驱动。"

        if self._wait_endpoint("topic", "/camera/color/image_raw", 0.5):
            results["camera_launch"] = "not started: camera topic already exists"
            results["camera"] = "already_ready"
        else:
            ok, message = self._spawn("camera", [
                "ros2", "launch", "orbbec_camera", "dabai.launch.py",
                "publish_tf:=false", "depth_registration:=true",
            ])
            results["camera_launch"] = message
            results["camera"] = "ready" if ok and self._wait_endpoint("topic", "/camera/color/image_raw", 3.0) else "started_but_not_ready"

        if self._wait_endpoint("node", "/handeye_publisher", 0.5):
            results["calib_launch"] = "not started: /handeye_publisher already exists"
            results["calib"] = "already_ready"
        else:
            ok, message = self._spawn("calib", [
                "ros2", "launch", "easy_handeye2", "publish.launch.py", f"name:={calib_name}",
            ])
            results["calib_launch"] = message
            results["calib"] = (
                "ready" if ok and self._wait_endpoint("node", "/handeye_publisher", 3.0)
                else "started_but_not_ready"
            )

        # The filter pipeline is optional.  Do not spend CPU filtering clouds or
        # populate MoveIt's occupancy map when fast planning is requested.
        if not octomap_enabled:
            results["pointcloud_filter_launch"] = "not started: OctoMap disabled by configuration"
            results["pointcloud_filter"] = "skipped_disabled"
        else:
            cloud = self.start_pointcloud_filter(enabled=True)
            results["pointcloud_filter_launch"] = cloud["message"]
            results["pointcloud_filter"] = "ready" if cloud["success"] else "started_but_not_ready"
        if results["pointcloud_filter"] in ("failed", "started_but_not_ready"):
            results["success"] = False
            results["hint"] = (
                "点云过滤器未发布 /octomap_cloud。检查 /camera/depth_registered/points "
                "和 /tmp/robot_arm_bringup/pointcloud_filter.log。"
            )

        return results

    def start_pointcloud_filter(self, enabled: bool = True) -> dict:
        """Start the controllable cloud pipeline on demand."""
        if self._wait_endpoint("service", "/octomap/set_enabled", 0.2):
            return {"success": True, "message": "not started: OctoMap gate already exists"}
        ok, message = self._spawn("pointcloud_filter", [
            "ros2", "launch", "pcl_filter", "pcl_filter.launch.py",
            "input_cloud:=/camera/depth_registered/points", "voxel_leaf:=0.02",
            f"octomap_enabled:={'true' if enabled else 'false'}",
        ])
        ready = ok and self._wait_endpoint("service", "/octomap/set_enabled", 5.0)
        return {"success": ready, "message": message}

    def status(self) -> dict:
        with self._lock:
            processes = {
                name: {"pid": process.pid, "running": process.poll() is None}
                for name, process in self._processes.items()
            }
        endpoints = {
            "move_action": self._wait_endpoint("action", "/move_action", 0.5),
            "camera_color": self._wait_endpoint("topic", "/camera/color/image_raw", 0.5),
            "camera_points": self._wait_endpoint("topic", "/camera/depth_registered/points", 0.5),
            "handeye_publisher": self._wait_endpoint("node", "/handeye_publisher", 0.5),
            "filtered_cloud": self._wait_endpoint("topic", "/filtered_cloud", 0.5),
            "octomap_cloud": self._wait_endpoint("topic", "/octomap_cloud", 0.5),
            "octomap_control": self._wait_endpoint("service", "/octomap/set_enabled", 0.5),
            "planning_scene_apply": self._wait_endpoint("service", "/apply_planning_scene", 0.5),
            "planning_scene_get": self._wait_endpoint("service", "/get_planning_scene", 0.5),
            "tf": self._wait_endpoint("topic", "/tf", 0.5),
        }
        return {
            "can": self.check_can(),
            "processes": processes,
            "endpoints": endpoints,
            "topics": {
                name: self._topic_graph(name)
                for name in (
                    "/camera/depth_registered/points",
                    "/filtered_cloud",
                    "/octomap_cloud",
                )
            },
        }

    def _topic_graph(self, topic: str) -> dict:
        node = self._node_provider()
        try:
            publishers = node.get_publishers_info_by_topic(topic)
            subscriptions = node.get_subscriptions_info_by_topic(topic)
            return {
                "publishers": len(publishers),
                "subscriptions": len(subscriptions),
                "publisher_nodes": sorted({info.node_name for info in publishers}),
                "subscriber_nodes": sorted({info.node_name for info in subscriptions}),
            }
        except Exception as exc:
            return {
                "publishers": 0,
                "subscriptions": 0,
                "publisher_nodes": [],
                "subscriber_nodes": [],
                "error": str(exc),
            }
