"""External ROS launch-process management and readiness checks."""

import os
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
                )
            self._processes[name] = process
            return True, f"started (pid={process.pid}, log={log_path})"

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
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def start(self, can_port: str = "can0", calib_name: str = "my_eih_calib_v6") -> dict:
        can_status = self.check_can(can_port)
        if not can_status["up"]:
            return {
                "success": False,
                "can": can_status,
                "hint": "CAN接口未就绪。请手动执行: sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up",
                "arm": "skipped",
                "camera": "skipped",
                "calib": "skipped",
            }

        results = {"success": True, "can": can_status}
        ok, message = self._spawn("arm", [
            "ros2", "launch", "agx_arm_ctrl", "start_single_agx_arm_moveit.launch.py",
            f"can_port:={can_port}", "arm_type:=nero", "effector_type:=agx_gripper",
        ])
        results["arm_launch"] = message
        if ok:
            results["arm"] = "ready" if self._wait_endpoint("action", "/move_action", 10.0) else "started_but_not_ready"
            if results["arm"] == "started_but_not_ready":
                results["hint"] = "MoveIt启动超时(10s)。检查CAN、机械臂电源和驱动。"
        else:
            results["arm"] = "already_ready" if self._wait_endpoint("action", "/move_action", 0.5) else "failed"

        ok, message = self._spawn("camera", [
            "ros2", "launch", "orbbec_camera", "dabai.launch.py", "publish_tf:=false",
        ])
        results["camera_launch"] = message
        if ok:
            results["camera"] = "ready" if self._wait_endpoint("topic", "/camera/color/image_raw", 3.0) else "started_but_not_ready"
        else:
            results["camera"] = "already_ready" if self._wait_endpoint("topic", "/camera/color/image_raw", 0.5) else "failed"

        ok, message = self._spawn("calib", [
            "ros2", "launch", "easy_handeye2", "publish.launch.py", f"name:={calib_name}",
        ])
        results["calib_launch"] = message
        if ok:
            results["calib"] = "ready" if self._wait_endpoint("topic", "/tf", 3.0) else "started_but_not_ready"
        else:
            results["calib"] = "already_ready" if self._wait_endpoint("topic", "/tf", 0.5) else "failed"
        return results

    def status(self) -> dict:
        with self._lock:
            processes = {
                name: {"pid": process.pid, "running": process.poll() is None}
                for name, process in self._processes.items()
            }
        return {
            "can": self.check_can(),
            "processes": processes,
            "endpoints": {
                "move_action": self._wait_endpoint("action", "/move_action", 0.5),
                "camera_color": self._wait_endpoint("topic", "/camera/color/image_raw", 0.5),
                "tf": self._wait_endpoint("topic", "/tf", 0.5),
            },
        }
