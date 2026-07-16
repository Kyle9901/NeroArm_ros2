"""Runtime control of the cloud gate and MoveIt occupancy map."""

from std_srvs.srv import Empty, SetBool

from .futures import wait_for_future


class OctomapControl:
    def __init__(self, node, callback_group) -> None:
        self._node = node
        self._gate = node.create_client(
            SetBool, "/octomap/set_enabled", callback_group=callback_group
        )
        self._clear = node.create_client(
            Empty, "/clear_octomap", callback_group=callback_group
        )

    def set_enabled(self, enabled: bool, timeout: float = 5.0) -> dict:
        enabled = bool(enabled)
        gate_wait = timeout if enabled else min(timeout, 0.1)
        if not self._gate.wait_for_service(timeout_sec=gate_wait):
            if not enabled:
                return self._clear_without_gate(timeout)
            return {
                "success": False,
                "enabled": enabled,
                "error": "/octomap/set_enabled unavailable; start pcl_filter",
            }

        gate_future = self._gate.call_async(SetBool.Request(data=enabled))
        if not wait_for_future(gate_future, timeout, context=self._node.context):
            return {"success": False, "enabled": enabled, "error": "OctoMap gate timeout"}
        gate_response = gate_future.result()
        if gate_response is None or not gate_response.success:
            message = gate_response.message if gate_response else "no response"
            return {"success": False, "enabled": enabled, "error": message}

        cleared = False
        if not enabled:
            if not self._clear.wait_for_service(timeout_sec=timeout):
                return {
                    "success": False,
                    "enabled": False,
                    "error": "cloud forwarding stopped, but /clear_octomap is unavailable",
                }
            clear_future = self._clear.call_async(Empty.Request())
            if not wait_for_future(clear_future, timeout, context=self._node.context):
                return {
                    "success": False,
                    "enabled": False,
                    "error": "cloud forwarding stopped, but clearing OctoMap timed out",
                }
            cleared = clear_future.result() is not None

        return {
            "success": True,
            "enabled": enabled,
            "cleared": cleared,
            "gate_available": True,
            "message": "OctoMap enabled" if enabled else "OctoMap disabled and cleared",
        }

    def _clear_without_gate(self, timeout: float) -> dict:
        """Disabled is already satisfied when no gate exists; clear stale voxels."""
        if not self._clear.wait_for_service(timeout_sec=timeout):
            return {
                "success": False,
                "enabled": False,
                "gate_available": False,
                "error": "OctoMap gate is absent and /clear_octomap is unavailable",
            }
        clear_future = self._clear.call_async(Empty.Request())
        if not wait_for_future(clear_future, timeout, context=self._node.context):
            return {
                "success": False,
                "enabled": False,
                "gate_available": False,
                "error": "Clearing the disabled OctoMap timed out",
            }
        cleared = clear_future.result() is not None
        return {
            "success": cleared,
            "enabled": False,
            "cleared": cleared,
            "gate_available": False,
            "message": "OctoMap disabled (no gate running) and cleared",
        }
