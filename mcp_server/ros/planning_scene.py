"""Optional MoveIt planning-scene objects."""

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive


class PlanningSceneService:
    def __init__(self, node, callback_group):
        self._node = node
        self._client = node.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
            callback_group=callback_group,
        )
        self._desk_added = False

    def _apply(self, scene: PlanningScene, timeout: float):
        request = ApplyPlanningScene.Request()
        request.scene = scene
        future = self._client.call_async(request)
        if not self._node._spin_until(future, timeout):
            return None
        return future.result()

    def add_desk(self, timeout: float = 5.0) -> bool:
        if self._desk_added:
            return True
        if not self._client.wait_for_service(timeout):
            self._node.get_logger().warn("apply_planning_scene not available — skip desk collision")
            return False

        desk_z = self._node._get_param("desk_z_surface")
        size = list(self._node._get_param("desk_size"))
        collision = CollisionObject()
        collision.id = "desk"
        collision.header.frame_id = self._node._get_param("base_frame")
        collision.operation = CollisionObject.ADD
        primitive = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=size)
        pose = Pose()
        pose.position.z = desk_z - size[2] / 2.0
        pose.orientation.w = 1.0
        collision.primitives.append(primitive)
        collision.primitive_poses.append(pose)
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(collision)

        self._node.get_logger().info(
            f"Adding desk collision: surface_z={desk_z:.3f}, size={size}"
        )
        response = self._apply(scene, timeout)
        if response is None or not response.success:
            self._node.get_logger().warn("add desk collision failed or timed out")
            return False
        self._desk_added = True
        self._node.get_logger().info("desk collision object added")
        return True
