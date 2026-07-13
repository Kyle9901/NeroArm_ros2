"""MoveIt planning-scene objects and allowed-collision rules."""

from geometry_msgs.msg import Pose
from moveit_msgs.msg import AllowedCollisionEntry, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive


_GRASP_ACM_LINKS = [
    "gripper_base", "gripper_link1", "gripper_link2", "tcp_link",
    "gripper_flange", "link7", "link5", "link6",
]


class PlanningSceneService:
    def __init__(self, node, callback_group):
        self._node = node
        self._client = node.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
            callback_group=callback_group,
        )
        self._publisher = node.create_publisher(PlanningScene, "/planning_scene", 10)
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

    def _publish_acm(self, object_id: str, allow_object: bool):
        names = [object_id, *_GRASP_ACM_LINKS]
        scene = PlanningScene(is_diff=True)
        scene.allowed_collision_matrix.entry_names = names
        for name_i in names:
            entry = AllowedCollisionEntry()
            entry.enabled = [
                allow_object if name_i == object_id or name_j == object_id else True
                for name_j in names
            ]
            scene.allowed_collision_matrix.entry_values.append(entry)
        self._publisher.publish(scene)

    def add_target(
        self,
        x: float,
        y: float,
        z: float,
        object_id: str = "target_object",
        size: tuple[float, float, float] = (0.06, 0.06, 0.08),
        shape: str = "BOX",
        timeout: float = 5.0,
    ) -> bool:
        if not self._client.wait_for_service(timeout):
            self._node.get_logger().warn("apply_planning_scene not available")
            return False
        collision = CollisionObject()
        collision.id = object_id
        collision.header.frame_id = self._node._get_param("base_frame")
        collision.operation = CollisionObject.ADD
        primitive = SolidPrimitive()
        if shape.upper() == "CYLINDER":
            primitive.type = SolidPrimitive.CYLINDER
            primitive.dimensions = [size[2], size[0] / 2.0]
        else:
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = list(size)
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = float(x), float(y), float(z)
        pose.orientation.w = 1.0
        collision.primitives.append(primitive)
        collision.primitive_poses.append(pose)
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(collision)

        self._node.get_logger().info(
            f"Adding target collision: id={object_id}, pos=({x:.3f},{y:.3f},{z:.3f}), "
            f"shape={shape}, size={size}"
        )
        response = self._apply(scene, timeout)
        if response is None or not response.success:
            self._node.get_logger().warn("add target collision failed or timed out")
            return False
        self._publish_acm(object_id, allow_object=True)
        self._node.get_logger().info(
            f"target collision '{object_id}' added, ACM ghost mode enabled"
        )
        return True

    def remove_target(self, object_id: str = "target_object", timeout: float = 5.0) -> bool:
        if not self._client.wait_for_service(timeout):
            return False
        collision = CollisionObject(id=object_id, operation=CollisionObject.REMOVE)
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(collision)
        self._publish_acm(object_id, allow_object=False)
        response = self._apply(scene, timeout)
        return response is not None and response.success
