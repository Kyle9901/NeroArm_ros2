"""Built-in deterministic task templates."""

from .place_resolver import resolve_place
from .types import ParamSpec, RetryConfig, Step, TaskTemplate


PICK_AND_PLACE = TaskTemplate(
    name="pick_and_place",
    description="抓取指定物体并放置到指定位置",
    match_patterns=["放到", "放在", "抓", "pick", "place"],
    required_params=[
        ParamSpec("target", "要抓取的物体描述"),
        ParamSpec("place", "放置位置，可以是方位词或base_link坐标"),
    ],
    pipeline=[
        Step("locate", skill="locate_object", args_from=["params.target"]),
        Step("grasp", skill="grasp_object", args_from=["locate.x", "locate.y", "locate.z"]),
        Step("resolve_place", fn=resolve_place, args_from=["params.place"]),
        Step("place", skill="place_object", args_from=["resolve_place.x", "resolve_place.y", "resolve_place.z"]),
    ],
    retry_policy={
        "locate": RetryConfig(max_attempts=3, recover="go_home"),
        "grasp": RetryConfig(max_attempts=2, recover="go_home"),
        "place": RetryConfig(max_attempts=2, recover="go_home"),
    },
    user_visible=[
        "locate.debug_image",
        "locate.x",
        "locate.y",
        "locate.z",
        "grasp.holding",
        "grasp.gripper_width",
        "place.place_x",
        "place.place_y",
        "place.place_z",
    ],
)


VISUAL_GRASP = TaskTemplate(
    name="visual_grasp",
    description="使用视觉伺服抓取指定物体",
    match_patterns=["视觉伺服", "visual", "直接抓", "抓取"],
    required_params=[ParamSpec("target", "要抓取的物体描述")],
    pipeline=[Step("visual_grasp", skill="visual_grasp", args_from=["params.target"])],
    retry_policy={"visual_grasp": RetryConfig(max_attempts=2, recover="go_home")},
    user_visible=["visual_grasp.holding", "visual_grasp.gripper_width", "visual_grasp.final_position"],
)


SCAN_SCENE = TaskTemplate(
    name="scan_scene",
    description="扫描桌面上的可见色块",
    match_patterns=["扫描", "识别", "检测", "scan", "detect", "物块"],
    required_params=[],
    pipeline=[Step("scan", skill="scan_scene")],
    retry_policy={"scan": RetryConfig(max_attempts=2, recover="go_home")},
    user_visible=["scan.count", "scan.blocks", "scan.debug_image"],
)


TEMPLATES = [PICK_AND_PLACE, VISUAL_GRASP, SCAN_SCENE]
