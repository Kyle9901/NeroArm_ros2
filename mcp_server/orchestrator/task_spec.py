"""Typed task semantics shared by deterministic and LLM task parsing.

Natural-language parsing must stop at :class:`TaskSpec`.  Only the compiler in
this module is allowed to turn task semantics into executable skill steps.
This keeps the fast parser and the LLM fallback from developing different
manipulation behavior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..object_types import uses_candidate_grasp


class TaskSpecError(ValueError):
    """The requested task is understood but is unsafe or incomplete."""


class Intent(str, Enum):
    PICK = "pick"
    PICK_PLACE = "pick_place"
    PLACE_HELD = "place_held"
    SCAN = "scan"
    GO_HOME = "go_home"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"
    WAVE = "wave"
    NOD = "nod"
    HANDSHAKE = "handshake"


class Relation(str, Enum):
    ON_TOP_OF = "on_top_of"
    RIGHT_OF = "right_of"
    LEFT_OF = "left_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"


@dataclass(frozen=True)
class ObjectRef:
    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise TaskSpecError("Object reference cannot be empty")


@dataclass(frozen=True)
class OriginalPose:
    kind: str = "original"


@dataclass(frozen=True)
class ConfiguredPlace:
    kind: str = "configured"


@dataclass(frozen=True)
class NamedZone:
    name: str
    kind: str = "named_zone"


@dataclass(frozen=True)
class AbsolutePose:
    x: float
    y: float
    z: float
    kind: str = "absolute"


@dataclass(frozen=True)
class RelativeDestination:
    reference: ObjectRef
    relation: Relation
    kind: str = "relative"


Destination = (
    OriginalPose
    | ConfiguredPlace
    | NamedZone
    | AbsolutePose
    | RelativeDestination
)


@dataclass(frozen=True)
class TaskSpec:
    intent: Intent
    source: ObjectRef | None = None
    destination: Destination | None = None
    times: int | None = None


_ZONE_ALIASES = {
    "right": "right", "右": "right", "右边": "right", "右侧": "right", "右方": "right",
    "left": "left", "左": "left", "左边": "left", "左侧": "left", "左方": "left",
    "center": "center", "middle": "center", "中间": "center", "中央": "center",
    "front": "front", "前": "front", "前面": "front", "前方": "front",
    "back": "back", "后": "back", "后面": "back", "后方": "back",
}

_ZONE_CANONICAL_ZH = {
    "right": "右边",
    "left": "左边",
    "center": "中间",
    "front": "前面",
    "back": "后面",
}

_RELATION_SUFFIXES = (
    (Relation.ON_TOP_OF, ("上方", "上面", "顶部", "上")),
    (Relation.RIGHT_OF, ("右边", "右侧", "右方")),
    (Relation.LEFT_OF, ("左边", "左侧", "左方")),
    (Relation.IN_FRONT_OF, ("前面", "前方")),
    (Relation.BEHIND, ("后面", "后方")),
)

_PICK = r"(?:抓取|拿起|pick\s*(?:up)?)"
_PLACE = r"(?:放到|放在|放回|放置到|移动到|place\s*(?:to|at)?)"


def _clean_text(value: str) -> str:
    return value.strip().strip("，,。.!！?？ ")


def parse_destination(text: str) -> Destination:
    """Parse a destination without silently converting unknown text."""
    raw = _clean_text(text)
    lowered = raw.lower()
    if any(token in lowered for token in ("原位", "原位置", "原处", "original position")):
        return OriginalPose()

    # Object-relative Chinese expressions must be checked before named zones:
    # "红色物块右边" is not the global right-hand drop zone.
    for relation, suffixes in _RELATION_SUFFIXES:
        for suffix in suffixes:
            if lowered.endswith(suffix):
                reference = _clean_text(raw[: -len(suffix)])
                reference = re.sub(r"(?:的)$", "", reference).strip()
                if reference and reference not in ("桌面", "工作区", "区域"):
                    return RelativeDestination(ObjectRef(reference), relation)

    english_patterns = (
        (Relation.ON_TOP_OF, r"^(?:on\s+top\s+of|above)\s+(.+)$"),
        (Relation.RIGHT_OF, r"^(?:to\s+the\s+)?right\s+of\s+(.+)$"),
        (Relation.LEFT_OF, r"^(?:to\s+the\s+)?left\s+of\s+(.+)$"),
        (Relation.IN_FRONT_OF, r"^in\s+front\s+of\s+(.+)$"),
        (Relation.BEHIND, r"^behind\s+(.+)$"),
    )
    for relation, pattern in english_patterns:
        match = re.match(pattern, lowered)
        if match:
            return RelativeDestination(ObjectRef(_clean_text(match.group(1))), relation)

    normalized_zone = lowered
    for prefix in ("桌面", "工作区", "区域"):
        if normalized_zone.startswith(prefix):
            normalized_zone = normalized_zone[len(prefix):].strip("的 ")
            break
    zone = _ZONE_ALIASES.get(normalized_zone)
    if zone:
        return NamedZone(zone)

    raise TaskSpecError(f"Unknown placement destination: {raw}")


def parse_fast_task(text: str) -> TaskSpec | None:
    """Parse deterministic common commands into a typed task description."""
    raw = _clean_text(text)
    lowered = raw.lower()

    match = re.search(
        rf"{_PICK}\s*(.+?)\s*(?:并|然后|再)?\s*{_PLACE}\s*(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        source = ObjectRef(_clean_text(match.group(1)))
        destination = parse_destination(match.group(2))
        return TaskSpec(Intent.PICK_PLACE, source=source, destination=destination)

    match = re.search(
        rf"(?:把|将)\s*(.+?)\s*{_PLACE}\s*(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        source = ObjectRef(_clean_text(match.group(1)))
        destination = parse_destination(match.group(2))
        return TaskSpec(Intent.PICK_PLACE, source=source, destination=destination)

    match = re.fullmatch(rf"{_PICK}\s*(.+)", raw, flags=re.IGNORECASE)
    if match:
        return TaskSpec(Intent.PICK, source=ObjectRef(_clean_text(match.group(1))))

    if re.fullmatch(r"(?:放回(?:原位|原位置|原处)|放到原位|放到原位置)", raw):
        return TaskSpec(Intent.PLACE_HELD, destination=OriginalPose())
    if lowered in ("放下", "放下来", "put down", "drop"):
        return TaskSpec(Intent.PLACE_HELD, destination=ConfiguredPlace())
    match = re.fullmatch(rf"{_PLACE}\s*(.+)", raw, flags=re.IGNORECASE)
    if match:
        return TaskSpec(Intent.PLACE_HELD, destination=parse_destination(match.group(1)))

    if any(word in lowered for word in ("扫描桌面", "scan scene", "scan table")):
        return TaskSpec(Intent.SCAN)
    if any(word in lowered for word in ("回home", "归位", "回家", "回到观察位", "go home")) or lowered == "home":
        return TaskSpec(Intent.GO_HOME)
    if any(word in lowered for word in ("打开夹爪", "松手", "释放", "open gripper", "release")):
        return TaskSpec(Intent.OPEN_GRIPPER)
    if any(word in lowered for word in ("闭合夹爪", "夹紧", "close gripper")):
        return TaskSpec(Intent.CLOSE_GRIPPER)
    if any(word in lowered for word in ("挥手", "wave", "摆动", "打招呼", "摇手")):
        return TaskSpec(Intent.WAVE)
    if any(word in lowered for word in ("点头", "nod")):
        return TaskSpec(Intent.NOD)
    if any(word in lowered for word in ("握手", "handshake")):
        return TaskSpec(Intent.HANDSHAKE)
    return None


def task_spec_from_dict(data: dict[str, Any]) -> TaskSpec:
    """Decode and validate an LLM-produced task specification."""
    try:
        intent = Intent(str(data["intent"]))
    except (KeyError, ValueError) as exc:
        raise TaskSpecError("Task spec has an invalid or missing intent") from exc

    source_data = data.get("source")
    source = None
    if source_data is not None:
        if isinstance(source_data, str):
            source = ObjectRef(source_data)
        elif isinstance(source_data, dict):
            source = ObjectRef(str(source_data.get("name", "")))
        else:
            raise TaskSpecError("Task source must be a string or object reference")

    destination_data = data.get("destination")
    destination: Destination | None = None
    if destination_data is not None:
        if not isinstance(destination_data, dict):
            raise TaskSpecError("Task destination must be an object")
        kind = destination_data.get("kind")
        if kind == "original":
            destination = OriginalPose()
        elif kind == "configured":
            destination = ConfiguredPlace()
        elif kind == "named_zone":
            zone = str(destination_data.get("name", ""))
            if zone not in _ZONE_CANONICAL_ZH:
                raise TaskSpecError(f"Unknown named zone: {zone}")
            destination = NamedZone(zone)
        elif kind == "absolute":
            try:
                destination = AbsolutePose(
                    float(destination_data["x"]),
                    float(destination_data["y"]),
                    float(destination_data["z"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise TaskSpecError("Absolute destination requires numeric x/y/z") from exc
        elif kind == "relative":
            reference = destination_data.get("reference")
            reference_name = reference.get("name", "") if isinstance(reference, dict) else reference
            try:
                relation = Relation(str(destination_data["relation"]))
            except (KeyError, ValueError) as exc:
                raise TaskSpecError("Relative destination has an invalid relation") from exc
            destination = RelativeDestination(ObjectRef(str(reference_name or "")), relation)
        else:
            raise TaskSpecError(f"Unknown destination kind: {kind}")

    spec = TaskSpec(intent, source=source, destination=destination, times=data.get("times"))
    _validate_task_spec(spec)
    return spec


def _validate_task_spec(spec: TaskSpec) -> None:
    if spec.intent in (Intent.PICK, Intent.PICK_PLACE) and spec.source is None:
        raise TaskSpecError(f"{spec.intent.value} requires a source object")
    if spec.intent in (Intent.PICK_PLACE, Intent.PLACE_HELD) and spec.destination is None:
        raise TaskSpecError(f"{spec.intent.value} requires a destination")
    if spec.intent == Intent.PICK and spec.destination is not None:
        raise TaskSpecError("pick must not contain a placement destination")
    if isinstance(spec.destination, RelativeDestination) and spec.source is not None:
        if spec.destination.reference.name == spec.source.name:
            raise TaskSpecError("Source and reference object must be different")
    if spec.times is not None:
        try:
            times = int(spec.times)
        except (TypeError, ValueError) as exc:
            raise TaskSpecError("Gesture repetition count must be an integer") from exc
        if times <= 0:
            raise TaskSpecError("Gesture repetition count must be positive")


def _grasp_step(target: str, locate_name: str = "locate") -> dict[str, Any]:
    return {
        "name": "grasp",
        "skill": "grasp_object",
        "args": {
            "x": f"${locate_name}.x",
            "y": f"${locate_name}.y",
            "z": f"${locate_name}.z",
            "geometry": f"${locate_name}.geometry",
            "target": target,
        },
    }


def _compile_destination(
    destination: Destination,
    *,
    bridge: Any,
    source_target: str | None,
    place_held: bool,
) -> list[dict[str, Any]]:
    if isinstance(destination, OriginalPose):
        if place_held:
            context = bridge.get_task_context()
            x = context.get("pick_x")
            y = context.get("pick_y")
            z = context.get("pick_z")
            if x is None or y is None or z is None:
                raise TaskSpecError("Cannot return to original pose: saved pick pose is missing")
            args: dict[str, Any] = {"x": x, "y": y, "z": z}
            selected = context.get("selected_candidate")
            if selected:
                args["reverse_candidate"] = selected
        else:
            args = {"x": "$grasp.pick_x", "y": "$grasp.pick_y", "z": "$grasp.pick_z"}
            if uses_candidate_grasp(source_target):
                args["reverse_candidate"] = "$grasp.selected_candidate"
        return [{"name": "place", "skill": "place_object", "args": args}]

    if isinstance(destination, ConfiguredPlace):
        pose = bridge.get_place_pose()
        return [{
            "name": "place",
            "skill": "place_object",
            "args": {"x": pose["x"], "y": pose["y"], "z": pose["z"]},
        }]

    if isinstance(destination, AbsolutePose):
        return [{
            "name": "place",
            "skill": "place_object",
            "args": {"x": destination.x, "y": destination.y, "z": destination.z},
        }]

    if isinstance(destination, NamedZone):
        place_name = _ZONE_CANONICAL_ZH[destination.name]
        return [
            {"name": "resolve", "skill": "resolve_place", "args": {"place": place_name}},
            {
                "name": "place",
                "skill": "place_object",
                "args": {"x": "$resolve.x", "y": "$resolve.y", "z": "$resolve.z"},
            },
        ]

    if isinstance(destination, RelativeDestination):
        if not source_target:
            raise TaskSpecError(
                "Relative placement verification requires a known source object"
            )
        if destination.relation == Relation.ON_TOP_OF:
            return [
                {
                    "name": "stack",
                    "skill": "stack_on",
                    "args": {
                        "source_geometry": "$grasp.geometry",
                        "support_geometry": "$locate_reference.geometry",
                        "selected_candidate": "$grasp.selected_candidate",
                    },
                },
                {
                    "name": "place",
                    "skill": "place_object",
                    "args": {
                        "x": "$stack.x",
                        "y": "$stack.y",
                        "z": "$stack.z",
                        "placement_candidate": "$stack.placement_candidate",
                    },
                },
                {"name": "verify_home", "skill": "go_home", "args": {}},
                {
                    "name": "verify_locate",
                    "skill": "locate_object",
                    "args": {"target": source_target},
                },
                {
                    "name": "verify_place",
                    "skill": "verify_placement",
                    "args": {
                        "observed_geometry": "$verify_locate.geometry",
                        "expected_x": "$stack.x",
                        "expected_y": "$stack.y",
                        "expected_surface_z": "$stack.expected_surface_z",
                    },
                },
            ]
        return [
            {
                "name": "relative_place",
                "skill": "offset_from",
                "args": {
                    "source_geometry": "$grasp.geometry",
                    "reference_geometry": "$locate_reference.geometry",
                    "selected_candidate": "$grasp.selected_candidate",
                    "relation": destination.relation.value,
                },
            },
            {
                "name": "place",
                "skill": "place_object",
                "args": {
                    "x": "$relative_place.x",
                    "y": "$relative_place.y",
                    "z": "$relative_place.z",
                    "placement_candidate": "$relative_place.placement_candidate",
                },
            },
            {"name": "verify_home", "skill": "go_home", "args": {}},
            {
                "name": "verify_locate",
                "skill": "locate_object",
                "args": {"target": source_target},
            },
            {
                "name": "verify_place",
                "skill": "verify_placement",
                "args": {
                    "observed_geometry": "$verify_locate.geometry",
                    "expected_x": "$relative_place.x",
                    "expected_y": "$relative_place.y",
                    "expected_surface_z": "$relative_place.expected_surface_z",
                },
            },
        ]
    raise TaskSpecError(f"Unsupported destination type: {type(destination).__name__}")


def compile_task_spec(spec: TaskSpec, bridge: Any) -> list[dict[str, Any]]:
    """Compile validated task semantics into the existing skill pipeline."""
    _validate_task_spec(spec)
    if spec.intent in (Intent.PICK, Intent.PICK_PLACE):
        assert spec.source is not None
        target = spec.source.name
        pipeline: list[dict[str, Any]] = [
            {"name": "go_home", "skill": "go_home", "args": {}},
            {"name": "locate", "skill": "locate_object", "args": {"target": target}},
        ]
        if isinstance(spec.destination, RelativeDestination):
            pipeline.append({
                "name": "locate_reference",
                "skill": "locate_object",
                "args": {"target": spec.destination.reference.name},
            })
        pipeline.append(_grasp_step(target))
        if spec.intent == Intent.PICK_PLACE:
            assert spec.destination is not None
            pipeline.extend(_compile_destination(
                spec.destination,
                bridge=bridge,
                source_target=target,
                place_held=False,
            ))
        return pipeline

    if spec.intent == Intent.PLACE_HELD:
        holding = bridge.get_holding()
        if holding is None:
            raise TaskSpecError("Cannot place while holding state is unknown")
        if holding is False:
            raise TaskSpecError("Cannot place because the gripper is not holding an object")
        assert spec.destination is not None
        if isinstance(spec.destination, RelativeDestination):
            raise TaskSpecError("Cannot detect a reference object after an object is already held")
        return _compile_destination(
            spec.destination,
            bridge=bridge,
            source_target=None,
            place_held=True,
        )

    direct = {
        Intent.SCAN: [{"name": "go_home", "skill": "go_home", "args": {}}, {"name": "scan", "skill": "scan_scene", "args": {}}],
        Intent.GO_HOME: [{"name": "home", "skill": "go_home", "args": {}}],
        Intent.OPEN_GRIPPER: [{"name": "open", "skill": "open_gripper", "args": {}}],
        Intent.CLOSE_GRIPPER: [{"name": "close", "skill": "close_gripper", "args": {}}],
        Intent.WAVE: [{"name": "wave", "skill": "wave", "args": {}}],
        Intent.NOD: [{"name": "nod", "skill": "nod", "args": {}}],
        Intent.HANDSHAKE: [{"name": "handshake", "skill": "handshake", "args": {}}],
    }
    try:
        pipeline = direct[spec.intent]
    except KeyError as exc:
        raise TaskSpecError(f"Unsupported task intent: {spec.intent.value}") from exc
    if spec.times is not None and spec.intent in (Intent.WAVE, Intent.NOD, Intent.HANDSHAKE):
        pipeline[0]["args"] = {"times": int(spec.times)}
    return pipeline


_REQUIRED_ARGS = {
    "locate_object": {"target"},
    "grasp_object": {"x", "y", "z", "geometry", "target"},
    "place_object": {"x", "y", "z"},
    "resolve_place": {"place"},
    "stack_on": {"source_geometry", "support_geometry", "selected_candidate"},
    "offset_from": {
        "source_geometry", "reference_geometry", "selected_candidate", "relation",
    },
    "verify_placement": {
        "observed_geometry", "expected_x", "expected_y", "expected_surface_z",
    },
}

_KNOWN_OUTPUTS = {
    "locate_object": {"x", "y", "z", "geometry", "bbox", "source", "debug_image"},
    "grasp_object": {
        "holding", "pick_x", "pick_y", "pick_z", "geometry",
        "gripper_width", "selected_candidate",
    },
    "resolve_place": {"x", "y", "z"},
    "stack_on": {
        "x", "y", "z", "object_center", "placement_candidate", "clearance",
        "expected_surface_z", "support_margin", "overhang",
    },
    "offset_from": {
        "x", "y", "z", "object_center", "placement_candidate", "clearance",
        "separation", "expected_surface_z",
    },
    "verify_placement": {
        "verified", "observed_x", "observed_y", "observed_surface_z",
        "xy_error", "surface_z_error",
    },
}


def validate_pipeline(pipeline: list[dict[str, Any]], valid_skills: set[str]) -> None:
    """Statically validate ordering, references, ports, and required inputs."""
    if not pipeline:
        raise TaskSpecError("Pipeline is empty")
    seen: dict[str, str] = {}
    grasp_active = False
    for index, step in enumerate(pipeline):
        name = step.get("name")
        skill = step.get("skill")
        if not isinstance(name, str) or not name:
            raise TaskSpecError(f"Pipeline step {index} has no valid name")
        if name in seen:
            raise TaskSpecError(f"Duplicate pipeline step name: {name}")
        if skill not in valid_skills:
            raise TaskSpecError(f"Unknown skill: {skill}")
        args = step.get("args")
        if not isinstance(args, dict):
            raise TaskSpecError(f"Step {name} args must be an object")
        missing = _REQUIRED_ARGS.get(skill, set()) - set(args)
        if missing:
            raise TaskSpecError(f"Step {name} missing args: {', '.join(sorted(missing))}")
        if grasp_active and skill in (
            "locate_object", "detect_by_color", "scan_scene"
        ):
            raise TaskSpecError(
                f"Perception step {name} cannot run while an object is held"
            )
        for value in args.values():
            if not isinstance(value, str) or not value.startswith("$"):
                continue
            parts = value[1:].split(".")
            if len(parts) != 2 or not all(parts):
                raise TaskSpecError(f"Invalid pipeline reference: {value}")
            source_name, field = parts
            source_skill = seen.get(source_name)
            if source_skill is None:
                raise TaskSpecError(f"Reference {value} does not point to an earlier step")
            known = _KNOWN_OUTPUTS.get(source_skill)
            if known is not None and field not in known:
                raise TaskSpecError(f"Skill {source_skill} does not declare output field {field}")
        seen[name] = str(skill)
        if skill == "grasp_object":
            grasp_active = True
        elif skill == "place_object":
            grasp_active = False
