import pytest

from mcp_server.orchestrator.task_spec import (
    Intent,
    Relation,
    RelativeDestination,
    TaskSpecError,
    compile_task_spec,
    parse_fast_task,
    task_spec_from_dict,
    validate_pipeline,
)
from mcp_server.orchestrator import planner
from mcp_server.orchestrator.graph import GraphExecutor
from mcp_server.skills import placement


class Bridge:
    def __init__(self, holding=False, context=None):
        self._holding = holding
        self._context = context or {}

    def get_holding(self):
        return self._holding

    def get_task_context(self):
        return dict(self._context)

    def get_place_pose(self):
        return {"x": -0.4, "y": -0.25, "z": 0.2}


VALID_SKILLS = {
    "go_home",
    "locate_object",
    "scan_scene",
    "grasp_object",
    "place_object",
    "resolve_place",
    "stack_on",
    "offset_from",
    "verify_placement",
    "open_gripper",
    "close_gripper",
    "wave",
    "nod",
    "handshake",
}


def test_water_bottle_pick_fast_route_preserves_target_for_shared_pipeline():
    spec = parse_fast_task("抓取水瓶")
    assert spec.intent == Intent.PICK
    assert spec.source.name == "水瓶"

    pipeline = compile_task_spec(spec, Bridge())
    assert [step["skill"] for step in pipeline] == [
        "go_home", "locate_object", "grasp_object",
    ]
    assert pipeline[1]["args"]["target"] == "水瓶"
    assert pipeline[2]["args"]["target"] == "水瓶"
    assert pipeline[2]["args"]["geometry"] == "$locate.geometry"


def test_stack_route_locates_both_objects_before_grasp():
    spec = parse_fast_task("抓取蓝色物块并放到红色物块上方")
    assert spec.intent == Intent.PICK_PLACE
    assert isinstance(spec.destination, RelativeDestination)
    assert spec.destination.relation == Relation.ON_TOP_OF

    pipeline = compile_task_spec(spec, Bridge())
    assert [step["skill"] for step in pipeline] == [
        "go_home",
        "locate_object",
        "locate_object",
        "grasp_object",
        "stack_on",
        "place_object",
        "go_home",
        "locate_object",
        "verify_placement",
    ]
    assert pipeline[2]["args"]["target"] == "红色物块"
    assert pipeline[4]["args"] == {
        "source_geometry": "$grasp.geometry",
        "support_geometry": "$locate_reference.geometry",
        "selected_candidate": "$grasp.selected_candidate",
    }
    assert pipeline[5]["args"]["placement_candidate"] == (
        "$stack.placement_candidate"
    )
    assert pipeline[7]["args"]["target"] == "蓝色物块"
    assert pipeline[8]["args"] == {
        "observed_geometry": "$verify_locate.geometry",
        "expected_x": "$stack.x",
        "expected_y": "$stack.y",
        "expected_surface_z": "$stack.expected_surface_z",
    }
    validate_pipeline(pipeline, VALID_SKILLS)


@pytest.mark.parametrize("command", [
    "把蓝色物块放到红色物块上",
    "把蓝色物块放到红色物块的上方",
    "抓取蓝色物块并放在红色物块上面",
])
def test_common_stack_phrasings_keep_the_reference_object(command):
    spec = parse_fast_task(command)
    assert isinstance(spec.destination, RelativeDestination)
    assert spec.destination.reference.name == "红色物块"
    assert spec.destination.relation == Relation.ON_TOP_OF


def test_object_relative_right_is_not_global_right_zone():
    spec = parse_fast_task("把蓝色物块放到红色物块右边")
    pipeline = compile_task_spec(spec, Bridge())

    assert pipeline[2]["args"]["target"] == "红色物块"
    relative = next(step for step in pipeline if step["skill"] == "offset_from")
    assert relative["args"]["relation"] == "right_of"
    assert all(step["skill"] != "resolve_place" for step in pipeline)


def test_global_right_zone_still_uses_named_zone_resolution():
    spec = parse_fast_task("把蓝色物块放到桌面右边")
    pipeline = compile_task_spec(spec, Bridge())

    assert [step["skill"] for step in pipeline].count("locate_object") == 1
    resolve = next(step for step in pipeline if step["skill"] == "resolve_place")
    assert resolve["args"]["place"] == "右边"


def test_unknown_destination_fails_instead_of_using_default_pose():
    with pytest.raises(TaskSpecError, match="Unknown placement destination"):
        parse_fast_task("把蓝色物块放到窗台")

    result = placement.resolve_place(Bridge(), "窗台")
    assert result.ok is False
    assert "Unknown named placement zone" in result.error


def test_return_held_object_requires_saved_pick_pose():
    spec = parse_fast_task("放回原位")
    with pytest.raises(TaskSpecError, match="saved pick pose is missing"):
        compile_task_spec(spec, Bridge(holding=True))


def test_place_held_distinguishes_false_and_unknown_holding_state():
    spec = parse_fast_task("放下")
    with pytest.raises(TaskSpecError, match="not holding"):
        compile_task_spec(spec, Bridge(holding=False))
    with pytest.raises(TaskSpecError, match="holding state is unknown"):
        compile_task_spec(spec, Bridge(holding=None))


def test_llm_task_spec_decoder_rejects_unknown_destination_kind():
    with pytest.raises(TaskSpecError, match="Unknown destination kind"):
        task_spec_from_dict({
            "intent": "pick_place",
            "source": {"name": "蓝色物块"},
            "destination": {"kind": "somewhere"},
        })


def test_pipeline_validator_rejects_forward_reference():
    pipeline = [
        {
            "name": "place",
            "skill": "place_object",
            "args": {"x": "$resolve.x", "y": "$resolve.y", "z": "$resolve.z"},
        },
        {"name": "resolve", "skill": "resolve_place", "args": {"place": "右边"}},
    ]
    with pytest.raises(TaskSpecError, match="earlier step"):
        validate_pipeline(pipeline, VALID_SKILLS)


def test_pipeline_validator_rejects_duplicate_names_and_post_grasp_perception():
    duplicate = [
        {"name": "same", "skill": "go_home", "args": {}},
        {"name": "same", "skill": "go_home", "args": {}},
    ]
    with pytest.raises(TaskSpecError, match="Duplicate"):
        validate_pipeline(duplicate, VALID_SKILLS)

    post_grasp = [
        {
            "name": "grasp",
            "skill": "grasp_object",
            "args": {"x": 0.0, "y": 0.0, "z": 0.0, "geometry": {}, "target": "物体"},
        },
        {"name": "late_locate", "skill": "locate_object", "args": {"target": "参照物"}},
    ]
    with pytest.raises(TaskSpecError, match="while an object is held"):
        validate_pipeline(post_grasp, VALID_SKILLS)

    post_release = [
        *post_grasp[:1],
        {
            "name": "place",
            "skill": "place_object",
            "args": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        {
            "name": "verify_locate",
            "skill": "locate_object",
            "args": {"target": "物体"},
        },
    ]
    validate_pipeline(post_release, VALID_SKILLS)


def test_pipeline_validator_checks_declared_output_fields():
    pipeline = [
        {"name": "locate", "skill": "locate_object", "args": {"target": "物体"}},
        {
            "name": "place",
            "skill": "place_object",
            "args": {"x": "$locate.no_such_field", "y": "$locate.y", "z": "$locate.z"},
        },
    ]
    with pytest.raises(TaskSpecError, match="does not declare output"):
        validate_pipeline(pipeline, VALID_SKILLS)


def test_llm_fallback_returns_the_same_typed_task_spec(monkeypatch):
    class Response:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {
                "choices": [{
                    "message": {
                        "content": (
                            '{"intent":"pick_place",'
                            '"source":{"name":"蓝色物块"},'
                            '"destination":{"kind":"relative",'
                            '"relation":"on_top_of",'
                            '"reference":{"name":"红色物块"}}}'
                        )
                    }
                }]
            }

    monkeypatch.setattr(planner, "PLANNING_LLM_CONFIG", {
        "api_key": "test-key",
        "api_url": "https://invalid.test",
        "model": "test-model",
        "timeout": 1,
        "retries": 0,
    })
    monkeypatch.setattr(planner.requests, "post", lambda *args, **kwargs: Response())

    spec = planner.plan_task_spec("请帮我叠放两个物块")
    assert isinstance(spec.destination, RelativeDestination)
    assert spec.destination.reference.name == "红色物块"
    pipeline = compile_task_spec(spec, Bridge())
    assert pipeline[2]["args"]["target"] == "红色物块"


def test_executor_reports_semantic_error_before_building_or_running_graph():
    result = GraphExecutor(Bridge(), vlm=None).execute_task(
        "把蓝色物块放到窗台"
    )
    assert result.status == "failed"
    assert "Unknown placement destination" in result.error
