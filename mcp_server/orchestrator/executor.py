"""A small synchronous executor for deterministic task templates.

This is the first orchestration skeleton. It executes template pipelines in order;
LangGraph interrupt/resume and agent planning can be layered on top later.
"""

from typing import Any, TYPE_CHECKING

from .router import route_template
from .types import Step, TaskResult, TaskTemplate
from ..components import motion
from ..skills import manipulation, perception, prepare as prepare_skill, visual
from ..skills.base import SkillResult

if TYPE_CHECKING:
    from ..ros_bridge import RobotBridge
    from ..vlm_client import VlmClient


_SKILLS = {
    "locate_object": perception.locate_object,
    "scan_scene": perception.scan_scene,
    "grasp_object": manipulation.grasp_object,
    "place_object": manipulation.place_object,
    "visual_grasp": visual.visual_grasp,
    "prepare": prepare_skill.prepare,
}


class TemplateExecutor:
    def __init__(self, bridge: "RobotBridge", vlm: "VlmClient"):
        self.bridge = bridge
        self.vlm = vlm

    def execute_task(self, task: str, params: dict[str, Any] | None = None,
                     template: TaskTemplate | None = None) -> TaskResult:
        template = template or route_template(task)
        if template is None:
            return TaskResult(status="failed", error="No matching template", messages=["未匹配到任务模板"])
        params = params or {}
        missing = [p.name for p in template.required_params if p.required and p.name not in params]
        if missing:
            return TaskResult(
                status="failed",
                template=template.name,
                error=f"Missing required params: {', '.join(missing)}",
                messages=[f"缺少参数: {', '.join(missing)}"],
            )

        outputs: dict[str, Any] = {"params": params}
        messages: list[str] = []
        for step in template.pipeline:
            result = self._run_step_with_retries(template, step, outputs)
            outputs[step.name] = result.data if isinstance(result, SkillResult) else result
            if isinstance(result, SkillResult) and result.holding is not None:
                outputs[step.name]["holding"] = result.holding
            if isinstance(result, SkillResult) and not result.ok:
                return TaskResult(
                    status="failed",
                    template=template.name,
                    outputs=outputs,
                    user_output=self._visible(template, outputs),
                    messages=messages + [f"步骤 {step.name} 失败: {result.error}"],
                    error=result.error,
                )
            messages.append(f"步骤 {step.name} 完成")

        return TaskResult(
            status="completed",
            template=template.name,
            outputs=outputs,
            user_output=self._visible(template, outputs),
            messages=messages,
        )

    def _run_step_with_retries(self, template: TaskTemplate, step: Step,
                               outputs: dict[str, Any]) -> SkillResult | dict:
        policy = template.retry_policy.get(step.name)
        max_attempts = policy.max_attempts if policy else 1
        last: SkillResult | dict | None = None
        for attempt in range(max_attempts):
            if attempt > 0 and policy and policy.recover == "go_home":
                motion.go_home(self.bridge)
            last = self._run_step(step, outputs)
            if not isinstance(last, SkillResult) or last.ok or not last.retryable:
                return last
        return last

    def _run_step(self, step: Step, outputs: dict[str, Any]) -> SkillResult | dict:
        args = [self._resolve_ref(outputs, ref) for ref in step.args_from]
        if step.fn:
            return step.fn(self.bridge, *args, **step.kwargs)
        if step.skill is None:
            return SkillResult.failure("Step has no skill or function", failed_step=step.name, retryable=False)
        skill = _SKILLS[step.skill]
        if step.skill in ("locate_object", "visual_grasp"):
            return skill(self.bridge, self.vlm, *args, **step.kwargs)
        return skill(self.bridge, *args, **step.kwargs)

    def _resolve_ref(self, outputs: dict[str, Any], ref: str) -> Any:
        cur: Any = outputs
        for part in ref.split("."):
            cur = cur[part]
        return cur

    def _visible(self, template: TaskTemplate, outputs: dict[str, Any]) -> dict[str, Any]:
        result = {}
        for ref in template.user_visible:
            try:
                result[ref] = self._resolve_ref(outputs, ref)
            except Exception:
                continue
        return result
