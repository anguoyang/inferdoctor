from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from inferdoctor.core.setup import normalize_goal, normalize_preference, normalize_runtime

QUICKSTART_GOALS = ["chatbot", "customer-service", "restaurant-ordering", "document-qa", "rag", "local-api", "not-sure"]
QUICKSTART_PREFERENCES = ["easiest", "performance"]
QUICKSTART_LOCATIONS = ["local", "lan", "endpoint"]
QUICKSTART_HARDWARE = ["auto", "cpu", "gpu"]
QUICKSTART_RUNTIMES = ["ollama", "vllm", "sglang", "xinference", "openai-compatible", "not-sure"]


@dataclass(frozen=True)
class QuickstartPlan:
    goal: str
    preference: str
    location: str
    hardware: str
    runtime: str
    endpoint: Optional[str]
    recommended_runtime: str
    template: str
    endpoint_guidance: List[str]
    validation_commands: List[str]
    performance_commands: List[str]
    safety_notes: List[str]


def normalize_quickstart_goal(value: Optional[str]) -> str:
    goal = normalize_goal(value)
    if goal == "document-qa" and value and value.strip().lower() == "rag":
        return "rag"
    if goal not in QUICKSTART_GOALS:
        return "not-sure"
    return goal


def _template_for_goal(goal: str) -> str:
    if goal == "customer-service":
        return "customer-service"
    if goal == "restaurant-ordering":
        return "restaurant-ordering"
    if goal in {"document-qa", "rag"}:
        return "local-doc-qa"
    if goal == "local-api":
        return "openai-compatible-api"
    return "customer-service" if goal == "not-sure" else "ollama-chat"


def _runtime_for(goal: str, preference: str, hardware: str, runtime: str) -> str:
    if runtime != "not-sure":
        return runtime
    if preference == "easiest" or hardware == "cpu":
        return "ollama or another OpenAI-compatible local endpoint"
    if goal == "local-api" and hardware == "gpu":
        return "vllm or sglang"
    if hardware == "gpu":
        return "ollama first, vllm or sglang when serving needs grow"
    return "ollama for easiest setup, vllm or sglang for performance serving"


def build_quickstart_plan(
    goal: Optional[str] = None,
    preference: Optional[str] = None,
    endpoint: Optional[str] = None,
    location: Optional[str] = None,
    hardware: Optional[str] = None,
    runtime: Optional[str] = None,
) -> QuickstartPlan:
    normalized_goal = normalize_quickstart_goal(goal)
    normalized_preference = normalize_preference(preference)
    if normalized_preference not in QUICKSTART_PREFERENCES:
        normalized_preference = "easiest"
    normalized_location = (location or ("endpoint" if endpoint else "local")).strip().lower().replace("_", "-")
    if normalized_location not in QUICKSTART_LOCATIONS:
        normalized_location = "local"
    normalized_hardware = (hardware or "auto").strip().lower().replace("_", "-")
    if normalized_hardware not in QUICKSTART_HARDWARE:
        normalized_hardware = "auto"
    normalized_runtime = normalize_runtime(runtime)
    if normalized_runtime not in QUICKSTART_RUNTIMES:
        normalized_runtime = "not-sure"

    template = _template_for_goal(normalized_goal)
    recommended_runtime = _runtime_for(normalized_goal, normalized_preference, normalized_hardware, normalized_runtime)
    endpoint_hint = endpoint or "http://127.0.0.1:8000/v1"
    endpoint_guidance = [
        "Use an OpenAI-compatible endpoint when possible; templates read it from .env or config.yaml.",
        "For Ollama OpenAI-compatible mode, use the endpoint your Ollama version exposes and validate it first.",
        "For vLLM or SGLang, the base URL usually ends with /v1.",
    ]
    if endpoint:
        endpoint_guidance.insert(0, "Configured endpoint hint: {0}".format(endpoint))
    elif normalized_location == "lan":
        endpoint_guidance.insert(0, "Use a private LAN endpoint only if you control it; pass the URL explicitly when measuring performance.")

    project_dir = "./{0}-demo".format(template)
    validation_commands = [
        "inferdoctor",
        "inferdoctor stack plan --goal {0}".format("document-qa" if normalized_goal == "rag" else normalized_goal),
        "inferdoctor template create {0} --output {1}".format(template, project_dir),
        "inferdoctor template validate {0}".format(project_dir),
        "inferdoctor template smoke-test {0}".format(project_dir),
    ]
    performance_commands = [
        "inferdoctor perf streaming --endpoint {0} --model local-model --format json --output before.json".format(endpoint_hint),
        "inferdoctor perf baseline create --report before.json --name before",
        "inferdoctor optimize plan --report before.json --goal {0}".format("document-qa" if normalized_goal == "rag" else normalized_goal),
        "inferdoctor perf compare before.json after.json",
    ]
    safety_notes = [
        "quickstart is advisory; it does not install runtimes, download models, start services, or send private documents.",
        "Performance commands are tiny smoke tests, not benchmarks.",
        "Use harmless prompts and avoid sending sensitive data to endpoints you do not control.",
    ]
    return QuickstartPlan(
        goal=normalized_goal,
        preference=normalized_preference,
        location=normalized_location,
        hardware=normalized_hardware,
        runtime=normalized_runtime,
        endpoint=endpoint,
        recommended_runtime=recommended_runtime,
        template=template,
        endpoint_guidance=endpoint_guidance,
        validation_commands=validation_commands,
        performance_commands=performance_commands,
        safety_notes=safety_notes,
    )


def render_quickstart_plan(plan: QuickstartPlan) -> str:
    lines = [
        "InferDoctor Quickstart",
        "=" * 57,
        "Goal: {0}".format(plan.goal),
        "Preference: {0}".format(plan.preference),
        "Endpoint location: {0}".format(plan.location),
        "Hardware: {0}".format(plan.hardware),
        "Existing runtime: {0}".format(plan.runtime),
        "",
        "Recommended path:",
        "- Runtime: {0}".format(plan.recommended_runtime),
        "- Template: {0}".format(plan.template),
        "",
        "Endpoint configuration guidance:",
    ]
    lines.extend("- {0}".format(item) for item in plan.endpoint_guidance)
    lines.extend(["", "Build and validate:"])
    lines.extend("{0}. {1}".format(index, item) for index, item in enumerate(plan.validation_commands, 1))
    lines.extend(["", "Measure and optimize after the app starts:"])
    lines.extend("{0}. {1}".format(index, item) for index, item in enumerate(plan.performance_commands, 1))
    lines.extend(["", "Safety notes:"])
    lines.extend("- {0}".format(item) for item in plan.safety_notes)
    return "\n".join(lines)
