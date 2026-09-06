from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from inferdoctor.i18n import t


GOALS = [
    "chatbot",
    "document-qa",
    "customer-service",
    "restaurant-ordering",
    "local-api",
    "not-sure",
]

PREFERENCES = ["easiest", "performance", "cpu", "gpu"]
RUNTIMES = ["ollama", "vllm", "sglang", "xinference", "not-sure"]


@dataclass(frozen=True)
class SetupPlan:
    goal: str
    preference: str
    runtime: str
    template: str
    recommended_runtime: str
    reason: str
    diagnosis_command: str
    template_command: str


def normalize_goal(value: Optional[str]) -> str:
    if not value:
        return "not-sure"
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "doc-qa": "document-qa",
        "docs": "document-qa",
        "rag": "document-qa",
        "api": "local-api",
        "customer": "customer-service",
        "restaurant": "restaurant-ordering",
        "unsure": "not-sure",
    }
    return aliases.get(normalized, normalized)


def normalize_preference(value: Optional[str]) -> str:
    if not value:
        return "easiest"
    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "easy": "easiest",
        "best-performance": "performance",
        "gpu-optimized": "gpu",
        "cpu-only": "cpu",
    }
    return aliases.get(normalized, normalized)


def normalize_runtime(value: Optional[str]) -> str:
    if not value:
        return "not-sure"
    normalized = value.strip().lower().replace("_", "-")
    aliases = {"none": "not-sure", "unknown": "not-sure"}
    return aliases.get(normalized, normalized)


def _template_for_goal(goal: str) -> str:
    if goal == "customer-service":
        return "customer-service"
    if goal == "restaurant-ordering":
        return "restaurant-ordering"
    if goal == "document-qa":
        return "local-doc-qa"
    if goal == "local-api":
        return "openai-compatible-api"
    return "ollama-chat"


def _runtime_for(preference: str, runtime: str, goal: str) -> str:
    if runtime != "not-sure":
        return runtime
    if preference in {"easiest", "cpu"}:
        return "ollama"
    if preference == "gpu" and goal == "local-api":
        return "vllm or sglang"
    if preference == "gpu":
        return "ollama first, vllm later"
    return "vllm for throughput, ollama for easiest setup"


def recommend_setup(goal: Optional[str], preference: Optional[str], runtime: Optional[str] = None) -> SetupPlan:
    normalized_goal = normalize_goal(goal)
    normalized_preference = normalize_preference(preference)
    normalized_runtime = normalize_runtime(runtime)
    if normalized_goal not in GOALS:
        normalized_goal = "not-sure"
    if normalized_preference not in PREFERENCES:
        normalized_preference = "easiest"
    if normalized_runtime not in RUNTIMES:
        normalized_runtime = "not-sure"

    template = _template_for_goal(normalized_goal)
    recommended_runtime = _runtime_for(normalized_preference, normalized_runtime, normalized_goal)
    reason = (
        "This path keeps setup simple and uses a local OpenAI-compatible endpoint."
        if normalized_preference in {"easiest", "cpu"}
        else "This path favors GPU-capable serving while keeping diagnosis separate from setup."
    )
    return SetupPlan(
        goal=normalized_goal,
        preference=normalized_preference,
        runtime=normalized_runtime,
        template=template,
        recommended_runtime=recommended_runtime,
        reason=reason,
        diagnosis_command="inferdoctor check",
        template_command="inferdoctor template create {0} --output ./{0}-demo".format(template),
    )


def _display_runtime(value: str) -> str:
    if value == "vllm":
        return "vLLM"
    if value == "sglang":
        return "SGLang"
    return value if " " in value else value.title()


def render_setup_plan(plan: SetupPlan, language: str = "auto") -> str:
    lines = [
        t("setup.title", language),
        "=" * 57,
        "Goal: {0}".format(plan.goal),
        "Preference: {0}".format(plan.preference),
        "Existing runtime: {0}".format(plan.runtime),
        "",
        t("setup.plain_english", language),
        "  Start with {0}.".format(_display_runtime(plan.recommended_runtime)),
        "  Use the {0} template.".format(plan.template),
        "  {0}".format(plan.reason),
        "",
        t("setup.step_by_step", language),
        "  1. Check this machine: {0}".format(plan.diagnosis_command),
        "  2. Inspect the template: inferdoctor template show {0}".format(plan.template),
        "  3. Create the starter: {0}".format(plan.template_command),
        "  4. Edit the generated .env or config.yaml for your local endpoint.",
        "  5. Follow the generated README to run the demo.",
        "",
        "Safety: init is advisory. It does not install runtimes, download models, run inference, or modify system settings.",
    ]
    return "\n".join(lines)
