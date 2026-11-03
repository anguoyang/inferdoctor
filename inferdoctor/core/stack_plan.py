from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from inferdoctor.core.recommendations import StackRecommendation, recommend_stack
from inferdoctor.core.templates import create_template_project
from inferdoctor.i18n import t


@dataclass(frozen=True)
class LocalAIStackPlan:
    recommendation: StackRecommendation
    required_components: List[str]
    optional_components: List[str]
    steps: List[str]
    commands: List[str]
    warnings: List[str]


@dataclass(frozen=True)
class StackBootstrapPlan:
    recommendation: StackRecommendation
    project_path: str
    commands: List[str]
    safe_actions: List[str]
    requires_approval: List[str]
    will_not_do: List[str]


@dataclass(frozen=True)
class StackBootstrapFiles:
    recommendation: StackRecommendation
    output_dir: str
    written: List[str]


def _required_components(goal: str, runtime: str) -> list[str]:
    components = ["Python 3.9+", "A local OpenAI-compatible model endpoint"]
    lowered = runtime.lower()
    if "ollama" in lowered:
        components.append("Ollama or another simple local runtime")
    if "vllm" in lowered or "sglang" in lowered:
        components.append("NVIDIA GPU with enough VRAM for the selected model")
    if goal == "document-qa":
        components.append("Local Markdown documents")
    return components


def _optional_components(goal: str, runtime: str) -> list[str]:
    components = ["Open WebUI for browser chat", "Docker if your preferred runtime uses containers"]
    if goal == "document-qa":
        components.extend(["Dify for a fuller RAG app", "Embedding service when you outgrow keyword retrieval"])
    if "vllm" not in runtime.lower():
        components.append("vLLM later for higher throughput")
    if "sglang" not in runtime.lower():
        components.append("SGLang later for OpenAI-compatible serving experiments")
    return components


def build_stack_plan(
    goal: Optional[str] = None,
    preference: Optional[str] = None,
    hardware: str = "auto",
    vram_gib: Optional[float] = None,
) -> LocalAIStackPlan:
    recommendation = recommend_stack(
        goal=goal,
        preference=preference,
        hardware=hardware,
        vram_gib=vram_gib,
    )
    template_dir = "./{0}-demo".format(recommendation.template)
    steps = [
        "Run the health check first so endpoint and GPU problems are visible early.",
        "Create the starter project for your goal.",
        "Validate the generated files before editing them.",
        "Configure LOCAL_AI_BASE_URL and LOCAL_AI_MODEL in .env or config.yaml.",
        "Run the generated app and keep InferDoctor nearby for endpoint troubleshooting.",
    ]
    commands = [
        "inferdoctor",
        "inferdoctor recommend --goal {0}".format(recommendation.goal),
        "inferdoctor template create {0} --output {1}".format(recommendation.template, template_dir),
        "inferdoctor template validate {0}".format(template_dir),
        "inferdoctor template smoke-test {0}".format(template_dir),
        "inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1",
        "inferdoctor check sglang --endpoint http://127.0.0.1:30000/v1",
    ]
    warnings = [
        "This is a heuristic setup plan, not a benchmark or hardware guarantee.",
        "InferDoctor does not install runtimes, download models, run inference, or modify system settings by default.",
    ]
    if recommendation.vram_gib is None:
        warnings.append("VRAM was not detected or provided; add --vram for a clearer model-size recommendation.")
    return LocalAIStackPlan(
        recommendation=recommendation,
        required_components=_required_components(recommendation.goal, recommendation.runtime),
        optional_components=_optional_components(recommendation.goal, recommendation.runtime),
        steps=steps,
        commands=commands,
        warnings=warnings,
    )


def render_stack_plan(plan: LocalAIStackPlan, language: str = "auto") -> str:
    rec = plan.recommendation
    vram = "unknown" if rec.vram_gib is None else "{0:g} GiB".format(rec.vram_gib)
    lines = [
        t("stack_plan.title", language),
        "=" * 57,
        "Goal: {0}".format(rec.goal),
        "Preference: {0}".format(rec.preference),
        "Hardware: {0}".format(rec.hardware),
        "VRAM: {0}".format(vram),
        "",
        "Recommended path:",
        "  Runtime: {0}".format(rec.runtime),
        "  Model size class: {0}".format(rec.model_size_class),
        "  Starter template: {0}".format(rec.template),
        "  Why: {0}".format(rec.why),
        "",
        "Use-case fit:",
    ]
    lines.extend("  - {0}".format(item) for item in rec.use_case_guidance)
    lines.extend([
        "",
        t("stack_plan.required_components", language),
    ])
    lines.extend("  - {0}".format(component) for component in plan.required_components)
    lines.extend(["", t("stack_plan.optional_components", language)])
    lines.extend("  - {0}".format(component) for component in plan.optional_components)
    lines.extend(["", t("stack_plan.step_by_step", language)])
    lines.extend("  {0}. {1}".format(index, step) for index, step in enumerate(plan.steps, start=1))
    lines.extend(["", "Commands to run:"])
    lines.extend("  {0}".format(command) for command in plan.commands)
    lines.extend(["", "Warnings and caveats:"])
    lines.extend("  - {0}".format(warning) for warning in plan.warnings)
    return "\n".join(lines)


def build_stack_bootstrap_plan(
    goal: Optional[str] = None,
    preference: Optional[str] = None,
    hardware: str = "auto",
    vram_gib: Optional[float] = None,
    output_dir: Optional[str] = None,
) -> StackBootstrapPlan:
    recommendation = recommend_stack(
        goal=goal,
        preference=preference,
        hardware=hardware,
        vram_gib=vram_gib,
    )
    project_path = output_dir or "./{0}-demo".format(recommendation.template)
    commands = [
        "inferdoctor",
        "inferdoctor recommend --goal {0}".format(recommendation.goal),
        "inferdoctor template create {0} --output {1}".format(recommendation.template, project_path),
        "inferdoctor template validate {0}".format(project_path),
        "inferdoctor template smoke-test {0}".format(project_path),
        "cd {0}".format(project_path),
        "cp .env.example .env",
        "python app.py --dry-run" if recommendation.template in {"customer-service", "restaurant-ordering"} else "python query.py --dry-run",
        "python app.py --check-config" if recommendation.template in {"customer-service", "restaurant-ordering"} else "python query.py --check-config",
    ]
    safe_actions = [
        "Show this plan without changing files.",
        "Generate a starter project only when the user runs the template create command.",
        "Validate files and run dry-run smoke commands without contacting endpoints.",
    ]
    requires_approval = [
        "Installing any local AI runtime or Python dependency.",
        "Starting Docker containers or background services.",
        "Calling a live model endpoint with user data.",
    ]
    will_not_do = [
        "Install Ollama, vLLM, SGLang, Xinference, CUDA, or GPU frameworks.",
        "Download models.",
        "Run model inference.",
        "Modify system settings or start services.",
    ]
    return StackBootstrapPlan(
        recommendation=recommendation,
        project_path=project_path,
        commands=commands,
        safe_actions=safe_actions,
        requires_approval=requires_approval,
        will_not_do=will_not_do,
    )


def render_stack_bootstrap_plan(plan: StackBootstrapPlan, language: str = "auto") -> str:
    rec = plan.recommendation
    lines = [
        "InferDoctor Stack Bootstrap Plan (Dry Run)",
        "=" * 57,
        "Goal: {0}".format(rec.goal),
        "Preference: {0}".format(rec.preference),
        "Recommended runtime: {0}".format(rec.runtime),
        "Recommended template: {0}".format(rec.template),
        "Generated project path: {0}".format(plan.project_path),
        "Model size class: {0}".format(rec.model_size_class),
        "",
        "Commands that would be run by a careful beginner:",
    ]
    lines.extend("  {0}. {1}".format(index, command) for index, command in enumerate(plan.commands, start=1))
    lines.extend(["", "What is safe in this dry run:"])
    lines.extend("  - {0}".format(item) for item in plan.safe_actions)
    lines.extend(["", "Requires explicit user approval:"])
    lines.extend("  - {0}".format(item) for item in plan.requires_approval)
    lines.extend(["", "InferDoctor will not do automatically:"])
    lines.extend("  - {0}".format(item) for item in plan.will_not_do)
    lines.extend([
        "",
        "Next command to actually create files:",
        "  inferdoctor template create {0} --output {1}".format(rec.template, plan.project_path),
    ])
    return "\n".join(lines)



def _bootstrap_readme(plan: StackBootstrapPlan) -> str:
    rec = plan.recommendation
    return """# InferDoctor Bootstrap Project: {goal}

This directory was generated by `inferdoctor stack bootstrap`.
It contains a lightweight starter template plus a safe setup plan.

## What Was Generated

- Starter template: `{template}`
- Recommended runtime: {runtime}
- Model size class: {model_size}
- `bootstrap_plan.md`: commands and safety boundaries
- `next_steps.md`: beginner next actions
- `config_summary.yaml`: non-secret setup summary

## Safety

InferDoctor generated files only. It did not install dependencies, start Docker,
download models, call model endpoints, or modify system settings.

## Validate

```bash
inferdoctor template validate .
inferdoctor template smoke-test .
```

## Next

Read `next_steps.md`, copy `.env.example` to `.env`, and point the endpoint at a
local OpenAI-compatible server you already run.
""".format(
        goal=rec.goal,
        template=rec.template,
        runtime=rec.runtime,
        model_size=rec.model_size_class,
    )


def _next_steps(plan: StackBootstrapPlan) -> str:
    rec = plan.recommendation
    return """# Next Steps

1. Run `inferdoctor` and fix the top stack issues first.
2. Copy `.env.example` to `.env`.
3. Set `LOCAL_AI_BASE_URL` and `LOCAL_AI_MODEL` for your local endpoint.
4. Run `inferdoctor template validate .`.
5. Run `inferdoctor template smoke-test .`.
6. Try the generated dry-run or config-check command before using a live endpoint.

Recommended runtime: {runtime}
Recommended template: {template}

InferDoctor will not install runtimes, download models, start services, or run
inference for you.
""".format(runtime=rec.runtime, template=rec.template)


def _config_summary(plan: StackBootstrapPlan) -> str:
    rec = plan.recommendation
    return """goal: {goal}
preference: {preference}
recommended_runtime: {runtime}
recommended_template: {template}
model_size_class: {model_size}
safety: generated_files_only
""".format(
        goal=rec.goal,
        preference=rec.preference,
        runtime=rec.runtime,
        template=rec.template,
        model_size=rec.model_size_class,
    )


def create_stack_bootstrap_project(
    goal: Optional[str] = None,
    preference: Optional[str] = None,
    hardware: str = "auto",
    vram_gib: Optional[float] = None,
    output_dir: str = "./inferdoctor-bootstrap-demo",
) -> StackBootstrapFiles:
    plan = build_stack_bootstrap_plan(
        goal=goal,
        preference=preference,
        hardware=hardware,
        vram_gib=vram_gib,
        output_dir=output_dir,
    )
    root = Path(output_dir).expanduser()
    written = create_template_project(plan.recommendation.template, str(root))
    extra_files = {
        "bootstrap_plan.md": render_stack_bootstrap_plan(plan),
        "next_steps.md": _next_steps(plan),
        "config_summary.yaml": _config_summary(plan),
    }
    readme = root / "README.md"
    if readme.exists():
        readme.write_text(readme.read_text(encoding="utf-8") + chr(10) + chr(10) + "---" + chr(10) + chr(10) + _bootstrap_readme(plan), encoding="utf-8")
    else:
        extra_files["README.md"] = _bootstrap_readme(plan)
    for relative, content in extra_files.items():
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written.append(str(destination))
    return StackBootstrapFiles(recommendation=plan.recommendation, output_dir=output_dir, written=written)


def render_stack_bootstrap_files(result: StackBootstrapFiles, language: str = "auto") -> str:
    lines = [
        "InferDoctor Stack Bootstrap Files Created",
        "=" * 57,
        "Goal: {0}".format(result.recommendation.goal),
        "Template: {0}".format(result.recommendation.template),
        "Output directory: {0}".format(result.output_dir),
        "",
        "Generated files:",
    ]
    lines.extend("  - {0}".format(path) for path in result.written)
    lines.extend([
        "",
        "Safety:",
        "  Files were generated only. No dependencies were installed, no services were started, and no endpoints were called.",
        "",
        "Next commands:",
        "  cd {0}".format(result.output_dir),
        "  inferdoctor template validate .",
        "  inferdoctor template smoke-test .",
        "  cp .env.example .env",
    ])
    return chr(10).join(lines)
