from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from inferdoctor.core.models import CheckResult, Status
from inferdoctor.i18n import t


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    title: str
    status: Status
    reason: str
    next_step: str


SCENARIO_ORDER = [
    "local-chatbot",
    "rag-app",
    "openai-compatible-server",
    "dify-local-rag",
    "gpu-inference",
    "cpu-only-fallback",
]

OPENAI_ENDPOINTS = ("vllm", "sglang", "llamacpp", "lmstudio")


SCENARIO_TITLES = {
    "local-chatbot": "Local chatbot",
    "rag-app": "RAG app",
    "openai-compatible-server": "OpenAI-compatible server",
    "dify-local-rag": "Dify local RAG",
    "gpu-inference": "GPU inference",
    "cpu-only-fallback": "CPU-only fallback",
}


def scenario_names() -> List[str]:
    return list(SCENARIO_ORDER)


def _by_name(results: Iterable[CheckResult]) -> Dict[str, CheckResult]:
    return {result.name: result for result in results}


def _is_pass(results: Dict[str, CheckResult], name: str) -> bool:
    result = results.get(name)
    return result is not None and result.status == Status.PASS


def _is_warn(results: Dict[str, CheckResult], name: str) -> bool:
    result = results.get(name)
    return result is not None and result.status == Status.WARN


def _any_pass(results: Dict[str, CheckResult], names) -> bool:
    return any(_is_pass(results, name) for name in names)


def _any_warn(results: Dict[str, CheckResult], names) -> bool:
    return any(_is_warn(results, name) for name in names)


def evaluate_scenarios(
    results: Iterable[CheckResult], target: Optional[str] = None
) -> List[ScenarioResult]:
    result_map = _by_name(results)
    scenarios = [
        _local_chatbot(result_map),
        _rag_app(result_map),
        _openai_compatible_server(result_map),
        _dify_local_rag(result_map),
        _gpu_inference(result_map),
        _cpu_only_fallback(result_map),
    ]
    if target is not None:
        scenarios = [scenario for scenario in scenarios if scenario.name == target]
    return scenarios


def _local_chatbot(results: Dict[str, CheckResult]) -> ScenarioResult:
    if _is_pass(results, "ollama"):
        return ScenarioResult(
            "local-chatbot",
            SCENARIO_TITLES["local-chatbot"],
            Status.PASS,
            "Ollama is available, so a local chatbot path is ready.",
            "Run inferdoctor check ollama if chat requests still fail.",
        )
    if _any_pass(results, OPENAI_ENDPOINTS):
        return ScenarioResult(
            "local-chatbot",
            SCENARIO_TITLES["local-chatbot"],
            Status.PASS,
            "An OpenAI-compatible local endpoint is available for chat clients.",
            "Point your chatbot client at the passing endpoint.",
        )
    return ScenarioResult(
        "local-chatbot",
        SCENARIO_TITLES["local-chatbot"],
        Status.WARN,
        "No local chat runtime responded, but the machine check completed.",
        "Start Ollama or configure a vLLM/SGLang OpenAI-compatible endpoint.",
    )


def _rag_app(results: Dict[str, CheckResult]) -> ScenarioResult:
    if _is_pass(results, "dify"):
        return ScenarioResult(
            "rag-app",
            SCENARIO_TITLES["rag-app"],
            Status.PASS,
            "Dify is reachable, so a local RAG app path is visible.",
            "Check your app-specific embedding and vector database settings next.",
        )
    if _any_pass(results, OPENAI_ENDPOINTS) or _is_pass(results, "ollama"):
        return ScenarioResult(
            "rag-app",
            SCENARIO_TITLES["rag-app"],
            Status.WARN,
            "A model endpoint is available, but Dify or embedding infrastructure was not confirmed.",
            "Configure Dify or verify your embedding endpoint and vector database.",
        )
    return ScenarioResult(
        "rag-app",
        SCENARIO_TITLES["rag-app"],
        Status.WARN,
        "No RAG application endpoint or model serving endpoint was confirmed.",
        "Configure Dify or an OpenAI-compatible model endpoint before debugging the app layer.",
    )


def _openai_compatible_server(results: Dict[str, CheckResult]) -> ScenarioResult:
    if _any_pass(results, OPENAI_ENDPOINTS):
        return ScenarioResult(
            "openai-compatible-server",
            SCENARIO_TITLES["openai-compatible-server"],
            Status.PASS,
            "At least one OpenAI-compatible server responded with a valid models route.",
            "Use inferdoctor report --format markdown to share endpoint details.",
        )
    if _any_warn(results, OPENAI_ENDPOINTS):
        return ScenarioResult(
            "openai-compatible-server",
            SCENARIO_TITLES["openai-compatible-server"],
            Status.WARN,
            "A server responded, but the OpenAI-compatible models route needs attention.",
            "Run inferdoctor explain openai-compatible-404 or retry with --endpoint.",
        )
    return ScenarioResult(
        "openai-compatible-server",
        SCENARIO_TITLES["openai-compatible-server"],
        Status.FAIL,
        "No OpenAI-compatible local endpoint was confirmed reachable.",
        "Run inferdoctor check vllm, sglang, llamacpp, or lmstudio with the expected endpoint.",
    )


def _dify_local_rag(results: Dict[str, CheckResult]) -> ScenarioResult:
    if _is_pass(results, "dify"):
        status = Status.PASS
        reason = "Dify is reachable at the configured endpoint."
        next_step = "Validate model provider and embedding settings inside Dify."
    else:
        status = Status.WARN
        reason = "Dify is optional and was not confirmed reachable."
        next_step = "Add endpoints.dify to inferdoctor.yaml or check Dify container ports."
    return ScenarioResult("dify-local-rag", SCENARIO_TITLES["dify-local-rag"], status, reason, next_step)


def _gpu_inference(results: Dict[str, CheckResult]) -> ScenarioResult:
    if _is_pass(results, "nvidia"):
        if _is_pass(results, "cuda"):
            reason = "NVIDIA GPU and CUDA toolkit were detected."
        else:
            reason = "NVIDIA GPU is visible; CUDA toolkit may be optional for prebuilt runtimes."
        return ScenarioResult(
            "gpu-inference",
            SCENARIO_TITLES["gpu-inference"],
            Status.PASS,
            reason,
            "Use inferdoctor capacity for rough workload readiness estimates.",
        )
    return ScenarioResult(
        "gpu-inference",
        SCENARIO_TITLES["gpu-inference"],
        Status.WARN,
        "No NVIDIA GPU was confirmed by nvidia-smi.",
        "Use CPU fallback or repair the NVIDIA driver if GPU inference is required.",
    )


def _cpu_only_fallback(results: Dict[str, CheckResult]) -> ScenarioResult:
    if _is_pass(results, "system"):
        return ScenarioResult(
            "cpu-only-fallback",
            SCENARIO_TITLES["cpu-only-fallback"],
            Status.PASS,
            "System diagnostics completed, so CPU-only fallback diagnostics are available.",
            "Use small quantized models and conservative context lengths on CPU-only machines.",
        )
    return ScenarioResult(
        "cpu-only-fallback",
        SCENARIO_TITLES["cpu-only-fallback"],
        Status.FAIL,
        "System diagnostics did not complete.",
        "Run inferdoctor check system --verbose first.",
    )


def render_scenarios(scenarios: Iterable[ScenarioResult], language: str = "auto") -> str:
    lines = [
        t("scenarios.title", language),
        "=" * 57,
        "Goal-oriented diagnosis for common local AI setups.",
        "",
    ]
    for scenario in scenarios:
        lines.extend(
            [
                "{0}:".format(scenario.title),
                "  Status: {0}".format(scenario.status.value.upper()),
                "  Reason: {0}".format(scenario.reason),
                "  Next step: {0}".format(scenario.next_step),
                "",
            ]
        )
    return "\n".join(lines).rstrip()
