from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from inferdoctor.i18n import t


@dataclass(frozen=True)
class Explanation:
    topic: str
    title: str
    meaning: str
    common_causes: List[str]
    next_steps: List[str]
    related_command: str


EXPLANATIONS: Dict[str, Explanation] = {
    "openai-compatible-404": Explanation(
        topic="openai-compatible-404",
        title="OpenAI-compatible endpoint returned 404",
        meaning=(
            "The server responded, but InferDoctor could not find the expected "
            "/v1/models route."
        ),
        common_causes=[
            "The endpoint is missing the /v1 prefix.",
            "The URL points to a web UI, proxy root, or health route instead of the API.",
            "The runtime exposes an OpenAI-compatible API on a different port.",
        ],
        next_steps=[
            "Try adding /v1 to the configured endpoint.",
            "Open the runtime logs and confirm which host and port expose /v1/models.",
            "Retry the component check with --endpoint.",
        ],
        related_command="inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1",
    ),
    "connection-refused": Explanation(
        topic="connection-refused",
        title="Endpoint connection refused",
        meaning=(
            "Nothing accepted the TCP connection at the configured host and port."
        ),
        common_causes=[
            "The service is not running.",
            "The service is listening on another port or interface.",
            "A container port is not published to the host.",
        ],
        next_steps=[
            "Start the runtime or web service you want to diagnose.",
            "Check the configured endpoint in inferdoctor.yaml.",
            "Retry with inferdoctor check <component> --endpoint <url>.",
        ],
        related_command="inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1",
    ),
    "cuda-toolkit-missing": Explanation(
        topic="cuda-toolkit-missing",
        title="CUDA toolkit was not found",
        meaning=(
            "InferDoctor could not find nvcc. The NVIDIA driver may still work, "
            "but the CUDA toolkit is not available on PATH."
        ),
        common_causes=[
            "Only the NVIDIA driver is installed.",
            "CUDA toolkit is installed but nvcc is not on PATH.",
            "The machine uses prebuilt runtimes that do not need nvcc.",
        ],
        next_steps=[
            "Run nvcc --version if you expect a toolkit to be installed.",
            "Install CUDA toolkit only if you compile CUDA code or use a runtime that requires nvcc.",
            "No action is needed for CPU-only usage or many prebuilt Ollama-style runtimes.",
        ],
        related_command="inferdoctor check cuda",
    ),
    "nvidia-driver-missing": Explanation(
        topic="nvidia-driver-missing",
        title="NVIDIA driver was not found",
        meaning="InferDoctor could not find or query nvidia-smi.",
        common_causes=[
            "This is a CPU-only or non-NVIDIA machine.",
            "The NVIDIA driver is not installed or not loaded.",
            "nvidia-smi is not available on PATH.",
        ],
        next_steps=[
            "Run nvidia-smi directly if this machine should have an NVIDIA GPU.",
            "Skip this issue on CPU-only machines.",
            "Repair the driver only if GPU inference is required.",
        ],
        related_command="inferdoctor check nvidia",
    ),
    "ollama-not-running": Explanation(
        topic="ollama-not-running",
        title="Ollama is not running or not reachable",
        meaning="The Ollama CLI or default API endpoint did not respond as expected.",
        common_causes=[
            "Ollama is not installed.",
            "The Ollama service is stopped.",
            "Ollama is listening on a non-default host or port.",
        ],
        next_steps=[
            "Start Ollama if you use it on this machine.",
            "Check endpoints.ollama in inferdoctor.yaml.",
            "Skip this issue if Ollama is not part of your stack.",
        ],
        related_command="inferdoctor check ollama --endpoint http://127.0.0.1:11434",
    ),
    "dify-connection-failed": Explanation(
        topic="dify-connection-failed",
        title="Dify endpoint connection failed",
        meaning="InferDoctor could not reach the configured Dify endpoint.",
        common_causes=[
            "Dify is not running on this machine.",
            "The configured Dify port is wrong.",
            "A Docker port mapping is missing or points to another service.",
        ],
        next_steps=[
            "Add endpoints.dify to inferdoctor.yaml if you want Dify diagnostics.",
            "Check Dify container and proxy port mappings.",
            "Skip this issue if Dify is not part of your stack.",
        ],
        related_command="inferdoctor check dify --endpoint http://127.0.0.1:5001",
    ),
    "sglang-endpoint-not-reachable": Explanation(
        topic="sglang-endpoint-not-reachable",
        title="SGLang endpoint is not reachable",
        meaning="InferDoctor could not reach the SGLang OpenAI-compatible API.",
        common_causes=[
            "SGLang is not running.",
            "The API is listening on another port.",
            "The endpoint is missing the /v1 base path.",
        ],
        next_steps=[
            "Confirm the SGLang server host, port, and /v1 path.",
            "Retry with --endpoint using the exact OpenAI-compatible base URL.",
            "Skip this issue if SGLang is not part of your stack.",
        ],
        related_command="inferdoctor check sglang --endpoint http://127.0.0.1:30000/v1",
    ),
    "vllm-endpoint-not-reachable": Explanation(
        topic="vllm-endpoint-not-reachable",
        title="vLLM endpoint is not reachable",
        meaning="InferDoctor could not reach the vLLM OpenAI-compatible API.",
        common_causes=[
            "vLLM is not running.",
            "The server uses a different host, port, or base path.",
            "A proxy or container mapping is routing the request elsewhere.",
        ],
        next_steps=[
            "Confirm the vLLM server host, port, and /v1 path.",
            "Try the /v1/models route directly from the same machine.",
            "Retry with --endpoint using the exact OpenAI-compatible base URL.",
        ],
        related_command="inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1",
    ),
    "xinference-endpoint-not-reachable": Explanation(
        topic="xinference-endpoint-not-reachable",
        title="Xinference endpoint is not reachable",
        meaning="InferDoctor could not reach the Xinference supervisor endpoint.",
        common_causes=[
            "Xinference is not running.",
            "The supervisor is listening on another host or port.",
            "A proxy or firewall is blocking local access.",
        ],
        next_steps=[
            "Confirm the Xinference supervisor address.",
            "Update endpoints.xinference in inferdoctor.yaml if needed.",
            "No Xinference SDK is required for this check.",
        ],
        related_command="inferdoctor check xinference --endpoint http://127.0.0.1:9997",
    ),
}


def explain_topics() -> List[str]:
    return sorted(EXPLANATIONS)


def render_explanation(topic: str, language: str = "auto") -> str:
    explanation = EXPLANATIONS[topic]
    lines = [
        t("explain.title", language, topic=explanation.title),
        "=" * 57,
        "",
        t("explain.what_it_means", language),
        "  {0}".format(explanation.meaning),
        "",
        t("explain.common_causes", language),
    ]
    lines.extend("  - {0}".format(item) for item in explanation.common_causes)
    lines.extend(["", t("explain.what_to_try_next", language)])
    lines.extend("  - {0}".format(item) for item in explanation.next_steps)
    lines.extend(["", t("explain.related_command", language), "  {0}".format(explanation.related_command)])
    return "\n".join(lines)
