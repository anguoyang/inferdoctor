from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from inferdoctor.core.capacity import detect_hardware, estimate_model_memory_gib, parse_model_size_b
from inferdoctor.i18n import t


@dataclass(frozen=True)
class ModelFitResult:
    size_b: float
    quant: str
    runtime: Optional[str]
    vram_gib: Optional[float]
    estimated_memory_gib: float
    fit: str
    reason: str
    easiest_path: str
    performance_path: str
    caveat: str
    unknowns: str
    use_cases: List[str]


def _reason(fit: str, vram_gib: Optional[float], estimated: float) -> str:
    if vram_gib is None:
        return "No VRAM figure was available, so this is only a rough memory class estimate."
    if fit == "LIKELY OK":
        return "Available VRAM is several GiB above the simple memory estimate."
    if fit == "MAYBE":
        return "Available VRAM is close to the estimate, so context length and runtime overhead matter."
    return "The simple memory estimate ({0:.1f} GiB) exceeds the provided VRAM.".format(estimated)


def _use_cases(size_b: float, fit: str, vram_gib: Optional[float]) -> list[str]:
    if fit == "UNLIKELY":
        return [
            "Not recommended for GPU optimized serving with the provided memory.",
            "Try CPU-only experimentation, a smaller model size class, or lower context length.",
        ]
    if size_b <= 8:
        return [
            "Likely suitable for personal chatbot, restaurant ordering assistant, and customer service demos.",
            "Good first step before document Q&A or local API serving.",
        ]
    if size_b <= 14:
        return [
            "Likely suitable for customer service chatbot, document Q&A, and local OpenAI-compatible API demos when endpoint checks pass.",
            "Maybe suitable for small team RAG depending on context length and embedding workload.",
        ]
    return [
        "Maybe suitable for local API and GPU optimized serving when VRAM headroom is real.",
        "Not the first choice for beginner templates unless smaller models already work.",
    ]


def estimate_model_fit(
    size: object,
    quant: str = "q4",
    runtime: Optional[str] = None,
    vram_gib: Optional[float] = None,
) -> ModelFitResult:
    size_b = parse_model_size_b(size)
    if size_b is None:
        size_b = 7.0
    resolved_vram = vram_gib
    if resolved_vram is None:
        resolved_vram = detect_hardware().vram_gib
    estimated = estimate_model_memory_gib(size_b, quant, runtime)
    if resolved_vram is None:
        fit = "MAYBE"
        caveat = "Use --vram for a clearer estimate. CPU-only fallback may work but can be slow."
    elif resolved_vram >= estimated + 4:
        fit = "LIKELY OK"
        caveat = "There appears to be memory headroom, but context length and runtime settings still matter."
    elif resolved_vram >= estimated:
        fit = "MAYBE"
        caveat = "Memory is close to the estimate; reduce context length or use a smaller quantization/runtime setting."
    else:
        fit = "UNLIKELY"
        caveat = "Estimated memory exceeds available VRAM; choose a smaller model, lower context, or CPU/offload fallback."
    easiest_path = "Try Ollama or LM Studio first with a quantized model size class that fits this estimate."
    performance_path = "Use vLLM or SGLang only after endpoint and VRAM checks pass; tune context and concurrency carefully."
    unknowns = "This estimate does not know exact model architecture, context length, KV cache size, batch size, or runtime flags."
    return ModelFitResult(
        size_b=size_b,
        quant=quant,
        runtime=runtime,
        vram_gib=resolved_vram,
        estimated_memory_gib=estimated,
        fit=fit,
        reason=_reason(fit, resolved_vram, estimated),
        easiest_path=easiest_path,
        performance_path=performance_path,
        caveat=caveat,
        unknowns=unknowns,
        use_cases=_use_cases(size_b, fit, resolved_vram),
    )


def render_model_fit(result: ModelFitResult, language: str = "auto") -> str:
    vram = "unknown" if result.vram_gib is None else "{0:g} GiB".format(result.vram_gib)
    runtime = result.runtime or "generic"
    lines = [
        t("model_fit.title", language),
        "=" * 57,
        "Heuristic estimate only. This is not a benchmark.",
        "",
        t("model_fit.request", language, description=""),
        "  Model size: {0:g}B".format(result.size_b),
        "  Quantization: {0}".format(result.quant.upper()),
        "  Runtime: {0}".format(runtime),
        "  VRAM: {0}".format(vram),
        "",
        t("model_fit.estimate", language, description=""),
        "  Estimated memory: {0:.1f} GiB".format(result.estimated_memory_gib),
        "  Fit: {0}".format(result.fit),
        "  Why: {0}".format(result.reason),
        "",
        t("model_fit.runtime_guidance", language),
        "  Easiest path: {0}".format(result.easiest_path),
        "  Performance path: {0}".format(result.performance_path),
        "",
        "Use-case guidance:",
    ]
    lines.extend("  - {0}".format(item) for item in result.use_cases)
    lines.extend([
        "",
        "Memory caveats:",
        "  {0}".format(result.caveat),
        "",
        "What InferDoctor does not know yet:",
        "  {0}".format(result.unknowns),
        "",
        "Next commands:",
        "  inferdoctor capacity --model-size {0:g}b --quant {1}{2}".format(
            result.size_b,
            result.quant,
            " --runtime " + result.runtime if result.runtime else "",
        ),
        "  inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1",
        "  inferdoctor check sglang --endpoint http://127.0.0.1:30000/v1",
    ])
    return "\n".join(lines)
