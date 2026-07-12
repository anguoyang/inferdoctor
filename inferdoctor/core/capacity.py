from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from typing import List, Optional

from inferdoctor.checkers.nvidia import NvidiaChecker
from inferdoctor.core.config import Config
from inferdoctor.core.models import Status
from inferdoctor.i18n import t

GPU_VRAM_PROFILES = {
    "rtx 4090": 24.0,
    "rtx 3090": 24.0,
    "rtx 4080": 16.0,
    "rtx 4070 ti": 12.0,
    "rtx 4070": 12.0,
    "rtx 4060 ti": 16.0,
    "rtx 3060": 12.0,
    "a100 80gb": 80.0,
    "a100": 40.0,
    "h100": 80.0,
    "l40s": 48.0,
    "t4": 16.0,
}

QUANT_BYTES_PER_PARAM = {
    "q4": 0.65,
    "q8": 1.1,
}


@dataclass(frozen=True)
class CapacityHardware:
    architecture: str
    total_ram_gib: Optional[float]
    available_ram_gib: Optional[float]
    vram_gib: Optional[float]
    gpu_name: Optional[str] = None
    vram_source: str = "detected"


@dataclass(frozen=True)
class CapacityRequest:
    model_size_b: Optional[float] = None
    quant: str = "q4"
    runtime: Optional[str] = None


@dataclass(frozen=True)
class CapacityRow:
    workload: str
    readiness: str
    note: str


def _meminfo_value(field: str) -> Optional[float]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024 / (1024 ** 3)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _detect_vram() -> tuple[Optional[float], Optional[str]]:
    result = NvidiaChecker().run(Config())
    if result.status != Status.PASS:
        return None, None
    gpus = result.raw_data.get("gpus") or []
    best_memory = None
    best_name = None
    for gpu in gpus:
        memory_mib = gpu.get("memory_total_mib")
        if memory_mib is None:
            continue
        if best_memory is None or memory_mib > best_memory:
            best_memory = memory_mib
            best_name = str(gpu.get("name") or "NVIDIA GPU")
    if best_memory is None:
        return None, best_name
    return best_memory / 1024, best_name


def infer_vram_from_gpu_name(gpu_name: Optional[str]) -> Optional[float]:
    if not gpu_name:
        return None
    normalized = gpu_name.lower().replace("nvidia", "").replace("geforce", "")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    explicit = re.search(r"(\d+(?:\.\d+)?)\s*g(?:i?b)?", normalized)
    if explicit:
        return float(explicit.group(1))

    for key, vram in sorted(GPU_VRAM_PROFILES.items(), key=lambda item: -len(item[0])):
        if key in normalized:
            return vram
    return None


def parse_model_size_b(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text.endswith("b"):
        text = text[:-1]
    return float(text)


def detect_hardware(
    vram_gib: Optional[float] = None,
    gpu_name: Optional[str] = None,
) -> CapacityHardware:
    detected_vram = vram_gib
    detected_gpu = gpu_name
    source = "override" if vram_gib is not None else "detected"
    if detected_vram is None and gpu_name is not None:
        detected_vram = infer_vram_from_gpu_name(gpu_name)
        if detected_vram is not None:
            source = "gpu-profile"
    if detected_vram is None:
        detected_vram, detected_gpu = _detect_vram()
    elif detected_gpu is None:
        detected_gpu = "manual VRAM override"
    return CapacityHardware(
        architecture=platform.machine() or "unknown",
        total_ram_gib=_meminfo_value("MemTotal"),
        available_ram_gib=_meminfo_value("MemAvailable"),
        vram_gib=detected_vram,
        gpu_name=detected_gpu,
        vram_source=source,
    )


def _at_least(value: Optional[float], threshold: float) -> bool:
    return value is not None and value >= threshold


def _either(
    vram: Optional[float],
    ram: Optional[float],
    vram_min: float,
    ram_min: float,
) -> bool:
    return _at_least(vram, vram_min) or _at_least(ram, ram_min)


def estimate_model_memory_gib(
    model_size_b: float,
    quant: str = "q4",
    runtime: Optional[str] = None,
) -> float:
    bytes_per_param = QUANT_BYTES_PER_PARAM.get(quant, QUANT_BYTES_PER_PARAM["q4"])
    memory = model_size_b * bytes_per_param + 2.0
    if runtime == "vllm":
        memory += max(2.0, model_size_b * 0.12)
    elif runtime == "ollama":
        memory += 1.0
    return round(memory, 1)


def estimate_requested_model(
    hardware: CapacityHardware, request: CapacityRequest
) -> Optional[CapacityRow]:
    if request.model_size_b is None and request.runtime is None:
        return None

    model_size = request.model_size_b or 7.0
    quant = request.quant or "q4"
    required = estimate_model_memory_gib(model_size, quant, request.runtime)
    memory = hardware.vram_gib if hardware.vram_gib is not None else hardware.total_ram_gib
    memory_label = "VRAM" if hardware.vram_gib is not None else "system RAM"

    if memory is None:
        readiness = "UNKNOWN"
        note = "No memory figure was available for this heuristic."
    elif memory >= required + 4:
        readiness = "LIKELY OK"
        note = "Estimated {0} need is about {1:.1f} GiB; {2} has headroom.".format(
            quant.upper(), required, memory_label
        )
    elif memory >= required:
        readiness = "MAYBE"
        note = "Estimated {0} need is about {1:.1f} GiB; reduce context if needed.".format(
            quant.upper(), required
        )
    else:
        readiness = "UNLIKELY"
        note = "Estimated {0} need is about {1:.1f} GiB, above available {2}.".format(
            quant.upper(), required, memory_label
        )

    runtime = request.runtime or "generic"
    return CapacityRow(
        "Requested {0:g}B {1} on {2}".format(model_size, quant.upper(), runtime),
        readiness,
        note,
    )


def _use_case_readiness(vram: Optional[float], ram: Optional[float]) -> list[CapacityRow]:
    rows: list[CapacityRow] = []
    cpu_ok = _at_least(ram, 8)
    small_gpu = _at_least(vram, 8)
    mid_gpu = _at_least(vram, 12)
    strong_gpu = _at_least(vram, 24)
    rows.append(
        CapacityRow(
            "Use case: personal chatbot",
            "LIKELY OK" if small_gpu or _at_least(ram, 16) else "MAYBE",
            "Try the customer-service or ollama-chat path with a small quantized model.",
        )
    )
    rows.append(
        CapacityRow(
            "Use case: customer service",
            "LIKELY OK" if small_gpu or _at_least(ram, 16) else "MAYBE",
            "Use the customer-service template, then validate and smoke-test before connecting an endpoint.",
        )
    )
    rows.append(
        CapacityRow(
            "Use case: document Q&A",
            "LIKELY OK" if mid_gpu or _at_least(ram, 16) else "MAYBE",
            "Start with local-doc-qa keyword retrieval; add embeddings later.",
        )
    )
    rows.append(
        CapacityRow(
            "Use case: restaurant ordering",
            "LIKELY OK" if small_gpu or cpu_ok else "MAYBE",
            "Use the restaurant-ordering template for a concrete local demo.",
        )
    )
    rows.append(
        CapacityRow(
            "Use case: local API",
            "LIKELY OK" if strong_gpu else "MAYBE" if mid_gpu else "NOT RECOMMENDED",
            "Validate /v1/models first; vLLM/SGLang need GPU headroom.",
        )
    )
    rows.append(
        CapacityRow(
            "Use case: small team RAG",
            "LIKELY OK" if strong_gpu else "MAYBE" if mid_gpu or _at_least(ram, 32) else "NOT RECOMMENDED",
            "Plan for embeddings, retrieval, and serving overhead beyond the chat model.",
        )
    )
    rows.append(
        CapacityRow(
            "Use case: CPU-only experiments",
            "LIKELY OK" if cpu_ok else "MAYBE",
            "Good for diagnosis, templates, keyword retrieval, and very small models.",
        )
    )
    rows.append(
        CapacityRow(
            "Use case: GPU optimized serving",
            "LIKELY OK" if strong_gpu else "MAYBE" if mid_gpu else "NOT RECOMMENDED",
            "Use vLLM or SGLang only after NVIDIA and endpoint checks pass.",
        )
    )
    return rows


def estimate_capacity(
    hardware: CapacityHardware, request: Optional[CapacityRequest] = None
) -> List[CapacityRow]:
    ram = hardware.total_ram_gib
    vram = hardware.vram_gib
    rows: List[CapacityRow] = []
    if request is not None:
        requested = estimate_requested_model(hardware, request)
        if requested is not None:
            rows.append(requested)

    rows.extend(_use_case_readiness(vram, ram))

    rows.append(
        CapacityRow(
            "CPU-only local AI",
            "LIMITED" if not _at_least(ram, 16) else "POSSIBLE",
            (
                "Useful for embeddings, small models, and diagnostics; "
                "expect slower generation without GPU."
            ),
        )
    )

    small_ok = _either(vram, ram, 6, 12)
    rows.append(
        CapacityRow(
            "Small local chat models",
            "LIKELY OK" if small_ok else "MAYBE",
            "Best with quantized models and conservative context length.",
        )
    )

    if _either(vram, ram, 8, 16):
        seven_b = "LIKELY OK"
    elif _either(vram, ram, 6, 12):
        seven_b = "MAYBE"
    else:
        seven_b = "UNLIKELY"
    rows.append(
        CapacityRow(
            "7B/8B quantized models",
            seven_b,
            "Heuristic assumes Q4-style quantization.",
        )
    )

    if _either(vram, ram, 16, 32):
        fourteen_b = "LIKELY OK"
    elif _either(vram, ram, 10, 20):
        fourteen_b = "MAYBE"
    else:
        fourteen_b = "UNLIKELY"
    rows.append(
        CapacityRow(
            "14B quantized models",
            fourteen_b,
            "May need lower context length or CPU offload.",
        )
    )

    if _either(vram, ram, 32, 64):
        thirty_two_b = "LIKELY OK"
    elif _either(vram, ram, 24, 48):
        thirty_two_b = "MAYBE"
    else:
        thirty_two_b = "UNLIKELY"
    rows.append(
        CapacityRow(
            "32B quantized models",
            thirty_two_b,
            "Memory headroom matters more than model name.",
        )
    )

    if _at_least(vram, 24):
        vllm = "LIKELY OK"
        vllm_note = (
            "Suitable for smaller FP16/BF16 serving; model and context still decide."
        )
    elif _at_least(vram, 16):
        vllm = "MEMORY LIMITED"
        vllm_note = "May work for smaller models with tight memory settings."
    else:
        vllm = "UNLIKELY"
        vllm_note = "vLLM FP16 serving usually needs substantial GPU memory."
    rows.append(CapacityRow("vLLM FP16 serving", vllm, vllm_note))

    if _either(vram, ram, 8, 16):
        ollama = "LIKELY OK"
    elif _either(vram, ram, 4, 8):
        ollama = "MAYBE"
    else:
        ollama = "UNLIKELY"
    rows.append(
        CapacityRow(
            "Ollama quantized usage",
            ollama,
            "Ollama-style quantized models are the intended fit for this estimate.",
        )
    )
    return rows


def _fmt_gib(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return "{0:.1f} GiB".format(value)


def render_capacity(
    vram_gib: Optional[float] = None,
    gpu_name: Optional[str] = None,
    model_size_b: Optional[object] = None,
    quant: str = "q4",
    runtime: Optional[str] = None,
    language: str = "auto",
) -> str:
    hardware = detect_hardware(vram_gib=vram_gib, gpu_name=gpu_name)
    request = CapacityRequest(
        model_size_b=parse_model_size_b(model_size_b),
        quant=quant,
        runtime=runtime,
    )
    rows = estimate_capacity(hardware, request)
    gpu_label = hardware.gpu_name or "none detected"
    vram_label = _fmt_gib(hardware.vram_gib)
    if hardware.vram_source == "override":
        vram_label += " (override)"
    elif hardware.vram_source == "gpu-profile":
        vram_label += " (GPU profile heuristic)"

    lines = [
        t("capacity.title", language),
        "=" * 57,
        "Heuristic estimate only. No models are downloaded or run.",
        "InferDoctor does not rank model names or benchmark throughput.",
        "",
        t("capacity.hardware_detected", language),
        "  CPU architecture: {0}".format(hardware.architecture),
        "  System RAM: {0} total, {1} available".format(
            _fmt_gib(hardware.total_ram_gib), _fmt_gib(hardware.available_ram_gib)
        ),
        "  NVIDIA VRAM: {0}".format(vram_label),
        "  GPU: {0}".format(gpu_label),
    ]
    if request.model_size_b is not None or request.runtime is not None:
        lines.extend(
            [
                "",
                "Requested estimate:",
                "  Model size: {0}".format(
                    "{0:g}B".format(request.model_size_b)
                    if request.model_size_b is not None
                    else "not specified"
                ),
                "  Quantization: {0}".format(request.quant.upper()),
                "  Runtime: {0}".format(request.runtime or "generic"),
            ]
        )
    lines.extend(
        [
            "",
            t("capacity.workload_readiness", language),
            "Workload                          Readiness       Notes",
            "--------------------------------  --------------  ----------------------------------------",
        ]
    )
    for row in rows:
        lines.append(
            "{0:<32}  {1:<14}  {2}".format(
                row.workload[:32], row.readiness, row.note
            )
        )
    lines.extend([
        "",
        "Capacity estimates are rough heuristics, not benchmarks.",
        "Use inferdoctor check for actual stack health diagnostics.",
    ])
    return "\n".join(lines)
