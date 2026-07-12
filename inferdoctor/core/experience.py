from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional

from inferdoctor.core.optimize import OptimizationReport
from inferdoctor.core.perf import PerfResult


@dataclass(frozen=True)
class ExperienceProfile:
    name: str
    title: str
    description: str
    what_matters: List[str]
    heuristic_ranges: List[str]
    progress_feedback: List[str]
    templates: List[str]
    checks: List[str]
    caveats: List[str]


PROFILES: Dict[str, ExperienceProfile] = {
    "interactive-chat": ExperienceProfile(
        name="interactive-chat",
        title="Interactive Chat",
        description="A user asks short questions and expects the answer to start quickly.",
        what_matters=["Low TTFT", "Streaming output", "Stable endpoint", "Reasonable total latency"],
        heuristic_ranges=["TTFT under ~1s feels responsive", "TTFT over ~3s often needs progress feedback", "TPS matters most for long answers"],
        progress_feedback=["Show connecting/thinking state", "Stream tokens as soon as possible", "Warm up before demos"],
        templates=["ollama-chat", "customer-service"],
        checks=["inferdoctor perf streaming", "inferdoctor optimize endpoint"],
        caveats=["Speed does not measure answer quality."],
    ),
    "customer-service": ExperienceProfile(
        name="customer-service",
        title="Customer Service Assistant",
        description="A support-style assistant where perceived responsiveness and reliability matter more than raw throughput.",
        what_matters=["TTFT", "Streaming", "Endpoint stability", "Short, helpful answers", "Clear fallback when unavailable"],
        heuristic_ranges=["TTFT under ~1.5s is usually comfortable for demos", "Use progress messages if retrieval or policy lookup happens first"],
        progress_feedback=["Show retrieval or policy-check phase", "Stream answer text", "Handle timeout with a friendly message"],
        templates=["customer-service"],
        checks=["inferdoctor template smoke-test", "inferdoctor perf baseline create", "inferdoctor perf compare"],
        caveats=["Do not send real customer data in smoke tests."],
    ),
    "restaurant-ordering": ExperienceProfile(
        name="restaurant-ordering",
        title="Restaurant Ordering Assistant",
        description="A transactional assistant where users expect quick turns, menu grounding, and clear policy handling.",
        what_matters=["Fast first response", "Reliable menu/policy grounding", "Streaming for longer explanations", "Short timeout budget"],
        heuristic_ranges=["TTFT over ~2s can feel slow in a kiosk or ordering flow", "Keep answers concise before checkout confirmation"],
        progress_feedback=["Show menu lookup progress", "Confirm constraints before generation", "Use short streamed responses"],
        templates=["restaurant-ordering"],
        checks=["inferdoctor stack plan", "inferdoctor template validate", "inferdoctor perf streaming"],
        caveats=["Template examples are not payment or production ordering systems."],
    ),
    "document-qa": ExperienceProfile(
        name="document-qa",
        title="Document Q&A",
        description="A local document assistant where retrieval latency and context size strongly affect perceived speed.",
        what_matters=["Retrieval progress", "Context budget", "TTFT after retrieval", "Citation clarity"],
        heuristic_ranges=["Retrieval above ~700ms should be shown to the user", "Too many chunks can delay TTFT"],
        progress_feedback=["Show searching documents", "Show selected context count", "Stream generation after context assembly"],
        templates=["local-doc-qa", "personal-kb"],
        checks=["inferdoctor optimize rag", "inferdoctor perf streaming"],
        caveats=["Retrieval quality is not measured by performance smoke tests."],
    ),
    "rag": ExperienceProfile(
        name="rag",
        title="RAG Application",
        description="A retrieval-augmented app where stage timing matters as much as model speed.",
        what_matters=["Embedding/query latency", "Retrieval latency", "Rerank latency", "Context assembly", "LLM TTFT", "Streaming"],
        heuristic_ranges=["Top-k over ~8 can increase prompt cost", "Rerank over ~1s should be justified by quality gains"],
        progress_feedback=["Expose retrieval phase", "Show rerank/progress if slow", "Stream answer after context is ready"],
        templates=["local-doc-qa", "dify-rag"],
        checks=["inferdoctor optimize rag", "inferdoctor stack plan --goal document-qa"],
        caveats=["InferDoctor does not profile your vector database unless you provide timings."],
    ),
    "local-api": ExperienceProfile(
        name="local-api",
        title="Local or LAN OpenAI-Compatible API",
        description="A self-hosted endpoint used by apps on the same machine or private network.",
        what_matters=["Reachability", "Auth clarity", "Streaming support", "Stability", "Throughput versus TTFT trade-off"],
        heuristic_ranges=["Use identical model/runtime settings for comparisons", "LAN tests should be explicit and harmless"],
        progress_feedback=["Return health quickly", "Log endpoint errors safely", "Expose streaming capability to clients"],
        templates=["openai-compatible-api", "vllm-api", "sglang-api"],
        checks=["inferdoctor check vllm", "inferdoctor check sglang", "inferdoctor perf endpoint"],
        caveats=["Do not send private prompts to endpoints you do not control."],
    ),
    "batch-processing": ExperienceProfile(
        name="batch-processing",
        title="Batch Processing",
        description="Offline or asynchronous processing where total throughput matters more than immediate TTFT.",
        what_matters=["Throughput", "Success rate", "Retry behavior", "Resource stability"],
        heuristic_ranges=["TTFT is less important than successful completion and sustained throughput", "Do not use InferDoctor smoke tests as load tests"],
        progress_feedback=["Show queue status", "Track failures", "Report completion progress"],
        templates=["openai-compatible-api"],
        checks=["inferdoctor perf endpoint --runs 3", "inferdoctor optimize endpoint"],
        caveats=["InferDoctor does not run concurrency or sustained benchmark workloads."],
    ),
    "internal-prototype": ExperienceProfile(
        name="internal-prototype",
        title="Internal Prototype",
        description="A small demo or internal app where clarity, quick setup, and safe failure modes matter most.",
        what_matters=["Easy setup", "Clear diagnostics", "Readable generated code", "Reasonable responsiveness"],
        heuristic_ranges=["A slower answer may be acceptable if streaming and progress feedback are clear", "Avoid oversized models for the first demo"],
        progress_feedback=["Use dry-run before live endpoint", "Show endpoint status", "Keep user prompts harmless"],
        templates=["customer-service", "restaurant-ordering", "local-doc-qa"],
        checks=["inferdoctor quickstart", "inferdoctor template smoke-test", "inferdoctor perf baseline create"],
        caveats=["Prototype defaults are not production security or reliability guarantees."],
    ),
}


def profile_names() -> List[str]:
    return sorted(PROFILES)


def get_profile(name: Optional[str]) -> Optional[ExperienceProfile]:
    if not name:
        return None
    return PROFILES[name]


def render_profile(profile: ExperienceProfile) -> str:
    lines = [
        "InferDoctor Experience Profile",
        "=" * 57,
        "Name: {0}".format(profile.name),
        "Title: {0}".format(profile.title),
        "Description: {0}".format(profile.description),
        "",
        "What matters most:",
    ]
    lines.extend("- {0}".format(item) for item in profile.what_matters)
    lines.extend(["", "Heuristic ranges:"])
    lines.extend("- {0}".format(item) for item in profile.heuristic_ranges)
    lines.extend(["", "Recommended progress feedback:"])
    lines.extend("- {0}".format(item) for item in profile.progress_feedback)
    lines.extend(["", "Relevant templates:"])
    lines.extend("- {0}".format(item) for item in profile.templates)
    lines.extend(["", "Relevant InferDoctor commands:"])
    lines.extend("- {0}".format(item) for item in profile.checks)
    lines.extend(["", "Caveats:"])
    lines.extend("- {0}".format(item) for item in profile.caveats)
    lines.extend(["", "These ranges are heuristics, not universal SLA guarantees."])
    return "\n".join(lines)


def profile_summary(profile: ExperienceProfile) -> str:
    return "Profile {0}: prioritize {1}.".format(profile.name, ", ".join(profile.what_matters[:3]))


def apply_profile_to_perf_result(result: PerfResult, profile_name: Optional[str]) -> PerfResult:
    profile = get_profile(profile_name)
    if not profile:
        return result
    suggestions = list(result.suggestions) + [profile_summary(profile)]
    warnings = list(result.warnings) + [
        "Experience profile {0} uses heuristic expectations, not benchmark thresholds.".format(profile.name)
    ]
    explanation = result.experience_explanation
    if explanation:
        explanation += " "
    explanation += "For {0}, {1}".format(profile.title, profile.heuristic_ranges[0])
    raw_data = dict(result.raw_data)
    raw_data["experience_profile"] = profile.name
    return replace(
        result,
        suggestions=suggestions,
        warnings=warnings,
        experience_explanation=explanation,
        raw_data=raw_data,
    )


def apply_profile_to_optimization_report(report: OptimizationReport, profile_name: Optional[str]) -> OptimizationReport:
    profile = get_profile(profile_name)
    if not profile:
        return report
    situation = list(report.situation) + [profile_summary(profile)]
    quick_wins = [
        "Observation: Experience profile is {0}.\nImpact: {1}\nAction: {2}\nNext: {3}\nLimitation: {4}".format(
            profile.title,
            profile.heuristic_ranges[0],
            profile.progress_feedback[0],
            profile.checks[0],
            profile.caveats[0],
        )
    ] + list(report.quick_wins)
    caveats = list(report.caveats) + ["Experience profile ranges are heuristics, not universal SLA guarantees."]
    return OptimizationReport(report.title, situation, list(report.bottlenecks), quick_wins[:5], list(report.next_commands), caveats)
