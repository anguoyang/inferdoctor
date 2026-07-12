from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from inferdoctor.core.experience import get_profile, profile_summary

from inferdoctor.core.perf_baseline import load_report_or_baseline
from inferdoctor.core.perf_compare import compare_performance_files

PLAN_SCHEMA_VERSION = "inferdoctor.optimize.plan.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str) -> Dict[str, Any]:
    try:
        data = json.loads(open(path, encoding="utf-8").read())
    except OSError as exc:
        raise ValueError("Could not read JSON file {0}: {1}".format(path, exc)) from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON file {0}: {1}".format(path, exc)) from exc
    if not isinstance(data, dict):
        raise ValueError("JSON document must be an object")
    return data


def _metric(data: Dict[str, Any], key: str) -> Optional[float]:
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    aliases = {
        "ttft": ("ttft_median", "ttft_seconds"),
        "total_latency": ("total_latency_median", "total_latency_seconds"),
        "generation_duration": ("generation_duration_median", "generation_duration_seconds"),
        "tps": ("generation_tps_median", "generation_tokens_per_second"),
    }
    for alias in aliases.get(key, (key,)):
        value = metrics.get(alias)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _append_unique(
    items: List[Dict[str, str]],
    priority: str,
    observation: str,
    action: str,
    verify: str,
    impact: str,
    limitation: str,
    evidence_level: str = "Possible",
) -> None:
    key = (priority, observation, action)
    existing = {(item["priority"], item["observation"], item["action"]) for item in items}
    if key in existing:
        return
    items.append({
        "priority": priority,
        "observation": observation,
        "action": action,
        "verify": verify,
        "expected_impact": impact,
        "limitation": limitation,
        "evidence_level": evidence_level,
    })


def _facts_from_report(path: str) -> Dict[str, Any]:
    try:
        data = _read_json(path)
    except ValueError:
        data = load_report_or_baseline(path)
    if data.get("schema_version") == "inferdoctor.perf.compare.v1":
        return {"kind": "comparison", "comparison": data}
    return {"kind": "report", "report": load_report_or_baseline(path)}


def build_optimization_plan(
    *,
    report_path: Optional[str] = None,
    baseline_path: Optional[str] = None,
    candidate_path: Optional[str] = None,
    runtime: Optional[str] = None,
    model_size: Optional[str] = None,
    vram_gib: Optional[float] = None,
    goal: Optional[str] = None,
    streaming: bool = False,
    retrieval_ms: Optional[float] = None,
    rerank_ms: Optional[float] = None,
    ttft: Optional[float] = None,
    profile: Optional[str] = None,
) -> Dict[str, Any]:
    observations: List[str] = []
    actions: List[Dict[str, str]] = []
    limitations: List[str] = [
        "This plan is heuristic and does not promise exact speedups.",
        "Smoke-test metrics are not formal benchmark results.",
        "Model quality is not measured by InferDoctor performance commands.",
    ]
    evidence: Dict[str, Any] = {
        "runtime": runtime,
        "model_size": model_size,
        "vram_gib": vram_gib,
        "goal": goal,
        "streaming_preferred": streaming,
        "experience_profile": profile,
        "user_provided_metrics": {
            "retrieval_ms": retrieval_ms,
            "rerank_ms": rerank_ms,
            "ttft": ttft,
        },
    }

    comparison: Optional[Dict[str, Any]] = None
    report: Optional[Dict[str, Any]] = None
    if baseline_path and candidate_path:
        comparison = compare_performance_files(baseline_path, candidate_path)
        evidence["comparison"] = comparison
        observations.extend(comparison.get("observations", []))
        if comparison.get("warnings"):
            _append_unique(
                actions,
                "Do now",
                "The compared runs are not directly comparable.",
                "Repeat both smoke tests with the same endpoint, model, test type, streaming mode, runs, and timeout.",
                "inferdoctor perf compare before.json after.json",
                "Higher confidence before changing runtime settings.",
                "A fair comparison still does not replace production monitoring.",
            )
        changes = comparison.get("metric_changes") if isinstance(comparison.get("metric_changes"), dict) else {}
        if changes.get("ttft_seconds", {}).get("direction") == "regression":
            _append_unique(actions, "Do now", "TTFT regressed after the change.", "Check cold start, context size, and streaming behavior before optimizing TPS.", "inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --warmup 1 --runs 2", "Improves perceived responsiveness if TTFT is the main bottleneck.", "The command uses a tiny prompt and may not match your full app prompt.")
        if changes.get("generation_tokens_per_second", {}).get("direction") == "regression":
            _append_unique(actions, "Test next", "Generation speed regressed after the change.", "Verify model size, quantization, runtime, and GPU placement.", "inferdoctor model fit --size {0} --quant q4 --vram {1}".format(model_size or "14b", vram_gib or 24), "Helps identify oversized model or memory pressure risk.", "Model fit is heuristic, not a profiler.")
    elif report_path:
        facts = _facts_from_report(report_path)
        report = facts.get("report")
        if report:
            evidence["report"] = report
            ttft = ttft if ttft is not None else _metric(report, "ttft")
            observations.append("Loaded performance smoke-test report for model {0}.".format(report.get("model") or "unknown"))
            if report.get("readiness_category"):
                observations.append("Readiness category: {0}.".format(report.get("readiness_category")))
    else:
        observations.append("No performance report or comparison was provided; using supplied hints only.")

    selected_profile = get_profile(profile)
    if selected_profile:
        observations.append(profile_summary(selected_profile))
        _append_unique(
            actions,
            "Do now",
            "Application profile is {0}.".format(selected_profile.title),
            selected_profile.progress_feedback[0],
            selected_profile.checks[0],
            selected_profile.heuristic_ranges[0],
            "Experience profile ranges are heuristics, not universal SLA guarantees.",
        )

    if not streaming:
        _append_unique(actions, "Do now", "Streaming is not confirmed or not requested.", "Enable streaming in the app and validate time to first visible content.", "inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model", "Usually improves perceived latency for chat and RAG apps.", "The smoke test cannot verify whether your frontend renders chunks progressively.")
    if ttft is not None and ttft > 2.0:
        _append_unique(actions, "Do now", "TTFT is above 2 seconds.", "Warm the endpoint, reduce context budget, and verify streaming first.", "inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --warmup 1 --runs 2", "Targets the delay users notice before the answer begins.", "TTFT depends on model cache state and prompt/context length.")
    if retrieval_ms is not None and retrieval_ms > 700:
        _append_unique(actions, "Test next", "Retrieval latency is high for an interactive RAG flow.", "Reduce top_k, cache repeated retrievals, and show retrieval progress before generation.", "inferdoctor optimize rag --retrieval-ms {0:g} --streaming".format(retrieval_ms), "Improves perceived responsiveness before generation begins.", "InferDoctor is using user-provided retrieval timing, not profiling your vector database.")
    if rerank_ms is not None and rerank_ms > 1000:
        _append_unique(actions, "Test next", "Reranking cost is high.", "Use rerank only when needed, reduce candidates, or degrade gracefully when rerank times out.", "inferdoctor optimize rag --rerank-ms {0:g} --top-k 6".format(rerank_ms), "Can reduce pre-generation delay in RAG apps.", "Relevance quality may change when reducing rerank work.")
    if vram_gib is not None and model_size:
        size_text = str(model_size).lower().rstrip("b")
        try:
            size = float(size_text)
        except ValueError:
            size = None
        if size and size >= 32 and vram_gib <= 24:
            _append_unique(actions, "Consider later", "Model size may be ambitious for the supplied VRAM.", "Use a smaller or quantized model for interactive demos before tuning large-model serving.", "inferdoctor model fit --size {0} --quant q4 --vram {1:g}".format(model_size, vram_gib), "Reduces memory pressure and cold-start risk.", "Actual fit depends on runtime overhead, KV cache, and context length.")
    if runtime == "vllm":
        _append_unique(actions, "Test next", "vLLM is optimized for serving but can trade throughput against interactive latency.", "Verify /v1 endpoint, max model length, and streaming behavior with tiny runs before app changes.", "inferdoctor check vllm --endpoint http://127.0.0.1:8000/v1", "Separates endpoint/config issues from model performance issues.", "InferDoctor does not inspect vLLM scheduler internals.")
    if runtime == "ollama":
        _append_unique(actions, "Test next", "Ollama is usually easiest for local demos.", "Warm the model and verify streaming before measuring user experience.", "inferdoctor check ollama", "Reduces demo surprises from model cold start or endpoint mismatch.", "InferDoctor does not pull models or configure Ollama.")

    if not actions:
        _append_unique(actions, "Not enough evidence", "No concrete bottleneck was supplied.", "Collect one bounded endpoint or streaming smoke-test report first.", "inferdoctor perf streaming --endpoint http://127.0.0.1:8000/v1 --model local-model --format json --output perf.json", "Gives a factual starting point for optimization.", "One tiny smoke test is still not a benchmark.")

    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "goal": goal or "not specified",
        "runtime": runtime or "not specified",
        "model_size": model_size,
        "vram_gib": vram_gib,
        "observations": observations,
        "actions": actions,
        "limitations": limitations,
        "evidence": evidence,
    }


def render_plan_json(plan: Dict[str, Any]) -> str:
    return json.dumps(plan, indent=2, sort_keys=True)


def _render_action(action: Dict[str, str]) -> List[str]:
    return [
        "- Evidence: {0}".format(action.get("evidence_level", "Possible")),
        "  Observation: {0}".format(action["observation"]),
        "  Action: {0}".format(action["action"]),
        "  Verify: {0}".format(action["verify"]),
        "  Expected impact: {0}".format(action["expected_impact"]),
        "  Limitation: {0}".format(action["limitation"]),
    ]


def render_plan_console(plan: Dict[str, Any]) -> str:
    lines = [
        "InferDoctor Optimization Plan",
        "=" * 57,
        "Goal: {0}".format(plan.get("goal") or "not specified"),
        "Runtime: {0}".format(plan.get("runtime") or "not specified"),
        "Model size: {0}".format(plan.get("model_size") or "not specified"),
        "VRAM: {0}".format("{0:g} GiB".format(plan["vram_gib"]) if plan.get("vram_gib") is not None else "not specified"),
        "",
        "Observations:",
    ]
    lines.extend("- {0}".format(item) for item in plan.get("observations", []))
    by_priority: Dict[str, List[Dict[str, str]]] = {}
    for action in plan.get("actions", []):
        by_priority.setdefault(action["priority"], []).append(action)
    for priority in ("Do now", "Test next", "Consider later", "Not enough evidence"):
        if priority not in by_priority:
            continue
        lines.extend(["", priority + ":"])
        for action in by_priority[priority]:
            lines.extend(_render_action(action))
    lines.extend(["", "Limitations:"])
    lines.extend("- {0}".format(item) for item in plan.get("limitations", []))
    return "\n".join(lines)


def render_plan_markdown(plan: Dict[str, Any]) -> str:
    lines = [
        "# InferDoctor Optimization Plan",
        "",
        "- Goal: `{0}`".format(plan.get("goal") or "not specified"),
        "- Runtime: `{0}`".format(plan.get("runtime") or "not specified"),
        "- Model size: `{0}`".format(plan.get("model_size") or "not specified"),
        "- VRAM: `{0}`".format("{0:g} GiB".format(plan["vram_gib"]) if plan.get("vram_gib") is not None else "not specified"),
        "",
        "## Observations",
        "",
    ]
    lines.extend("- {0}".format(item) for item in plan.get("observations", []))
    by_priority: Dict[str, List[Dict[str, str]]] = {}
    for action in plan.get("actions", []):
        by_priority.setdefault(action["priority"], []).append(action)
    for priority in ("Do now", "Test next", "Consider later", "Not enough evidence"):
        if priority not in by_priority:
            continue
        lines.extend(["", "## " + priority, ""])
        for action in by_priority[priority]:
            lines.extend(_render_action(action))
    lines.extend(["", "## Limitations", ""])
    lines.extend("- {0}".format(item) for item in plan.get("limitations", []))
    return "\n".join(lines)


def render_optimization_plan(plan: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return render_plan_json(plan)
    if output_format == "markdown":
        return render_plan_markdown(plan)
    return render_plan_console(plan)
