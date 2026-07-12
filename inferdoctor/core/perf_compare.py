from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from inferdoctor.core.perf import sanitize_endpoint
from inferdoctor.core.perf_baseline import load_report_or_baseline

COMPARE_SCHEMA_VERSION = "inferdoctor.perf.compare.v1"

LOWER_IS_BETTER = {
    "ttft_seconds": "TTFT",
    "total_latency_seconds": "Total latency",
    "generation_duration_seconds": "Generation duration",
}
HIGHER_IS_BETTER = {
    "generation_tokens_per_second": "Generation TPS",
    "success_rate": "Successful-run rate",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metric_value(baseline: Dict[str, Any], key: str) -> Optional[float]:
    metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), dict) else {}
    aliases = {
        "ttft_seconds": ("ttft_median", "ttft_seconds"),
        "total_latency_seconds": ("total_latency_median", "total_latency_seconds"),
        "generation_duration_seconds": ("generation_duration_median", "generation_duration_seconds"),
        "generation_tokens_per_second": ("generation_tps_median", "generation_tokens_per_second"),
    }
    if key == "success_rate":
        success = _number(baseline.get("successful_runs")) or 0.0
        failed = _number(baseline.get("failed_runs")) or 0.0
        total = success + failed
        return success / total if total > 0 else None
    for alias in aliases.get(key, (key,)):
        value = _number(metrics.get(alias))
        if value is not None:
            return value
    return None


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _change(baseline: Optional[float], candidate: Optional[float], *, higher_is_better: bool) -> Dict[str, Any]:
    if baseline is None or candidate is None:
        return {
            "baseline": baseline,
            "candidate": candidate,
            "absolute_change": None,
            "percent_change": None,
            "direction": "inconclusive",
            "reason": "missing comparable metric",
        }
    absolute = candidate - baseline
    percent = (absolute / baseline * 100.0) if baseline != 0 else None
    threshold = 0.05
    if baseline == 0:
        direction = "inconclusive"
        reason = "baseline is zero"
    else:
        ratio = absolute / baseline
        if higher_is_better:
            if ratio > threshold:
                direction = "improvement"
            elif ratio < -threshold:
                direction = "regression"
            else:
                direction = "similar"
        else:
            if ratio < -threshold:
                direction = "improvement"
            elif ratio > threshold:
                direction = "regression"
            else:
                direction = "similar"
        reason = "higher is better" if higher_is_better else "lower is better"
    return {
        "baseline": round(baseline, 6),
        "candidate": round(candidate, 6),
        "absolute_change": round(absolute, 6),
        "percent_change": round(percent, 2) if percent is not None else None,
        "direction": direction,
        "reason": reason,
    }


def _summary_item(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "endpoint": sanitize_endpoint(str(data.get("endpoint") or "")),
        "model": data.get("model"),
        "test_type": data.get("test_type"),
        "streaming_requested": data.get("streaming_requested"),
        "streaming_state": data.get("streaming_state"),
        "successful_runs": data.get("successful_runs"),
        "failed_runs": data.get("failed_runs"),
        "readiness_category": data.get("readiness_category"),
        "metric_quality": data.get("metric_quality") if isinstance(data.get("metric_quality"), dict) else {},
    }


def _compatibility(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    checks = (
        ("endpoint", "Endpoint differs"),
        ("model", "Model differs"),
        ("test_type", "Test type differs"),
        ("streaming_requested", "Streaming request mode differs"),
    )
    for key, message in checks:
        if baseline.get(key) != candidate.get(key):
            warnings.append("{0}: baseline={1!r}, candidate={2!r}.".format(message, baseline.get(key), candidate.get(key)))
    if baseline.get("streaming_state") != candidate.get("streaming_state"):
        warnings.append(
            "Observed streaming state differs: baseline={0!r}, candidate={1!r}.".format(
                baseline.get("streaming_state"), candidate.get("streaming_state")
            )
        )
    baseline_runs = (_number(baseline.get("successful_runs")) or 0.0) + (_number(baseline.get("failed_runs")) or 0.0)
    candidate_runs = (_number(candidate.get("successful_runs")) or 0.0) + (_number(candidate.get("failed_runs")) or 0.0)
    if baseline_runs != candidate_runs:
        warnings.append("Run count differs: baseline={0:g}, candidate={1:g}.".format(baseline_runs, candidate_runs))
    baseline_quality = baseline.get("metric_quality") if isinstance(baseline.get("metric_quality"), dict) else {}
    candidate_quality = candidate.get("metric_quality") if isinstance(candidate.get("metric_quality"), dict) else {}
    if baseline_quality != candidate_quality:
        warnings.append("Metric quality differs: baseline={0!r}, candidate={1!r}.".format(baseline_quality, candidate_quality))
    return not warnings, warnings


def compare_performance(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    comparable, compatibility_warnings = _compatibility(baseline, candidate)
    metric_changes: Dict[str, Dict[str, Any]] = {}
    for key in LOWER_IS_BETTER:
        metric_changes[key] = _change(_metric_value(baseline, key), _metric_value(candidate, key), higher_is_better=False)
    for key in HIGHER_IS_BETTER:
        metric_changes[key] = _change(_metric_value(baseline, key), _metric_value(candidate, key), higher_is_better=True)

    improvements = [key for key, value in metric_changes.items() if value["direction"] == "improvement"]
    regressions = [key for key, value in metric_changes.items() if value["direction"] == "regression"]
    if not comparable:
        verdict = "inconclusive"
    elif regressions and not improvements:
        verdict = "regression"
    elif improvements and not regressions:
        verdict = "improvement"
    elif improvements and regressions:
        verdict = "mixed"
    else:
        verdict = "similar"

    observations = _observations(metric_changes, compatibility_warnings)
    suggestions = _suggestions(verdict, metric_changes, compatibility_warnings)
    return {
        "schema_version": COMPARE_SCHEMA_VERSION,
        "timestamp": _now(),
        "baseline": _summary_item(baseline),
        "candidate": _summary_item(candidate),
        "compatible": comparable,
        "verdict": verdict,
        "metric_changes": metric_changes,
        "observations": observations,
        "suggestions": suggestions,
        "warnings": compatibility_warnings,
    }


def compare_performance_files(baseline_path: str, candidate_path: str) -> Dict[str, Any]:
    return compare_performance(load_report_or_baseline(baseline_path), load_report_or_baseline(candidate_path))


def _observations(metric_changes: Dict[str, Dict[str, Any]], warnings: List[str]) -> List[str]:
    observations: List[str] = []
    if warnings:
        observations.append("The two inputs are not directly comparable; review compatibility warnings first.")
    labels = {**LOWER_IS_BETTER, **HIGHER_IS_BETTER}
    for key, label in labels.items():
        change = metric_changes[key]
        if change["direction"] in ("improvement", "regression"):
            pct = change["percent_change"]
            observations.append("{0}: {1} ({2:+.2f}% change).".format(label, change["direction"], pct or 0.0))
    if not observations:
        observations.append("No clear performance movement was detected from the available metrics.")
    return observations


def _suggestions(verdict: str, metric_changes: Dict[str, Dict[str, Any]], warnings: List[str]) -> List[str]:
    if warnings:
        return ["Re-run both smoke tests with the same endpoint, model, test type, streaming mode, runs, and timeout before drawing conclusions."]
    suggestions: List[str] = []
    if metric_changes["ttft_seconds"]["direction"] == "regression":
        suggestions.append("TTFT regressed; check cold start, context size, streaming behavior, and endpoint path changes.")
    if metric_changes["generation_tokens_per_second"]["direction"] == "regression":
        suggestions.append("Generation speed regressed; verify the model size, quantization, GPU usage, and runtime configuration.")
    if metric_changes["success_rate"]["direction"] == "regression":
        suggestions.append("Successful-run rate regressed; inspect timeouts, endpoint stability, and server logs.")
    if verdict == "improvement":
        suggestions.append("Keep the changed configuration and verify once more with the same bounded runs before sharing results.")
    if not suggestions:
        suggestions.append("Use inferdoctor optimize plan --baseline before.json --candidate after.json for prioritized next steps.")
    return suggestions[:3]


def render_comparison_json(comparison: Dict[str, Any]) -> str:
    return json.dumps(comparison, indent=2, sort_keys=True)


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return "{0:.3f}{1}".format(value, suffix)
    return "{0}{1}".format(value, suffix)


def render_comparison_console(comparison: Dict[str, Any]) -> str:
    lines = [
        "InferDoctor Performance Comparison",
        "=" * 57,
        "Verdict: {0}".format(comparison.get("verdict", "inconclusive")),
        "Comparable: {0}".format("yes" if comparison.get("compatible") else "no"),
        "",
        "Metric                         Baseline        Candidate       Change          Result",
        "-" * 86,
    ]
    labels = {**LOWER_IS_BETTER, **HIGHER_IS_BETTER}
    for key, label in labels.items():
        change = comparison["metric_changes"][key]
        pct = change.get("percent_change")
        pct_text = "unknown" if pct is None else "{0:+.2f}%".format(pct)
        lines.append(
            "{0:<30} {1:<15} {2:<15} {3:<15} {4}".format(
                label,
                _fmt(change.get("baseline")),
                _fmt(change.get("candidate")),
                pct_text,
                change.get("direction"),
            )
        )
    if comparison.get("warnings"):
        lines.extend(["", "Compatibility warnings:"])
        lines.extend("- {0}".format(item) for item in comparison["warnings"])
    lines.extend(["", "Top observations:"])
    lines.extend("- {0}".format(item) for item in comparison.get("observations", []))
    lines.extend(["", "Suggested next action:"])
    lines.extend("- {0}".format(item) for item in comparison.get("suggestions", []))
    lines.extend(["", "Note: This compares bounded smoke-test outputs, not benchmark results."])
    return "\n".join(lines)


def render_comparison_markdown(comparison: Dict[str, Any]) -> str:
    lines = [
        "# InferDoctor Performance Comparison",
        "",
        "- Verdict: **{0}**".format(comparison.get("verdict", "inconclusive")),
        "- Comparable: `{0}`".format("yes" if comparison.get("compatible") else "no"),
        "",
        "| Metric | Baseline | Candidate | Change | Result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    labels = {**LOWER_IS_BETTER, **HIGHER_IS_BETTER}
    for key, label in labels.items():
        change = comparison["metric_changes"][key]
        pct = change.get("percent_change")
        pct_text = "unknown" if pct is None else "{0:+.2f}%".format(pct)
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} |".format(
                label,
                _fmt(change.get("baseline")),
                _fmt(change.get("candidate")),
                pct_text,
                change.get("direction"),
            )
        )
    if comparison.get("warnings"):
        lines.extend(["", "## Compatibility warnings", ""])
        lines.extend("- {0}".format(item) for item in comparison["warnings"])
    lines.extend(["", "## Observations", ""])
    lines.extend("- {0}".format(item) for item in comparison.get("observations", []))
    lines.extend(["", "## Suggested next action", ""])
    lines.extend("- {0}".format(item) for item in comparison.get("suggestions", []))
    lines.extend(["", "This compares bounded smoke-test outputs, not benchmark results."])
    return "\n".join(lines)


def render_comparison(comparison: Dict[str, Any], output_format: str = "console") -> str:
    if output_format == "json":
        return render_comparison_json(comparison)
    if output_format == "markdown":
        return render_comparison_markdown(comparison)
    return render_comparison_console(comparison)
