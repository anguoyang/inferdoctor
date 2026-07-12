from __future__ import annotations

import json
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from inferdoctor.core.perf import REPORT_SCHEMA_VERSION, sanitize_endpoint

BASELINE_SCHEMA_VERSION = "inferdoctor.perf.baseline.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_baseline_dir() -> Path:
    override = os.environ.get("INFERDOCTOR_BASELINE_DIR")
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME")
    root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return root / "inferdoctor" / "baselines"


def safe_baseline_name(name: Optional[str], report: Dict[str, Any]) -> str:
    raw = name or "{0}-{1}".format(report.get("test_type", "perf"), report.get("model") or "unknown-model")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw).strip()).strip("-._")
    return cleaned or "baseline"


def _read_json(path: str | Path) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError("Could not read JSON file {0}: {1}".format(path, exc)) from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON file {0}: {1}".format(path, exc)) from exc
    if not isinstance(data, dict):
        raise ValueError("JSON document must be an object")
    return data


def load_report_or_baseline(path: str | Path) -> Dict[str, Any]:
    try:
        resolved = resolve_baseline(path)
    except ValueError:
        resolved = Path(path).expanduser()
    data = _read_json(resolved)
    if data.get("schema_version") == BASELINE_SCHEMA_VERSION:
        return data
    return baseline_from_report(data)


def _metrics_from_report(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    aggregate = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), dict) else {}
    return {
        "ttft_seconds": metrics.get("ttft_seconds"),
        "total_latency_seconds": metrics.get("total_latency_seconds"),
        "generation_duration_seconds": metrics.get("generation_duration_seconds"),
        "generation_tokens_per_second": metrics.get("generation_tokens_per_second"),
        "output_tokens_exact": metrics.get("output_tokens_exact"),
        "output_tokens_estimate": metrics.get("output_tokens_estimate"),
        "ttft_min": aggregate.get("ttft_min"),
        "ttft_median": aggregate.get("ttft_median"),
        "ttft_max": aggregate.get("ttft_max"),
        "total_latency_median": aggregate.get("total_latency_median"),
        "generation_duration_median": aggregate.get("generation_duration_median"),
        "generation_tps_median": aggregate.get("generation_tps_median"),
    }


def _is_supported_perf_report_schema(schema_version: str) -> bool:
    if schema_version == REPORT_SCHEMA_VERSION:
        return True
    return schema_version.startswith("inferdoctor.perf.v")


def baseline_from_report(
    report: Dict[str, Any],
    name: Optional[str] = None,
    runtime: Optional[str] = None,
    hardware_summary: Optional[str] = None,
) -> Dict[str, Any]:
    schema_version = str(report.get("schema_version", ""))
    if schema_version == BASELINE_SCHEMA_VERSION:
        baseline = dict(report)
        if name:
            baseline["name"] = safe_baseline_name(name, report)
        if runtime:
            baseline["runtime"] = runtime
        if hardware_summary:
            baseline["hardware_summary"] = hardware_summary
        return baseline
    if schema_version.startswith("inferdoctor.perf.baseline."):
        raise ValueError("Unsupported InferDoctor baseline schema version: {0}".format(schema_version))
    if not _is_supported_perf_report_schema(schema_version):
        raise ValueError("Expected an InferDoctor endpoint or streaming performance report JSON document")
    experience = report.get("experience_read") if isinstance(report.get("experience_read"), dict) else {}
    baseline = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "created_at": _now(),
        "name": safe_baseline_name(name, report),
        "source_schema_version": report.get("schema_version"),
        "source_timestamp": report.get("timestamp"),
        "endpoint": sanitize_endpoint(str(report.get("endpoint") or "")),
        "model": report.get("model"),
        "runtime": runtime,
        "hardware_summary": hardware_summary or "{0} {1}".format(platform.system(), platform.machine()).strip(),
        "test_type": report.get("test_type"),
        "streaming_requested": report.get("streaming_requested"),
        "streaming_state": report.get("streaming_observed"),
        "successful_runs": report.get("successful_runs", 0),
        "failed_runs": report.get("failed_runs", 0),
        "metrics": _metrics_from_report(report),
        "metric_quality": report.get("metric_quality") if isinstance(report.get("metric_quality"), dict) else {},
        "readiness_category": experience.get("category") or report.get("user_experience"),
        "readiness_confidence": experience.get("confidence"),
        "warnings": list(report.get("warnings") or []),
        "errors": list(report.get("errors") or []),
    }
    return baseline


def baseline_path(name: str, directory: Optional[str | Path] = None) -> Path:
    root = Path(directory) if directory else default_baseline_dir()
    return root / (safe_baseline_name(name, {}) + ".json")


def save_baseline(
    baseline: Dict[str, Any],
    output: Optional[str | Path] = None,
    directory: Optional[str | Path] = None,
) -> Path:
    target = Path(output).expanduser() if output else baseline_path(str(baseline.get("name") or "baseline"), directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(baseline, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(payload)
    temp_path.replace(target)
    return target


def create_baseline_from_report_file(
    report_path: str | Path,
    name: Optional[str] = None,
    runtime: Optional[str] = None,
    output: Optional[str | Path] = None,
    directory: Optional[str | Path] = None,
    hardware_summary: Optional[str] = None,
) -> tuple[Dict[str, Any], Path]:
    report = _read_json(report_path)
    baseline = baseline_from_report(report, name=name, runtime=runtime, hardware_summary=hardware_summary)
    return baseline, save_baseline(baseline, output=output, directory=directory)


def list_baselines(directory: Optional[str | Path] = None) -> List[Path]:
    root = Path(directory) if directory else default_baseline_dir()
    if not root.exists():
        return []
    return sorted(path for path in root.glob("*.json") if path.is_file())


def resolve_baseline(identifier: str | Path, directory: Optional[str | Path] = None) -> Path:
    path = Path(identifier).expanduser()
    if path.exists():
        return path
    named = baseline_path(str(identifier), directory)
    if named.exists():
        return named
    raise ValueError("Baseline not found: {0}".format(identifier))


def delete_baseline(identifier: str | Path, directory: Optional[str | Path] = None) -> Path:
    path = resolve_baseline(identifier, directory)
    path.unlink()
    return path


def render_baseline_summary(baseline: Dict[str, Any], path: Optional[str | Path] = None) -> str:
    metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), dict) else {}
    lines = [
        "InferDoctor Performance Baseline",
        "=" * 57,
        "Name: {0}".format(baseline.get("name") or "unnamed"),
        "Path: {0}".format(path) if path else "Path: not saved",
        "Endpoint: {0}".format(baseline.get("endpoint") or "not provided"),
        "Model: {0}".format(baseline.get("model") or "not provided"),
        "Runtime: {0}".format(baseline.get("runtime") or "not provided"),
        "Test type: {0}".format(baseline.get("test_type") or "unknown"),
        "Streaming: {0}".format(baseline.get("streaming_state") or "unknown"),
        "TTFT: {0}".format(metrics.get("ttft_seconds")),
        "Total latency: {0}".format(metrics.get("total_latency_seconds")),
        "Generation duration: {0}".format(metrics.get("generation_duration_seconds")),
        "Generation TPS: {0}".format(metrics.get("generation_tokens_per_second")),
        "Successful runs: {0}".format(baseline.get("successful_runs", 0)),
        "Failed runs: {0}".format(baseline.get("failed_runs", 0)),
        "Readiness: {0}".format(baseline.get("readiness_category") or "unknown"),
    ]
    warnings = baseline.get("warnings") or []
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend("- {0}".format(item) for item in warnings)
    return "\n".join(lines)


def render_baseline_list(paths: List[Path]) -> str:
    lines = ["InferDoctor Performance Baselines", "=" * 57]
    if not paths:
        lines.append("No baselines found.")
        lines.append("Create one with: inferdoctor perf baseline create --report perf.json --name before")
        return "\n".join(lines)
    for path in paths:
        try:
            data = _read_json(path)
            lines.append("- {0}: {1} | {2} | {3}".format(data.get("name") or path.stem, data.get("model") or "model unknown", data.get("test_type") or "test unknown", path))
        except ValueError:
            lines.append("- {0}: unreadable baseline".format(path))
    return "\n".join(lines)


def render_baseline_markdown(baseline: Dict[str, Any], path: Optional[str | Path] = None) -> str:
    metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), dict) else {}
    lines = [
        "# InferDoctor Performance Baseline",
        "",
        "- Name: `{0}`".format(baseline.get("name") or "unnamed"),
        "- Path: `{0}`".format(path or "not saved"),
        "- Endpoint: `{0}`".format(baseline.get("endpoint") or "not provided"),
        "- Model: `{0}`".format(baseline.get("model") or "not provided"),
        "- Test type: `{0}`".format(baseline.get("test_type") or "unknown"),
        "- Streaming: `{0}`".format(baseline.get("streaming_state") or "unknown"),
        "- TTFT: `{0}`".format(metrics.get("ttft_seconds")),
        "- Total latency: `{0}`".format(metrics.get("total_latency_seconds")),
        "- Generation TPS: `{0}`".format(metrics.get("generation_tokens_per_second")),
        "- Readiness: **{0}**".format(baseline.get("readiness_category") or "unknown"),
    ]
    return "\n".join(lines)
