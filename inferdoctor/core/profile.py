from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List
from urllib.parse import urlsplit, urlunsplit

from inferdoctor import __version__
from inferdoctor.core.config import Config
from inferdoctor.core.health import recommend_fixes
from inferdoctor.core.models import CheckResult
from inferdoctor.i18n import t

SENSITIVE_KEY_PARTS = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "bearer",
)

COMMAND_PATH_KEYS = {
    "nvidia_smi_path": "nvidia-smi",
    "nvcc_path": "nvcc",
    "ollama_path": "ollama",
    "docker_path": "docker",
}


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def _redact_path(value: str) -> str:
    if value.startswith("/home/") or value.startswith("/root/"):
        parts = value.split("/")
        return "~/.../{0}".format(parts[-1]) if parts[-1] else "~/..."
    return value


def _redact_url(value: str) -> str:
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value

    host = parts.hostname or parts.netloc
    if parts.port:
        host = "{0}:{1}".format(host, parts.port)
    path = _redact_path(parts.path)
    query = "<redacted>" if parts.query else ""
    return urlunsplit((parts.scheme, host, path, query, ""))


def redact_value(value: Any, key: str = "") -> Any:
    if _is_sensitive_key(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): redact_value(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(item, key) for item in value]
    if isinstance(value, tuple):
        return [redact_value(item, key) for item in value]
    if isinstance(value, str):
        if "authorization:" in value.lower() or "bearer " in value.lower():
            return "<redacted>"
        if value.startswith("http://") or value.startswith("https://"):
            return _redact_url(value)
        return _redact_path(value)
    return value


def _memory_gib(value: Any) -> Any:
    if not isinstance(value, int):
        return None
    return round(value / (1024 ** 3), 1)


def _result_map(results: Iterable[CheckResult]) -> Dict[str, CheckResult]:
    return {result.name: result for result in results}


def _system_profile(results: Dict[str, CheckResult]) -> Dict[str, Any]:
    raw = results.get("system").raw_data if results.get("system") else {}
    memory = raw.get("available_memory_bytes")
    return {
        "os": redact_value(raw.get("os")),
        "python_version": raw.get("python_version"),
        "architecture": raw.get("architecture"),
        "available_memory_bytes": memory,
        "available_memory_gib": _memory_gib(memory),
    }


def _gpu_profile(results: Dict[str, CheckResult]) -> List[Dict[str, Any]]:
    raw = results.get("nvidia").raw_data if results.get("nvidia") else {}
    gpus = raw.get("gpus") if isinstance(raw, dict) else None
    if not isinstance(gpus, list):
        return []
    safe_gpus = []
    for gpu in gpus:
        if not isinstance(gpu, dict):
            continue
        memory_mib = gpu.get("memory_total_mib")
        safe_gpus.append(
            {
                "name": gpu.get("name"),
                "memory_total_mib": memory_mib,
                "memory_total_gib": round(memory_mib / 1024, 1)
                if isinstance(memory_mib, int)
                else None,
                "driver_version": gpu.get("driver_version"),
            }
        )
    return safe_gpus


def _command_profile(results: Dict[str, CheckResult]) -> Dict[str, Dict[str, Any]]:
    commands: Dict[str, Dict[str, Any]] = {}
    for result in results.values():
        for key, command in COMMAND_PATH_KEYS.items():
            if key in result.raw_data:
                path = result.raw_data.get(key)
                commands[command] = {
                    "available": bool(path),
                    "path": redact_value(path) if path else None,
                }
    for command in ("nvidia-smi", "nvcc", "ollama", "docker"):
        commands.setdefault(command, {"available": False, "path": None})
    return commands


def _endpoint_profile(config: Config) -> Dict[str, str]:
    return {name: redact_value(url) for name, url in sorted(config.endpoints.items())}


def _checker_summary(results: Iterable[CheckResult]) -> List[Dict[str, str]]:
    return [
        {
            "name": result.name,
            "status": result.status.value,
            "summary": redact_value(result.summary),
        }
        for result in results
    ]


def _top_fixes(results: List[CheckResult], config: Config) -> List[Dict[str, Any]]:
    fixes = []
    for fix in recommend_fixes(results, config):
        fixes.append(
            {
                "component": fix.component,
                "issue": redact_value(fix.issue),
                "likely_cause": redact_value(fix.likely_cause),
                "impact": redact_value(fix.impact),
                "next_command": redact_value(fix.next_command),
                "config_hint": redact_value(fix.config_hint),
            }
        )
    return fixes


def build_profile(results: Iterable[CheckResult], config: Config) -> Dict[str, Any]:
    result_list = list(results)
    mapped = _result_map(result_list)
    return {
        "tool": "InferDoctor",
        "version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "safe_to_share": True,
        "redaction_policy": [
            "Sensitive keys such as token, secret, password, api_key, and authorization are redacted.",
            "Home-directory paths are shortened.",
            "Endpoint credentials and query strings are redacted.",
        ],
        "system": _system_profile(mapped),
        "gpus": _gpu_profile(mapped),
        "commands": _command_profile(mapped),
        "configured_endpoints": _endpoint_profile(config),
        "checker_summary": _checker_summary(result_list),
        "top_fixes": _top_fixes(result_list, config),
    }


def render_profile_json(results: Iterable[CheckResult], config: Config) -> str:
    return json.dumps(build_profile(results, config), indent=2, sort_keys=True)


def _format_value(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def render_profile_markdown(results: Iterable[CheckResult], config: Config, language: str = "auto") -> str:
    profile = build_profile(results, config)
    lines = [
        t("profile.title", language),
        "",
        "Generated: `{0}`".format(profile["generated_at"]),
        "",
        "> Safe to share by default: secrets, endpoint credentials, query strings, and home paths are redacted.",
        "",
        t("profile.section_system", language),
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    system = profile["system"]
    for key in ("os", "python_version", "architecture", "available_memory_gib"):
        lines.append("| {0} | {1} |".format(key, _format_value(system.get(key))))

    lines.extend(["", t("profile.section_gpu", language), ""])
    if profile["gpus"]:
        lines.extend(["| GPU | VRAM | Driver |", "| --- | --- | --- |"])
        for gpu in profile["gpus"]:
            lines.append(
                "| {0} | {1} GiB | {2} |".format(
                    _format_value(gpu.get("name")),
                    _format_value(gpu.get("memory_total_gib")),
                    _format_value(gpu.get("driver_version")),
                )
            )
    else:
        lines.append("No NVIDIA GPU data was detected.")

    lines.extend(["", t("profile.section_commands", language), "", "| Command | Available | Path |", "| --- | --- | --- |"])
    for command, data in sorted(profile["commands"].items()):
        lines.append(
            "| {0} | {1} | {2} |".format(
                command,
                "yes" if data.get("available") else "no",
                _format_value(data.get("path")),
            )
        )

    lines.extend(["", t("profile.section_endpoints", language), "", "| Name | URL |", "| --- | --- |"])
    for name, url in profile["configured_endpoints"].items():
        lines.append("| {0} | `{1}` |".format(name, url))

    lines.extend(["", "## Checker Summary", "", "| Check | Status | Summary |", "| --- | --- | --- |"])
    for item in profile["checker_summary"]:
        lines.append(
            "| {0} | **{1}** | {2} |".format(
                item["name"], item["status"].upper(), item["summary"]
            )
        )

    lines.extend(["", "## Top Fixes", ""])
    if profile["top_fixes"]:
        for index, fix in enumerate(profile["top_fixes"], start=1):
            lines.extend(
                [
                    "{0}. **{1}**: {2}".format(index, fix["component"], fix["issue"]),
                    "   - Likely cause: {0}".format(fix["likely_cause"]),
                    "   - Try: `{0}`".format(fix["next_command"]),
                ]
            )
            if fix.get("config_hint"):
                lines.append("   - Config hint: `{0}`".format(fix["config_hint"]))
    else:
        lines.append("No immediate fixes recommended.")
    lines.append("")
    return "\n".join(lines)
