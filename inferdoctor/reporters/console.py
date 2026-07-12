from __future__ import annotations

import json
from textwrap import shorten
from typing import Dict, Iterable, List

from inferdoctor.core.config import Config
from inferdoctor.core.health import calculate_health, recommend_fixes
from inferdoctor.core.models import CheckResult
from inferdoctor.i18n import t

DISPLAY_NAMES: Dict[str, str] = {
    "system": "System",
    "nvidia": "NVIDIA",
    "cuda": "CUDA",
    "ollama": "Ollama",
    "vllm": "vLLM",
    "sglang": "SGLang",
    "llamacpp": "llama.cpp",
    "lmstudio": "LM Studio",
    "xinference": "Xinference",
    "dify": "Dify",
    "openwebui": "Open WebUI",
    "docker": "Docker",
}

HEALTH_LABEL_KEYS: Dict[str, str] = {
    "Healthy": "health.label_healthy",
    "Mostly healthy": "health.label_mostly_healthy",
    "Needs attention": "health.label_needs_attention",
    "Unhealthy": "health.label_unhealthy",
}


def _translated_summary(result: CheckResult, language: str) -> str:
    """Return the localized (or English fallback) summary for a CheckResult."""
    if not result.translation_key:
        return result.summary
    translated = t(
        "checker." + result.translation_key,
        language,
        **(result.translation_args or {}),
    )
    # t() returns the key itself if nothing was found — use English fallback
    if translated == "checker." + result.translation_key:
        return result.summary
    return translated


def _status_summary(health, language: str) -> str:
    return t(
        "dashboard.stack_summary",
        language,
        pass_count=health.counts.get("pass", 0),
        warn_count=health.counts.get("warn", 0),
        skip_count=health.counts.get("skip", 0),
        fail_count=health.counts.get("fail", 0),
    )


def _doctor_read(health, language: str) -> str:
    if health.counts.get("fail", 0):
        return t("dashboard.doctor_read_fail", language)
    if health.counts.get("warn", 0):
        return t("dashboard.doctor_read_warn", language)
    if health.counts.get("skip", 0):
        return t("dashboard.doctor_read_skip", language)
    return t("dashboard.doctor_read_pass", language)


def render_console(
    results: Iterable[CheckResult], verbose: bool = False, language: str = "auto"
) -> str:
    lines = []
    for result in results:
        summary = _translated_summary(result, language)
        lines.append(
            t(
                "console.status",
                language,
                status=result.status.value.upper(),
                name=result.name,
                summary=summary,
            )
        )
        for detail in result.details:
            lines.append(t("console.detail", language, detail=detail))
        for suggestion in result.suggestions:
            lines.append(t("console.suggestion", language, suggestion=suggestion))
        if verbose:
            raw_data = json.dumps(result.raw_data, indent=2, sort_keys=True)
            lines.append(t("console.raw_data", language))
            lines.extend(
                "         {0}".format(line) for line in raw_data.splitlines()
            )
    return "\n".join(lines)


def render_dashboard(
    results: Iterable[CheckResult], config: Config, verbose: bool = False, language: str = "auto"
) -> str:
    result_list = list(results)
    health = calculate_health(result_list)
    translated_label = t(
        HEALTH_LABEL_KEYS.get(health.label, ""),
        language,
    ) or health.label
    lines: List[str] = [
        t("dashboard.title", language),
        "=" * 57,
        t("dashboard.health", language, score=health.score, label=translated_label),
        _status_summary(health, language),
        _doctor_read(health, language),
        t("dashboard.scores", language),
        "",
        t("dashboard.header", language),
        t("dashboard.divider", language),
    ]
    for result in result_list:
        summary = _translated_summary(result, language)
        lines.append(
            "{0:<11} {1:<8} {2}".format(
                DISPLAY_NAMES.get(result.name, result.name.title()),
                result.status.value.upper(),
                shorten(summary, width=50, placeholder="..."),
            )
        )

    fixes = recommend_fixes(result_list, config)
    lines.extend(["", t("dashboard.top_fixes", language)])
    if not fixes:
        lines.append(t("dashboard.no_fixes", language))
    for index, fix in enumerate(fixes, start=1):
        lines.extend(
            [
                t("dashboard.fix", language, index=index, component=fix.component, issue=fix.issue),
                t("dashboard.likely_cause", language, cause=fix.likely_cause),
                t("dashboard.impact", language, impact=fix.impact),
                t("dashboard.try", language, command=fix.next_command),
            ]
        )
        if fix.config_hint:
            hint_label = t(
                "dashboard.hint_config"
                if fix.config_hint.startswith("endpoints.")
                else "dashboard.hint_note",
                language,
            )
            lines.append(
                t("dashboard.config", language, hint_label=hint_label, config_hint=fix.config_hint)
            )

    if verbose:
        lines.extend(["", t("dashboard.detail", language), render_console(result_list, True, language=language)])
    return "\n".join(lines)
