from __future__ import annotations

import json
from textwrap import shorten
from typing import Iterable, List

from inferdoctor.core.config import Config
from inferdoctor.core.health import calculate_health, recommend_fixes
from inferdoctor.core.models import CheckResult
from inferdoctor.i18n import t


DISPLAY_NAMES = {
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


def _status_summary(health, language: str) -> str:
    return t(
        "dashboard_stack_summary",
        language,
        pass_count=health.counts.get("pass", 0),
        warn_count=health.counts.get("warn", 0),
        skip_count=health.counts.get("skip", 0),
        fail_count=health.counts.get("fail", 0),
    )


def _doctor_read(health, language: str) -> str:
    if health.counts.get("fail", 0):
        return t("dashboard_doctor_read_fail", language)
    if health.counts.get("warn", 0):
        return t("dashboard_doctor_read_warn", language)
    if health.counts.get("skip", 0):
        return t("dashboard_doctor_read_skip", language)
    return t("dashboard_doctor_read_pass", language)


def render_console(
    results: Iterable[CheckResult], verbose: bool = False, language: str = "auto"
) -> str:
    lines = []
    for result in results:
        lines.append(
            t(
                "console_status",
                language,
                status=t("console_status_" + result.status.value, language),
                name=result.name,
                summary=result.summary,
            )
        )
        for detail in result.details:
            lines.append(t("console_detail", language, detail=detail))
        for suggestion in result.suggestions:
            lines.append(t("console_suggestion", language, suggestion=suggestion))
        if verbose:
            raw_data = json.dumps(result.raw_data, indent=2, sort_keys=True)
            lines.append(t("console_raw_data", language))
            lines.extend(
                "         {0}".format(line) for line in raw_data.splitlines()
            )
    return "\n".join(lines)


def render_dashboard(
    results: Iterable[CheckResult], config: Config, verbose: bool = False, language: str = "auto"
) -> str:
    result_list = list(results)
    health = calculate_health(result_list)
    lines: List[str] = [
        t("dashboard_title", language),
        "=" * 57,
        t("dashboard_health", language, score=health.score, label=health.label),
        _status_summary(health, language),
        _doctor_read(health, language),
        t("dashboard_scores", language),
        "",
        t("dashboard_header", language),
        t("dashboard_divider", language),
    ]
    for result in result_list:
        lines.append(
            t(
                "dashboard_row",
                language,
                name=(
                    t("console_component_system", language)
                    if result.name == "system"
                    else DISPLAY_NAMES.get(result.name, result.name.title())
                ),
                status=t("console_status_" + result.status.value, language),
                summary=shorten(result.summary, width=50, placeholder="..."),
            )
        )

    fixes = recommend_fixes(result_list, config)
    lines.extend(["", t("dashboard_top_fixes", language)])
    if not fixes:
        lines.append(t("dashboard_no_fixes", language))
    for index, fix in enumerate(fixes, start=1):
        lines.extend(
            [
                t("dashboard_fix", language, index=index, component=fix.component, issue=fix.issue),
                t("dashboard_likely_cause", language, cause=fix.likely_cause),
                t("dashboard_impact", language, impact=fix.impact),
                t("dashboard_try", language, command=fix.next_command),
            ]
        )
        if fix.config_hint:
            hint_label = t(
                "dashboard_hint_config"
                if fix.config_hint.startswith("endpoints.")
                else "dashboard_hint_note",
                language,
            )
            lines.append(
                t("dashboard_config", language, hint_label=hint_label, config_hint=fix.config_hint)
            )

    if verbose:
        lines.extend(["", t("dashboard_detail", language), render_console(result_list, True, language=language)])
    return "\n".join(lines)
