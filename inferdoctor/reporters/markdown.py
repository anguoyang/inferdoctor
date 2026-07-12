from __future__ import annotations

import json
from typing import Iterable, List

from inferdoctor.core.models import CheckResult
from inferdoctor.i18n import t
from inferdoctor.reporters.common import report_document


def _translated_summary(result: CheckResult, language: str) -> str:
    """Return the localized (or English fallback) summary for a CheckResult."""
    if not result.translation_key:
        return result.summary
    translated = t(
        "checker." + result.translation_key,
        language,
        **(result.translation_args or {}),
    )
    if translated == "checker." + result.translation_key:
        return result.summary
    return translated


def render_markdown(
    results: Iterable[CheckResult], verbose: bool = False, language: str = "auto"
) -> str:
    result_list = list(results)
    document = report_document(result_list)
    lines: List[str] = [
        "# " + t("markdown.title", language),
        "",
        t("markdown.generated_at", language, datetime=document["generated_at"]),
        "",
        t("markdown.table_header", language),
        t("markdown.table_divider", language),
    ]
    for result in result_list:
        summary = _translated_summary(result, language).replace("|", "\\|")
        lines.append(
            t(
                "markdown.table_row",
                language,
                name=result.name,
                status=result.status.value.upper(),
                summary=summary,
            )
        )

    for result in result_list:
        lines.extend(["", t("markdown.section_title", language, name=result.name), "", _translated_summary(result, language)])
        if result.details:
            lines.extend(["", t("markdown.details_title", language), ""])
            lines.extend(t("markdown.list_item", language, item=item) for item in result.details)
        if result.suggestions:
            lines.extend(["", t("markdown.suggestions_title", language), ""])
            lines.extend(t("markdown.list_item", language, item=item) for item in result.suggestions)
        if verbose:
            lines.extend(
                [
                    "",
                    t("markdown.raw_data_title", language),
                    "",
                    "```json",
                    json.dumps(result.raw_data, indent=2, sort_keys=True),
                    "```",
                ]
            )
    lines.append("")
    return "\n".join(lines)
