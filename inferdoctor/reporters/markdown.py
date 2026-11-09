from __future__ import annotations

import json
from typing import Iterable, List

from inferdoctor.core.models import CheckResult
from inferdoctor.i18n import t
from inferdoctor.reporters.common import report_document


def render_markdown(
    results: Iterable[CheckResult], verbose: bool = False, language: str = "auto"
) -> str:
    result_list = list(results)
    document = report_document(result_list)
    lines: List[str] = [
        t("markdown_title", language),
        "",
        t("markdown_generated", language, generated_at=document["generated_at"]),
        "",
        t("markdown_header", language),
        "| --- | --- | --- |",
    ]
    for result in result_list:
        summary = result.summary.replace("|", "\\|")
        lines.append(
            "| {0} | **{1}** | {2} |".format(
                result.name,
                t("console_status_" + result.status.value, language),
                summary,
            )
        )

    for result in result_list:
        lines.extend(["", "## {0}".format(result.name), "", result.summary])
        if result.details:
            lines.extend(["", t("markdown_details", language), ""])
            lines.extend("- {0}".format(item) for item in result.details)
        if result.suggestions:
            lines.extend(["", t("markdown_suggestions", language), ""])
            lines.extend("- {0}".format(item) for item in result.suggestions)
        if verbose:
            lines.extend(
                [
                    "",
                    t("markdown_raw_data", language),
                    "",
                    "```json",
                    json.dumps(result.raw_data, indent=2, sort_keys=True),
                    "```",
                ]
            )
    lines.append("")
    return "\n".join(lines)
