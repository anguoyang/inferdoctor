import json
from dataclasses import asdict
from unittest.mock import patch

import pytest

from inferdoctor.cli import main
from inferdoctor.core.config import Config
from inferdoctor.core.models import CheckResult, Status
from inferdoctor.i18n import t
from inferdoctor.reporters import render_json, render_markdown


@pytest.mark.parametrize("status", [Status.PASS, Status.WARN, Status.FAIL, Status.SKIP])
def test_markdown_english_output_is_byte_identical(monkeypatch, status):
    fixed_doc = {
        "tool": "InferDoctor",
        "version": "0.8.1",
        "generated_at": "2026-11-09T00:00:00+00:00",
        "results": [],
    }
    monkeypatch.setattr(
        "inferdoctor.reporters.markdown.report_document",
        lambda results: fixed_doc,
    )
    result = CheckResult(
        name="system",
        status=status,
        summary="Shared | diagnostic summary",
        details=["First detail", "Second detail"],
        suggestions=["First suggestion"],
        raw_data={"b": 2, "a": 1},
    )

    expected = "\n".join(
        [
            "# InferDoctor Report",
            "",
            "Generated: `2026-11-09T00:00:00+00:00`",
            "",
            "| Check | Status | Summary |",
            "| --- | --- | --- |",
            "| system | **{0}** | Shared \\| diagnostic summary |".format(
                status.value.upper()
            ),
            "",
            "## system",
            "",
            "Shared | diagnostic summary",
            "",
            "**Details**",
            "",
            "- First detail",
            "- Second detail",
            "",
            "**Suggestions**",
            "",
            "- First suggestion",
            "",
            "**Raw data**",
            "",
            "```json",
            json.dumps({"b": 2, "a": 1}, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )

    rendered_en = render_markdown([result], verbose=True, language="en")
    assert rendered_en == expected

    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    rendered_auto = render_markdown([result], verbose=True)
    assert rendered_auto == expected


@pytest.mark.parametrize(
    "language, title, generated_prefix, header, details_label, suggestions_label, raw_data_label",
    [
        (
            "en",
            "# InferDoctor Report",
            "Generated: ",
            "| Check | Status | Summary |",
            "**Details**",
            "**Suggestions**",
            "**Raw data**",
        ),
        (
            "zh",
            "# InferDoctor 报告",
            "生成时间：",
            "| 检查项 | 状态 | 摘要 |",
            "**详情**",
            "**建议**",
            "**原始数据**",
        ),
        (
            "ja",
            "# InferDoctor レポート",
            "生成日時：",
            "| チェック | ステータス | 概要 |",
            "**詳細**",
            "**提案**",
            "**生データ**",
        ),
    ],
)
@pytest.mark.parametrize("status", [Status.PASS, Status.WARN, Status.FAIL, Status.SKIP])
def test_markdown_localization_and_shared_data_preservation(
    monkeypatch,
    language,
    title,
    generated_prefix,
    header,
    details_label,
    suggestions_label,
    raw_data_label,
    status,
):
    timestamp = "2026-11-09T09:00:00+00:00"
    monkeypatch.setattr(
        "inferdoctor.reporters.markdown.report_document",
        lambda results: {
            "tool": "InferDoctor",
            "version": "0.8.1",
            "generated_at": timestamp,
            "results": [],
        },
    )
    result = CheckResult(
        name="system",
        status=status,
        summary="Shared diagnostic summary",
        details=["Shared detail line"],
        suggestions=["Shared suggestion line"],
        raw_data={"key": "shared_raw_value"},
    )
    original = asdict(result)
    expected_status = t("console_status_" + status.value, language)

    markdown_non_verbose = render_markdown([result], verbose=False, language=language)
    markdown_verbose = render_markdown([result], verbose=True, language=language)

    assert markdown_verbose.startswith(title + "\n\n")
    assert "{0}`{1}`".format(generated_prefix, timestamp) in markdown_verbose
    assert header in markdown_verbose
    assert "| --- | --- | --- |" in markdown_verbose
    assert (
        "| system | **{0}** | Shared diagnostic summary |".format(expected_status)
        in markdown_verbose
    )
    assert "## system\n\nShared diagnostic summary" in markdown_verbose
    assert "\n{0}\n\n- Shared detail line\n".format(details_label) in markdown_verbose
    assert (
        "\n{0}\n\n- Shared suggestion line\n".format(suggestions_label)
        in markdown_verbose
    )
    assert raw_data_label not in markdown_non_verbose
    assert (
        "\n{0}\n\n```json\n{{\n  \"key\": \"shared_raw_value\"\n}}\n```".format(
            raw_data_label
        )
        in markdown_verbose
    )

    assert asdict(result) == original
    report_result = json.loads(render_json([result]))["results"][0]
    assert report_result["name"] == "system"
    assert report_result["status"] == status.value
    assert report_result["summary"] == original["summary"]
    assert report_result["details"] == original["details"]
    assert report_result["suggestions"] == original["suggestions"]
    assert report_result["raw_data"] == original["raw_data"]


@pytest.mark.parametrize(
    "locale, expected_title, expected_status",
    [
        ("en_US.UTF-8", "# InferDoctor Report", "**PASS**"),
        ("zh_CN.UTF-8", "# InferDoctor 报告", "**通过**"),
        ("ja_JP.UTF-8", "# InferDoctor レポート", "**成功**"),
    ],
)
def test_markdown_auto_language_follows_locale(
    monkeypatch, locale, expected_title, expected_status
):
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", locale)

    result = CheckResult(
        name="system",
        status=Status.PASS,
        summary="System information collected",
    )
    rendered = render_markdown([result])
    assert expected_title in rendered
    assert "| system | {0} | System information collected |".format(expected_status) in rendered


@pytest.mark.parametrize(
    "language, expected_title, expected_header, expected_status",
    [
        ("zh", "# InferDoctor 报告", "| 检查项 | 状态 | 摘要 |", "**通过**"),
        ("ja", "# InferDoctor レポート", "| チェック | ステータス | 概要 |", "**成功**"),
    ],
)
def test_report_cli_markdown_uses_config_language(
    capsys, language, expected_title, expected_header, expected_status
):
    sample_result = CheckResult(
        name="system",
        status=Status.PASS,
        summary="System information collected",
    )
    config = Config(language=language)

    with patch(
        "inferdoctor.cli._results_for_target",
        return_value=([sample_result], config),
    ) as mocked:
        exit_code = main(["report", "--format", "markdown"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert expected_title in output
    assert expected_header in output
    assert "| system | {0} | System information collected |".format(expected_status) in output
    mocked.assert_called_once_with(None, None, None, None)
