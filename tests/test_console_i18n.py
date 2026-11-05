import json
from dataclasses import asdict

import pytest

from inferdoctor.cli import main
from inferdoctor.core.config import Config
from inferdoctor.core.models import CheckResult, Status
from inferdoctor.reporters import render_console, render_dashboard, render_json, render_markdown


@pytest.mark.parametrize(
    "language, labels, system",
    [
        ("en", ("PASS", "WARN", "FAIL", "SKIP"), "System"),
        ("ja", ("成功", "注意", "失敗", "スキップ"), "システム"),
        ("zh", ("通过", "警告", "失败", "跳过"), "系统"),
    ],
)
@pytest.mark.parametrize("status", [Status.PASS, Status.WARN, Status.FAIL, Status.SKIP])
def test_console_status_localization_preserves_shared_report_data(language, labels, system, status):
    result = CheckResult(
        name="system",
        status=status,
        summary="Shared diagnostic summary",
        details=["Shared detail"],
        suggestions=["Shared suggestion"],
    )
    original = asdict(result)
    label = dict(zip((Status.PASS, Status.WARN, Status.FAIL, Status.SKIP), labels))[status]

    console = render_console([result], language=language)
    dashboard = render_dashboard([result], Config(), verbose=True, language=language)

    assert console.startswith("[{0:<4}] system     Shared diagnostic summary".format(label))
    assert "{0:<11} {1:<8} Shared diagnostic summary".format(system, label) in dashboard
    assert console in dashboard
    for legend_label, score in zip(labels, (100, 60, 0, 85)):
        assert "{0} {1}".format(legend_label, score) in dashboard
    assert asdict(result) == original
    report_result = json.loads(render_json([result]))["results"][0]
    assert report_result["status"] == status.value
    assert report_result["summary"] == original["summary"]
    assert "| system | **{0}** | Shared diagnostic summary |".format(status.value.upper()) in render_markdown([result])


@pytest.mark.parametrize(
    "locale, expected",
    [("en_US.UTF-8", "System      PASS"), ("ja_JP.UTF-8", "システム        成功")],
)
def test_check_cli_localizes_from_locale(monkeypatch, capsys, locale, expected):
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", locale)

    assert main(["check", "system"]) == 0
    output = capsys.readouterr().out
    assert expected in output
    assert "System information collected" in output


def test_check_explicit_english_overrides_japanese_locale(monkeypatch, capsys):
    monkeypatch.setenv("LANGUAGE", "ja_JP.UTF-8")

    assert main(["check", "system", "--language", "en"]) == 0
    assert "System      PASS" in capsys.readouterr().out
