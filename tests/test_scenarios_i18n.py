from dataclasses import asdict
from unittest.mock import patch

import pytest

from inferdoctor.cli import main
from inferdoctor.core.config import Config
from inferdoctor.core.models import CheckResult, Status
from inferdoctor.core.scenarios import (
    ScenarioResult,
    evaluate_scenarios,
    render_scenarios,
)
from inferdoctor.i18n import t


@pytest.mark.parametrize("status", [Status.PASS, Status.WARN, Status.FAIL, Status.SKIP])
def test_scenarios_english_output_is_byte_identical(monkeypatch, status):
    scenario = ScenarioResult(
        name="local-chatbot",
        title="Local chatbot",
        status=status,
        reason="System and runtime checks completed.",
        next_step="Run inferdoctor check ollama --verbose.",
    )

    expected = "\n".join(
        [
            "InferDoctor Scenario Readiness",
            "=" * 57,
            "Goal-oriented diagnosis for common local AI setups.",
            "",
            "Local chatbot:",
            "  Status: {0}".format(status.value.upper()),
            "  Reason: System and runtime checks completed.",
            "  Next step: Run inferdoctor check ollama --verbose.",
        ]
    )

    rendered_en = render_scenarios([scenario], language="en")
    assert rendered_en == expected

    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    rendered_auto = render_scenarios([scenario])
    assert rendered_auto == expected


@pytest.mark.parametrize(
    "language, title, subtitle, status_prefix, reason_prefix, next_step_prefix",
    [
        (
            "en",
            "InferDoctor Scenario Readiness",
            "Goal-oriented diagnosis for common local AI setups.",
            "  Status: ",
            "  Reason: ",
            "  Next step: ",
        ),
        (
            "zh",
            "InferDoctor 场景就绪度",
            "针对常见本地 AI 设置的目标导向诊断。",
            "  状态：",
            "  原因：",
            "  下一步：",
        ),
        (
            "ja",
            "InferDoctor シナリオ準備状況",
            "一般的なローカルAI構成向けの目的別診断。",
            "  ステータス: ",
            "  理由: ",
            "  次のステップ: ",
        ),
    ],
)
@pytest.mark.parametrize("status", [Status.PASS, Status.WARN, Status.FAIL, Status.SKIP])
def test_scenarios_localization_and_shared_data_preservation(
    language,
    title,
    subtitle,
    status_prefix,
    reason_prefix,
    next_step_prefix,
    status,
):
    scenario = ScenarioResult(
        name="gpu-inference",
        title="GPU inference",
        status=status,
        reason="NVIDIA GPU is available; CUDA toolkit may be optional for bundled runtimes.",
        next_step="If you build or run CUDA-dependent kernels, install CUDA toolkit; otherwise test your runtime directly.",
    )
    original = asdict(scenario)
    expected_status = t("console_status_" + status.value, language)

    rendered = render_scenarios([scenario], language=language)

    assert rendered.startswith("{0}\n{1}\n{2}\n\n".format(title, "=" * 57, subtitle))
    assert "GPU inference:" in rendered
    assert "{0}{1}".format(status_prefix, expected_status) in rendered
    assert "{0}{1}".format(reason_prefix, scenario.reason) in rendered
    assert "{0}{1}".format(next_step_prefix, scenario.next_step) in rendered
    assert asdict(scenario) == original


@pytest.mark.parametrize(
    "locale, expected_title, expected_status_line",
    [
        ("en_US.UTF-8", "InferDoctor Scenario Readiness", "  Status: PASS"),
        ("zh_CN.UTF-8", "InferDoctor 场景就绪度", "  状态：通过"),
        ("ja_JP.UTF-8", "InferDoctor シナリオ準備状況", "  ステータス: 成功"),
    ],
)
def test_scenarios_auto_language_follows_locale(
    monkeypatch, locale, expected_title, expected_status_line
):
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", locale)

    scenarios = evaluate_scenarios(
        [CheckResult(name="system", status=Status.PASS, summary="System ok")],
        "cpu-only-fallback",
    )
    rendered = render_scenarios(scenarios)

    assert expected_title in rendered
    assert "CPU-only fallback:" in rendered
    assert expected_status_line in rendered


@pytest.mark.parametrize("command", ["scenario", "scenarios"])
@pytest.mark.parametrize(
    "language, expected_title, expected_status_line, expected_next_step_prefix",
    [
        ("zh", "InferDoctor 场景就绪度", "  状态：通过", "  下一步："),
        ("ja", "InferDoctor シナリオ準備状況", "  ステータス: 成功", "  次のステップ: "),
    ],
)
def test_scenario_cli_uses_config_language(
    capsys,
    command,
    language,
    expected_title,
    expected_status_line,
    expected_next_step_prefix,
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
        exit_code = main([command, "cpu-only-fallback"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert expected_title in output
    assert "CPU-only fallback:" in output
    assert expected_status_line in output
    assert expected_next_step_prefix in output
    mocked.assert_called_once_with(None, None, None, None)
