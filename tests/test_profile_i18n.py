import json
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

import inferdoctor.core.profile as profile_module
from inferdoctor.core.config import Config
from inferdoctor.core.models import CheckResult, Status
from inferdoctor.core.profile import (
    build_profile,
    render_profile_json,
    render_profile_markdown,
)
from inferdoctor.i18n import t


FIXED_TIMESTAMP = "2026-11-10T00:00:00+00:00"


class _FixedDateTime:
    @staticmethod
    def now(tz=None):
        return datetime(2026, 11, 10, 0, 0, 0, tzinfo=timezone.utc)


def _results_with_gpu_and_fix():
    return [
        CheckResult(
            name="system",
            status=Status.PASS,
            summary="System information collected",
            raw_data={
                "os": "Linux-test",
                "python_version": "3.12.0",
                "architecture": "x86_64",
                "available_memory_bytes": 8 * 1024 ** 3,
            },
        ),
        CheckResult(
            name="nvidia",
            status=Status.PASS,
            summary="1 NVIDIA GPU(s) detected",
            raw_data={
                "nvidia_smi_path": "/home/alice/bin/nvidia-smi",
                "gpus": [
                    {
                        "name": "NVIDIA RTX 4090",
                        "memory_total_mib": 24576,
                        "driver_version": "550.54.14",
                    }
                ],
            },
        ),
        CheckResult(
            name="ollama",
            status=Status.FAIL,
            summary="Ollama was not found and its API is not reachable",
            raw_data={"ollama_path": None},
        ),
    ]


def _results_without_gpu_or_fixes():
    return [
        CheckResult(
            name="system",
            status=Status.PASS,
            summary="System information collected",
            raw_data={
                "os": "Linux-test",
                "python_version": "3.12.0",
                "architecture": "x86_64",
                "available_memory_bytes": 16 * 1024 ** 3,
            },
        ),
    ]


def test_profile_markdown_english_exact_with_gpus_and_fixes(monkeypatch):
    monkeypatch.setattr(profile_module, "datetime", _FixedDateTime)
    config = Config(
        endpoints={"ollama": "http://user:pass@127.0.0.1:11434?token=abc"},
        language="en",
    )

    rendered = render_profile_markdown(_results_with_gpu_and_fix(), config)

    expected = "\n".join(
        [
            "# InferDoctor Safe Diagnostic Profile",
            "",
            "Generated: `{0}`".format(FIXED_TIMESTAMP),
            "",
            "> Safe to share by default: secrets, endpoint credentials, query strings, and home paths are redacted.",
            "",
            "## System",
            "",
            "| Field | Value |",
            "| --- | --- |",
            "| os | Linux-test |",
            "| python_version | 3.12.0 |",
            "| architecture | x86_64 |",
            "| available_memory_gib | 8.0 |",
            "",
            "## GPUs",
            "",
            "| GPU | VRAM | Driver |",
            "| --- | --- | --- |",
            "| NVIDIA RTX 4090 | 24.0 GiB | 550.54.14 |",
            "",
            "## Commands",
            "",
            "| Command | Available | Path |",
            "| --- | --- | --- |",
            "| docker | no | unknown |",
            "| nvcc | no | unknown |",
            "| nvidia-smi | yes | ~/.../nvidia-smi |",
            "| ollama | no | unknown |",
            "",
            "## Configured Endpoints",
            "",
            "| Name | URL |",
            "| --- | --- |",
            "| ollama | `http://127.0.0.1:11434?<redacted>` |",
            "",
            "## Checker Summary",
            "",
            "| Check | Status | Summary |",
            "| --- | --- | --- |",
            "| system | **PASS** | System information collected |",
            "| nvidia | **PASS** | 1 NVIDIA GPU(s) detected |",
            "| ollama | **FAIL** | Ollama was not found and its API is not reachable |",
            "",
            "## Top Fixes",
            "",
            "1. **Ollama**: Ollama was not found and its API is not reachable",
            "   - Likely cause: The service may not be running, or it may be listening on another port.",
            "   - Try: `inferdoctor check ollama --endpoint http://user:pass@127.0.0.1:11434?token=abc`",
            "   - Config hint: `endpoints.ollama: http://user:pass@127.0.0.1:11434?token=abc`",
            "",
        ]
    )

    assert rendered == expected


def test_profile_markdown_english_exact_without_gpus_or_fixes(monkeypatch):
    monkeypatch.setattr(profile_module, "datetime", _FixedDateTime)
    config = Config(
        endpoints={"ollama": "http://127.0.0.1:11434"},
        language="en",
    )

    rendered = render_profile_markdown(_results_without_gpu_or_fixes(), config)

    expected = "\n".join(
        [
            "# InferDoctor Safe Diagnostic Profile",
            "",
            "Generated: `{0}`".format(FIXED_TIMESTAMP),
            "",
            "> Safe to share by default: secrets, endpoint credentials, query strings, and home paths are redacted.",
            "",
            "## System",
            "",
            "| Field | Value |",
            "| --- | --- |",
            "| os | Linux-test |",
            "| python_version | 3.12.0 |",
            "| architecture | x86_64 |",
            "| available_memory_gib | 16.0 |",
            "",
            "## GPUs",
            "",
            "No NVIDIA GPU data was detected.",
            "",
            "## Commands",
            "",
            "| Command | Available | Path |",
            "| --- | --- | --- |",
            "| docker | no | unknown |",
            "| nvcc | no | unknown |",
            "| nvidia-smi | no | unknown |",
            "| ollama | no | unknown |",
            "",
            "## Configured Endpoints",
            "",
            "| Name | URL |",
            "| --- | --- |",
            "| ollama | `http://127.0.0.1:11434` |",
            "",
            "## Checker Summary",
            "",
            "| Check | Status | Summary |",
            "| --- | --- | --- |",
            "| system | **PASS** | System information collected |",
            "",
            "## Top Fixes",
            "",
            "No immediate fixes recommended.",
            "",
        ]
    )

    assert rendered == expected


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
@pytest.mark.parametrize("status", [Status.PASS, Status.WARN, Status.FAIL, Status.SKIP])
def test_profile_markdown_localizes_labels_and_preserves_data(
    monkeypatch, language, status
):
    monkeypatch.setattr(profile_module, "datetime", _FixedDateTime)
    results = _results_with_gpu_and_fix()
    results[-1] = CheckResult(
        name="ollama",
        status=status,
        summary="Ollama status check summary",
        raw_data={"ollama_path": None},
    )
    originals = [asdict(item) for item in results]
    config = Config(
        endpoints={"ollama": "http://user:pass@127.0.0.1:11434?token=abc"},
        language=language,
    )

    rendered = render_profile_markdown(results, config)

    assert rendered.startswith(t("profile_title", language) + "\n\n")
    assert t("profile_generated", language, generated_at=FIXED_TIMESTAMP) in rendered
    assert t("profile_note", language) in rendered
    assert t("profile_section_system", language) in rendered
    assert t("profile_system_header", language) in rendered
    assert t("profile_section_gpus", language) in rendered
    assert t("profile_gpus_header", language) in rendered
    assert t("profile_section_commands", language) in rendered
    assert t("profile_commands_header", language) in rendered
    assert t("profile_section_endpoints", language) in rendered
    assert t("profile_endpoints_header", language) in rendered
    assert t("profile_section_checker_summary", language) in rendered
    assert t("markdown_header", language) in rendered
    assert t("profile_section_top_fixes", language) in rendered

    # System field identifiers and values stay unchanged across languages
    assert "| os | Linux-test |" in rendered
    assert "| python_version | 3.12.0 |" in rendered
    assert "| architecture | x86_64 |" in rendered
    assert "| available_memory_gib | 8.0 |" in rendered

    # GPU data values stay unchanged across languages
    assert "| NVIDIA RTX 4090 | 24.0 GiB | 550.54.14 |" in rendered

    # Command names and paths stay unchanged; availability yes/no is localized
    yes_label = t("profile_available_yes", language)
    no_label = t("profile_available_no", language)
    assert "| nvidia-smi | {0} | ~/.../nvidia-smi |".format(yes_label) in rendered
    assert "| ollama | {0} | unknown |".format(no_label) in rendered

    # Endpoint name and redacted URL stay unchanged
    assert "| ollama | `http://127.0.0.1:11434?<redacted>` |" in rendered

    # Checker Summary status cells use console_status_* and preserve name/summary
    assert (
        "| ollama | **{0}** | Ollama status check summary |".format(
            t("console_status_" + status.value, language)
        )
        in rendered
    )

    # Top Fixes entry vs no-fixes line
    if status == Status.PASS:
        assert t("profile_no_fixes", language) in rendered
    else:
        assert "1. **Ollama**: Ollama status check summary" in rendered
        assert (
            t(
                "profile_likely_cause",
                language,
                cause="The service may not be running, or it may be listening on another port.",
            )
            in rendered
        )
        assert (
            t(
                "profile_try",
                language,
                command="inferdoctor check ollama --endpoint http://user:pass@127.0.0.1:11434?token=abc",
            )
            in rendered
        )
        assert (
            t(
                "profile_config_hint",
                language,
                config_hint="endpoints.ollama: http://user:pass@127.0.0.1:11434?token=abc",
            )
            in rendered
        )

    assert [asdict(item) for item in results] == originals


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_profile_markdown_localizes_no_gpu_and_no_fixes_lines(monkeypatch, language):
    monkeypatch.setattr(profile_module, "datetime", _FixedDateTime)
    config = Config(
        endpoints={"ollama": "http://127.0.0.1:11434"},
        language=language,
    )

    rendered = render_profile_markdown(_results_without_gpu_or_fixes(), config)

    assert t("profile_no_gpus", language) in rendered
    assert t("profile_no_fixes", language) in rendered


@pytest.mark.parametrize(
    "locale, expected_title, expected_status",
    [
        ("en_US.UTF-8", "# InferDoctor Safe Diagnostic Profile", "**PASS**"),
        ("zh_CN.UTF-8", "# InferDoctor 安全诊断档案", "**通过**"),
        ("ja_JP.UTF-8", "# InferDoctor 安全診断プロファイル", "**成功**"),
    ],
)
def test_profile_markdown_auto_language_follows_locale(
    monkeypatch, locale, expected_title, expected_status
):
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", locale)

    rendered = render_profile_markdown(_results_without_gpu_or_fixes(), Config())
    assert expected_title in rendered
    assert (
        "| system | {0} | System information collected |".format(expected_status)
        in rendered
    )


def test_profile_json_and_build_profile_are_unaffected_by_config_language(monkeypatch):
    monkeypatch.setattr(profile_module, "datetime", _FixedDateTime)
    results = _results_with_gpu_and_fix()

    en_config = Config(
        endpoints={"ollama": "http://user:pass@127.0.0.1:11434?token=abc"},
        language="en",
    )
    zh_config = Config(
        endpoints={"ollama": "http://user:pass@127.0.0.1:11434?token=abc"},
        language="zh",
    )
    ja_config = Config(
        endpoints={"ollama": "http://user:pass@127.0.0.1:11434?token=abc"},
        language="ja",
    )

    en_profile = build_profile(results, en_config)
    zh_profile = build_profile(results, zh_config)
    ja_profile = build_profile(results, ja_config)

    assert zh_profile == en_profile
    assert ja_profile == en_profile

    en_json = render_profile_json(results, en_config)
    zh_json = render_profile_json(results, zh_config)
    ja_json = render_profile_json(results, ja_config)

    assert zh_json == en_json
    assert ja_json == en_json

    parsed = json.loads(en_json)
    assert parsed["redaction_policy"] == [
        "Sensitive keys such as token, secret, password, api_key, and authorization are redacted.",
        "Home-directory paths are shortened.",
        "Endpoint credentials and query strings are redacted.",
    ]
    assert parsed["checker_summary"][0]["status"] == "pass"
    assert parsed["checker_summary"][2]["status"] == "fail"
