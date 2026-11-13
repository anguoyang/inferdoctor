from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from inferdoctor.core.perf import (
    PerfCheck,
    PerfResult,
    PerfRun,
    perf_result_to_dict,
    render_perf_json,
    render_perf_markdown,
    render_perf_result,
)
from inferdoctor.core.perf_baseline import (
    baseline_from_report,
    render_baseline_list,
    render_baseline_markdown,
    render_baseline_summary,
)
from inferdoctor.core.perf_compare import (
    compare_performance,
    render_comparison,
    render_comparison_console,
    render_comparison_json,
    render_comparison_markdown,
)
from inferdoctor.i18n import TRANSLATIONS, t


FIXED_TIMESTAMP = "2026-11-10T00:00:00+00:00"


def _sample_perf_result() -> PerfResult:
    return PerfResult(
        mode="streaming",
        endpoint="http://user:secret@127.0.0.1:8000/v1?api_key=hidden&mode=fast",
        model="qwen2.5-7b-instruct",
        reachable=True,
        openai_compatible="yes",
        streaming_supported="confirmed",
        ttft_seconds=0.45,
        total_latency_seconds=1.80,
        rough_tokens_per_second=24.5,
        output_tokens_estimate=None,
        user_experience="Responsive for interactive use",
        checks=[
            PerfCheck(
                name="models",
                status="PASS",
                summary="/models returned 1 model(s).",
                details=["Probe URL: http://127.0.0.1:8000/v1/models"],
            ),
            PerfCheck(
                name="streaming",
                status="WARN",
                summary="1 measured run(s) succeeded, 1 failed.",
                details=["Second run timed out."],
            ),
        ],
        suggestions=[
            "Keep streaming enabled.",
            "Use a warmup prompt before demos.",
        ],
        raw_data={"endpoint": "http://127.0.0.1:8000/v1"},
        generation_duration_seconds=1.35,
        output_tokens_exact=33,
        output_token_source="api_usage",
        tps_quality="exact",
        successful_runs=1,
        failed_runs=1,
        runs=[
            PerfRun(
                index=1,
                warmup=True,
                success=True,
                streaming_requested=True,
                streaming_observed="confirmed",
                status_code=200,
                ttft_seconds=0.60,
                total_latency_seconds=2.00,
                generation_duration_seconds=1.40,
                output_tokens=33,
                output_token_source="api_usage",
                tokens_per_second=23.5,
                tps_quality="exact",
                content_chunks=5,
            ),
            PerfRun(
                index=2,
                warmup=False,
                success=True,
                streaming_requested=True,
                streaming_observed="confirmed",
                status_code=200,
                ttft_seconds=0.45,
                total_latency_seconds=1.80,
                generation_duration_seconds=1.35,
                output_tokens=33,
                output_token_source="api_usage",
                tokens_per_second=24.5,
                tps_quality="exact",
                content_chunks=6,
            ),
            PerfRun(
                index=3,
                warmup=False,
                success=False,
                streaming_requested=True,
                streaming_observed="failed",
                status_code=500,
                ttft_seconds=None,
                total_latency_seconds=0.90,
                generation_duration_seconds=None,
                output_tokens=None,
                output_token_source="unknown",
                tokens_per_second=None,
                tps_quality="unknown",
                content_chunks=0,
                error="HTTP 500",
            ),
        ],
        aggregate_metrics={
            "ttft_min": 0.45,
            "ttft_median": 0.45,
            "ttft_max": 0.45,
            "total_latency_median": 1.80,
            "generation_duration_median": 1.35,
            "generation_tps_median": 24.5,
        },
        metric_quality={"tokens": "exact", "tps": "exact"},
        experience_explanation="The first non-empty generated content arrived quickly and streaming was confirmed.",
        confidence="medium",
        warnings=["1 malformed SSE event(s) were ignored."],
        errors=["HTTP 500"],
        timestamp=FIXED_TIMESTAMP,
    )


def _sample_baseline_dict() -> dict:
    return {
        "schema_version": "inferdoctor.perf.baseline.v1",
        "created_at": FIXED_TIMESTAMP,
        "name": "before-qwen",
        "source_schema_version": "inferdoctor.perf.v1",
        "source_timestamp": FIXED_TIMESTAMP,
        "endpoint": "http://127.0.0.1:8000/v1?api_key=REDACTED&mode=fast",
        "model": "qwen2.5-7b-instruct",
        "runtime": "vllm",
        "hardware_summary": "Linux x86_64",
        "test_type": "streaming",
        "streaming_requested": True,
        "streaming_state": "confirmed",
        "successful_runs": 2,
        "failed_runs": 0,
        "metrics": {
            "ttft_seconds": 0.45,
            "total_latency_seconds": 1.8,
            "generation_duration_seconds": 1.35,
            "generation_tokens_per_second": 24.5,
            "output_tokens_exact": 33,
            "output_tokens_estimate": None,
            "ttft_min": 0.42,
            "ttft_median": 0.45,
            "ttft_max": 0.48,
            "total_latency_median": 1.8,
            "generation_duration_median": 1.35,
            "generation_tps_median": 24.5,
        },
        "metric_quality": {"tokens": "exact", "tps": "exact"},
        "readiness_category": "Responsive for interactive use",
        "readiness_confidence": "medium",
        "warnings": ["single-machine smoke test"],
        "errors": [],
    }


def _sample_comparison_dict(with_warnings: bool = True) -> dict:
    before = _sample_baseline_dict()
    after = copy.deepcopy(before)
    after["metrics"]["ttft_seconds"] = 0.30
    after["metrics"]["ttft_median"] = 0.30
    after["metrics"]["total_latency_seconds"] = 1.20
    after["metrics"]["total_latency_median"] = 1.20
    after["metrics"]["generation_duration_seconds"] = 0.90
    after["metrics"]["generation_duration_median"] = 0.90
    after["metrics"]["generation_tokens_per_second"] = 36.0
    after["metrics"]["generation_tps_median"] = 36.0
    if with_warnings:
        after["model"] = "qwen2.5-14b-instruct"
    comparison = compare_performance(before, after)
    comparison["timestamp"] = FIXED_TIMESTAMP
    return comparison


def test_translations_table_has_matching_keys_across_languages():
    assert set(TRANSLATIONS["en"]) == set(TRANSLATIONS["zh"])
    assert set(TRANSLATIONS["en"]) == set(TRANSLATIONS["ja"])


def test_render_perf_result_english_byte_identical(monkeypatch):
    result = _sample_perf_result()
    expected = "\n".join(
        [
            "InferDoctor Performance UX Smoke Test",
            "=" * 57,
            "Mode: streaming",
            "Endpoint: http://127.0.0.1:8000/v1?api_key=REDACTED&mode=fast",
            "Model: qwen2.5-7b-instruct",
            "Reachable: yes",
            "OpenAI-compatible check: yes",
            "Streaming observed: confirmed",
            "Successful measured runs: 1",
            "Failed measured runs: 1",
            "TTFT median: 0.45s",
            "Total latency median: 1.80s",
            "Generation duration median: 1.35s",
            "Generation speed: ~24.5 tokens/sec",
            "Token metric quality: exact",
            "User experience read: Responsive for interactive use",
            "Confidence: medium",
            "Why: The first non-empty generated content arrived quickly and streaming was confirmed.",
            "",
            "Aggregate metrics:",
            "- TTFT min/median/max: 0.45s / 0.45s / 0.45s",
            "- Total latency median: 1.80s",
            "- Generation TPS median: ~24.5 tokens/sec",
            "",
            "Warnings:",
            "- 1 malformed SSE event(s) were ignored.",
            "",
            "Errors:",
            "- HTTP 500",
            "",
            "Checks:",
            "- [PASS] models: /models returned 1 model(s).",
            "  - Probe URL: http://127.0.0.1:8000/v1/models",
            "- [WARN] streaming: 1 measured run(s) succeeded, 1 failed.",
            "  - Second run timed out.",
            "",
            "Measured runs:",
            "- Run 2: PASS, streaming=confirmed, TTFT=0.45s, total=1.80s, TPS=~24.5 tokens/sec",
            "- Run 3: FAIL, streaming=failed, TTFT=unknown, total=0.90s, TPS=unknown",
            "",
            "Top optimization suggestions:",
            "1. Keep streaming enabled.",
            "2. Use a warmup prompt before demos.",
            "",
            "Note: This is a timeout-bounded smoke test, not a benchmark. Results vary by prompt, model, runtime, cache state, and hardware.",
        ]
    )

    assert render_perf_result(result, language="en") == expected

    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert render_perf_result(result) == expected


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_render_perf_result_localizes_labels_and_preserves_data(language):
    result = _sample_perf_result()
    original = asdict(result)

    rendered = render_perf_result(result, language=language)

    assert rendered.startswith(t("perf_title", language) + "\n" + "=" * 57 + "\n")
    assert t("perf_mode", language, mode="streaming") in rendered
    assert (
        t(
            "perf_endpoint",
            language,
            endpoint="http://127.0.0.1:8000/v1?api_key=REDACTED&mode=fast",
        )
        in rendered
    )
    assert t("perf_model", language, model="qwen2.5-7b-instruct") in rendered
    assert t("perf_reachable", language, reachable="yes") in rendered
    assert t("perf_openai_compatible", language, openai_compatible="yes") in rendered
    assert t("perf_streaming_observed", language, streaming="confirmed") in rendered
    assert t("perf_successful_runs", language, count=1) in rendered
    assert t("perf_failed_runs", language, count=1) in rendered
    assert t("perf_ttft_median", language, value="0.45s") in rendered
    assert t("perf_total_latency_median", language, value="1.80s") in rendered
    assert t("perf_generation_duration_median", language, value="1.35s") in rendered
    assert t("perf_generation_speed", language, value="~24.5 tokens/sec") in rendered
    assert t("perf_token_metric_quality", language, quality="exact") in rendered
    assert (
        t("perf_user_experience", language, experience="Responsive for interactive use")
        in rendered
    )
    assert t("perf_confidence", language, confidence="medium") in rendered
    assert (
        t(
            "perf_why",
            language,
            explanation="The first non-empty generated content arrived quickly and streaming was confirmed.",
        )
        in rendered
    )
    assert t("perf_aggregate_metrics", language) in rendered
    assert (
        t("perf_aggregate_ttft", language, min="0.45", median="0.45", max="0.45")
        in rendered
    )
    assert t("perf_aggregate_total_latency", language, median="1.80") in rendered
    assert (
        t("perf_aggregate_generation_tps", language, value="~24.5 tokens/sec")
        in rendered
    )
    assert t("perf_warnings", language) in rendered
    assert "- 1 malformed SSE event(s) were ignored." in rendered
    assert t("perf_errors", language) in rendered
    assert "- HTTP 500" in rendered
    assert t("perf_checks", language) in rendered
    assert "- [PASS] models: /models returned 1 model(s)." in rendered
    assert "  - Probe URL: http://127.0.0.1:8000/v1/models" in rendered
    assert "- [WARN] streaming: 1 measured run(s) succeeded, 1 failed." in rendered
    assert t("perf_measured_runs", language) in rendered
    assert (
        t(
            "perf_measured_run_line",
            language,
            index=2,
            status=t("console_status_pass", language),
            streaming="confirmed",
            ttft="0.45s",
            total="1.80s",
            tps="~24.5 tokens/sec",
        )
        in rendered
    )
    assert (
        t(
            "perf_measured_run_line",
            language,
            index=3,
            status=t("console_status_fail", language),
            streaming="failed",
            ttft="unknown",
            total="0.90s",
            tps="unknown",
        )
        in rendered
    )
    assert t("perf_top_suggestions", language) in rendered
    assert "1. Keep streaming enabled." in rendered
    assert "2. Use a warmup prompt before demos." in rendered
    assert t("perf_note", language) in rendered

    assert asdict(result) == original


def test_render_perf_result_preserves_raw_check_status_across_languages():
    base = _sample_perf_result()
    result = PerfResult(
        **{
            **base.__dict__,
            "checks": [
                PerfCheck(
                    name="reachability",
                    status="pass",
                    summary="Endpoint responded",
                )
            ],
        }
    )

    for language in ("en", "zh", "ja"):
        rendered = render_perf_result(result, language=language)
        assert "- [pass] reachability: Endpoint responded" in rendered.splitlines()


def test_render_perf_markdown_english_byte_identical(monkeypatch):
    result = _sample_perf_result()
    expected = "\n".join(
        [
            "# InferDoctor Performance Smoke Test",
            "",
            "- Endpoint: `http://127.0.0.1:8000/v1?api_key=REDACTED&mode=fast`",
            "- Model: `qwen2.5-7b-instruct`",
            "- Test type: `streaming`",
            "- Streaming observed: `confirmed`",
            "- Successful runs: `1`",
            "- Failed runs: `1`",
            "- TTFT median: `0.45s`",
            "- Total latency median: `1.80s`",
            "- Generation speed: `~24.5 tokens/sec`",
            "- Experience: **Responsive for interactive use**",
            "",
            "## Notes",
            "",
            "This is a timeout-bounded smoke test, not a benchmark. Do not use it as a throughput claim.",
            "",
            "## Suggestions",
            "",
            "1. Keep streaming enabled.",
            "2. Use a warmup prompt before demos.",
            "",
            "## Warnings",
            "",
            "- 1 malformed SSE event(s) were ignored.",
            "",
            "## Errors",
            "",
            "- HTTP 500",
        ]
    )

    assert render_perf_markdown(result, language="en") == expected

    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert render_perf_markdown(result) == expected


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_render_perf_markdown_localizes_labels_and_preserves_data(language):
    result = _sample_perf_result()
    original = asdict(result)

    rendered = render_perf_markdown(result, language=language)

    assert rendered.startswith(t("perf_markdown_title", language) + "\n\n")
    assert (
        t(
            "perf_markdown_endpoint",
            language,
            endpoint="http://127.0.0.1:8000/v1?api_key=REDACTED&mode=fast",
        )
        in rendered
    )
    assert t("perf_markdown_model", language, model="qwen2.5-7b-instruct") in rendered
    assert t("perf_markdown_test_type", language, mode="streaming") in rendered
    assert (
        t("perf_markdown_streaming_observed", language, streaming="confirmed")
        in rendered
    )
    assert t("perf_markdown_successful_runs", language, count=1) in rendered
    assert t("perf_markdown_failed_runs", language, count=1) in rendered
    assert t("perf_markdown_ttft_median", language, value="0.45s") in rendered
    assert t("perf_markdown_total_latency_median", language, value="1.80s") in rendered
    assert (
        t("perf_markdown_generation_speed", language, value="~24.5 tokens/sec")
        in rendered
    )
    assert (
        t(
            "perf_markdown_experience",
            language,
            experience="Responsive for interactive use",
        )
        in rendered
    )
    assert t("perf_markdown_section_notes", language) in rendered
    assert t("perf_markdown_note", language) in rendered
    assert t("perf_markdown_section_suggestions", language) in rendered
    assert "1. Keep streaming enabled." in rendered
    assert "2. Use a warmup prompt before demos." in rendered
    assert t("perf_markdown_section_warnings", language) in rendered
    assert "- 1 malformed SSE event(s) were ignored." in rendered
    assert t("perf_markdown_section_errors", language) in rendered
    assert "- HTTP 500" in rendered

    assert asdict(result) == original


def test_perf_json_and_dict_invariant_across_languages(monkeypatch):
    result = _sample_perf_result()
    baseline_dict = perf_result_to_dict(result)
    baseline_json = render_perf_json(result)

    for locale, lang in [
        ("en_US.UTF-8", "en"),
        ("zh_CN.UTF-8", "zh"),
        ("ja_JP.UTF-8", "ja"),
    ]:
        monkeypatch.setenv("LANG", locale)
        render_perf_result(result, language=lang)
        render_perf_markdown(result, language=lang)
        assert perf_result_to_dict(result) == baseline_dict
        assert render_perf_json(result) == baseline_json


def test_render_baseline_summary_english_byte_identical(monkeypatch):
    baseline = _sample_baseline_dict()
    expected_with_path = "\n".join(
        [
            "InferDoctor Performance Baseline",
            "=" * 57,
            "Name: before-qwen",
            "Path: /tmp/baselines/before-qwen.json",
            "Endpoint: http://127.0.0.1:8000/v1?api_key=REDACTED&mode=fast",
            "Model: qwen2.5-7b-instruct",
            "Runtime: vllm",
            "Test type: streaming",
            "Streaming: confirmed",
            "TTFT: 0.45",
            "Total latency: 1.8",
            "Generation duration: 1.35",
            "Generation TPS: 24.5",
            "Successful runs: 2",
            "Failed runs: 0",
            "Readiness: Responsive for interactive use",
            "",
            "Warnings:",
            "- single-machine smoke test",
        ]
    )

    assert (
        render_baseline_summary(
            baseline, path="/tmp/baselines/before-qwen.json", language="en"
        )
        == expected_with_path
    )

    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert (
        render_baseline_summary(baseline, path="/tmp/baselines/before-qwen.json")
        == expected_with_path
    )

    without_path = render_baseline_summary(baseline, path=None, language="en")
    assert "Path: not saved" in without_path


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_render_baseline_summary_localizes_labels_and_preserves_data(language):
    baseline = _sample_baseline_dict()
    original = copy.deepcopy(baseline)
    path = Path("/tmp/baselines/before-qwen.json")

    rendered = render_baseline_summary(baseline, path=path, language=language)

    assert rendered.startswith(
        t("perf_baseline_title", language) + "\n" + "=" * 57 + "\n"
    )
    assert t("perf_baseline_name", language, name="before-qwen") in rendered
    assert t("perf_baseline_path", language, path=path) in rendered
    assert (
        t(
            "perf_baseline_endpoint",
            language,
            endpoint="http://127.0.0.1:8000/v1?api_key=REDACTED&mode=fast",
        )
        in rendered
    )
    assert t("perf_baseline_model", language, model="qwen2.5-7b-instruct") in rendered
    assert t("perf_baseline_runtime", language, runtime="vllm") in rendered
    assert t("perf_baseline_test_type", language, test_type="streaming") in rendered
    assert t("perf_baseline_streaming", language, streaming="confirmed") in rendered
    assert t("perf_baseline_ttft", language, value=0.45) in rendered
    assert t("perf_baseline_total_latency", language, value=1.8) in rendered
    assert t("perf_baseline_generation_duration", language, value=1.35) in rendered
    assert t("perf_baseline_generation_tps", language, value=24.5) in rendered
    assert t("perf_baseline_successful_runs", language, count=2) in rendered
    assert t("perf_baseline_failed_runs", language, count=0) in rendered
    assert (
        t(
            "perf_baseline_readiness",
            language,
            readiness="Responsive for interactive use",
        )
        in rendered
    )
    assert t("perf_baseline_warnings", language) in rendered
    assert "- single-machine smoke test" in rendered

    rendered_unsaved = render_baseline_summary(baseline, path=None, language=language)
    assert t("perf_baseline_path", language, path="not saved") in rendered_unsaved

    assert baseline == original


def test_render_baseline_list_english_byte_identical(tmp_path, monkeypatch):
    expected_empty = "\n".join(
        [
            "InferDoctor Performance Baselines",
            "=" * 57,
            "No baselines found.",
            "Create one with: inferdoctor perf baseline create --report perf.json --name before",
        ]
    )
    assert render_baseline_list([], language="en") == expected_empty

    valid_path = tmp_path / "before-qwen.json"
    valid_path.write_text(json.dumps(_sample_baseline_dict()), encoding="utf-8")
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{not-valid-json", encoding="utf-8")

    expected_populated = "\n".join(
        [
            "InferDoctor Performance Baselines",
            "=" * 57,
            "- before-qwen: qwen2.5-7b-instruct | streaming | {0}".format(valid_path),
            "- {0}: unreadable baseline".format(broken_path),
        ]
    )
    assert (
        render_baseline_list([valid_path, broken_path], language="en")
        == expected_populated
    )

    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert render_baseline_list([valid_path, broken_path]) == expected_populated


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_render_baseline_list_localizes_labels_and_preserves_data(tmp_path, language):
    empty_rendered = render_baseline_list([], language=language)
    assert empty_rendered.startswith(
        t("perf_baseline_list_title", language) + "\n" + "=" * 57 + "\n"
    )
    assert t("perf_baseline_none", language) in empty_rendered
    assert t("perf_baseline_create_hint", language) in empty_rendered
    assert (
        "inferdoctor perf baseline create --report perf.json --name before"
        in empty_rendered
    )

    valid_path = tmp_path / "before-qwen.json"
    baseline_payload = json.dumps(_sample_baseline_dict())
    valid_path.write_text(baseline_payload, encoding="utf-8")
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{not-valid-json", encoding="utf-8")

    populated_rendered = render_baseline_list(
        [valid_path, broken_path], language=language
    )
    assert populated_rendered.startswith(
        t("perf_baseline_list_title", language) + "\n" + "=" * 57 + "\n"
    )
    assert (
        "- before-qwen: qwen2.5-7b-instruct | streaming | {0}".format(valid_path)
        in populated_rendered
    )
    assert (
        t("perf_baseline_unreadable", language, path=broken_path) in populated_rendered
    )
    assert valid_path.read_text(encoding="utf-8") == baseline_payload


def test_render_baseline_markdown_english_byte_identical(monkeypatch):
    baseline = _sample_baseline_dict()
    expected = "\n".join(
        [
            "# InferDoctor Performance Baseline",
            "",
            "- Name: `before-qwen`",
            "- Path: `/tmp/baselines/before-qwen.json`",
            "- Endpoint: `http://127.0.0.1:8000/v1?api_key=REDACTED&mode=fast`",
            "- Model: `qwen2.5-7b-instruct`",
            "- Test type: `streaming`",
            "- Streaming: `confirmed`",
            "- TTFT: `0.45`",
            "- Total latency: `1.8`",
            "- Generation TPS: `24.5`",
            "- Readiness: **Responsive for interactive use**",
        ]
    )

    assert (
        render_baseline_markdown(
            baseline, path="/tmp/baselines/before-qwen.json", language="en"
        )
        == expected
    )

    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert (
        render_baseline_markdown(baseline, path="/tmp/baselines/before-qwen.json")
        == expected
    )


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_render_baseline_markdown_localizes_labels_and_preserves_data(language):
    baseline = _sample_baseline_dict()
    original = copy.deepcopy(baseline)
    path = "/tmp/baselines/before-qwen.json"

    rendered = render_baseline_markdown(baseline, path=path, language=language)

    assert rendered.startswith(t("perf_baseline_markdown_title", language) + "\n\n")
    assert t("perf_baseline_markdown_name", language, name="before-qwen") in rendered
    assert t("perf_baseline_markdown_path", language, path=path) in rendered
    assert (
        t(
            "perf_baseline_markdown_endpoint",
            language,
            endpoint="http://127.0.0.1:8000/v1?api_key=REDACTED&mode=fast",
        )
        in rendered
    )
    assert (
        t("perf_baseline_markdown_model", language, model="qwen2.5-7b-instruct")
        in rendered
    )
    assert (
        t("perf_baseline_markdown_test_type", language, test_type="streaming")
        in rendered
    )
    assert (
        t("perf_baseline_markdown_streaming", language, streaming="confirmed")
        in rendered
    )
    assert t("perf_baseline_markdown_ttft", language, value=0.45) in rendered
    assert t("perf_baseline_markdown_total_latency", language, value=1.8) in rendered
    assert t("perf_baseline_markdown_generation_tps", language, value=24.5) in rendered
    assert (
        t(
            "perf_baseline_markdown_readiness",
            language,
            readiness="Responsive for interactive use",
        )
        in rendered
    )

    rendered_unsaved = render_baseline_markdown(baseline, path=None, language=language)
    assert t("perf_baseline_markdown_path", language, path="not saved") in rendered_unsaved

    assert baseline == original


def test_render_comparison_console_and_markdown_english_byte_identical(monkeypatch):
    comparison = _sample_comparison_dict(with_warnings=True)

    expected_console = "\n".join(
        [
            "InferDoctor Performance Comparison",
            "=" * 57,
            "Verdict: inconclusive",
            "Comparable: no",
            "",
            "Metric                         Baseline        Candidate       Change          Result",
            "-" * 86,
            "TTFT                           0.450           0.300           -33.33%         improvement",
            "Total latency                  1.800           1.200           -33.33%         improvement",
            "Generation duration            1.350           0.900           -33.33%         improvement",
            "Generation TPS                 24.500          36.000          +46.94%         improvement",
            "Successful-run rate            1.000           1.000           +0.00%          similar",
            "",
            "Compatibility warnings:",
            "- Model differs: baseline='qwen2.5-7b-instruct', candidate='qwen2.5-14b-instruct'.",
            "",
            "Top observations:",
            "- The two inputs are not directly comparable; review compatibility warnings first.",
            "- TTFT: improvement (-33.33% change).",
            "- Total latency: improvement (-33.33% change).",
            "- Generation duration: improvement (-33.33% change).",
            "- Generation TPS: improvement (+46.94% change).",
            "",
            "Suggested next action:",
            "- Re-run both smoke tests with the same endpoint, model, test type, streaming mode, runs, and timeout before drawing conclusions.",
            "",
            "Note: This compares bounded smoke-test outputs, not benchmark results.",
        ]
    )

    expected_markdown = "\n".join(
        [
            "# InferDoctor Performance Comparison",
            "",
            "- Verdict: **inconclusive**",
            "- Comparable: `no`",
            "",
            "| Metric | Baseline | Candidate | Change | Result |",
            "| --- | ---: | ---: | ---: | --- |",
            "| TTFT | 0.450 | 0.300 | -33.33% | improvement |",
            "| Total latency | 1.800 | 1.200 | -33.33% | improvement |",
            "| Generation duration | 1.350 | 0.900 | -33.33% | improvement |",
            "| Generation TPS | 24.500 | 36.000 | +46.94% | improvement |",
            "| Successful-run rate | 1.000 | 1.000 | +0.00% | similar |",
            "",
            "## Compatibility warnings",
            "",
            "- Model differs: baseline='qwen2.5-7b-instruct', candidate='qwen2.5-14b-instruct'.",
            "",
            "## Observations",
            "",
            "- The two inputs are not directly comparable; review compatibility warnings first.",
            "- TTFT: improvement (-33.33% change).",
            "- Total latency: improvement (-33.33% change).",
            "- Generation duration: improvement (-33.33% change).",
            "- Generation TPS: improvement (+46.94% change).",
            "",
            "## Suggested next action",
            "",
            "- Re-run both smoke tests with the same endpoint, model, test type, streaming mode, runs, and timeout before drawing conclusions.",
            "",
            "This compares bounded smoke-test outputs, not benchmark results.",
        ]
    )

    assert render_comparison(comparison, "console", language="en") == expected_console
    assert render_comparison_console(comparison, language="en") == expected_console
    assert render_comparison(comparison, "markdown", language="en") == expected_markdown
    assert render_comparison_markdown(comparison, language="en") == expected_markdown

    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert render_comparison(comparison, "console") == expected_console
    assert render_comparison(comparison, "markdown") == expected_markdown


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_render_comparison_localizes_labels_and_preserves_data(language):
    comparison = _sample_comparison_dict(with_warnings=True)
    original = copy.deepcopy(comparison)

    console = render_comparison(comparison, "console", language=language)
    markdown = render_comparison(comparison, "markdown", language=language)

    # Console labels
    assert console.startswith(
        t("perf_compare_title", language) + "\n" + "=" * 57 + "\n"
    )
    assert t("perf_compare_verdict", language, verdict="inconclusive") in console
    assert t("perf_compare_comparable", language, comparable="no") in console
    assert t("perf_compare_header", language) in console
    assert t("perf_compare_warnings", language) in console
    assert t("perf_compare_observations", language) in console
    assert t("perf_compare_suggestions", language) in console
    assert t("perf_compare_note", language) in console

    # Console data rows and computation outputs preserved
    assert (
        "TTFT                           0.450           0.300           -33.33%         improvement"
        in console
    )
    assert (
        "Generation TPS                 24.500          36.000          +46.94%         improvement"
        in console
    )
    assert (
        "- Model differs: baseline='qwen2.5-7b-instruct', candidate='qwen2.5-14b-instruct'."
        in console
    )
    assert "- TTFT: improvement (-33.33% change)." in console

    # Markdown labels
    assert markdown.startswith(t("perf_compare_markdown_title", language) + "\n\n")
    assert (
        t("perf_compare_markdown_verdict", language, verdict="inconclusive")
        in markdown
    )
    assert (
        t("perf_compare_markdown_comparable", language, comparable="no") in markdown
    )
    assert t("perf_compare_markdown_header", language) in markdown
    assert t("perf_compare_markdown_section_warnings", language) in markdown
    assert t("perf_compare_markdown_section_observations", language) in markdown
    assert t("perf_compare_markdown_section_suggestions", language) in markdown
    assert t("perf_compare_markdown_note", language) in markdown

    # Markdown data rows and computation outputs preserved
    assert "| TTFT | 0.450 | 0.300 | -33.33% | improvement |" in markdown
    assert "| Generation TPS | 24.500 | 36.000 | +46.94% | improvement |" in markdown
    assert (
        "- Model differs: baseline='qwen2.5-7b-instruct', candidate='qwen2.5-14b-instruct'."
        in markdown
    )
    assert "- TTFT: improvement (-33.33% change)." in markdown

    assert comparison == original


@pytest.mark.parametrize("language", ["en", "zh", "ja"])
def test_render_comparison_json_mode_is_invariant_across_languages(
    monkeypatch, language
):
    comparison = _sample_comparison_dict(with_warnings=False)
    original = copy.deepcopy(comparison)
    expected_json = render_comparison_json(comparison)

    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    rendered_json = render_comparison(comparison, "json", language=language)

    assert rendered_json == expected_json
    parsed = json.loads(rendered_json)
    assert parsed["schema_version"] == "inferdoctor.perf.compare.v1"
    assert parsed["verdict"] == "improvement"
    assert parsed["compatible"] is True
    assert comparison == original


@pytest.mark.parametrize(
    "locale, expected_perf_title, expected_perf_md_title, expected_base_title, expected_list_title, expected_base_md_title, expected_cmp_title, expected_cmp_md_title",
    [
        (
            "en_US.UTF-8",
            "InferDoctor Performance UX Smoke Test",
            "# InferDoctor Performance Smoke Test",
            "InferDoctor Performance Baseline",
            "InferDoctor Performance Baselines",
            "# InferDoctor Performance Baseline",
            "InferDoctor Performance Comparison",
            "# InferDoctor Performance Comparison",
        ),
        (
            "zh_CN.UTF-8",
            "InferDoctor 性能体验冒烟测试",
            "# InferDoctor 性能冒烟测试",
            "InferDoctor 性能基线",
            "InferDoctor 性能基线列表",
            "# InferDoctor 性能基线",
            "InferDoctor 性能对比",
            "# InferDoctor 性能对比",
        ),
        (
            "ja_JP.UTF-8",
            "InferDoctor パフォーマンス UX スモークテスト",
            "# InferDoctor パフォーマンススモークテスト",
            "InferDoctor パフォーマンスベースライン",
            "InferDoctor パフォーマンスベースライン一覧",
            "# InferDoctor パフォーマンスベースライン",
            "InferDoctor パフォーマンス比較",
            "# InferDoctor パフォーマンス比較",
        ),
    ],
)
def test_all_six_perf_render_surfaces_auto_language_follows_locale(
    monkeypatch,
    locale,
    expected_perf_title,
    expected_perf_md_title,
    expected_base_title,
    expected_list_title,
    expected_base_md_title,
    expected_cmp_title,
    expected_cmp_md_title,
):
    for variable in ("LANGUAGE", "LC_ALL", "LC_MESSAGES"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("LANG", locale)

    result = _sample_perf_result()
    baseline = baseline_from_report(perf_result_to_dict(result), name="before")
    comparison = _sample_comparison_dict(with_warnings=False)

    assert render_perf_result(result).startswith(expected_perf_title + "\n")
    assert render_perf_markdown(result).startswith(expected_perf_md_title + "\n")
    assert render_baseline_summary(baseline).startswith(expected_base_title + "\n")
    assert render_baseline_list([]).startswith(expected_list_title + "\n")
    assert render_baseline_markdown(baseline).startswith(expected_base_md_title + "\n")
    assert render_comparison(comparison, "console").startswith(
        expected_cmp_title + "\n"
    )
    assert render_comparison(comparison, "markdown").startswith(
        expected_cmp_md_title + "\n"
    )
