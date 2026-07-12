from __future__ import annotations

import json

from inferdoctor.cli import main
from inferdoctor.core.perf import PerfResult, perf_result_to_dict
from inferdoctor.core.perf_compare import compare_performance, render_comparison


def _report(*, endpoint="http://127.0.0.1:8000/v1", model="local-model", ttft=1.0, total=4.0, duration=3.0, tps=20.0, success=2, failed=0, streaming="confirmed") -> dict:
    return perf_result_to_dict(
        PerfResult(
            mode="streaming",
            endpoint=endpoint,
            model=model,
            reachable=True,
            openai_compatible="yes",
            streaming_supported=streaming,
            ttft_seconds=ttft,
            total_latency_seconds=total,
            generation_duration_seconds=duration,
            rough_tokens_per_second=tps,
            output_tokens_estimate=60,
            tps_quality="estimated",
            successful_runs=success,
            failed_runs=failed,
            aggregate_metrics={
                "ttft_median": ttft,
                "total_latency_median": total,
                "generation_duration_median": duration,
                "generation_tps_median": tps,
            },
            metric_quality={"tokens": "estimated", "tps": "estimated"},
            user_experience="Usable with streaming",
            confidence="medium",
        )
    )


def test_compare_performance_detects_improvement():
    before = _report(ttft=1.2, total=5.0, duration=3.8, tps=14.0, success=2, failed=1)
    after = _report(ttft=0.7, total=3.0, duration=2.1, tps=26.0, success=3, failed=0)

    comparison = compare_performance(before, after)

    assert comparison["schema_version"] == "inferdoctor.perf.compare.v1"
    assert comparison["compatible"] is True
    assert comparison["verdict"] == "improvement"
    assert comparison["metric_changes"]["ttft_seconds"]["direction"] == "improvement"
    assert comparison["metric_changes"]["generation_tokens_per_second"]["direction"] == "improvement"
    assert comparison["metric_changes"]["success_rate"]["direction"] == "improvement"


def test_compare_performance_warns_for_incompatible_inputs():
    before = _report(endpoint="http://127.0.0.1:8000/v1", model="a")
    after = _report(endpoint="http://127.0.0.1:30000/v1", model="b")

    comparison = compare_performance(before, after)

    assert comparison["compatible"] is False
    assert comparison["verdict"] == "inconclusive"
    assert any("Endpoint differs" in warning for warning in comparison["warnings"])
    assert any("Model differs" in warning for warning in comparison["warnings"])


def test_render_comparison_formats():
    comparison = compare_performance(_report(ttft=1.0), _report(ttft=0.8))

    assert "Performance Comparison" in render_comparison(comparison)
    assert "# InferDoctor Performance Comparison" in render_comparison(comparison, "markdown")
    parsed = json.loads(render_comparison(comparison, "json"))
    assert parsed["schema_version"] == "inferdoctor.perf.compare.v1"


def test_perf_compare_cli_with_positional_paths(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_report(ttft=1.4, total=5.5, tps=12.0)), encoding="utf-8")
    after.write_text(json.dumps(_report(ttft=0.8, total=3.2, tps=24.0)), encoding="utf-8")

    exit_code = main(["perf", "compare", str(before), str(after)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Verdict: improvement" in output
    assert "TTFT" in output


def test_perf_compare_cli_writes_json_output(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    out = tmp_path / "comparison.json"
    before.write_text(json.dumps(_report(ttft=1.4)), encoding="utf-8")
    after.write_text(json.dumps(_report(ttft=0.8)), encoding="utf-8")

    exit_code = main([
        "perf",
        "compare",
        "--baseline",
        str(before),
        "--candidate",
        str(after),
        "--format",
        "json",
        "--output",
        str(out),
    ])

    assert exit_code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["verdict"] == "improvement"


def test_perf_compare_cli_requires_two_inputs(capsys):
    exit_code = main(["perf", "compare", "before.json"])

    assert exit_code == 2
    assert "requires a baseline and candidate" in capsys.readouterr().err


def test_compare_performance_warns_for_different_run_counts_and_metric_quality():
    before = _report(success=1, failed=0)
    after = _report(success=3, failed=0)
    after["metric_quality"] = {"tokens": "exact", "tps": "exact"}

    comparison = compare_performance(before, after)

    assert comparison["compatible"] is False
    assert comparison["verdict"] == "inconclusive"
    assert any("Run count differs" in warning for warning in comparison["warnings"])
    assert any("Metric quality differs" in warning for warning in comparison["warnings"])
