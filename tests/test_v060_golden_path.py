from __future__ import annotations

import json

from inferdoctor.cli import main
from inferdoctor.core.perf import PerfResult, perf_result_to_dict


def _report(endpoint: str = "http://user:secret@127.0.0.1:8000/v1?api_key=hidden", *, ttft: float = 1.2, total: float = 4.0, tps: float = 18.0) -> dict:
    return perf_result_to_dict(
        PerfResult(
            mode="streaming",
            endpoint=endpoint,
            model="local-model",
            reachable=True,
            openai_compatible="yes",
            streaming_supported="confirmed",
            ttft_seconds=ttft,
            total_latency_seconds=total,
            generation_duration_seconds=max(total - ttft, 0.1),
            rough_tokens_per_second=tps,
            output_tokens_estimate=64,
            tps_quality="estimated",
            successful_runs=2,
            failed_runs=0,
            aggregate_metrics={
                "ttft_median": ttft,
                "total_latency_median": total,
                "generation_duration_median": max(total - ttft, 0.1),
                "generation_tps_median": tps,
            },
            metric_quality={"tokens": "estimated", "tps": "estimated"},
            user_experience="Usable with streaming",
            confidence="medium",
            warnings=["bounded smoke test"],
        )
    )


def test_v060_closed_loop_golden_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INFERDOCTOR_BASELINE_DIR", str(tmp_path / "baselines"))
    project = tmp_path / "customer-service-demo"

    assert main(["quickstart", "customer-service", "--preference", "easiest"]) == 0
    assert "InferDoctor Quickstart" in capsys.readouterr().out

    assert main(["template", "create", "customer-service", "--output", str(project)]) == 0
    assert (project / "README.md").exists()

    assert main(["template", "validate", str(project)]) == 0
    validate_output = capsys.readouterr().out
    assert "Project Readiness" in validate_output

    assert main(["template", "smoke-test", str(project)]) == 0
    smoke_output = capsys.readouterr().out
    assert "Template Smoke Test" in smoke_output

    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_report(ttft=1.2, total=4.0, tps=18.0)), encoding="utf-8")
    after.write_text(
        json.dumps(_report("http://127.0.0.1:8000/v1", ttft=0.8, total=3.1, tps=24.0)),
        encoding="utf-8",
    )

    assert main(["perf", "baseline", "create", "--report", str(before), "--name", "before"]) == 0
    create_output = capsys.readouterr().out
    assert "Performance Baseline" in create_output
    assert "secret" not in create_output

    assert main(["perf", "baseline", "show", "before"]) == 0
    show_output = capsys.readouterr().out
    assert "api_key=REDACTED" in show_output
    assert "secret" not in show_output

    assert main(["perf", "baseline", "list"]) == 0
    assert "before" in capsys.readouterr().out

    comparison = tmp_path / "comparison.json"
    assert main(["perf", "compare", str(before), str(after), "--format", "json", "--output", str(comparison)]) == 0
    comparison_data = json.loads(comparison.read_text(encoding="utf-8"))
    assert comparison_data["schema_version"] == "inferdoctor.perf.compare.v1"
    assert comparison_data["verdict"] in {"improvement", "inconclusive"}

    plan = tmp_path / "plan.md"
    assert main([
        "optimize",
        "plan",
        "--baseline",
        str(before),
        "--candidate",
        str(after),
        "--profile",
        "customer-service",
        "--format",
        "markdown",
        "--output",
        str(plan),
    ]) == 0
    plan_text = plan.read_text(encoding="utf-8")
    assert "# InferDoctor Optimization Plan" in plan_text
    assert "Limitations" in plan_text

    assert main(["experience", "profile", "customer-service"]) == 0
    assert "customer-service" in capsys.readouterr().out

    assert main(["perf", "baseline", "delete", "before", "--yes"]) == 0
    assert "Deleted performance baseline" in capsys.readouterr().out
