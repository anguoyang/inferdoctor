from __future__ import annotations

import json

from inferdoctor.cli import main
from inferdoctor.core.optimization_plan import build_optimization_plan, render_optimization_plan
from inferdoctor.core.perf import PerfResult, perf_result_to_dict


def _report(ttft=2.8, total=7.0, tps=14.0):
    return perf_result_to_dict(
        PerfResult(
            mode="streaming",
            endpoint="http://127.0.0.1:8000/v1",
            model="local-model",
            reachable=True,
            openai_compatible="yes",
            streaming_supported="confirmed",
            ttft_seconds=ttft,
            total_latency_seconds=total,
            generation_duration_seconds=max(total - ttft, 0.1),
            rough_tokens_per_second=tps,
            successful_runs=1,
            failed_runs=0,
            aggregate_metrics={"ttft_median": ttft, "total_latency_median": total, "generation_tps_median": tps},
            metric_quality={"tokens": "estimated", "tps": "estimated"},
            user_experience="Likely frustrating without progress feedback",
            confidence="medium",
        )
    )


def test_build_optimization_plan_from_supplied_metrics():
    plan = build_optimization_plan(
        runtime="vllm",
        model_size="14b",
        vram_gib=24,
        goal="customer-service",
        streaming=False,
        retrieval_ms=900,
        ttft=2.5,
    )

    assert plan["schema_version"] == "inferdoctor.optimize.plan.v1"
    assert plan["goal"] == "customer-service"
    priorities = {action["priority"] for action in plan["actions"]}
    assert "Do now" in priorities
    assert "Test next" in priorities
    assert any("TTFT" in action["observation"] for action in plan["actions"])
    assert any("Retrieval" in action["observation"] for action in plan["actions"])


def test_render_optimization_plan_formats():
    plan = build_optimization_plan(runtime="ollama", streaming=True)

    assert "InferDoctor Optimization Plan" in render_optimization_plan(plan)
    assert "# InferDoctor Optimization Plan" in render_optimization_plan(plan, "markdown")
    parsed = json.loads(render_optimization_plan(plan, "json"))
    assert parsed["schema_version"] == "inferdoctor.optimize.plan.v1"


def test_optimize_plan_cli_from_report(tmp_path, capsys):
    report = tmp_path / "perf.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")

    exit_code = main(["optimize", "plan", "--report", str(report), "--goal", "customer-service", "--runtime", "vllm"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "InferDoctor Optimization Plan" in output
    assert "Loaded performance smoke-test report" in output
    assert "Do now" in output


def test_optimize_plan_cli_from_comparison(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    out = tmp_path / "plan.md"
    before.write_text(json.dumps(_report(ttft=1.0, total=4.0, tps=30.0)), encoding="utf-8")
    after.write_text(json.dumps(_report(ttft=2.4, total=6.0, tps=12.0)), encoding="utf-8")

    exit_code = main([
        "optimize",
        "plan",
        "--baseline",
        str(before),
        "--candidate",
        str(after),
        "--format",
        "markdown",
        "--output",
        str(out),
    ])

    assert exit_code == 0
    text = out.read_text(encoding="utf-8")
    assert "# InferDoctor Optimization Plan" in text
    assert "TTFT regressed" in text


def test_optimize_plan_requires_baseline_and_candidate_together(capsys):
    exit_code = main(["optimize", "plan", "--baseline", "before.json"])

    assert exit_code == 2
    assert "requires both --baseline and --candidate" in capsys.readouterr().err


def test_optimization_plan_actions_include_evidence_level():
    plan = build_optimization_plan(runtime="vllm", model_size="14b", vram_gib=24, ttft=2.4)

    assert plan["actions"]
    assert all(action.get("evidence_level") for action in plan["actions"])
    rendered = render_optimization_plan(plan)
    assert "Evidence:" in rendered
